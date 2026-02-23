import logging
import os
from typing import Any, Callable, Optional, Type, Union

import hydra
import numpy as np
import torch
import torch.nn as nn
from ase import Atoms
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.graph.compute import generate_graph
from fairchem.core.models.base import GraphModelMixin
from huggingface_hub import hf_hub_download
from torch_geometric.data import Batch, Data
from triforces.models.chemistry import stress_to_voigt_6, voigt_6_to_stress

from triforces.models.base import Model
from triforces.models.model_outputs import ModelOutputs
from triforces.models.normalization import NormalizationState, NormalizationType

logger = logging.getLogger("triforces")


class eSEN(Model):
    """eSEN interaction model wrapper.

    Parameters
    ----------
    targets : list[str], optional
        Targets predicted by the model.
    model : nn.Module, optional
        Pre-initialized model to use. If ``None``, loads from a pretrained checkpoint.
    model_type : str, optional
        Pretrained model identifier (e.g., ``"esen_30m_oam"``).
    use_ema : bool, default=False
        Whether to wrap the model with EMA weights.
    use_normalizers : bool, default=True
        Whether to keep pretrained normalizers from checkpoints.
    device : str, default="cpu"
        Device to load the model on.
    hook_fns : dict[str, Callable], default={}
        Hook functions registered on modules.
    freeze_weights : list[str], default=[]
        Module paths whose parameters should be frozen.
    direct_forces : bool, default=False
        Whether forces are computed directly by the backbone.
    disable_forces : bool, default=False
        Whether to disable force prediction.
    disable_stress : bool, default=False
        Whether to disable stress prediction.
    **kwargs : Any
        Additional keyword arguments forwarded to model initialization.
    """

    def __init__(
        self,
        targets: Optional[list[str]] = None,
        model: Optional[Type[nn.Module]] = None,
        model_type: str | None = "esen_30m_oam",
        use_ema: bool = False,
        use_normalizers: bool = True,
        device: str = "cpu",
        hook_fns: dict[str, Callable] = {},
        freeze_weights: list[str] = [],
        direct_forces: bool = False,
        disable_forces: bool = False,
        disable_stress: bool = False,
        **kwargs,
    ):
        super().__init__(
            targets=targets, hook_fns=hook_fns, freeze_weights=freeze_weights
        )

        # Initialize model components
        if model is not None:
            self.model = model.model
            # Pre-trained checkpoints from FAIRChem have their own normalizers
            self.tasks = model.tasks
            if use_normalizers:
                # This is float64 otherwise
                for key in self.tasks:
                    self.tasks[key].normalizer = self.tasks[key].normalizer.float()
                    if self.tasks[key].element_references is not None:
                        self.tasks[key].element_references = self.tasks[
                            key
                        ].element_references.float()
            else:
                self.tasks = None

            self.backbone = self.model.module.backbone
            self.output_heads = self.model.module.output_heads
            self.hook_fns = {
                "model.module.backbone": eSEN.backbone_hook_fn,
                **hook_fns,
            }

        else:
            # Load pretrained model settings but not the weights
            # override the model config with the kwargs
            checkpoint = self.from_pretrained(
                model_name=model_type, device=device, return_checkpoint=True
            )
            # if kwargs:
            #     checkpoint.model_config = update_configs(
            #         checkpoint.model_config, kwargs
            #     )
            if "num_layers" in kwargs:
                checkpoint.model_config["backbone"]["num_layers"] = kwargs["num_layers"]
            if "edge_channels" in kwargs:
                checkpoint.model_config["backbone"]["edge_channels"] = kwargs[
                    "edge_channels"
                ]
            if "sphere_channels" in kwargs:
                checkpoint.model_config["backbone"]["sphere_channels"] = kwargs[
                    "sphere_channels"
                ]
            self.model = hydra.utils.instantiate(checkpoint.model_config)
            if use_ema:
                self.model = torch.optim.swa_utils.AveragedModel(model)
            # We handle normalization in the regular pipeline
            self.tasks = None

            self.backbone = self.model.backbone
            self.output_heads = self.model.output_heads
            self.hook_fns = {
                "model.backbone": eSEN.backbone_hook_fn,
                **hook_fns,
            }

            if "direct" in model_type:
                direct_forces = True

        self.direct_forces = direct_forces
        self.disable_forces = disable_forces
        self.disable_stress = disable_stress

        self.backbone.direct_forces = direct_forces
        self.backbone.regress_forces = not disable_forces
        self.backbone.regress_stress = not disable_stress

        self.requires_grad_for_inference = not direct_forces
        self.backbone.otf_graph = False

        self._post_init()

    @property
    def possible_targets(self):
        return ["energy", "forces", "stress"]

    @classmethod
    def get_model_name(cls):
        return "eSEN"

    def _compute_displacement(
        self,
        batch: Batch,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Compute displacement tensors for stress via autograd.

        Parameters
        ----------
        batch : Batch
            Batched graph with positions and cell tensors.

        Returns
        -------
        torch.Tensor
            Displacement tensor with shape ``(B, 3, 3)`` and ``requires_grad=True``.
        torch.Tensor
            Original cell tensor before strain application.

        Notes
        -----
        This is needed when using energy-conserving heads with direct eSEN models.
        The method mutates ``batch.pos`` and ``batch.cell`` in-place by applying a
        symmetric strain.
        """
        device = batch.pos.device
        dtype = batch.pos.dtype
        B = batch.ptr.size(0) - 1

        # Create displacement tensor (symmetric strain)
        displacement = torch.zeros((3, 3), dtype=dtype, device=device)
        displacement = displacement.view(-1, 3, 3).expand(B, 3, 3).clone()
        displacement.requires_grad = True

        # Make displacement symmetric
        symmetric_displacement = 0.5 * (displacement + displacement.transpose(-1, -2))

        # Ensure positions have gradients
        if batch.pos.requires_grad is False:
            batch.pos = batch.pos.detach().requires_grad_(True)

        # Apply strain to positions: pos_new = pos + pos @ symmetric_displacement
        batch_displacement = torch.index_select(symmetric_displacement, 0, batch.batch)
        batch.pos = batch.pos + torch.bmm(
            batch.pos.unsqueeze(-2), batch_displacement
        ).squeeze(-2)

        # Apply strain to cell: cell_new = cell + cell @ symmetric_displacement
        # Cell shape is (B, 3, 3) - keep it that way for FAIRChem compatibility
        orig_cell = batch.cell.clone()
        batch.cell = batch.cell + torch.bmm(batch.cell, symmetric_displacement)

        return displacement, orig_cell

    def forward(
        self,
        batch: Batch,
        training: bool = False,
        transform: Callable[..., Data] | None = None,
        skip_displacement: bool = False,
    ):
        # For energy-conserving heads, we need displacement for stress computation.
        # Direct eSEN models don't compute displacement internally, so we do it here.
        # This must be done BEFORE the forward pass so gradients flow through.
        # skip_displacement=True is used when the caller already computed it.
        displacement = None
        orig_cell = None
        if self.requires_grad_for_inference and not skip_displacement:
            # Check if backbone already handles displacement (conservative mode)
            backbone_handles_displacement = (
                hasattr(self.backbone, "regress_stress")
                and self.backbone.regress_stress
                and not getattr(self.backbone, "direct_forces", True)
            )
            if not backbone_handles_displacement:
                displacement, orig_cell = self._compute_displacement(batch)

        batch.batch_full = batch.batch
        batch.atomic_numbers_full = batch.atomic_numbers

        atomic_batch = AtomicData(
            pos=batch.pos,
            atomic_numbers=batch.atomic_numbers,
            cell=batch.cell,
            pbc=batch.pbc,
            natoms=batch.natoms,
            edge_index=batch.edge_index,
            cell_offsets=batch.cell_offsets,
            nedges=batch.get("nedges", None),
            charge=batch.get("charge", None),
            spin=batch.get("spin", None),
            fixed=batch.get("fixed", None),
            tags=batch.get("tags", None),
            energy=batch.get("energy", None),
            forces=batch.get("forces", None),
            stress=voigt_6_to_stress(batch.get("stress", None)),
            batch=batch.get("batch", None),
            dataset=batch.get("dataset", None),
            sid=[""] * batch.num_graphs,
        )

        outputs = self.model(atomic_batch)

        # TODO(Ramlaoui): This might be dangerous if
        # there are mixed datasets in batches
        dataset = batch.dataset[0]
        outputs = {
            key: outputs[f"{dataset}_{key}"][key]
            for key in self.targets
            if f"{dataset}_{key}" in outputs
        }

        if "stress" in outputs:
            outputs["stress"] = stress_to_voigt_6(outputs["stress"].reshape(-1, 3, 3))

        output_keys = list(outputs.keys())
        # We only go here for FAIRChem's pre-trained checkpoints, not ours
        if self.tasks is not None:
            for key in output_keys:
                task = self.tasks[f"{dataset}_{key}"]
                # At this stage outputs are all normalized, batch are all denormalized
                key_denormed = outputs[key]
                outputs[f"{key}_normalized"] = key_denormed

                device = key_denormed.device
                if task.element_references is not None:
                    task.element_references = task.element_references.to(device)

                if task.normalizer is not None:
                    task.normalizer.mean = task.normalizer.mean.to(device)
                    task.normalizer.rmsd = task.normalizer.rmsd.to(device)

                if key in batch and training:
                    # Normalize the batch key, but keep the denormalized version
                    batch[f"{key}_denormalized"] = batch[key]
                    if task.element_references is not None:
                        batch[key] = task.element_references.apply_refs(
                            batch, batch[key]
                        )
                    if task.normalizer is not None:
                        batch[key] = task.normalizer.norm(batch[key])

                if task.normalizer is not None:
                    key_denormed = task.normalizer.denorm(key_denormed)

                if task.element_references is not None:
                    # Denormalize the output key
                    key_denormed = task.element_references.undo_refs(
                        batch, key_denormed
                    )

                # In training mode, we want to return the denormalized outputs
                # In inference mode, we return the normalized outputs but
                # then we need to denormalize during loss computation for
                # pre-trained models.
                if training:
                    outputs[f"{key}_denormalized"] = key_denormed
                    outputs[f"is_normalized_{key}"] = True
                else:
                    outputs[key] = key_denormed
                    outputs[f"is_normalized_{key}"] = False

        outputs = self.add_hook_features(outputs)

        # Add displacement for energy-conserving heads (stress computation)
        # If we computed it locally (for direct models), use that; otherwise use hook's
        if displacement is not None:
            outputs["displacement"] = displacement
            outputs["pos"] = batch.pos  # Strained positions with gradients
            outputs["cell"] = batch.cell  # Strained cell

        return outputs

    def format_model_outputs(
        self, batch: Batch, outputs: dict[str, torch.Tensor], add_keys: list[str] = []
    ) -> ModelOutputs:
        # Convert stress to Voigt-6 if it's in (B, 3, 3) format
        # This handles stress from EnergyConservingHead
        if "stress" in outputs and outputs["stress"] is not None:
            stress = outputs["stress"]
            if stress.dim() == 3 and stress.shape[-2:] == (3, 3):
                outputs["stress"] = stress_to_voigt_6(stress)

        batch_size = batch.num_graphs if hasattr(batch, "num_graphs") else 1
        kwargs = {
            "batch": batch.batch if hasattr(batch, "batch") else None,
            "batch_size": batch_size,
            "ptr": batch.ptr if hasattr(batch, "ptr") else None,
            "attributes": {},
        }

        normalization_state = NormalizationState()

        if "energy" in outputs:
            kwargs["energy"] = outputs["energy"]
            if "is_normalized_energy" not in outputs or outputs["is_normalized_energy"]:
                normalization_state.add_transform(
                    NormalizationType.MODEL_OUTPUT,
                    params={"model": self.get_model_name(), "key": "energy"},
                )
        if "forces" in outputs:
            kwargs["forces"] = outputs["forces"]
            if "is_normalized_forces" not in outputs or outputs["is_normalized_forces"]:
                normalization_state.add_transform(
                    NormalizationType.MODEL_OUTPUT,
                    params={"model": self.get_model_name(), "key": "forces"},
                )

        for key, value in outputs.items():
            if key not in ["energy", "forces"]:
                kwargs["attributes"][key] = value
                if (
                    f"is_normalized_{key}" not in outputs
                    or outputs[f"is_normalized_{key}"]
                ):
                    normalization_state.add_transform(
                        NormalizationType.MODEL_OUTPUT,
                        params={"model": self.get_model_name(), "key": key},
                    )

        model_outputs = ModelOutputs(**kwargs)
        return self._format_model_outputs(
            batch, outputs, model_outputs, add_keys=add_keys
        )

    def disable_heads(self, disable_attributes: list[str] | None = None):
        super().disable_heads(disable_attributes)
        if "forces" in disable_attributes:
            self.disable_forces = True
            self.backbone.regress_forces = False
        if "stress" in disable_attributes:
            self.disable_stress = True
            self.backbone.regress_stress = False

        for key in self.output_heads:
            self.output_heads[key].regress_forces = self.backbone.regress_forces
            self.output_heads[key].regress_stress = self.backbone.regress_stress

        self.requires_grad_for_inference = (
            not self.disable_forces or not self.disable_stress
        ) and self.backbone.direct_forces

    def get_backbone_outputs(self, outputs: dict[str, torch.Tensor]):
        backbone_outputs = {}
        backbone_outputs["node_feats"] = outputs["node_feats"]
        if "node_feats_equivariant" in outputs:
            backbone_outputs["node_feats_equivariant"] = outputs[
                "node_feats_equivariant"
            ]

        # Include pos/displacement/cell for energy-conserving heads
        if "displacement" in outputs:
            backbone_outputs["displacement"] = outputs["displacement"]
        if "pos" in outputs:
            backbone_outputs["pos"] = outputs["pos"]
        if "cell" in outputs:
            backbone_outputs["cell"] = outputs["cell"]

        return backbone_outputs

    @staticmethod
    def backbone_hook_fn(m, input_embeddings, output_embeddings, hook_features):
        # node_embedding shape: [N, num_sh_coeffs, sphere_channels]
        # For lmax=2: num_sh_coeffs = 9 (1 + 3 + 5)
        # Index 0: l=0 (scalar)
        # Indices 1-3: l=1 (vector)
        # Indices 4-8: l=2 (rank-2 tensor)
        node_embedding = output_embeddings["node_embedding"]

        # l=0 scalars (invariant features)
        hook_features["node_feats"] = node_embedding[:, 0, :]  # [N, sphere_channels]

        # l=1 vectors for equivariant head
        # Shape: [N, 3, sphere_channels] -> [N, sphere_channels * 3]
        # Reshape to have vectors in e3nn format: [N, num_vectors * 3]
        esen_l1_vectors = node_embedding[:, 1:4, :]  # [N, 3, sphere_channels]
        # Transpose to [N, sphere_channels, 3] then flatten to [N, sphere_channels * 3]
        node_feats_l1 = esen_l1_vectors.transpose(1, 2).reshape(
            node_embedding.shape[0], -1
        )
        hook_features["node_feats_l1"] = node_feats_l1

        # Equivariant features: scalars (l=0) + vectors (l=1) concatenated
        # This matches the format used by the equivariant head.
        # irreps: "128x0e+128x1o" (sphere_channels scalars + sphere_channels vectors)
        hook_features["node_feats_equivariant"] = torch.cat(
            [hook_features["node_feats"], node_feats_l1], dim=-1
        )
        hook_features["displacement"] = output_embeddings["displacement"]

    def get_readout(self) -> tuple[nn.Module, int, Optional[nn.Module]]:
        identity = nn.Identity()  # Features are already invariant
        return (identity, self.backbone.sphere_channels, None)

    @staticmethod
    def get_default_transforms(
        model: Optional["eSEN"] = None,
        cutoff: float = 6.0,
        max_neighbors: int = 20,
        use_pbc: bool = True,
        enforce_max_neighbors_strictly: bool = True,
    ):
        """Return the default graph transform for eSEN.

        Parameters
        ----------
        model : eSEN, optional
            Model instance to extract parameters from.
        cutoff : float, default=6.0
            Cutoff radius for neighbor search.
        max_neighbors : int, default=20
            Maximum number of neighbors.
        use_pbc : bool, default=True
            Whether to use periodic boundary conditions.
        enforce_max_neighbors_strictly : bool, default=True
            Whether to enforce max neighbors strictly.

        Returns
        -------
        EsenGraph
            Graph builder configured for the model.
        """
        if model is not None and hasattr(model, "backbone"):
            cutoff = getattr(model.backbone, "cutoff", cutoff)
            max_neighbors = getattr(model.backbone, "max_neighbors", max_neighbors)
            use_pbc = getattr(model.backbone, "use_pbc", use_pbc)
            enforce_max_neighbors_strictly = getattr(
                model.backbone,
                "enforce_max_neighbors_strictly",
                enforce_max_neighbors_strictly,
            )

        return EsenGraph(
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            use_pbc=use_pbc,
            enforce_max_neighbors_strictly=enforce_max_neighbors_strictly,
        )

    @staticmethod
    def from_pretrained(
        model_name: str = "esen_30m_oam",
        device: Union[str, torch.device] = "cpu",
        return_checkpoint: bool = False,
        return_calculator: bool = False,
        disable_forces: bool = False,
        disable_stress: bool = False,
    ) -> Union["eSEN", Any]:
        existing_models = [
            "esen_30m_oam",
            "esen_md_direct_all",
            "esen_sm_direct_all",
            "esen_sm_conserving_all",
        ]
        assert model_name in existing_models, (
            f"Model {model_name} not found, available models: {existing_models}"
        )

        if model_name == "esen_30m_oam":
            default_repo_id = "fairchem/OMAT24"
        else:
            default_repo_id = "facebook/OMol25"
            model_name = f"checkpoints/{model_name}"

        model_path = None
        try:
            hf_model_repo_id = os.environ.get("HF_MODEL_REPO_ID", default_repo_id)
            hf_model_path = os.environ.get("HF_MODEL_PATH", f"{model_name}.pt")
            model_path = hf_hub_download(
                repo_id=hf_model_repo_id, filename=hf_model_path
            )
        except Exception as e:
            logger.error(f"Error downloading model: {e}")

        if model_path is None:
            raise ValueError("Model path not found")

        # Convert torch.device to string if needed
        if isinstance(device, torch.device):
            device = str(device)

        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit

        predict_unit = load_predict_unit(
            path=model_path,
            inference_settings="default",
            overrides={"backbone": {"always_use_pbc": False}},
            device=device,
        )
        predict_unit.move_to_device()

        if return_checkpoint:
            return torch.load(model_path, map_location="cpu", weights_only=False)
        elif return_calculator:
            return FAIRChemCalculator(predict_unit)
        else:
            return eSEN(
                model=predict_unit,
                disable_forces=disable_forces,
                disable_stress=disable_stress,
            )


def _atoms_to_disconnected_graph(atoms: Atoms, **kwargs) -> Data:
    pos = torch.as_tensor(atoms.get_positions(wrap=True), dtype=torch.float32)
    atomic_numbers = torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    cell = torch.as_tensor(np.asarray(atoms.cell.array), dtype=torch.float32).view(
        1, 3, 3
    )
    tags = torch.as_tensor(atoms.get_tags(), dtype=torch.long)
    fixed = kwargs.pop("fixed", np.zeros(pos.shape[0], dtype=bool))
    fixed = torch.as_tensor(np.asarray(fixed).astype(int), dtype=torch.long)
    pbc = torch.as_tensor(np.asarray(atoms.pbc), dtype=torch.bool)

    return Data(
        atomic_numbers=atomic_numbers,
        pos=pos,
        cell=cell,
        natoms=len(atoms),
        tags=tags,
        fixed=fixed,
        pbc=pbc,
        **kwargs,
    )


class EsenGraph:
    """Graph builder using FAIRChem ``GraphModelMixin`` for eSEN.

    Parameters
    ----------
    cutoff : float, default=6.0
        Cutoff radius for neighbor search.
    max_neighbors : int, default=20
        Maximum number of neighbors.
    use_pbc : bool, default=True
        Whether to use periodic boundary conditions.
    enforce_max_neighbors_strictly : bool, default=True
        Whether to enforce max neighbors strictly.
    use_pbc_single : bool, default=False
        Whether to use single PBC handling in FAIRChem.
    """

    def __init__(
        self,
        cutoff: float = 6.0,
        max_neighbors: int = 20,
        use_pbc: bool = True,
        enforce_max_neighbors_strictly: bool = True,
        use_pbc_single: bool = False,
    ):
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.radius = cutoff
        self.max_num_neighbors = max_neighbors
        self.use_pbc = use_pbc
        self.enforce_max_neighbors_strictly = enforce_max_neighbors_strictly
        self.use_pbc_single = use_pbc_single
        self.graph_mixin = GraphModelMixin()
        self.graph_mixin.use_pbc = use_pbc
        self.graph_mixin.use_pbc_single = use_pbc_single

        self.converter_attrs = [
            "edge_index",
            "edge_distance",
            "edge_distance_vec",
            "cell_offsets",
            "offset_distances",
            "neighbors",
        ]

    def __call__(self, atoms: Atoms, **kwargs) -> Data:
        data = _atoms_to_disconnected_graph(atoms, **kwargs)

        cutoff = getattr(self, "radius", self.cutoff)
        max_neighbors = getattr(self, "max_num_neighbors", self.max_neighbors)

        out = self.graph_mixin.generate_graph(
            Batch.from_data_list([data]),
            cutoff,
            max_neighbors,
            use_pbc=self.use_pbc,
            otf_graph=True,
            enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
        )

        data.edge_index = out.edge_index
        data.edge_distance = out.edge_distance
        data.edge_distance_vec = out.edge_distance_vec
        data.cell_offsets = out.cell_offsets
        data.offset_distances = out.offset_distances
        data.neighbors = out.neighbors
        data.node_offset = out.node_offset
        data.batch_full = out.batch_full
        data.atomic_numbers_full = out.atomic_numbers_full
        data.nedges = out.edge_index.shape[1]

        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(cutoff={self.cutoff}, "
            f"max_neighbors={self.max_neighbors})"
        )


class UMAGraph:
    """Build UMA/eSEN-compatible graphs from ASE ``Atoms``.

    Parameters
    ----------
    radius : float, default=6.0
        Cutoff radius for neighbor search.
    max_num_neighbors : int, default=300
        Maximum number of neighbors.
    enforce_max_neighbors_strictly : bool, default=False
        Whether to enforce max neighbors strictly.
    radius_pbc_version : int, default=1
        Radius-PBC version for FAIRChem graph generation.
    dataset : str, default="omat"
        Dataset name used in the graph metadata.
    """

    def __init__(
        self,
        radius: float = 6.0,
        max_num_neighbors: int = 300,
        enforce_max_neighbors_strictly: bool = False,
        radius_pbc_version: int = 1,
        dataset: str = "omat",
    ):
        self.radius = radius
        self.max_num_neighbors = max_num_neighbors
        self.enforce_max_neighbors_strictly = enforce_max_neighbors_strictly
        self.radius_pbc_version = radius_pbc_version
        self.dataset = dataset

        self.converter_attrs = [
            "edge_index",
            "edge_distance",
            "edge_distance_vec",
            "cell_offsets",
            "offset_distances",
            "neighbors",
        ]

    def __call__(self, atoms: Atoms, **kwargs) -> Data:
        sid = kwargs.pop("sid", None)
        data_dict = AtomicData.from_ase(
            atoms, sid=sid, r_data_keys=["charge", "spin", "composition", "data_id"]
        )
        data_dict["sid"] = sid

        graph_dict = generate_graph(
            data_dict,
            cutoff=self.radius,
            max_neighbors=self.max_num_neighbors,
            enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
            radius_pbc_version=self.radius_pbc_version,
            pbc=torch.as_tensor(atoms.get_pbc(), dtype=torch.bool).unsqueeze(0),
        )

        atomic_data_dict = data_dict.to_dict()
        del atomic_data_dict["edge_index"]
        del atomic_data_dict["cell_offsets"]
        atomic_data_dict["nedges"] = graph_dict["edge_index"].shape[1]
        if "energy" in atomic_data_dict:
            del atomic_data_dict["energy"]
        if "forces" in atomic_data_dict:
            del atomic_data_dict["forces"]
        if "stress" in atomic_data_dict:
            del atomic_data_dict["stress"]

        collide = set(kwargs.keys()) & (
            set(atomic_data_dict.keys()) | set(graph_dict.keys())
        )
        for key in collide:
            kwargs.pop(key)

        data = Data(**{**atomic_data_dict, **graph_dict, **kwargs})
        data.dataset = self.dataset
        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(radius={self.radius}, "
            f"max_num_neighbors={self.max_num_neighbors}, dataset={self.dataset})"
        )


def esen_graph(
    *,
    radius: float = 6.0,
    max_num_neighbors: int = 300,
    enforce_max_neighbors_strictly: bool = False,
    radius_pbc_version: int = 1,
    dataset: str = "omat",
):
    """Return the UMA graph transform used for eSEN collation.

    Parameters
    ----------
    radius : float, default=6.0
        Cutoff radius for neighbor search.
    max_num_neighbors : int, default=300
        Maximum number of neighbors.
    enforce_max_neighbors_strictly : bool, default=False
        Whether to enforce max neighbors strictly.
    radius_pbc_version : int, default=1
        Radius-PBC version for FAIRChem graph generation.
    dataset : str, default="omat"
        Dataset name used in the graph metadata.

    Returns
    -------
    UMAGraph
        Graph builder instance.
    """
    return UMAGraph(
        radius=radius,
        max_num_neighbors=max_num_neighbors,
        enforce_max_neighbors_strictly=enforce_max_neighbors_strictly,
        radius_pbc_version=radius_pbc_version,
        dataset=dataset,
    )


def uma_graph(
    *,
    radius: float = 6.0,
    max_num_neighbors: int = 300,
    enforce_max_neighbors_strictly: bool = False,
    radius_pbc_version: int = 1,
    dataset: str = "omat",
):
    """Alias for UMA graph transforms.

    Parameters
    ----------
    radius : float, default=6.0
        Cutoff radius for neighbor search.
    max_num_neighbors : int, default=300
        Maximum number of neighbors.
    enforce_max_neighbors_strictly : bool, default=False
        Whether to enforce max neighbors strictly.
    radius_pbc_version : int, default=1
        Radius-PBC version for FAIRChem graph generation.
    dataset : str, default="omat"
        Dataset name used in the graph metadata.

    Returns
    -------
    UMAGraph
        Graph builder instance.
    """
    return esen_graph(
        radius=radius,
        max_num_neighbors=max_num_neighbors,
        enforce_max_neighbors_strictly=enforce_max_neighbors_strictly,
        radius_pbc_version=radius_pbc_version,
        dataset=dataset,
    )
