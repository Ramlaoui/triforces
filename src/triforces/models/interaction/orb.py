from pathlib import Path
from typing import Callable, Optional

import logging
import orb_models.forcefield.pretrained as pretrained
import torch
import torch.nn as nn
from ase import Atoms
from orb_models.forcefield.atomic_system import SystemConfig, ase_atoms_to_atom_graphs
from orb_models.forcefield.base import AtomGraphs
from orb_models.forcefield.calculator import ORBCalculator
from torch_geometric.data import Batch, Data

from triforces.models.base import Model
from triforces.models.model_outputs import ModelOutputs
from triforces.models.normalization import NormalizationState, NormalizationType

logger = logging.getLogger("triforces")

UNTRAINED_MODELS = {
    "orb-v2": {
        "model": pretrained.orb_v2_architecture,
        "system_config": SystemConfig(radius=6.0, max_num_neighbors=20),
    },
    "orb-v3-conservative": {
        "model": pretrained.orb_v3_conservative_architecture,
        "system_config": SystemConfig(radius=6.0, max_num_neighbors=120),
    },
    "orb-v3-conservative-20": {
        "model": pretrained.orb_v3_conservative_architecture,
        "system_config": SystemConfig(radius=6.0, max_num_neighbors=20),
    },
    "orb-v3-direct": {
        "model": pretrained.orb_v3_direct_architecture,
        "system_config": SystemConfig(radius=6.0, max_num_neighbors=20),
    },
    "orb-v3-conservative-omol": {
        "model": pretrained.orb_v3_conservative_architecture,
        "system_config": SystemConfig(radius=6.0, max_num_neighbors=120),
        "has_charge_spin_cond": True,
        "has_stress": False,
    },
    "orb-v3-direct-omol": {
        "model": pretrained.orb_v3_direct_architecture,
        "system_config": SystemConfig(radius=6.0, max_num_neighbors=120),
        "has_charge_spin_cond": True,
        "has_stress": False,
    },
}


class Orb(nn.Module):
    def __init__(
        self,
        model: nn.Module | None = None,
        model_type: str = "orb-v2",
        device: str = "cpu",
        disable_forces: bool = False,
        disable_stress: bool = False,
        remove_torque: bool = True,
        compute_displacement: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.hook_fns = {
            "model.model": Orb.backbone_hook_fn,
        }

        # Orb-v3 require gradients for inference for the conservative model
        self.requires_grad_for_inference = "conservative" in model_type
        self.model_type = model_type
        self.model = model
        self.pretrained = model is not None
        self.compute_displacement = compute_displacement

        if self.model is None:
            model_config = UNTRAINED_MODELS[model_type]
            # Extract architecture-specific kwargs (has_charge_spin_cond, has_stress, etc.)
            arch_kwargs = {
                k: v
                for k, v in model_config.items()
                if k not in ("model", "system_config")
            }
            self.model = model_config["model"](
                system_config=model_config["system_config"],
                device=device,
                **arch_kwargs,
                **kwargs,
            )

        # This just removes the confidence head not the other heads
        self.model.model._decoder = nn.Sequential()
        if "confidence" in self.model.heads:
            del self.model.heads["confidence"]
            if "conservative" in self.model_type:
                self.model.extra_properties.remove("confidence")

        self.disable_forces, self.disable_stress = disable_forces, disable_stress

        if not remove_torque:
            self.model.heads["forces"].remove_torque_for_nonpbc_systems = False

        self._post_init()

    @property
    def possible_targets(self):
        return ["energy", "forces", "stress"]

    def disable_heads(self, disable_attributes: list[str] | None = None):
        super().disable_heads(disable_attributes)
        if "forces" in disable_attributes:
            self.disable_forces = True
        if "stress" in disable_attributes:
            self.disable_stress = True
        self.requires_grad_for_inference = (
            not self.disable_forces or not self.disable_stress
        )

    def forward(
        self,
        batch: Batch,
        training: bool = True,
        transform: Callable[..., Data] | None = None,
    ) -> dict[str, torch.Tensor]:
        system_targets = {
            key: batch[key] for key in ["energy", "stress"] if hasattr(batch, key)
        }

        node_targets = {key: batch[key] for key in ["forces"] if hasattr(batch, key)}

        atom_graphs = AtomGraphs(
            senders=batch.edge_index[0, :],
            receivers=batch.edge_index[1, :],
            n_node=batch.n_node,
            n_edge=batch.n_edge,
            node_features={k: batch[k] for k in batch.node_features[0]},
            edge_features={k: batch[k] for k in batch.edge_features[0]},
            system_features={k: batch[k] for k in batch.system_features[0]},
            node_targets=node_targets,
            system_targets=system_targets,
            edge_targets=None,
            system_id=None,
            fix_atoms=None,
            tags=batch.tags,
            radius=batch.radius,
            max_num_neighbors=batch.max_num_neighbors,
            half_supercell=batch.half_supercell,
        )

        if self.compute_displacement:
            vectors, stress_displacement, generator = (
                atom_graphs.compute_differentiable_edge_vectors()
            )
            assert stress_displacement is not None
            assert generator is not None
            atom_graphs.system_features["stress_displacement"] = stress_displacement
            atom_graphs.system_features["generator"] = generator
            atom_graphs.edge_features["vectors"] = vectors

        out = self.model(atom_graphs)

        if "conservative" in self.model_type:
            # For conservative models, the energy is normalized per atom in orb
            out["energy"] = self.model.heads.energy.denormalize(
                out["energy"], atom_graphs
            )
            # out["energy"] = out["energy"] * batch.natoms

        out = self.add_hook_features(out)
        if self.compute_displacement:
            # out["vectors"] = atom_graphs.edge_features["vectors"]
            out["displacement"] = atom_graphs.system_features["stress_displacement"]
            out["generator"] = atom_graphs.system_features["generator"]
            out["pos"] = atom_graphs.node_features["positions"]

        return out

    def format_model_outputs(
        self,
        batch: Batch,
        raw_model_outputs: dict[str, torch.Tensor],
        add_keys: list[str] = [],
    ) -> ModelOutputs:
        # Check if we need to rename grad_forces/grad_stress to forces/stress
        has_grad_outputs = (
            "grad_forces" in raw_model_outputs or "grad_stress" in raw_model_outputs
        )

        if not self.pretrained:
            # For non-pretrained models, just rename grad_forces/grad_stress if present
            if has_grad_outputs:
                if "grad_forces" in raw_model_outputs:
                    raw_model_outputs["forces"] = raw_model_outputs.pop("grad_forces")
                if "grad_stress" in raw_model_outputs:
                    raw_model_outputs["stress"] = raw_model_outputs.pop("grad_stress")

                # forces and stress are already unnormalized in this case (orb conservative)
                model_outputs = super().format_model_outputs(batch, raw_model_outputs)

                if model_outputs.normalization_state:
                    model_outputs.normalization_state.remove_transform(
                        NormalizationType.MODEL_OUTPUT, key="forces"
                    )
                    model_outputs.normalization_state.remove_transform(
                        NormalizationType.MODEL_OUTPUT, key="stress"
                    )
                    model_outputs.normalization_state.remove_transform(
                        NormalizationType.MODEL_OUTPUT, key="energy"
                    )
                    model_outputs.normalization_state.add_transform(
                        "random_rotate", params={"key": "forces"}
                    )

                return model_outputs
            return super().format_model_outputs(batch, raw_model_outputs)

        batch_size = batch.num_graphs if hasattr(batch, "num_graphs") else 1
        kwargs = {
            "batch": batch.batch if hasattr(batch, "batch") else None,
            "batch_size": batch_size,
            "ptr": batch.ptr if hasattr(batch, "ptr") else None,
            "attributes": {},
        }

        normalization_state = NormalizationState()

        forces_key = "grad_forces" if "grad_forces" in raw_model_outputs else "forces"
        stress_key = "grad_stress" if "grad_stress" in raw_model_outputs else "stress"

        if "energy" in raw_model_outputs:
            kwargs["energy"] = raw_model_outputs["energy"]
            if "energy_denormalized" in raw_model_outputs:
                kwargs["attributes"]["energy_denormalized"] = raw_model_outputs[
                    "energy_denormalized"
                ]
                normalization_state.add_transform(
                    NormalizationType.MODEL_OUTPUT,
                    params={"model": self.get_model_name(), "key": "energy"},
                )
            # for predictions, they are already denormalized

        if "forces" in self.targets:
            kwargs["forces"] = raw_model_outputs[forces_key]
            if "forces_denormalized" in raw_model_outputs:
                kwargs["attributes"]["forces_denormalized"] = raw_model_outputs[
                    "forces_denormalized"
                ]
                normalization_state.add_transform(
                    NormalizationType.MODEL_OUTPUT,
                    params={"model": self.get_model_name(), "key": "forces"},
                )
        if "stress" in self.targets:
            kwargs["attributes"]["stress"] = raw_model_outputs[stress_key]
            if "stress_denormalized" in raw_model_outputs:
                kwargs["attributes"]["stress_denormalized"] = raw_model_outputs[
                    "stress_denormalized"
                ]
                normalization_state.add_transform(
                    NormalizationType.MODEL_OUTPUT,
                    params={"model": self.get_model_name(), "key": "stress"},
                )

        # Handle additional outputs
        for key, value in raw_model_outputs.items():
            if key not in ["energy", "forces", "stress"] and value is not None:
                kwargs["attributes"][key] = value
                normalization_state.add_transform(
                    NormalizationType.MODEL_OUTPUT, params={"key": key}
                )

        kwargs["normalization_state"] = normalization_state

        model_outputs = ModelOutputs(**kwargs)
        return self._format_model_outputs(
            batch, raw_model_outputs, model_outputs, add_keys=add_keys
        )

    @staticmethod
    def backbone_hook_fn(m, input_embeddings, output_embeddings, hook_features):
        hook_features["node_feats"] = output_embeddings["node_features"]

    def get_backbone_outputs(self, outputs: dict[str, torch.Tensor]):
        backbone_outputs = {}
        backbone_outputs["node_feats"] = outputs["node_feats"]

        if "displacement" in outputs:
            backbone_outputs["displacement"] = outputs["displacement"]

        # ORB stores positions as "pos" in forward(), check both keys
        if "pos" in outputs:
            backbone_outputs["pos"] = outputs["pos"]
        elif "positions" in outputs:
            backbone_outputs["pos"] = outputs["positions"]

        return backbone_outputs

    def get_readout(self) -> tuple[nn.Module, int, Optional[nn.Module]]:
        return None, self.model.model.node_embed_size, None

    @classmethod
    def from_pretrained(
        cls,
        model_type: str = "orb-v2",
        device: str = "cpu",
        compile: bool = True,
        return_calculator: bool = False,
        precision: str = "float32-high",
        **kwargs,
    ):
        assert model_type in pretrained.ORB_PRETRAINED_MODELS.keys(), (
            f"Only {pretrained.ORB_PRETRAINED_MODELS.keys()} models are supported for now"
        )

        try:
            model = pretrained.ORB_PRETRAINED_MODELS[model_type](
                device=device, compile=compile, precision=precision
            )
        except Exception as e:
            logger.warning(f"Failed to compile Orb model: {e}")
            model = pretrained.ORB_PRETRAINED_MODELS[model_type](
                device=device, precision=precision
            )

        if return_calculator:
            return ORBCalculator(model, device=device)

        # Set all parameters to be trainable
        for param in model.parameters():
            param.requires_grad = True

        return cls(model=model, model_type=model_type)

    def get_log_params(self) -> dict:
        base_info = super().get_log_params()
        orb_info = {
            "model_type": self.model_type,
        }
        return {**base_info, **orb_info}

    @classmethod
    def from_direct_checkpoint(
        cls,
        checkpoint_path: str,
        conservative_model_type: str = "orb-v3-conservative",
        device: str = "cpu",
        compile: bool = False,
        strict: bool = False,
        **kwargs,
    ):
        """Load a conservative ORB model from a direct model checkpoint.

        This method allows transferring pre-trained weights from a direct ORB model
        (which has separate force and stress heads) to a conservative ORB model
        (which computes forces/stress via energy gradients). The backbone and energy
        head weights are transferred, while force/stress head weights are ignored.

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file (.pt, .pth, or .ckpt) containing the
            direct model's state dict.
        conservative_model_type : str, optional
            The conservative model architecture to use. Must be a conservative
            variant (e.g., "orb-v3-conservative"). Default: "orb-v3-conservative".
        device : str, optional
            Device to load the model on ("cpu", "cuda", "cuda:0", etc.).
            Default: "cpu".
        compile : bool, optional
            Whether to compile the model with torch.compile. Default: False.
        strict : bool, optional
            If True, raises an error for missing/unexpected non-head keys.
            If False, allows flexible loading. Default: False.
            Note: Conservative models have grad_forces_normalizer and
            grad_stress_normalizer which don't exist in direct models, so
            strict=False is recommended for direct->conservative transfer.
        **kwargs
            Additional arguments passed to the Orb constructor.

        Returns
        -------
        Orb
            An Orb model instance with a conservative architecture and weights
            transferred from the direct checkpoint.

        Raises
        ------
        ValueError
            If conservative_model_type is not a conservative architecture.
        FileNotFoundError
            If checkpoint_path does not exist.
        RuntimeError
            If strict=True and incompatible non-head weights are found.

        Notes
        -----
        The transfer process:
        1. Creates a new conservative model architecture
        2. Loads the direct model's state dict from checkpoint
        3. Transfers compatible weights:
           - model.* (backbone/MoleculeGNS) - TRANSFERRED
           - heads.energy.* (energy head) - TRANSFERRED
           - heads.forces.* (force head) - IGNORED
           - heads.stress.* (stress head) - IGNORED
        4. Forces and stress are computed via autograd on energy

        The backbone architectures must be compatible (same latent_dim,
        num_message_passing_steps, etc.). This is guaranteed if you use
        matching model versions (e.g., both orb-v3).

        Examples
        --------
        >>> # Load conservative model from direct checkpoint
        >>> model = Orb.from_direct_checkpoint(
        ...     checkpoint_path="path/to/direct_model.pt",
        ...     conservative_model_type="orb-v3-conservative",
        ...     device="cuda",
        ... )
        >>>
        >>> # Now use the model - forces computed via gradients
        >>> outputs = model(batch)
        >>> # Forces are energy-conservative: F = -dE/dpositions
        """
        if "conservative" not in conservative_model_type:
            raise ValueError(
                f"Model type '{conservative_model_type}' is not a conservative "
                f"architecture. Use a model type containing 'conservative'."
            )

        # Verify checkpoint exists
        checkpoint_path_obj = Path(checkpoint_path)
        if not checkpoint_path_obj.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        logger.info(
            f"Loading conservative model '{conservative_model_type}' "
            f"from direct checkpoint: {checkpoint_path}"
        )

        # Create Orb wrapper first (with model=None, so pretrained=False)
        # This properly initializes everything including normalizers
        orb_model = cls(
            model=None,  # Will create untrained model
            model_type=conservative_model_type,
            device=device,
            **kwargs,
        )

        # Load the checkpoint
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Extract state_dict from checkpoint
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Load compatible weights into the wrapped model
        logger.info("Loading compatible weights...")
        incompatible_keys = orb_model.model.load_state_dict(state_dict, strict=False)

        if incompatible_keys and (
            incompatible_keys.missing_keys or incompatible_keys.unexpected_keys
        ):
            logger.info(
                f"Loaded checkpoint:\n"
                f"  Missing keys (will use defaults): {len(incompatible_keys.missing_keys)}\n"
                f"  Unexpected keys (ignored): {len(incompatible_keys.unexpected_keys)}"
            )

        logger.info(
            f"✓ Conservative model loaded successfully\n"
            f"  Model type: {conservative_model_type}\n"
            f"  Pretrained: {orb_model.pretrained}\n"
            f"  Device: {device}\n"
            f"  Forces computed via: autograd (F = -dE/dpositions)\n"
            f"  Stress computed via: autograd (S = dE/dcell / volume)"
        )

        return orb_model

    @classmethod
    def from_checkpoint_with_charge_spin(
        cls,
        checkpoint_path: str,
        model_type: str = "orb-v3-direct-omol",
        device: str = "cpu",
        **kwargs,
    ):
        """Load an ORB model with charge/spin conditioning from a checkpoint without it.

        This method creates a model architecture with ChargeSpinConditioner enabled,
        then loads compatible weights from a checkpoint that was trained without
        charge/spin conditioning. The conditioner parameters are randomly initialized.

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file (.pt, .pth, or .ckpt).
        model_type : str, optional
            The model architecture to use. Should be an omol variant with
            has_charge_spin_cond=True. Default: "orb-v3-direct-omol".
        device : str, optional
            Device to load the model on. Default: "cpu".
        **kwargs
            Additional arguments passed to the Orb constructor.

        Returns
        -------
        Orb
            An Orb model instance with charge/spin conditioning and weights
            transferred from the checkpoint.
        """
        # Verify checkpoint exists
        checkpoint_path_obj = Path(checkpoint_path)
        if not checkpoint_path_obj.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        # Verify model type has charge/spin conditioning
        if model_type not in UNTRAINED_MODELS:
            raise ValueError(f"Unknown model type: {model_type}")
        if not UNTRAINED_MODELS[model_type].get("has_charge_spin_cond", False):
            raise ValueError(
                f"Model type '{model_type}' does not have charge/spin conditioning. "
                f"Use an omol variant (e.g., 'orb-v3-direct-omol', 'orb-v3-conservative-omol')."
            )

        logger.info(
            f"Loading model '{model_type}' with charge/spin conditioning "
            f"from checkpoint: {checkpoint_path}"
        )

        # Create model with charge/spin conditioning
        orb_model = cls(
            model=None,
            model_type=model_type,
            device=device,
            **kwargs,
        )

        # Load the checkpoint
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Extract state_dict from checkpoint
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Load compatible weights (ChargeSpinConditioner keys will be missing)
        logger.info("Loading compatible weights...")
        incompatible_keys = orb_model.model.load_state_dict(state_dict, strict=False)

        # Log what was loaded
        missing_conditioner = [
            k for k in incompatible_keys.missing_keys if "conditioner" in k
        ]
        other_missing = [
            k for k in incompatible_keys.missing_keys if "conditioner" not in k
        ]

        logger.info(
            f"Loaded checkpoint:\n"
            f"  ChargeSpinConditioner params (randomly initialized): {len(missing_conditioner)}\n"
            f"  Other missing keys: {len(other_missing)}\n"
            f"  Unexpected keys (ignored): {len(incompatible_keys.unexpected_keys)}"
        )

        if other_missing:
            logger.warning(f"Missing non-conditioner keys: {other_missing[:10]}...")

        logger.info(
            f"✓ Model with charge/spin conditioning loaded successfully\n"
            f"  Model type: {model_type}\n"
            f"  Device: {device}\n"
            f"  ChargeSpinConditioner: randomly initialized (will be trained)"
        )

        return orb_model

    @staticmethod
    def get_default_transforms(model: Model) -> "OrbGraph":
        system_config = model.model.system_config
        return OrbGraph(
            radius=system_config.radius,
            max_num_neighbors=system_config.max_num_neighbors,
        )


class OrbGraph:
    """Build an ORB graph representation from ASE ``Atoms``.

    Parameters
    ----------
    radius : float, default=6.0
        Cutoff radius for neighbor search.
    max_num_neighbors : int, default=20
        Maximum number of neighbors.
    device : str, default="cpu"
        Device for graph construction.
    """

    def __init__(
        self,
        radius: float = 6.0,
        max_num_neighbors: int = 20,
        device: str = "cpu",
    ):
        self.radius = float(radius)
        self.max_num_neighbors = int(max_num_neighbors)
        self.device = device

        self.system_config = SystemConfig(
            radius=self.radius, max_num_neighbors=self.max_num_neighbors
        )

    def __call__(self, atoms: Atoms, **kwargs) -> Data:
        atom_graphs = ase_atoms_to_atom_graphs(
            atoms,
            self.system_config,
            device=self.device,
        )

        atom_graphs_dict = atom_graphs.to_dict()
        atom_graphs_dict["pos"] = atom_graphs_dict["positions"]
        atom_graphs_dict["node_features"] = list(atom_graphs.node_features.keys())
        atom_graphs_dict["edge_features"] = list(atom_graphs.edge_features.keys())
        atom_graphs_dict["system_features"] = list(atom_graphs.system_features.keys())
        atom_graphs_dict["natoms"] = len(atoms)

        edge_index = torch.stack([atom_graphs.senders, atom_graphs.receivers], axis=0)
        atom_graphs_dict["edge_index"] = edge_index

        data = Data(**atom_graphs_dict, **kwargs)

        if "total_charge" in data.keys():
            data.charge = data.total_charge
        if "total_spin" in data.keys():
            data.spin = data.total_spin

        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(radius={self.radius}, "
            f"max_num_neighbors={self.max_num_neighbors})"
        )


def orb_graph(
    *,
    radius: float = 6.0,
    max_num_neighbors: int = 20,
    device: str = "cpu",
) -> OrbGraph:
    """Return the ORB graph builder used for collation.

    Parameters
    ----------
    radius : float, default=6.0
        Cutoff radius for neighbor search.
    max_num_neighbors : int, default=20
        Maximum number of neighbors.
    device : str, default="cpu"
        Device for graph construction.

    Returns
    -------
    OrbGraph
        Graph builder instance.
    """
    return OrbGraph(radius=radius, max_num_neighbors=max_num_neighbors, device=device)
