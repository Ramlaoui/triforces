import logging
from typing import Callable

import orb_models.forcefield.pretrained as pretrained
import torch
import torch.nn as nn
from ase import Atoms
from orb_models.forcefield.atomic_system import SystemConfig, ase_atoms_to_atom_graphs
from orb_models.forcefield.base import AtomGraphs
from orb_models.forcefield.calculator import ORBCalculator
from torch_geometric.data import Batch, Data

from triforces.models.base import Model
from triforces.models.outputs import BackboneOutputs

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


class Orb(Model):
    triforces_graph_backend = "orb"

    def __init__(
        self,
        model: nn.Module | None = None,
        model_type: str = "orb-v2",
        device: str = "cpu",
        disable_forces: bool = False,
        disable_stress: bool = False,
        remove_torque: bool = True,
        compute_displacement: bool = True,
        targets: list[str] | None = None,
        hook_fns: dict[str, Callable] | None = None,
        freeze_weights: list[str] | None = None,
        **kwargs,
    ):
        merged_hook_fns = {"model.model": Orb.backbone_hook_fn}
        if hook_fns:
            merged_hook_fns.update(hook_fns)
        super().__init__(
            targets=targets,
            hook_fns=merged_hook_fns,
            freeze_weights=freeze_weights,
        )

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
        if disable_attributes is None:
            return
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
    ) -> BackboneOutputs:
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
        node_feats = out.get("node_feats")
        if node_feats is None:
            raise ValueError("ORB interaction output is missing `node_feats`.")
        graph_feats = out.get("graph_feats")
        if graph_feats is None:
            batch_idx = getattr(batch, "batch", None)
            if batch_idx is None:
                graph_feats = node_feats.mean(dim=0, keepdim=True)
            else:
                num_graphs = getattr(batch, "num_graphs", None)
                if num_graphs is None:
                    num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() else 0
                num_graphs = int(num_graphs)
                graph_feats = node_feats.new_zeros((num_graphs, node_feats.size(-1)))
                graph_feats.index_add_(0, batch_idx, node_feats)
                counts = torch.bincount(batch_idx, minlength=num_graphs).clamp_min(1)
                graph_feats = graph_feats / counts.to(graph_feats.dtype).unsqueeze(1)
        extras = {k: v for k, v in out.items() if k not in {"node_feats", "graph_feats"}}
        return BackboneOutputs(
            node_feats=node_feats,
            graph_feats=graph_feats,
            extras=extras,
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

    def get_readout(self) -> tuple[nn.Module, int, nn.Module | None]:
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

    triforces_graph_backend = "orb"

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
        atom_graphs_dict["pos"] = torch.as_tensor(
            atom_graphs_dict["positions"], dtype=torch.float32
        )
        atom_graphs_dict["z"] = torch.as_tensor(atoms.numbers, dtype=torch.long)
        atom_graphs_dict["node_features"] = list(atom_graphs.node_features.keys())
        atom_graphs_dict["edge_features"] = list(atom_graphs.edge_features.keys())
        atom_graphs_dict["system_features"] = list(atom_graphs.system_features.keys())
        atom_graphs_dict["natoms"] = len(atoms)

        edge_index = torch.stack([atom_graphs.senders, atom_graphs.receivers], axis=0)
        atom_graphs_dict["edge_index"] = edge_index
        edge_vec = torch.as_tensor(atom_graphs_dict["vectors"], dtype=torch.float32)
        edge_dist = edge_vec.norm(dim=-1)
        atom_graphs_dict["edge_vec"] = edge_vec
        atom_graphs_dict["edge_dist"] = edge_dist

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


orb_graph.triforces_graph_backend = "orb"
