from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from ase import Atoms
from ase.io import write
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from triforces.data.ase_contrastive import AtomsSample
from triforces.models.outputs import BackboneOutputs
from triforces.train import run


def _write_cif(path: Path, *, shift: float) -> None:
    atoms = Atoms(
        symbols=["Si", "Si"],
        positions=[[0.0 + shift, 0.0, 0.0], [1.35 + shift, 1.35, 1.35]],
        cell=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        pbc=True,
    )
    write(path.as_posix(), atoms)


class _DummyInteractionBackbone(nn.Module):
    def __init__(self, embed_dim: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(119, embed_dim)

    def forward(self, batch, training: bool = False, transform=None):
        node_feats = self.embed(batch.z)
        num_graphs = batch.num_graphs
        graph_feats = node_feats.new_zeros((num_graphs, node_feats.size(-1)))
        graph_feats.index_add_(0, batch.batch, node_feats)
        count = torch.bincount(batch.batch, minlength=num_graphs).clamp_min(1)
        graph_feats = graph_feats / count.to(graph_feats.dtype).unsqueeze(1)
        return BackboneOutputs(node_feats=node_feats, graph_feats=graph_feats)


class _ToySupervisedDataset(Dataset[AtomsSample]):
    def __init__(self) -> None:
        self._atoms = [
            Atoms(
                symbols=["Si", "Si"],
                positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
                cell=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
                pbc=True,
            ),
            Atoms(
                symbols=["Si", "Si"],
                positions=[[0.1, 0.0, 0.0], [1.3, 0.0, 0.0]],
                cell=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
                pbc=True,
            ),
        ]
        for i, atoms in enumerate(self._atoms):
            atoms.info["energy"] = float(i + 1)

    def __len__(self) -> int:
        return len(self._atoms)

    def __getitem__(self, idx: int) -> AtomsSample:
        return AtomsSample(atoms=self._atoms[int(idx)].copy(), pair_id=int(idx))


class _ToySupervisedForcesDataset(Dataset[AtomsSample]):
    def __init__(self) -> None:
        self._atoms = [
            Atoms(
                symbols=["Si", "Si"],
                positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
                cell=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
                pbc=True,
            ),
            Atoms(
                symbols=["Si", "Si"],
                positions=[[0.1, 0.0, 0.0], [1.3, 0.0, 0.0]],
                cell=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
                pbc=True,
            ),
        ]
        for i, atoms in enumerate(self._atoms):
            atoms.info["energy"] = float(i + 1)
            atoms.arrays["forces"] = np.zeros((len(atoms), 3), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._atoms)

    def __getitem__(self, idx: int) -> AtomsSample:
        return AtomsSample(atoms=self._atoms[int(idx)].copy(), pair_id=int(idx))


def test_train_run_with_local_cif_dataset(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch_cluster") is None:
        pytest.skip("torch-cluster is required for radius_graph in smoke training.")

    _write_cif(tmp_path / "sample_0.cif", shift=0.0)
    _write_cif(tmp_path / "sample_1.cif", shift=0.1)

    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "dataset": {
                "_target_": "triforces.data.ase_contrastive.CifFolderDataset",
                "root": str(tmp_path),
                "glob": "**/*.cif",
            },
            "contrastive": {
                "_target_": "triforces.datasets.ContrastiveDataset",
                "return_pairs": True,
                "n_augmentation_views": 2,
                "n_opposite_views": 0,
                "include_original": False,
                "apply_augmentations_prob": 0.0,
                "augmentations": {},
            },
            "collate": {
                "_target_": "triforces.data.pyg_collate",
                "_partial_": True,
                "graph": {
                    "_target_": "triforces.data.simple_graph",
                    "radius": 4.0,
                    "max_num_neighbors": 8,
                },
            },
            "model": {
                "_target_": "triforces.models.adapter_model.AdapterModel",
                "backbone": {
                    "_target_": "triforces.models.triforces.TriForcesModel",
                    "interaction": {
                        "_target_": "tests.test_train_standalone_smoke._DummyInteractionBackbone",
                        "embed_dim": 16,
                    },
                    "interaction_dim": 16,
                    "interaction_name": "dummy",
                    "enable_composition": False,
                    "enable_structural": False,
                    "use_final_mlp": False,
                },
                "heads": {
                    "proj": {
                        "_target_": "triforces.models.heads.ProjectionHead",
                        "input_dim": 16,
                        "node_projection_dim": 8,
                        "graph_projection_dim": 8,
                        "projection_hidden_dims": [16],
                        "use_batch_norm": False,
                        "dropout": 0.0,
                    }
                },
            },
            "loss": {
                "_target_": "triforces.losses.ContrastiveLoss",
                "temperature_graph": 0.1,
                "lambda_node": 0.0,
                "lambda_graph": 1.0,
            },
            "train": {
                "epochs": 1,
                "batch_size": 1,
                "lr": 1e-3,
                "tqdm": False,
                "log_every": 1,
            },
            "logger": {"enabled": False},
        }
    )

    assert run(cfg) == 0


def test_train_run_supervised_without_pair_collate() -> None:
    if importlib.util.find_spec("torch_cluster") is None:
        pytest.skip("torch-cluster is required for radius_graph in smoke training.")

    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "dataset": {
                "_target_": "tests.test_train_standalone_smoke._ToySupervisedDataset",
            },
            "collate": {
                "_target_": "triforces.data.pyg_collate",
                "_partial_": True,
                "contrastive": False,
                "graph": {
                    "_target_": "triforces.data.simple_graph",
                    "radius": 4.0,
                    "max_num_neighbors": 8,
                },
            },
            "model": {
                "_target_": "triforces.models.adapter_model.AdapterModel",
                "backbone": {
                    "_target_": "triforces.models.triforces.TriForcesModel",
                    "interaction": {
                        "_target_": "tests.test_train_standalone_smoke._DummyInteractionBackbone",
                        "embed_dim": 16,
                    },
                    "interaction_dim": 16,
                    "interaction_name": "dummy",
                    "enable_composition": False,
                    "enable_structural": False,
                    "use_final_mlp": False,
                },
                "heads": {
                    "pred": {
                        "_target_": "triforces.models.heads.DirectSupervisedHead",
                        "input_dim": 16,
                        "predict_energy": True,
                        "predict_forces": False,
                        "predict_stress": False,
                        "hidden_dims": [16],
                    }
                },
            },
            "loss": {
                "_target_": "triforces.losses.SupervisedLoss",
                "energy_weight": 1.0,
                "forces_weight": 0.0,
                "stress_weight": 0.0,
            },
            "train": {
                "epochs": 1,
                "batch_size": 2,
                "lr": 1e-3,
                "tqdm": False,
                "log_every": 1,
            },
            "logger": {"enabled": False},
        }
    )

    assert run(cfg) == 0


@pytest.mark.parametrize(
    ("head_target", "compute_displacement"),
    [
        ("triforces.models.heads.DirectSupervisedHead", False),
        ("triforces.models.heads.EnergyConservingHead", True),
    ],
)
def test_train_run_supervised_orb_heads(
    head_target: str, compute_displacement: bool
) -> None:
    if importlib.util.find_spec("orb_models") is None:
        pytest.skip("orb_models is required for ORB smoke training.")

    head_cfg = {
        "_target_": head_target,
        "_partial_": True,
        "predict_forces": True,
        "predict_stress": False,
        "hidden_dims": [64],
    }
    if head_target.endswith("DirectSupervisedHead"):
        head_cfg["predict_energy"] = True

    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "dataset": {
                "_target_": "tests.test_train_standalone_smoke._ToySupervisedForcesDataset",
            },
            "collate": {
                "_target_": "triforces.data.pyg_collate",
                "_partial_": True,
                "contrastive": False,
                "graph": {
                    "_target_": "triforces.models.interaction.orb.orb_graph",
                    "radius": 6.0,
                    "max_num_neighbors": 20,
                    "device": "cpu",
                },
            },
            "model": {
                "_target_": "triforces.models.adapter_model.AdapterModel",
                "backbone": {
                    "_target_": "triforces.models.triforces.TriForcesModel",
                    "interaction": {
                        "_target_": "triforces.models.interaction.orb.Orb",
                        "model_type": "orb-v3-direct",
                        "disable_forces": True,
                        "disable_stress": True,
                        "compute_displacement": compute_displacement,
                    },
                    "interaction_name": "orb",
                    "interaction_dim": None,
                    "enable_composition": False,
                    "enable_structural": False,
                    "use_final_mlp": False,
                },
                "heads": {
                    "pred": head_cfg,
                },
            },
            "loss": {
                "_target_": "triforces.losses.SupervisedLoss",
                "energy_weight": 1.0,
                "forces_weight": 1.0,
                "stress_weight": 0.0,
            },
            "train": {
                "epochs": 1,
                "batch_size": 2,
                "lr": 1e-3,
                "tqdm": False,
                "log_every": 1,
            },
            "logger": {"enabled": False},
        }
    )

    assert run(cfg) == 0
