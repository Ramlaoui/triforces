from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from omegaconf import OmegaConf
from torch_geometric.data import Data

from triforces.data.asedb_dataset import ASEDBDataset
from triforces.models.outputs import BackboneOutputs
from triforces.train import run


def _write_asedb(path: Path, *, n_rows: int, with_calculator: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(str(path)) as db:
        for i in range(n_rows):
            atoms = Atoms(
                symbols=["Si", "Si"],
                positions=[[0.0 + 0.1 * i, 0.0, 0.0], [1.35 + 0.1 * i, 1.35, 1.35]],
                cell=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
                pbc=True,
            )
            if with_calculator:
                atoms.calc = SinglePointCalculator(
                    atoms,
                    energy=-1.0 - 0.1 * i,
                    forces=np.zeros((len(atoms), 3), dtype=np.float32),
                    stress=np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float32),
                )
            db.write(atoms, data={"sample_id": i, "tag": f"s{i}"})


def test_asedb_dataset_reads_targets_and_extract_keys(tmp_path: Path):
    db_path = tmp_path / "single.db"
    _write_asedb(db_path, n_rows=1, with_calculator=True)

    dataset = ASEDBDataset(
        path=db_path,
        add_targets=["energy", "energy_per_atom", "forces", "stress"],
        extract_keys=["sample_id"],
    )
    sample = dataset[0]

    assert sample.pair_id == 0
    assert sample.atoms.info["energy"] == pytest.approx(-1.0)
    assert sample.atoms.info["energy_per_atom"] == pytest.approx(-0.5)
    assert tuple(sample.atoms.arrays["forces"].shape) == (2, 3)
    assert tuple(sample.atoms.info["stress"].shape) == (6,)
    assert sample.atoms.info["sample_id"] == 0


def test_asedb_dataset_directory_and_node_counts(tmp_path: Path):
    db_dir = tmp_path / "dbs"
    _write_asedb(db_dir / "part_000.db", n_rows=1, with_calculator=True)
    _write_asedb(db_dir / "part_001.db", n_rows=1, with_calculator=True)

    dataset = ASEDBDataset(path=db_dir)
    node_counts = dataset.get_node_counts()

    assert len(dataset) == 2
    assert dataset[1].pair_id == 1
    assert node_counts.tolist() == [2, 2]


def test_asedb_dataset_missing_extract_key_raises(tmp_path: Path):
    db_path = tmp_path / "missing_key.db"
    _write_asedb(db_path, n_rows=1, with_calculator=True)

    dataset = ASEDBDataset(path=db_path, extract_keys=["does_not_exist"])
    with pytest.raises(KeyError, match="does_not_exist"):
        _ = dataset[0]


def test_asedb_dataset_missing_target_raises(tmp_path: Path):
    db_path = tmp_path / "no_calc.db"
    _write_asedb(db_path, n_rows=1, with_calculator=False)

    dataset = ASEDBDataset(path=db_path, add_targets=["energy"])
    with pytest.raises(ValueError, match="Target 'energy'"):
        _ = dataset[0]


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


class _NoEdgeGraph:
    def __call__(self, atoms: Atoms, **kwargs) -> Data:
        z = torch.as_tensor(atoms.numbers, dtype=torch.long)
        pos = torch.as_tensor(np.asarray(atoms.positions), dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        return Data(z=z, atomic_numbers=z, pos=pos, edge_index=edge_index, **kwargs)


def test_train_run_with_asedb_dataset(tmp_path: Path):
    db_path = tmp_path / "train.db"
    _write_asedb(db_path, n_rows=2, with_calculator=True)

    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "dataset": {
                "_target_": "triforces.data.asedb_dataset.ASEDBDataset",
                "path": str(db_path),
                "add_targets": [],
                "extract_keys": [],
                "keep_db_open": True,
            },
            "collate": {
                "_target_": "triforces.data.pyg_collate",
                "_partial_": True,
                "graph": {
                    "_target_": "tests.test_asedb_dataset._NoEdgeGraph",
                },
            },
            "model": {
                "_target_": "triforces.models.adapter_model.AdapterModel",
                "backbone": {
                    "_target_": "triforces.models.triforces.TriForcesModel",
                    "interaction": {
                        "_target_": "tests.test_asedb_dataset._DummyInteractionBackbone",
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
