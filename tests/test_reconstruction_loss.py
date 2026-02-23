import numpy as np
import pytest
import torch
import torch.nn as nn
from ase import Atoms
from torch_geometric.data import Batch

from triforces.data.ase_contrastive import AtomsSample, atoms_sample_to_pyg_data
from triforces.losses import ReconstructionLoss


class DummyBaseLoss(nn.Module):
    def forward(self, data, preds, step=0):
        return preds["base"].sum(), {"dummy_metric": 1.0}


class ZeroBaseLoss(nn.Module):
    def forward(self, data, preds, step=0):
        return torch.tensor(0.0), {}


class CountingBaseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, data, preds, step=0):
        self.calls += 1
        return preds["base"].sum(), {"dummy_metric": 1.0}


def test_reconstruction_loss_adds_noise_loss():
    loss_fn = ReconstructionLoss(DummyBaseLoss(), noise_weight=2.0)

    preds = {
        "base": torch.ones(1),
        "noise_displacement": torch.zeros((3, 3)),
    }
    data = type("Data", (), {"noise_displacement": torch.ones((3, 3))})()

    loss, metrics = loss_fn(data, preds)

    assert loss.item() == pytest.approx(7.0)
    assert metrics["loss/base"] == pytest.approx(1.0)
    assert metrics["loss/noise"] == pytest.approx(3.0)
    assert metrics["noise_rmse"] == pytest.approx(np.sqrt(3.0))
    assert metrics["noise_cosine_similarity"] == pytest.approx(0.0)
    assert metrics["total_loss"] == pytest.approx(7.0)


def test_reconstruction_loss_applies_base_weight():
    loss_fn = ReconstructionLoss(DummyBaseLoss(), base_weight=0.5, noise_weight=2.0)

    preds = {
        "base": torch.ones(1),
        "noise_displacement": torch.zeros((3, 3)),
    }
    data = type("Data", (), {"noise_displacement": torch.ones((3, 3))})()

    loss, metrics = loss_fn(data, preds)

    # base=1 -> weighted base=0.5; noise=3 with weight=2 -> 6.0; total=6.5
    assert loss.item() == pytest.approx(6.5)
    assert metrics["loss/base"] == pytest.approx(1.0)
    assert metrics["loss/base_weight"] == pytest.approx(0.5)
    assert metrics["loss/noise"] == pytest.approx(3.0)
    assert metrics["total_loss"] == pytest.approx(6.5)


def test_reconstruction_loss_skips_base_loss_when_weight_is_zero():
    base_loss = CountingBaseLoss()
    loss_fn = ReconstructionLoss(base_loss, base_weight=0.0, noise_weight=1.0)

    preds = {
        "base": torch.ones(1),
        "noise_displacement": torch.zeros((2, 3)),
    }
    data = type("Data", (), {"noise_displacement": torch.ones((2, 3))})()

    loss, metrics = loss_fn(data, preds)

    assert base_loss.calls == 0
    assert metrics["loss/base_weight"] == pytest.approx(0.0)
    assert "loss/base" not in metrics
    assert "loss/base_skipped" not in metrics
    assert "dummy_metric" not in metrics
    assert loss.item() == pytest.approx(metrics["loss/noise"])
    assert metrics["total_loss"] == pytest.approx(loss.item())


def test_reconstruction_loss_respects_noise_mask():
    loss_fn = ReconstructionLoss(ZeroBaseLoss(), noise_weight=1.0)

    preds = {"noise_displacement": torch.zeros((3, 3))}
    target = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    mask = torch.tensor([True, False, True])
    data = type("Data", (), {"noise_displacement": target, "noise_mask": mask})()

    loss, metrics = loss_fn(data, preds)

    # Per-node squared error sums: [1, 4, 16] -> mask keeps [1, 16] -> mean = 8.5
    assert loss.item() == pytest.approx(8.5)
    assert metrics["loss/noise"] == pytest.approx(8.5)
    assert metrics["noise_rmse"] == pytest.approx(np.sqrt(8.5))
    assert metrics["noise_cosine_similarity"] == pytest.approx(0.0)


def test_reconstruction_loss_logs_noise_cosine_similarity():
    loss_fn = ReconstructionLoss(ZeroBaseLoss(), noise_weight=1.0)

    preds = {
        "noise_displacement": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        )
    }
    target = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [2.0, 2.0, 0.0]])
    mask = torch.tensor([True, True, False])
    data = type("Data", (), {"noise_displacement": target, "noise_mask": mask})()

    _, metrics = loss_fn(data, preds)

    # Cosine similarities on masked nodes are [1, -1] -> mean = 0
    assert metrics["noise_cosine_similarity"] == pytest.approx(0.0)


def test_reconstruction_loss_adds_atom_type_loss():
    loss_fn = ReconstructionLoss(
        ZeroBaseLoss(),
        noise_weight=0.0,
        atom_type_weight=1.0,
    )

    logits = torch.zeros((4, 6), dtype=torch.float32)
    logits[0, 1] = 8.0
    logits[2, 4] = 8.0
    preds = {"atom_type_logits": logits}
    data = type(
        "Data",
        (),
        {
            "original_numbers": torch.tensor([1, 2, 4, 3], dtype=torch.long),
            "atom_mask": torch.tensor([True, False, True, False]),
        },
    )()

    loss, metrics = loss_fn(data, preds)

    assert loss.item() == pytest.approx(metrics["loss/atom_type"])
    assert metrics["atom_type_accuracy_masked"] == pytest.approx(1.0)
    assert metrics["n_masked_atoms"] == pytest.approx(2.0)


def test_pyg_collate_tracks_noise_displacement():
    atoms1 = Atoms(numbers=[1, 1], positions=[[0, 0, 0], [1, 0, 0]])
    atoms1.arrays["noise_displacement"] = np.array(
        [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32
    )
    atoms2 = Atoms(numbers=[1, 1], positions=[[0, 0, 1], [1, 0, 1]])

    batch = Batch.from_data_list(
        [
            atoms_sample_to_pyg_data(AtomsSample(atoms=atoms1, pair_id=0)),
            atoms_sample_to_pyg_data(AtomsSample(atoms=atoms2, pair_id=1)),
        ]
    )

    assert batch.noise_displacement is not None
    assert batch.noise_displacement.shape == (4, 3)
    assert batch.noise_mask is not None
    assert batch.noise_mask.shape == (4,)
    assert batch.noise_mask[:2].all()
    assert (~batch.noise_mask[2:]).all()
