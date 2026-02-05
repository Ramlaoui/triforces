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
    assert metrics["total_loss"] == pytest.approx(7.0)


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
