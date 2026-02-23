from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from triforces.losses.supervised import SupervisedLoss


def test_supervised_loss_masks_missing_targets() -> None:
    preds = {
        "energy": torch.tensor([2.0, 5.0], dtype=torch.float32),
        "forces": torch.tensor(
            [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float32
        ),
    }
    data = SimpleNamespace(
        energy=torch.tensor([1.0, float("nan")], dtype=torch.float32),
        forces=torch.tensor(
            [[0.0, 0.0, 0.0], [float("nan"), float("nan"), float("nan")]],
            dtype=torch.float32,
        ),
    )
    loss_fn = SupervisedLoss(
        energy_weight=1.0,
        forces_weight=2.0,
        stress_weight=0.0,
        energy_loss="mse",
        forces_loss="mse",
    )
    loss, metrics = loss_fn(data, preds)

    # energy mse on first item: (2-1)^2 = 1
    # forces mse on first node: mean([(1-0)^2, 0, 0]) = 1/3
    expected = 1.0 + 2.0 * (1.0 / 3.0)
    assert torch.isclose(loss, torch.tensor(expected, dtype=loss.dtype), atol=1e-6)
    assert metrics["n_energy"] == 1.0
    assert metrics["n_forces"] == 1.0


def test_supervised_loss_handles_stress_matrix_and_voigt() -> None:
    preds = {
        "stress": torch.tensor([[1.0, 2.0, 3.0, 0.3, 0.2, 0.1]], dtype=torch.float32)
    }
    stress_matrix = torch.tensor(
        [[[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]]], dtype=torch.float32
    )
    data = SimpleNamespace(stress=stress_matrix)
    loss_fn = SupervisedLoss(energy_weight=0.0, forces_weight=0.0, stress_weight=1.0)
    loss, metrics = loss_fn(data, preds)

    assert torch.isclose(loss, torch.tensor(0.0, dtype=loss.dtype), atol=1e-7)
    assert metrics["n_stress"] == 1.0


def test_supervised_loss_returns_zero_when_no_targets() -> None:
    preds = {"energy": torch.tensor([1.0], dtype=torch.float32)}
    data = SimpleNamespace()
    loss_fn = SupervisedLoss(energy_weight=1.0, forces_weight=1.0, stress_weight=1.0)
    loss, metrics = loss_fn(data, preds)

    assert torch.isclose(loss, torch.tensor(0.0, dtype=loss.dtype))
    assert metrics["n_energy"] == 0.0
    assert metrics["n_forces"] == 0.0
    assert metrics["n_stress"] == 0.0


def test_supervised_loss_applies_energy_reference_and_standardization() -> None:
    preds = {
        "energy": torch.tensor([1.8, 2.0], dtype=torch.float32),
    }
    data = SimpleNamespace(
        energy=torch.tensor([1.6, 2.4], dtype=torch.float32),
        atomic_numbers=torch.tensor([1, 8, 1, 1], dtype=torch.long),
        batch=torch.tensor([0, 0, 1, 1], dtype=torch.long),
    )
    loss_fn = SupervisedLoss(
        energy_weight=1.0,
        forces_weight=0.0,
        stress_weight=0.0,
        energy_references={1: 0.1, 8: 0.5},
        standardization={"energy": {"mean": 1.0, "std": 2.0}},
    )
    loss, metrics = loss_fn(data, preds)

    # Base MSE: mean([(1.8-1.6)^2, (2.0-2.4)^2]) = 0.1
    # Standardization with std=2 scales error by 1/4 -> 0.025
    assert torch.isclose(loss, torch.tensor(0.025, dtype=loss.dtype), atol=1e-7)
    assert metrics["n_energy"] == 2.0


def test_supervised_loss_energy_reference_requires_batch_metadata() -> None:
    preds = {"energy": torch.tensor([1.0], dtype=torch.float32)}
    data = SimpleNamespace(energy=torch.tensor([1.0], dtype=torch.float32))
    loss_fn = SupervisedLoss(
        energy_weight=1.0,
        energy_references={1: 0.1},
    )

    with pytest.raises(ValueError, match="requires `atomic_numbers`"):
        loss_fn(data, preds)


def test_supervised_loss_checkpoint_state_roundtrip() -> None:
    loss_fn = SupervisedLoss(
        energy_weight=1.0,
        energy_references={1: 0.1, 8: 0.5},
        standardization={
            "energy": {"mean": 1.0, "std": 2.0},
            "forces": {"mean": 0.0, "std": 0.5},
        },
    )
    state = loss_fn.get_checkpoint_state()

    restored = SupervisedLoss(energy_weight=1.0)
    restored.load_checkpoint_state(state)
    restored_state = restored.get_checkpoint_state()

    assert "energy_references" in restored_state
    assert "standardization" in restored_state
    assert set(restored_state["standardization"].keys()) == {"energy", "forces"}
