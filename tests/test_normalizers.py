from __future__ import annotations

import torch

from triforces.normalizers import EnergyReferenceNormalizer, StandardizationNormalizer


def test_energy_reference_normalizer_roundtrip() -> None:
    normalizer = EnergyReferenceNormalizer.from_state({1: 0.1, 8: 0.5})
    energy = torch.tensor([1.8, 2.5], dtype=torch.float32)
    atomic_numbers = torch.tensor([1, 8, 1, 1], dtype=torch.long)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    normalized = normalizer.normalize(
        energy, atomic_numbers=atomic_numbers, batch=batch
    )
    restored = normalizer.denormalize(
        normalized, atomic_numbers=atomic_numbers, batch=batch
    )

    expected_normalized = torch.tensor([1.2, 2.3], dtype=torch.float32)
    assert torch.allclose(normalized, expected_normalized, atol=1e-6)
    assert torch.allclose(restored, energy, atol=1e-6)


def test_standardization_normalizer_roundtrip() -> None:
    normalizer = StandardizationNormalizer.from_state({"mean": 2.0, "std": 4.0})
    values = torch.tensor([0.0, 2.0, 6.0], dtype=torch.float32)
    normalized = normalizer.normalize(values)
    restored = normalizer.denormalize(normalized)

    expected_normalized = torch.tensor([-0.5, 0.0, 1.0], dtype=torch.float32)
    assert torch.allclose(normalized, expected_normalized, atol=1e-6)
    assert torch.allclose(restored, values, atol=1e-6)
