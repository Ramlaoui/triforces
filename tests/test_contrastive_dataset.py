from __future__ import annotations

from dataclasses import dataclass

import pytest
from ase import Atoms
from torch.utils.data import Dataset

from triforces.datasets import ContrastiveDataset


@dataclass
class _Sample:
    atoms: Atoms
    pair_id: int


class _ToyAtomsDataset(Dataset[_Sample]):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> _Sample:
        atoms = Atoms(
            symbols=["Si", "Si"],
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            cell=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
            pbc=True,
        )
        return _Sample(atoms=atoms, pair_id=int(idx))


class _MarkAugmentation:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, atoms: Atoms) -> Atoms:
        out = atoms.copy()
        applied = list(out.info.get("applied", []))
        applied.append(self.name)
        out.info["applied"] = applied
        return out


def _applied_names(sample: _Sample) -> list[str]:
    return list(sample.atoms.info.get("applied", []))


def test_global_probability_can_disable_all_augmentations() -> None:
    dataset = ContrastiveDataset(
        dataset=_ToyAtomsDataset(),
        augmentations={
            "a": _MarkAugmentation("a"),
            "b": _MarkAugmentation("b"),
        },
        augmentation_probabilities={"a": 1.0, "b": 1.0},
        n_augmentation_views=1,
        apply_augmentations_prob=0.0,
        return_pairs=False,
    )

    sample = dataset[0]
    assert _applied_names(sample) == []


def test_uses_per_augmentation_probabilities_for_optional_augmentations() -> None:
    dataset = ContrastiveDataset(
        dataset=_ToyAtomsDataset(),
        augmentations={
            "a": _MarkAugmentation("a"),
            "b": _MarkAugmentation("b"),
        },
        augmentation_probabilities={"a": 1.0, "b": 0.0},
        n_augmentation_views=1,
        apply_augmentations_prob=1.0,
        return_pairs=False,
    )

    sample = dataset[0]
    assert _applied_names(sample) == ["a"]


def test_rejects_invalid_augmentation_probability_config() -> None:
    with pytest.raises(ValueError, match="unknown augmentation name"):
        ContrastiveDataset(
            dataset=_ToyAtomsDataset(),
            augmentations={"a": _MarkAugmentation("a")},
            augmentation_probabilities={"missing": 1.0},
        )

    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        ContrastiveDataset(
            dataset=_ToyAtomsDataset(),
            augmentations={"a": _MarkAugmentation("a")},
            augmentation_probabilities={"a": 1.5},
        )
