from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import triforces.data.atompack_dataset as atompack_module
from triforces.data.atompack_dataset import AtompackDataset


@dataclass
class FakeMolecule:
    atomic_numbers: np.ndarray
    positions: np.ndarray
    energy: float | None = None
    forces: np.ndarray | None = None
    stress: np.ndarray | None = None
    cell: np.ndarray | None = None
    pbc: tuple[bool, bool, bool] | None = None
    props: dict | None = None

    def has_property(self, key: str) -> bool:
        return self.props is not None and key in self.props

    def get_property(self, key: str):
        return self.props[key]


def _fake_atompack(monkeypatch: pytest.MonkeyPatch, mapping: dict[str, list]) -> dict:
    calls = {"open": [], "open_mmap": []}

    class FakeDatabase:
        @staticmethod
        def open(path: str):
            calls["open"].append(path)
            return list(mapping[path])

        @staticmethod
        def open_mmap(path: str):
            calls["open_mmap"].append(path)
            return list(mapping[path])

    class FakeAtompack:
        Database = FakeDatabase

    monkeypatch.setattr(atompack_module, "atompack", FakeAtompack)
    return calls


def test_atompack_dataset_reads_targets_and_properties(monkeypatch, tmp_path: Path):
    shard = tmp_path / "data.atp"
    shard.touch()

    molecule = FakeMolecule(
        atomic_numbers=np.array([1, 8], dtype=np.uint8),
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        energy=-5.0,
        forces=np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]], dtype=np.float32),
        stress=np.eye(3, dtype=np.float32),
        cell=np.eye(3, dtype=np.float32) * 5.0,
        pbc=(True, True, False),
        props={"charge": 1, "spin": 2, "sample_id": 7},
    )

    calls = _fake_atompack(monkeypatch, {str(shard): [molecule]})
    dataset = AtompackDataset(
        path=shard,
        use_mmap=False,
        add_targets=["energy", "energy_per_atom", "forces", "stress"],
        extract_keys=["sample_id"],
    )

    sample = dataset[0]

    assert sample.pair_id == 0
    assert tuple(sample.atoms.numbers.tolist()) == (1, 8)
    assert sample.atoms.info["energy"] == pytest.approx(-5.0)
    assert sample.atoms.info["energy_per_atom"] == pytest.approx(-2.5)
    assert tuple(sample.atoms.arrays["forces"].shape) == (2, 3)
    assert tuple(sample.atoms.info["stress"].shape) == (3, 3)
    assert sample.atoms.info["charge"] == pytest.approx(1.0)
    assert sample.atoms.info["spin"] == pytest.approx(2.0)
    assert sample.atoms.info["sample_id"] == pytest.approx(7.0)
    assert sample.atoms.pbc.tolist() == [True, True, False]
    assert calls["open"] and not calls["open_mmap"]


def test_atompack_dataset_can_read_multiple_shards_with_mmap(monkeypatch, tmp_path: Path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_0 = shard_dir / "part_000.atp"
    shard_1 = shard_dir / "part_001.atp"
    shard_0.touch()
    shard_1.touch()

    molecule_0 = FakeMolecule(
        atomic_numbers=np.array([1], dtype=np.uint8),
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
    )
    molecule_1 = FakeMolecule(
        atomic_numbers=np.array([8], dtype=np.uint8),
        positions=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    )

    calls = _fake_atompack(
        monkeypatch,
        {
            str(shard_0): [molecule_0],
            str(shard_1): [molecule_1],
        },
    )
    dataset = AtompackDataset(path=shard_dir, use_mmap=True)

    assert len(dataset) == 2
    assert dataset[1].pair_id == 1
    assert dataset[1].atoms.numbers.tolist() == [8]
    assert calls["open_mmap"] and not calls["open"]


def test_atompack_dataset_raises_for_missing_requested_target(monkeypatch, tmp_path: Path):
    shard = tmp_path / "missing_energy.atp"
    shard.touch()
    molecule = FakeMolecule(
        atomic_numbers=np.array([1], dtype=np.uint8),
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
    )
    _fake_atompack(monkeypatch, {str(shard): [molecule]})

    dataset = AtompackDataset(path=shard, add_targets=["energy"])
    with pytest.raises(ValueError, match="Target 'energy'"):
        _ = dataset[0]
