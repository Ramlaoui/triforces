"""Minimal LeMat-Bulk dataset wrapper using HuggingFace datasets."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from datasets import load_dataset
from torch.utils.data import Dataset

from triforces.utils.stress import stress_array_to_voigt_6

from .ase_contrastive import AtomsSample

DEFAULT_DATASET_NAME = "LeMaterial/LeMat-Bulk"
DEFAULT_CONFIG = "compatible_pbe"
DEFAULT_SPLIT = "train"
DEFAULT_METADATA_KEYS = (
    "immutable_id",
    "nsites",
    "functional",
    "database",
    "formula",
    "elements",
)


def lemat_item_to_ase(
    item: dict,
    *,
    add_targets: Sequence[str] | None = None,
    add_metadata: bool = True,
) -> Atoms:
    """Convert a LeMat-Bulk item to an ASE Atoms object.

    Parameters
    ----------
    item : dict
        Dataset row from HuggingFace LeMat-Bulk.
    add_targets : Sequence[str], optional
        Targets to attach: "energy", "energy_per_atom", "forces", "stress".
    add_metadata : bool, default=True
        Whether to copy common metadata into atoms.info.
    Returns
    -------
    Atoms
        ASE atoms object with optional targets and metadata attached.
    """
    add_targets = set(add_targets or [])

    sites = item["species_at_sites"]
    coords = item["cartesian_site_positions"]
    cell = item["lattice_vectors"]

    if sites and isinstance(sites[0], (int, np.integer)):
        atoms = Atoms(numbers=sites, positions=coords, cell=cell, pbc=True)
    else:
        atoms = Atoms(symbols=sites, positions=coords, cell=cell, pbc=True)

    if add_metadata:
        for key in DEFAULT_METADATA_KEYS:
            if key in item and item[key] is not None:
                atoms.info[key] = item[key]

    energy = None
    forces = None
    stress_voigt = None
    stress_tensor = None

    if "energy" in add_targets or "energy_per_atom" in add_targets:
        energy = item.get("energy")
        if energy is not None:
            atoms.info["energy"] = float(energy)

    if "energy_per_atom" in add_targets and energy is not None:
        atoms.info["energy_per_atom"] = float(energy) / max(len(atoms), 1)

    if "forces" in add_targets:
        raw_forces = item.get("forces")
        if raw_forces is not None and len(raw_forces) > 0:
            forces = np.asarray(raw_forces, dtype=np.float32)
            atoms.arrays["forces"] = forces

    if "stress" in add_targets:
        raw_stress = item.get("stress_tensor")
        if raw_stress is not None and len(raw_stress) > 0:
            stress_tensor = np.asarray(raw_stress, dtype=np.float32)
            if stress_tensor.shape != (3, 3):
                stress_tensor = stress_tensor.reshape(3, 3)
            stress_voigt = stress_array_to_voigt_6(stress_tensor)
            atoms.info["stress"] = stress_voigt
            atoms.info["stress_tensor"] = stress_tensor

    if energy is not None or forces is not None or stress_voigt is not None:
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=energy,
            forces=forces,
            stress=stress_voigt,
        )

    return atoms


class LeMatBulkDataset(Dataset[AtomsSample]):
    """Minimal random-access LeMat-Bulk dataset that yields ASE Atoms.

    Parameters
    ----------
    name : str, default="compatible_pbe"
        LeMat-Bulk config name (compatible_pbe, compatible_pbesol, etc.).
    split : str, default="train"
        Dataset split.
    dataset_name : str, default="LeMaterial/LeMat-Bulk"
        HuggingFace dataset name.
    cache_dir : str, optional
        HuggingFace cache directory.
    max_samples : int, optional
        Limit number of samples for quick runs.
    add_targets : Sequence[str], optional
        Targets to attach to atoms: "energy", "energy_per_atom", "forces", "stress".
    add_metadata : bool, default=True
        Whether to copy common metadata into atoms.info.
    """

    def __init__(
        self,
        *,
        name: str = DEFAULT_CONFIG,
        split: str = DEFAULT_SPLIT,
        dataset_name: str = DEFAULT_DATASET_NAME,
        cache_dir: str | None = None,
        max_samples: int | None = None,
        add_targets: Sequence[str] | None = None,
        add_metadata: bool = True,
    ):
        self.dataset = load_dataset(
            dataset_name,
            name,
            split=split,
            cache_dir=cache_dir,
        )
        if max_samples is not None:
            self.dataset = self.dataset.select(range(int(max_samples)))

        self.name = name
        self.split = split
        self.dataset_name = dataset_name
        self.add_targets = list(add_targets or [])
        self.add_metadata = bool(add_metadata)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> AtomsSample:
        item = self.dataset[int(idx)]
        atoms = lemat_item_to_ase(
            item, add_targets=self.add_targets, add_metadata=self.add_metadata
        )
        return AtomsSample(atoms=atoms, pair_id=int(idx))

    def get_node_counts(self) -> np.ndarray:
        """Return number of sites per structure when available.

        Returns
        -------
        np.ndarray
            Array of node counts.
        """
        if "nsites" in self.dataset.column_names:
            return np.asarray(self.dataset.select_columns(["nsites"])["nsites"])
        return np.array([len(self[i].atoms) for i in range(len(self))], dtype=np.int64)
