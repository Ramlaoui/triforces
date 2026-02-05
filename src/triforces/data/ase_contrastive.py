from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.io import read as ase_read
from torch.utils.data import Dataset
from torch_geometric.data import Data


@dataclass
class AtomsSample:
    atoms: Atoms
    pair_id: int


class CifFolderDataset(Dataset[AtomsSample]):
    """Load structures from a folder of CIF files.

    Parameters
    ----------
    root : str or Path
        Root directory containing CIF files.
    glob : str, default="**/*.cif"
        Glob pattern used to find CIF files.

    Notes
    -----
    Each file becomes one sample with ``pair_id == index``.
    """

    def __init__(self, root: str | Path, *, glob: str = "**/*.cif"):
        self.root = Path(root)
        self.paths = sorted(self.root.glob(glob))
        if not self.paths:
            raise FileNotFoundError(f"No CIF files found under: {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> AtomsSample:
        path = self.paths[idx]
        atoms = ase_read(path.as_posix())
        if not isinstance(atoms, Atoms):
            # ase.io.read can return a list for multi-image files; take the first.
            atoms = atoms[0]
        return AtomsSample(atoms=atoms.copy(), pair_id=int(idx))


def atoms_to_pyg_data(atoms: Atoms, *, pair_id: int) -> Data:
    """Convert an ASE ``Atoms`` object into a PyG ``Data`` graph.

    Parameters
    ----------
    atoms : Atoms
        ASE atoms object.
    pair_id : int
        Pair ID for contrastive batching.

    Returns
    -------
    Data
        PyG data object containing positions, atomic numbers, and correspondence.

    Notes
    -----
    Uses ``atoms.info["node_correspondence"]`` when present (e.g., crops/subgraphs),
    otherwise defaults to ``arange(num_atoms)``.
    """
    n = len(atoms)
    z = torch.as_tensor(atoms.numbers, dtype=torch.long)
    pos = torch.as_tensor(np.asarray(atoms.positions), dtype=torch.float32)

    corr = atoms.info.get("node_correspondence", None)
    if corr is None:
        corr = np.arange(n, dtype=np.int64)
    corr = torch.as_tensor(np.asarray(corr), dtype=torch.long)
    if corr.numel() != n:
        raise ValueError(
            f"node_correspondence length mismatch: got {corr.numel()} expected {n}"
        )

    if hasattr(atoms, "arrays") and "noise_displacement" in atoms.arrays:
        noise = torch.as_tensor(
            np.asarray(atoms.arrays["noise_displacement"]), dtype=torch.float32
        )
        if noise.shape != (n, 3):
            raise ValueError(
                f"noise_displacement shape mismatch: got {tuple(noise.shape)} expected ({n}, 3)"
            )
        noise_mask = torch.ones((n,), dtype=torch.bool)
    else:
        noise = torch.zeros((n, 3), dtype=torch.float32)
        noise_mask = torch.zeros((n,), dtype=torch.bool)

    return Data(
        z=z,
        pos=pos,
        pair_id=torch.tensor([int(pair_id)], dtype=torch.long),
        node_correspondence=corr,
        noise_displacement=noise,
        noise_mask=noise_mask,
    )


def atoms_sample_to_pyg_data(sample: Any) -> Data:
    atoms = getattr(sample, "atoms", None)
    if atoms is None and isinstance(sample, Atoms):
        atoms = sample
    if atoms is None:
        raise TypeError("Expected an ASE Atoms object or sample with `.atoms`.")

    pair_id = getattr(sample, "pair_id", None)
    if pair_id is None and hasattr(atoms, "info"):
        pair_id = atoms.info.get("pair_id", None)
    if pair_id is None:
        raise ValueError("pair_id is required for contrastive batching.")

    return atoms_to_pyg_data(atoms, pair_id=int(pair_id))


class PyGAtomsDataset(Dataset[Data]):
    """Wrap an ASE-based dataset to emit PyG ``Data`` objects.

    Parameters
    ----------
    dataset : Dataset[Any]
        Dataset yielding ASE atoms samples.
    """

    def __init__(self, dataset: Dataset[Any]):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Data:
        return atoms_sample_to_pyg_data(self.dataset[idx])
