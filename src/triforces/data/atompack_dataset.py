from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np
import torch
from ase import Atoms
from torch.utils.data import Dataset


def _require_atompack():
    try:
        import atompack  # type: ignore

        return atompack
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "AtompackDataset requires the `atompack` Python package."
        ) from e


@dataclass
class AtompackSample:
    atoms: Atoms
    pair_id: int


class AtompackDataset(Dataset[AtompackSample]):
    """Atompack dataset wrapper that yields ASE ``Atoms``.

    Parameters
    ----------
    path : str or Path or Sequence[str | Path]
        A single ``.atp`` file, a directory containing ``.atp`` files, or a list
        of files/directories.
    use_mmap : bool, default=False
        Whether to open Atompack databases with memory mapping.
    """

    def __init__(
        self,
        path: Union[str, Path, Sequence[Union[str, Path]]],
        *,
        use_mmap: bool = False,
    ):
        self.use_mmap = bool(use_mmap)
        self.paths: List[Path] = self._expand_paths(path)
        if not self.paths:
            raise FileNotFoundError(f"No .atp files found for: {path}")

        # Precompute per-file lengths (once in main process).
        atompack = _require_atompack()
        self.file_lengths: List[int] = []
        for p in self.paths:
            db = (
                atompack.Database.open_mmap(str(p))
                if self.use_mmap
                else atompack.Database.open(str(p))
            )
            self.file_lengths.append(int(len(db)))

        self._cum = np.cumsum([0] + self.file_lengths).tolist()

        # Worker-local connections.
        self._worker_id: int | None = None
        self._dbs: Dict[Path, object] = {}

    def _expand_paths(
        self, path: Union[str, Path, Sequence[Union[str, Path]]]
    ) -> List[Path]:
        if isinstance(path, (str, Path)):
            path_list = [path]
        else:
            path_list = list(path)

        out: List[Path] = []
        for p in path_list:
            pp = Path(p)
            if pp.is_dir():
                out.extend(sorted(pp.glob("**/*.atp")))
            else:
                out.append(pp)

        # Filter to existing .atp files
        out = [p for p in out if p.exists() and p.suffix == ".atp"]
        return out

    def __len__(self) -> int:
        return int(self._cum[-1])

    def _get_worker_dbs(self) -> Dict[Path, object]:
        atompack = _require_atompack()
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else None

        if self._worker_id == wid and self._dbs:
            return self._dbs

        self._dbs.clear()
        self._worker_id = wid

        for p in self.paths:
            self._dbs[p] = (
                atompack.Database.open_mmap(str(p))
                if self.use_mmap
                else atompack.Database.open(str(p))
            )

        return self._dbs

    def _locate(self, idx: int) -> tuple[int, int]:
        # Find file id such that cum[file] <= idx < cum[file+1]
        lo, hi = 0, len(self.paths)
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self._cum[mid] <= idx:
                lo = mid
            else:
                hi = mid
        file_i = lo
        local = idx - self._cum[file_i]
        return file_i, int(local)

    def __getitem__(self, idx: int) -> AtompackSample:
        file_i, local = self._locate(int(idx))
        p = self.paths[file_i]
        dbs = self._get_worker_dbs()
        mol = dbs[p][local]

        info = {}
        # Optional fields commonly used by downstream transforms.
        if hasattr(mol, "has_property") and mol.has_property("charge"):
            info["charge"] = mol.get_property("charge")
        if hasattr(mol, "has_property") and mol.has_property("spin"):
            info["spin"] = mol.get_property("spin")

        atoms = Atoms(
            numbers=mol.atomic_numbers, positions=mol.positions, info=info or None
        )

        if getattr(mol, "cell", None) is not None:
            atoms.set_cell(mol.cell)
            if hasattr(mol, "has_property") and mol.has_property("pbc"):
                atoms.set_pbc(mol.get_property("pbc"))
            elif getattr(mol, "pbc", None) is not None:
                atoms.set_pbc(mol.pbc)
            else:
                atoms.set_pbc(True)
        else:
            atoms.set_pbc(False)

        return AtompackSample(atoms=atoms, pair_id=int(idx))
