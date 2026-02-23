from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch
from ase import Atoms
from ase.db import connect
from torch.utils.data import Dataset

from triforces.utils.stress import stress_array_to_voigt_6

from .ase_contrastive import AtomsSample

_SUPPORTED_SUFFIXES = (".db", ".aselmdb")
_DB_OPEN_KWARGS = {
    ".db": {},
    ".aselmdb": {"readonly": True, "use_lock_file": False},
}


def _require_hf_hub() -> Any:
    try:
        import huggingface_hub  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "huggingface_hub is required for ASE DB Hugging Face integration.\n"
            "Install with: uv pip install huggingface_hub"
        ) from exc
    return huggingface_hub


def _resolve_asedb_path_from_hf(
    *,
    repo_id: str,
    path_in_repo: str | None = None,
    revision: str | None = None,
    repo_type: str = "dataset",
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    local_files_only: bool = False,
    force_download: bool = False,
) -> Path:
    hf = _require_hf_hub()
    normalized = None
    if path_in_repo is not None:
        normalized = path_in_repo.strip().strip("/")
        if not normalized:
            raise ValueError("path_in_repo cannot be empty")

    common_kwargs = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "revision": revision,
        "token": token,
        "cache_dir": cache_dir,
        "local_dir": local_dir,
        "local_files_only": local_files_only,
        "force_download": force_download,
    }

    if normalized and normalized.endswith(_SUPPORTED_SUFFIXES):
        return Path(hf.hf_hub_download(filename=normalized, **common_kwargs))

    allow_patterns = (
        [
            f"{normalized}/*.db",
            f"{normalized}/**/*.db",
            f"{normalized}/*.aselmdb",
            f"{normalized}/**/*.aselmdb",
        ]
        if normalized
        else ["*.db", "**/*.db", "*.aselmdb", "**/*.aselmdb"]
    )
    snapshot_root = Path(
        hf.snapshot_download(allow_patterns=allow_patterns, **common_kwargs)
    )

    if normalized is None:
        found = sorted(snapshot_root.glob("**/*.db")) + sorted(
            snapshot_root.glob("**/*.aselmdb")
        )
        if not found:
            raise FileNotFoundError(
                f"No ASE DB files found in Hugging Face repo '{repo_id}'. "
                "Provide path_in_repo to the dataset folder/file."
            )
        return snapshot_root

    resolved = snapshot_root / normalized
    if not resolved.exists():
        raise FileNotFoundError(
            f"Resolved path does not exist in downloaded snapshot: {resolved}"
        )
    return resolved


class _ASEDBHandle:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.db = connect(str(path), **_DB_OPEN_KWARGS.get(self.path.suffix, {}))

    def __len__(self) -> int:
        return len(self.db)

    def get_row(self, local_idx: int):
        row_iter = self.db.select(limit=1, offset=int(local_idx))
        row = next(row_iter, None)
        if row is None:
            raise IndexError(
                f"Local index {local_idx} out of bounds for database {self.path}"
            )
        return row

    def close(self) -> None:
        close_fn = getattr(self.db, "close", None)
        if callable(close_fn):
            close_fn()


class ASEDBDataset(Dataset[AtomsSample]):
    """ASE DB dataset wrapper that yields ASE ``Atoms``.

    Parameters
    ----------
    path : str or Path or Sequence[str | Path] or None
        A single ``.db``/``.aselmdb`` file, a directory containing database files,
        or a list of files/directories.
    repo_id : str, optional
        Hugging Face repo id. If set, dataset path is resolved from the Hub.
    path_in_repo : str, optional
        Relative file/folder path in the Hugging Face repo.
    add_targets : Sequence[str], optional
        Targets to attach to output ASE atoms (e.g. ``energy``, ``forces``,
        ``stress``, ``energy_per_atom``).
    extract_keys : Sequence[str], optional
        Additional row data keys to attach to output ASE atoms.
    keep_db_open : bool, default=True
        Keep worker-local DB handles open for faster repeated access.
    """

    def __init__(
        self,
        path: Union[str, Path, Sequence[Union[str, Path]], None] = None,
        *,
        repo_id: str | None = None,
        path_in_repo: str | None = None,
        revision: str | None = None,
        repo_type: str = "dataset",
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_dir: str | Path | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
        add_targets: Sequence[str] | None = None,
        extract_keys: Sequence[str] | None = None,
        keep_db_open: bool = True,
    ):
        if path is not None and repo_id is not None:
            raise ValueError("Provide either `path` or `repo_id`, not both.")
        if path is None and repo_id is None:
            raise ValueError("Either `path` or `repo_id` must be provided.")

        if repo_id is not None:
            path = _resolve_asedb_path_from_hf(
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                revision=revision,
                repo_type=repo_type,
                token=token,
                cache_dir=cache_dir,
                local_dir=local_dir,
                local_files_only=local_files_only,
                force_download=force_download,
            )

        self.add_targets = list(add_targets or [])
        self.extract_keys = list(extract_keys or [])
        self.keep_db_open = bool(keep_db_open)

        self.paths: List[Path] = self._expand_paths(path)
        if not self.paths:
            raise FileNotFoundError(f"No ASE DB files found for: {path}")

        self.file_lengths: List[int] = []
        for p in self.paths:
            db = _ASEDBHandle(p)
            try:
                self.file_lengths.append(int(len(db)))
            finally:
                db.close()

        self._cum = np.cumsum([0] + self.file_lengths).tolist()

        # Worker-local connections.
        self._worker_id: int | None = None
        self._dbs: Dict[Path, _ASEDBHandle] = {}

    def _expand_paths(
        self, path: Union[str, Path, Sequence[Union[str, Path]], None]
    ) -> List[Path]:
        if path is None:
            return []

        if isinstance(path, (str, Path)):
            path_list = [path]
        else:
            path_list = list(path)

        out: List[Path] = []
        for p in path_list:
            pp = Path(p)
            if pp.is_dir():
                for suffix in _SUPPORTED_SUFFIXES:
                    out.extend(sorted(pp.glob(f"**/*{suffix}")))
            else:
                out.append(pp)

        return [
            p
            for p in out
            if p.exists() and p.suffix in _SUPPORTED_SUFFIXES and p.is_file()
        ]

    def __len__(self) -> int:
        return int(self._cum[-1])

    def _get_worker_dbs(self) -> Dict[Path, _ASEDBHandle]:
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else None

        if self._worker_id == wid and self._dbs:
            return self._dbs

        self.close()
        self._dbs.clear()
        self._worker_id = wid

        for p in self.paths:
            self._dbs[p] = _ASEDBHandle(p)

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

    @classmethod
    def from_huggingface(
        cls,
        repo_id: str,
        path_in_repo: str | None = None,
        *,
        revision: str | None = None,
        repo_type: str = "dataset",
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_dir: str | Path | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
        add_targets: Sequence[str] | None = None,
        extract_keys: Sequence[str] | None = None,
        keep_db_open: bool = True,
    ) -> "ASEDBDataset":
        local_path = _resolve_asedb_path_from_hf(
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            revision=revision,
            repo_type=repo_type,
            token=token,
            cache_dir=cache_dir,
            local_dir=local_dir,
            local_files_only=local_files_only,
            force_download=force_download,
        )
        return cls(
            path=local_path,
            add_targets=add_targets,
            extract_keys=extract_keys,
            keep_db_open=keep_db_open,
        )

    @staticmethod
    def _row_data(row: Any) -> dict[str, Any]:
        data = getattr(row, "data", None)
        if isinstance(data, dict):
            return data
        return {}

    @classmethod
    def _row_get_value(cls, row: Any, key: str) -> Any:
        data = cls._row_data(row)
        if key in data:
            return data[key]
        if hasattr(row, "get"):
            value = row.get(key, None)
            if value is not None:
                return value
        return getattr(row, key, None)

    @classmethod
    def _get_target_value(cls, atoms: Atoms, row: Any, target: str) -> Any:
        if target == "energy":
            try:
                return float(atoms.get_potential_energy())
            except Exception:
                return cls._row_get_value(row, "energy")

        if target == "forces":
            try:
                return np.asarray(atoms.get_forces(), dtype=np.float32)
            except Exception:
                return cls._row_get_value(row, "forces")

        if target == "stress":
            try:
                return np.asarray(atoms.get_stress(voigt=True), dtype=np.float32)
            except Exception:
                return cls._row_get_value(row, "stress")

        return cls._row_get_value(row, target)

    @staticmethod
    def _attach_value(atoms: Atoms, key: str, value: Any) -> None:
        if key == "forces":
            atoms.arrays["forces"] = np.asarray(value, dtype=np.float32)
            return
        if key == "stress":
            stress = np.asarray(value, dtype=np.float32)
            if stress.shape == (3, 3):
                atoms.info["stress_tensor"] = stress
                stress = stress_array_to_voigt_6(stress)
            atoms.info["stress"] = stress.astype(np.float32, copy=False)
            return

        array_value = np.asarray(value)
        if array_value.ndim == 0:
            atoms.info[key] = array_value.item()
            return

        # Keep atom-wise arrays in atoms.arrays for downstream transforms/collate.
        if array_value.shape[0] == len(atoms):
            atoms.arrays[key] = array_value
            return
        atoms.info[key] = array_value

    def _row_to_atoms(self, row: Any) -> Atoms:
        atoms = row.toatoms()

        for key in self.extract_keys:
            value = self._row_get_value(row, key)
            if value is None:
                raise KeyError(f"Key '{key}' not found in ASE DB row data")
            self._attach_value(atoms, key, value)

        for target in self.add_targets:
            if target == "energy_per_atom":
                energy = self._get_target_value(atoms, row, "energy")
                if energy is None:
                    raise ValueError(
                        "Target 'energy_per_atom' was requested but 'energy' was not found in ASE DB row"
                    )
                atoms.info["energy_per_atom"] = float(energy) / max(len(atoms), 1)
                continue

            value = self._get_target_value(atoms, row, target)
            if value is None:
                raise ValueError(
                    f"Target '{target}' was requested but not found in ASE DB row"
                )
            self._attach_value(atoms, target, value)

        return atoms

    def __getitem__(self, idx: int) -> AtomsSample:
        file_i, local = self._locate(int(idx))
        p = self.paths[file_i]

        if self.keep_db_open:
            dbs = self._get_worker_dbs()
            row = dbs[p].get_row(local)
        else:
            db = _ASEDBHandle(p)
            try:
                row = db.get_row(local)
            finally:
                db.close()

        atoms = self._row_to_atoms(row)
        return AtomsSample(atoms=atoms, pair_id=int(idx))

    def get_node_counts(self) -> np.ndarray:
        counts: list[int] = []
        for path in self.paths:
            db = _ASEDBHandle(path)
            try:
                for row in db.db.select():
                    natoms = getattr(row, "natoms", None)
                    counts.append(int(natoms) if natoms is not None else len(row.toatoms()))
            finally:
                db.close()
        return np.asarray(counts, dtype=np.int64)

    def close(self) -> None:
        dbs = getattr(self, "_dbs", None)
        if not isinstance(dbs, dict):
            return
        for db in dbs.values():
            db.close()
        dbs.clear()

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass
