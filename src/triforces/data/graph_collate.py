"""Graph-building collate helpers for Atoms-based datasets."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import torch
from ase import Atoms
from torch_geometric.data import Data

from triforces.utils.stress import stress_array_to_voigt_6

from .pyg_collate import pyg_contrastive_collate, pyg_supervised_collate

AtomsTransform = Callable[..., Data]


DEFAULT_INFO_KEYS = (
    "energy",
    "energy_per_atom",
    "stress",
    "stress_tensor",
    "rotation_matrix",
    "cell_noise_displacement",
    "charge",
    "spin",
)
DEFAULT_ARRAY_KEYS = (
    "forces",
    "noise_displacement",
    "original_numbers",
    "atom_mask",
)


def _as_tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def _has_array(atoms: Atoms, key: str) -> bool:
    return hasattr(atoms, "arrays") and key in atoms.arrays


def _get_info(atoms: Atoms, key: str) -> Any:
    info = getattr(atoms, "info", None) or {}
    return info.get(key, None)


def _extract_node_correspondence(atoms: Atoms, n: int) -> torch.Tensor:
    corr = _get_info(atoms, "node_correspondence")
    if corr is None:
        corr = np.arange(n, dtype=np.int64)
    return _as_tensor(corr, dtype=torch.long)


def _extract_noise(
    atoms: Atoms, n: int, has_noise_any: bool
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not has_noise_any:
        return None, None
    if _has_array(atoms, "noise_displacement"):
        noise = np.asarray(atoms.arrays["noise_displacement"], dtype=np.float32)
        mask = np.ones((n,), dtype=bool)
    else:
        noise = np.zeros((n, 3), dtype=np.float32)
        mask = np.zeros((n,), dtype=bool)
    return _as_tensor(noise, dtype=torch.float32), _as_tensor(mask, dtype=torch.bool)


def _extract_forces(atoms: Atoms, n: int, has_forces_any: bool) -> torch.Tensor | None:
    if not has_forces_any:
        return None
    if _has_array(atoms, "forces"):
        forces = np.asarray(atoms.arrays["forces"], dtype=np.float32)
    else:
        forces = np.full((n, 3), np.nan, dtype=np.float32)
    return _as_tensor(forces, dtype=torch.float32)


def _extract_original_numbers(
    atoms: Atoms, n: int, has_original_any: bool
) -> torch.Tensor | None:
    if not has_original_any:
        return None
    if _has_array(atoms, "original_numbers"):
        original_numbers = np.asarray(atoms.arrays["original_numbers"], dtype=np.int64)
    else:
        original_numbers = np.asarray(atoms.numbers, dtype=np.int64)
    if original_numbers.shape != (n,):
        original_numbers = original_numbers.reshape(-1)[:n]
    return _as_tensor(original_numbers, dtype=torch.long)


def _extract_atom_mask(atoms: Atoms, n: int, has_mask_any: bool) -> torch.Tensor | None:
    if not has_mask_any:
        return None
    if _has_array(atoms, "atom_mask"):
        atom_mask = np.asarray(atoms.arrays["atom_mask"], dtype=bool)
    else:
        atom_mask = np.zeros((n,), dtype=bool)
    if atom_mask.shape != (n,):
        atom_mask = atom_mask.reshape(-1)[:n]
    return _as_tensor(atom_mask, dtype=torch.bool)


def _extract_info_tensor(
    atoms: Atoms,
    key: str,
    *,
    n: int,
    has_any: bool,
) -> torch.Tensor | None:
    if not has_any:
        return None

    value = _get_info(atoms, key)
    if value is None:
        if key == "rotation_matrix":
            value = np.eye(3, dtype=np.float32)
        elif key == "cell_noise_displacement":
            value = np.zeros((3, 3), dtype=np.float32)
        elif key == "stress":
            value = np.full((6,), np.nan, dtype=np.float32)
        elif key == "stress_tensor":
            value = np.full((3, 3), np.nan, dtype=np.float32)
        if key == "energy_per_atom":
            energy = _get_info(atoms, "energy")
            if energy is not None:
                value = float(energy) / max(n, 1)
        if value is None:
            value = np.nan

    if key == "stress" and value is not None and np.asarray(value).shape == (3, 3):
        value = stress_array_to_voigt_6(np.asarray(value))
    if key == "stress_tensor" and value is not None and np.asarray(value).shape == (6,):
        value = np.asarray(value, dtype=np.float32).reshape(3, 3)

    dtype = torch.float32 if key not in ("charge", "spin") else torch.float32
    return _as_tensor(value, dtype=dtype)


def _atoms_to_graph(
    atoms: Atoms,
    *,
    transform: AtomsTransform,
    pair_id: int | None,
    has_any: dict[str, bool],
) -> Data:
    n = len(atoms)

    kwargs: dict[str, Any] = {}
    kwargs["node_correspondence"] = _extract_node_correspondence(atoms, n)

    noise_disp, noise_mask = _extract_noise(atoms, n, has_any["noise_displacement"])
    if noise_disp is not None:
        kwargs["noise_displacement"] = noise_disp
        kwargs["noise_mask"] = noise_mask

    forces = _extract_forces(atoms, n, has_any["forces"])
    if forces is not None:
        kwargs["forces"] = forces

    original_numbers = _extract_original_numbers(atoms, n, has_any["original_numbers"])
    if original_numbers is not None:
        kwargs["original_numbers"] = original_numbers

    atom_mask = _extract_atom_mask(atoms, n, has_any["atom_mask"])
    if atom_mask is not None:
        kwargs["atom_mask"] = atom_mask

    for key in DEFAULT_INFO_KEYS:
        tensor = _extract_info_tensor(atoms, key, n=n, has_any=has_any[key])
        if tensor is not None:
            kwargs[key] = tensor

    data = transform(atoms, **kwargs)

    if pair_id is not None:
        data.pair_id = torch.as_tensor(int(pair_id), dtype=torch.long)

    # Ensure kwargs are present even if transform ignores them
    for key, value in kwargs.items():
        if not hasattr(data, key):
            setattr(data, key, value)

    return data


def _flatten_samples(samples: Sequence[Any]) -> list[Any]:
    flat_samples: list[Any] = []
    for sample in samples:
        if isinstance(sample, (list, tuple)):
            flat_samples.extend(sample)
        else:
            flat_samples.append(sample)
    return flat_samples


def _atoms_samples_to_data_list(
    flat_samples: Sequence[Any],
    *,
    transform: AtomsTransform,
    include_pair_id: bool,
) -> list[Data]:
    if not flat_samples:
        raise ValueError("Empty batch")

    has_any = {key: False for key in (*DEFAULT_INFO_KEYS, *DEFAULT_ARRAY_KEYS)}
    for sample in flat_samples:
        atoms = sample.atoms
        for key in DEFAULT_ARRAY_KEYS:
            if _has_array(atoms, key):
                has_any[key] = True
        for key in DEFAULT_INFO_KEYS:
            if _get_info(atoms, key) is not None:
                has_any[key] = True

    data_list = [
        _atoms_to_graph(
            sample.atoms,
            transform=transform,
            pair_id=(getattr(sample, "pair_id", None) if include_pair_id else None),
            has_any=has_any,
        )
        for sample in flat_samples
    ]
    return data_list


def graph_supervised_collate(
    samples: Sequence[Any],
    *,
    transform: AtomsTransform,
):
    """Collate an atoms-based batch into a PyG batch without pair metadata."""
    flat_samples = _flatten_samples(samples)
    data_list = _atoms_samples_to_data_list(
        flat_samples, transform=transform, include_pair_id=False
    )
    return pyg_supervised_collate(data_list)


def graph_contrastive_collate(
    samples: Sequence[Any],
    *,
    transform: AtomsTransform,
):
    """Collate an atoms-based batch into a PyG contrastive batch."""
    flat_samples = _flatten_samples(samples)
    data_list = _atoms_samples_to_data_list(
        flat_samples, transform=transform, include_pair_id=True
    )
    return pyg_contrastive_collate(data_list)


def pyg_collate(
    samples: Sequence[Any],
    *,
    graph: AtomsTransform,
    contrastive: bool = True,
):
    """Collate atoms samples using ``graph`` transform.

    Parameters
    ----------
    samples : Sequence[Any]
        Batch samples.
    graph : Callable[..., Data]
        Graph builder to use for each sample.
    contrastive : bool, default=True
        If True, require ``pair_id`` and add contrastive pair indices.
    Returns
    -------
    Batch
        Collated PyG batch.
    """
    if contrastive:
        return graph_contrastive_collate(samples, transform=graph)
    return graph_supervised_collate(samples, transform=graph)


def build_graph_collate(
    transform: AtomsTransform,
    *,
    contrastive: bool = True,
) -> Callable[[Sequence[Any]], Any]:
    """Build a closure suitable to pass as a DataLoader ``collate_fn``.

    Parameters
    ----------
    transform : Callable[..., Data]
        Graph builder converting ASE ``Atoms`` to PyG ``Data``.
    contrastive : bool, default=True
        If True, require pair metadata and add contrastive pair indices.
    Returns
    -------
    Callable[[Sequence[Any]], Any]
        Collate function for a PyTorch DataLoader.
    """

    def _collate(samples: Sequence[Any]):
        return pyg_collate(samples, graph=transform, contrastive=contrastive)

    return _collate
