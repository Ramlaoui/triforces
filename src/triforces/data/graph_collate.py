"""Graph-building collate helpers for Atoms-based datasets."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Protocol, Sequence

import numpy as np
import torch
from ase import Atoms
from torch_geometric.data import Data

from .pyg_collate import pyg_contrastive_collate


class AtomsTransformLike(Protocol):
    def __call__(self, atoms: Atoms, **kwargs) -> Data:  # pragma: no cover - protocol
        ...


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
)
GRAPH_OVERRIDE_KEYS = (
    "graph_params",
    "graph_radius",
    "graph_max_num_neighbors",
    "graph_max_neigh",
    "graph_loop",
    "graph_enforce_max_neighbors_strictly",
    "graph_radius_pbc_version",
    "graph_dataset",
    "graph_device",
    "graph_r_max",
)


def _stress_tensor_to_voigt(stress: np.ndarray) -> np.ndarray:
    stress = np.asarray(stress, dtype=np.float32)
    if stress.shape != (3, 3):
        stress = stress.reshape(3, 3)
    return np.array(
        [
            stress[0, 0],
            stress[1, 1],
            stress[2, 2],
            stress[1, 2],
            stress[0, 2],
            stress[0, 1],
        ],
        dtype=np.float32,
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


def _extract_graph_overrides(atoms: Atoms) -> dict[str, Any]:
    info = getattr(atoms, "info", None) or {}
    overrides: dict[str, Any] = {}

    params = info.get("graph_params")
    if isinstance(params, dict):
        overrides.update(params)

    if "graph_radius" in info:
        overrides["radius"] = info["graph_radius"]
    if "graph_max_num_neighbors" in info:
        overrides["max_num_neighbors"] = info["graph_max_num_neighbors"]
    if "graph_max_neigh" in info:
        overrides["max_neigh"] = info["graph_max_neigh"]
    if "graph_loop" in info:
        overrides["loop"] = info["graph_loop"]
    if "graph_enforce_max_neighbors_strictly" in info:
        overrides["enforce_max_neighbors_strictly"] = info[
            "graph_enforce_max_neighbors_strictly"
        ]
    if "graph_radius_pbc_version" in info:
        overrides["radius_pbc_version"] = info["graph_radius_pbc_version"]
    if "graph_dataset" in info:
        overrides["dataset"] = info["graph_dataset"]
    if "graph_device" in info:
        overrides["device"] = info["graph_device"]
    if "graph_r_max" in info:
        overrides["r_max"] = info["graph_r_max"]

    return overrides


def _apply_overrides(transform: AtomsTransformLike, overrides: dict[str, Any]):
    if not overrides:
        return {}

    saved: dict[str, Any] = {}
    for key, value in overrides.items():
        if hasattr(transform, key):
            saved[key] = getattr(transform, key)
            setattr(transform, key, value)

    if ("radius" in overrides or "max_num_neighbors" in overrides) and hasattr(
        transform, "system_config"
    ):
        try:
            from orb_models.forcefield.atomic_system import SystemConfig

            radius = overrides.get("radius", getattr(transform, "radius", None))
            max_num_neighbors = overrides.get(
                "max_num_neighbors", getattr(transform, "max_num_neighbors", None)
            )
            if radius is not None and max_num_neighbors is not None:
                saved["system_config"] = getattr(transform, "system_config", None)
                transform.system_config = SystemConfig(
                    radius=float(radius), max_num_neighbors=int(max_num_neighbors)
                )
        except Exception:
            pass

    return saved


def _restore_overrides(transform: AtomsTransformLike, saved: dict[str, Any]):
    if not saved:
        return
    for key, value in saved.items():
        if hasattr(transform, key):
            setattr(transform, key, value)


def _extract_noise(
    atoms: Atoms, n: int, has_noise_any: bool
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not has_noise_any:
        return None, None
    if _has_array(atoms, "noise_displacement"):
        noise = np.asarray(atoms.arrays["noise_displacement"], dtype=np.float32)
        mask = np.ones((n,), dtype=bool)
    else:
        noise = np.zeros((n, 3), dtype=np.float32)
        mask = np.zeros((n,), dtype=bool)
    return _as_tensor(noise, dtype=torch.float32), _as_tensor(mask, dtype=torch.bool)


def _extract_forces(
    atoms: Atoms, n: int, has_forces_any: bool
) -> Optional[torch.Tensor]:
    if not has_forces_any:
        return None
    if _has_array(atoms, "forces"):
        forces = np.asarray(atoms.arrays["forces"], dtype=np.float32)
    else:
        forces = np.full((n, 3), np.nan, dtype=np.float32)
    return _as_tensor(forces, dtype=torch.float32)


def _extract_info_tensor(
    atoms: Atoms,
    key: str,
    *,
    n: int,
    has_any: bool,
) -> Optional[torch.Tensor]:
    if not has_any:
        return None

    value = _get_info(atoms, key)
    if value is None:
        if key == "energy_per_atom":
            energy = _get_info(atoms, "energy")
            if energy is not None:
                value = float(energy) / max(n, 1)
        if value is None:
            value = np.nan

    if key == "stress" and value is not None and np.asarray(value).shape == (3, 3):
        value = _stress_tensor_to_voigt(value)
    if key == "stress_tensor" and value is not None and np.asarray(value).shape == (6,):
        value = np.asarray(value, dtype=np.float32).reshape(3, 3)

    dtype = torch.float32 if key not in ("charge", "spin") else torch.float32
    return _as_tensor(value, dtype=dtype)


def _atoms_to_graph(
    atoms: Atoms,
    *,
    transform: AtomsTransformLike,
    pair_id: int | None,
    info_keys: Sequence[str],
    array_keys: Sequence[str],
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

    for key in info_keys:
        if key not in DEFAULT_INFO_KEYS:
            continue
        tensor = _extract_info_tensor(atoms, key, n=n, has_any=has_any[key])
        if tensor is not None:
            kwargs[key] = tensor

    overrides = _extract_graph_overrides(atoms)
    saved = _apply_overrides(transform, overrides)
    try:
        data = transform(atoms, **kwargs)
    finally:
        _restore_overrides(transform, saved)

    if pair_id is not None:
        data.pair_id = torch.as_tensor(int(pair_id), dtype=torch.long)

    if not hasattr(data, "z") and hasattr(data, "atomic_numbers"):
        data.z = data.atomic_numbers
    if not hasattr(data, "pos") and hasattr(data, "positions"):
        data.pos = data.positions

    # Ensure kwargs are present even if transform ignores them
    for key, value in kwargs.items():
        if not hasattr(data, key):
            setattr(data, key, value)

    return data


def graph_contrastive_collate(
    samples: Sequence[Any],
    *,
    transform: AtomsTransformLike,
    info_keys: Optional[Sequence[str]] = None,
    array_keys: Optional[Sequence[str]] = None,
):
    flat_samples: list[Any] = []
    for sample in samples:
        if isinstance(sample, (list, tuple)):
            flat_samples.extend(sample)
        else:
            flat_samples.append(sample)

    if not flat_samples:
        raise ValueError("Empty batch")

    info_keys = tuple(info_keys or DEFAULT_INFO_KEYS)
    array_keys = tuple(array_keys or DEFAULT_ARRAY_KEYS)

    has_any = {key: False for key in (*info_keys, *array_keys)}
    for sample in flat_samples:
        atoms = sample.atoms
        for key in array_keys:
            if _has_array(atoms, key):
                has_any[key] = True
        for key in info_keys:
            if _get_info(atoms, key) is not None:
                has_any[key] = True

    data_list = [
        _atoms_to_graph(
            sample.atoms,
            transform=transform,
            pair_id=getattr(sample, "pair_id", None),
            info_keys=info_keys,
            array_keys=array_keys,
            has_any=has_any,
        )
        for sample in flat_samples
    ]

    return pyg_contrastive_collate(data_list)


def pyg_collate(
    samples: Sequence[Any],
    *,
    graph: AtomsTransformLike,
    info_keys: Optional[Sequence[str]] = None,
    array_keys: Optional[Sequence[str]] = None,
):
    return graph_contrastive_collate(
        samples, transform=graph, info_keys=info_keys, array_keys=array_keys
    )


def build_graph_collate(
    transform: AtomsTransformLike,
    *,
    info_keys: Optional[Sequence[str]] = None,
    array_keys: Optional[Sequence[str]] = None,
) -> Callable[[Sequence[Any]], Any]:
    def _collate(samples: Sequence[Any]):
        return graph_contrastive_collate(
            samples, transform=transform, info_keys=info_keys, array_keys=array_keys
        )

    return _collate
