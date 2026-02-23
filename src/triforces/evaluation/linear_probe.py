from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from ase.data import atomic_masses, covalent_radii
from torch.utils.data import DataLoader

__all__ = ["LinearProbeEvaluator", "LinearProbeResult"]


CRYSTAL_SYSTEMS: tuple[tuple[range, str], ...] = (
    (range(1, 3), "triclinic"),
    (range(3, 16), "monoclinic"),
    (range(16, 75), "orthorhombic"),
    (range(75, 143), "tetragonal"),
    (range(143, 168), "trigonal"),
    (range(168, 195), "hexagonal"),
    (range(195, 231), "cubic"),
)


@dataclass(frozen=True)
class LinearProbeResult:
    property_name: str
    metrics: dict[str, float]
    task: str


def _get_crystal_system(space_group: int) -> str:
    for sg_range, system in CRYSTAL_SYSTEMS:
        if int(space_group) in sg_range:
            return system
    return "unknown"


def _get_chemical_family(z_values: np.ndarray) -> str:
    unique_z = np.unique(z_values.astype(np.int64))
    if unique_z.size == 0:
        return "unknown"
    if unique_z.size == 1:
        return "element"
    if 8 in unique_z:
        if np.all(unique_z == 8):
            return "element"
        return (
            "oxide"
            if not np.any(np.isin(unique_z, np.array([16, 34, 52], dtype=np.int64)))
            else "chalcogenide"
        )
    if np.any(np.isin(unique_z, np.array([16, 34, 52], dtype=np.int64))):
        return "chalcogenide"
    if np.any(np.isin(unique_z, np.array([9, 17, 35, 53], dtype=np.int64))):
        return "halide"
    if 7 in unique_z:
        return "nitride"
    if 15 in unique_z:
        return "phosphide"
    if 6 in unique_z:
        return "carbide"
    if not np.any(
        np.isin(
            unique_z,
            np.array([1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 52, 53], dtype=np.int64),
        )
    ):
        return "intermetallic"
    return "other"


def _safe_float_array(values: list[Any]) -> np.ndarray:
    out = np.full((len(values),), np.nan, dtype=np.float64)
    for i, value in enumerate(values):
        if value is None:
            continue
        try:
            out[i] = float(value)
        except Exception:
            continue
    return out


def _tensor_to_graph_values(value: Any, *, num_graphs: int) -> list[Any]:
    if value is None:
        return [None] * num_graphs
    if isinstance(value, (int, float, str)):
        return [value] * num_graphs
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim == 0:
            return [tensor.item()] * num_graphs
        if tensor.ndim == 1 and tensor.shape[0] == num_graphs:
            return [tensor[i].item() for i in range(num_graphs)]
        if tensor.ndim == 2 and tensor.shape[0] == num_graphs and tensor.shape[1] == 1:
            return [tensor[i, 0].item() for i in range(num_graphs)]
        return [None] * num_graphs
    if isinstance(value, (list, tuple)):
        if len(value) == num_graphs:
            return list(value)
        return [None] * num_graphs
    return [None] * num_graphs


def _num_graphs_from_batch(batch: object) -> int:
    num_graphs = getattr(batch, "num_graphs", None)
    if isinstance(num_graphs, int):
        return int(num_graphs)

    batch_index = getattr(batch, "batch", None)
    if isinstance(batch_index, torch.Tensor) and batch_index.numel() > 0:
        return int(batch_index.max().item()) + 1
    return 0


def _graph_embeddings_from_outputs(
    outputs: Any, embedding_key: str
) -> torch.Tensor | None:
    if isinstance(outputs, dict):
        value = outputs.get(embedding_key)
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            return value
        return None

    value = getattr(outputs, embedding_key, None)
    if isinstance(value, torch.Tensor) and value.ndim == 2:
        return value
    return None


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    mean_true = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - mean_true) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def _standardize_train_test(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std


def _random_split_indices(
    n: int, *, test_size: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n, dtype=np.int64)
    rng.shuffle(indices)
    n_test = int(round(float(test_size) * n))
    n_test = max(1, min(n - 1, n_test))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return train_idx, test_idx


def _stratified_split_indices(
    labels: np.ndarray, *, test_size: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray] | None:
    classes = np.unique(labels)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        if cls_idx.size < 2:
            return None
        cls_idx = cls_idx.copy()
        rng.shuffle(cls_idx)
        n_test_cls = int(round(float(test_size) * cls_idx.size))
        n_test_cls = max(1, min(cls_idx.size - 1, n_test_cls))
        test_parts.append(cls_idx[:n_test_cls])
        train_parts.append(cls_idx[n_test_cls:])

    if not train_parts or not test_parts:
        return None
    train_idx = np.concatenate(train_parts, axis=0)
    test_idx = np.concatenate(test_parts, axis=0)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def _fit_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    alpha: float,
    fit_intercept: bool = True,
) -> np.ndarray:
    x_aug = x_train
    if fit_intercept:
        x_aug = np.concatenate(
            [np.ones((x_train.shape[0], 1), dtype=np.float64), x_train], axis=1
        )
    n_features = x_aug.shape[1]
    eye = np.eye(n_features, dtype=np.float64)
    if fit_intercept:
        eye[0, 0] = 0.0
    xtx = x_aug.T @ x_aug
    xty = x_aug.T @ y_train
    try:
        return np.linalg.solve(xtx + float(alpha) * eye, xty)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(xtx + float(alpha) * eye, xty, rcond=None)[0]


class LinearProbeEvaluator:
    """Run lightweight linear probes on graph-level embeddings."""

    def __init__(
        self,
        *,
        regression_properties: list[str] | None = None,
        classification_properties: list[str] | None = None,
        test_size: float = 0.2,
        random_seed: int = 42,
        ridge_alpha: float = 1.0,
        min_samples: int = 24,
    ) -> None:
        self.regression_properties = regression_properties or [
            "n_atoms",
            "mean_atomic_number",
            "volume_per_atom",
            "density",
            "lattice_anisotropy",
            "mean_nn_distance",
            "packing_fraction",
            "energy_per_atom",
            "formation_energy_per_atom",
        ]
        self.classification_properties = classification_properties or [
            "chemical_family",
            "crystal_system",
        ]
        self.test_size = float(test_size)
        self.random_seed = int(random_seed)
        self.ridge_alpha = float(ridge_alpha)
        self.min_samples = int(min_samples)

    @torch.no_grad()
    def _extract_batch_targets(self, batch: object) -> dict[str, np.ndarray]:
        num_graphs = _num_graphs_from_batch(batch)
        if num_graphs <= 0:
            return {}

        batch_index = getattr(batch, "batch", None)
        if not isinstance(batch_index, torch.Tensor):
            return {}
        batch_index = batch_index.detach().cpu().to(torch.long)

        z = getattr(batch, "atomic_numbers", None)
        if z is None:
            z = getattr(batch, "z", None)
        if not isinstance(z, torch.Tensor):
            return {}
        z = z.detach().cpu().to(torch.long)

        n_atoms_t = torch.bincount(batch_index, minlength=num_graphs).to(torch.float64)
        z_sum = torch.zeros((num_graphs,), dtype=torch.float64)
        z_sum.index_add_(0, batch_index, z.to(torch.float64))
        mean_atomic_number_t = z_sum / torch.clamp(n_atoms_t, min=1.0)

        targets: dict[str, np.ndarray] = {
            "n_atoms": n_atoms_t.numpy(),
            "mean_atomic_number": mean_atomic_number_t.numpy(),
        }

        # Geometry-derived targets (volume, density, anisotropy, packing fraction)
        volume = np.full((num_graphs,), np.nan, dtype=np.float64)
        lattice_anisotropy = np.full((num_graphs,), np.nan, dtype=np.float64)
        packing_fraction = np.full((num_graphs,), np.nan, dtype=np.float64)
        density = np.full((num_graphs,), np.nan, dtype=np.float64)

        cell = getattr(batch, "cell", None)
        if isinstance(cell, torch.Tensor):
            cell_t = cell.detach().cpu().to(torch.float64)
            if cell_t.ndim == 2 and cell_t.shape == (3, 3):
                cell_t = cell_t.unsqueeze(0).repeat(num_graphs, 1, 1)
            if cell_t.ndim == 3 and cell_t.shape[0] == num_graphs:
                volume = torch.det(cell_t).abs().numpy()
                vec_norms = torch.linalg.norm(cell_t, dim=-1)
                min_norm = torch.clamp(torch.min(vec_norms, dim=1).values, min=1e-8)
                max_norm = torch.max(vec_norms, dim=1).values
                lattice_anisotropy = (max_norm / min_norm).numpy()

                z_np = z.numpy()
                node_graph = batch_index.numpy()
                for g in range(num_graphs):
                    mask = node_graph == g
                    if not np.any(mask):
                        continue
                    v = float(volume[g])
                    if v <= 1e-12:
                        continue
                    z_g = z_np[mask]

                    mass_amu = float(np.sum(atomic_masses[z_g]))
                    density[g] = 1.66053906660 * mass_amu / v

                    radii = covalent_radii[z_g]
                    sphere_volume = float(
                        np.sum((4.0 / 3.0) * np.pi * np.power(radii, 3))
                    )
                    packing_fraction[g] = sphere_volume / v

        volume_per_atom = volume / np.clip(targets["n_atoms"], a_min=1.0, a_max=None)
        targets["volume_per_atom"] = volume_per_atom
        targets["density"] = density
        targets["lattice_anisotropy"] = lattice_anisotropy
        targets["packing_fraction"] = packing_fraction

        # Mean nearest-neighbor distance from edges.
        mean_nn = np.full((num_graphs,), np.nan, dtype=np.float64)
        edge_index = getattr(batch, "edge_index", None)
        edge_dist = getattr(batch, "edge_dist", None)
        if (
            isinstance(edge_index, torch.Tensor)
            and edge_index.ndim == 2
            and edge_index.shape[0] == 2
        ):
            src = edge_index[0].detach().cpu().to(torch.long)
            if isinstance(edge_dist, torch.Tensor):
                dist = edge_dist.detach().cpu().reshape(-1).to(torch.float64)
            else:
                edge_vec = getattr(batch, "edge_vec", None)
                if isinstance(edge_vec, torch.Tensor):
                    dist = torch.linalg.norm(
                        edge_vec.detach().cpu().to(torch.float64), dim=-1
                    )
                else:
                    dist = None
            if dist is not None and dist.numel() == src.numel():
                node_min = torch.full((z.shape[0],), float("inf"), dtype=torch.float64)
                node_min.scatter_reduce_(0, src, dist, reduce="amin", include_self=True)
                finite = torch.isfinite(node_min)
                if finite.any():
                    sums = torch.zeros((num_graphs,), dtype=torch.float64)
                    counts = torch.zeros((num_graphs,), dtype=torch.float64)
                    valid_graph = batch_index[finite]
                    sums.index_add_(0, valid_graph, node_min[finite])
                    counts.index_add_(
                        0,
                        valid_graph,
                        torch.ones_like(valid_graph, dtype=torch.float64),
                    )
                    valid_counts = counts > 0
                    mean_nn_t = torch.full(
                        (num_graphs,), float("nan"), dtype=torch.float64
                    )
                    mean_nn_t[valid_counts] = sums[valid_counts] / counts[valid_counts]
                    mean_nn = mean_nn_t.numpy()
        targets["mean_nn_distance"] = mean_nn

        # Energy-like labels if available.
        energy_values = _tensor_to_graph_values(
            getattr(batch, "energy", None), num_graphs=num_graphs
        )
        energy_per_atom_values = _tensor_to_graph_values(
            getattr(batch, "energy_per_atom", None), num_graphs=num_graphs
        )
        formation_energy_values = _tensor_to_graph_values(
            getattr(batch, "formation_energy", None), num_graphs=num_graphs
        )
        formation_energy_pa_values = _tensor_to_graph_values(
            getattr(batch, "formation_energy_per_atom", None), num_graphs=num_graphs
        )

        energy = _safe_float_array(energy_values)
        energy_per_atom = _safe_float_array(energy_per_atom_values)
        formation_energy = _safe_float_array(formation_energy_values)
        formation_energy_per_atom = _safe_float_array(formation_energy_pa_values)

        n_atoms = targets["n_atoms"]
        missing_e_pa = np.isnan(energy_per_atom) & np.isfinite(energy)
        energy_per_atom[missing_e_pa] = energy[missing_e_pa] / np.clip(
            n_atoms[missing_e_pa], a_min=1.0, a_max=None
        )
        targets["energy_per_atom"] = energy_per_atom

        missing_f_pa = np.isnan(formation_energy_per_atom) & np.isfinite(
            formation_energy
        )
        formation_energy_per_atom[missing_f_pa] = formation_energy[
            missing_f_pa
        ] / np.clip(n_atoms[missing_f_pa], a_min=1.0, a_max=None)
        targets["formation_energy_per_atom"] = formation_energy_per_atom

        # Classification labels.
        chemical_family: list[str | None] = []
        z_np = z.numpy()
        graph_idx_np = batch_index.numpy()
        for g in range(num_graphs):
            mask = graph_idx_np == g
            if not np.any(mask):
                chemical_family.append(None)
                continue
            chemical_family.append(_get_chemical_family(z_np[mask]))

        crystal_system_values = _tensor_to_graph_values(
            getattr(batch, "crystal_system", None), num_graphs=num_graphs
        )
        if all(v is None for v in crystal_system_values):
            sg_values = _tensor_to_graph_values(
                getattr(batch, "space_group", None), num_graphs=num_graphs
            )
            crystal_system_values = [
                _get_crystal_system(int(v)) if v is not None else None
                for v in sg_values
            ]

        targets["chemical_family"] = np.array(chemical_family, dtype=object)
        targets["crystal_system"] = np.array(crystal_system_values, dtype=object)
        return targets

    @torch.no_grad()
    def collect(
        self,
        *,
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        embedding_key: str = "graph_projections",
        max_samples: int | None = None,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
        progress_every_batches: int | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        was_training = model.training
        model.eval()

        embedding_chunks: list[np.ndarray] = []
        target_chunks: dict[str, list[np.ndarray]] = {}
        collected = 0
        batches_seen = 0
        progress_interval = (
            None
            if progress_every_batches is None
            else max(1, int(progress_every_batches))
        )

        if callable(progress_callback):
            progress_callback(
                "collect_start",
                {
                    "max_samples": (None if max_samples is None else int(max_samples)),
                },
            )

        for batches_seen, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            outputs = model(batch, training=False)

            embeddings_t = _graph_embeddings_from_outputs(outputs, embedding_key)
            if embeddings_t is None:
                continue
            embeddings = embeddings_t.detach().cpu().to(torch.float64).numpy()
            if embeddings.shape[0] == 0:
                continue

            targets = self._extract_batch_targets(batch)
            if not targets:
                continue

            graph_count = embeddings.shape[0]
            embedding_chunks.append(embeddings)
            for key, arr in targets.items():
                target_chunks.setdefault(key, []).append(arr)

            collected += graph_count
            if (
                callable(progress_callback)
                and progress_interval is not None
                and (batches_seen % progress_interval) == 0
            ):
                progress_callback(
                    "collect_progress",
                    {
                        "batches_seen": int(batches_seen),
                        "samples_collected": int(collected),
                        "max_samples": (
                            None if max_samples is None else int(max_samples)
                        ),
                    },
                )
            if max_samples is not None and collected >= int(max_samples):
                break

        if was_training:
            model.train()

        if not embedding_chunks:
            if callable(progress_callback):
                progress_callback(
                    "collect_done",
                    {
                        "batches_seen": int(batches_seen),
                        "samples_collected": 0,
                    },
                )
            return np.empty((0, 0), dtype=np.float64), {}

        x = np.concatenate(embedding_chunks, axis=0)
        y: dict[str, np.ndarray] = {}
        for key, chunks in target_chunks.items():
            y[key] = np.concatenate(chunks, axis=0)

        if max_samples is not None and x.shape[0] > int(max_samples):
            rng = np.random.default_rng(self.random_seed)
            idx = rng.choice(x.shape[0], size=int(max_samples), replace=False)
            x = x[idx]
            for key in y:
                y[key] = y[key][idx]

        if callable(progress_callback):
            progress_callback(
                "collect_done",
                {
                    "batches_seen": int(batches_seen),
                    "samples_collected": int(x.shape[0]),
                },
            )
        return x, y

    def _evaluate_regression_property(
        self, x: np.ndarray, y: np.ndarray
    ) -> dict[str, float] | None:
        mask = np.isfinite(y)
        if int(mask.sum()) < self.min_samples:
            return None
        xx = x[mask]
        yy = y[mask].astype(np.float64)
        if xx.shape[0] < 3:
            return None

        rng = np.random.default_rng(self.random_seed)
        train_idx, test_idx = _random_split_indices(
            xx.shape[0], test_size=self.test_size, rng=rng
        )
        x_train, x_test = xx[train_idx], xx[test_idx]
        y_train, y_test = yy[train_idx], yy[test_idx]
        x_train, x_test = _standardize_train_test(x_train, x_test)

        w = _fit_ridge(x_train, y_train, alpha=self.ridge_alpha)
        x_train_aug = np.concatenate(
            [np.ones((x_train.shape[0], 1), dtype=np.float64), x_train], axis=1
        )
        x_test_aug = np.concatenate(
            [np.ones((x_test.shape[0], 1), dtype=np.float64), x_test], axis=1
        )
        y_pred_train = x_train_aug @ w
        y_pred_test = x_test_aug @ w
        mae = float(np.mean(np.abs(y_test - y_pred_test)))
        rmse = float(np.sqrt(np.mean((y_test - y_pred_test) ** 2)))
        return {
            "r2": _r2_score(y_test, y_pred_test),
            "r2_train": _r2_score(y_train, y_pred_train),
            "mae": mae,
            "rmse": rmse,
            "n_samples": float(xx.shape[0]),
            "n_train": float(x_train.shape[0]),
            "n_test": float(x_test.shape[0]),
        }

    def _evaluate_classification_property(
        self, x: np.ndarray, y: np.ndarray
    ) -> dict[str, float] | None:
        values = np.asarray(y, dtype=object)
        valid = np.array(
            [
                v is not None and not (isinstance(v, float) and np.isnan(v))
                for v in values
            ],
            dtype=bool,
        )
        if int(valid.sum()) < self.min_samples:
            return None

        xx = x[valid]
        yy_obj = values[valid]
        unique = np.unique(yy_obj)
        if unique.size < 2:
            return None
        label_map = {label: i for i, label in enumerate(unique.tolist())}
        yy = np.array([label_map[v] for v in yy_obj], dtype=np.int64)

        rng = np.random.default_rng(self.random_seed)
        split = _stratified_split_indices(yy, test_size=self.test_size, rng=rng)
        if split is None:
            split = _random_split_indices(
                xx.shape[0], test_size=self.test_size, rng=rng
            )
        train_idx, test_idx = split
        if train_idx.size == 0 or test_idx.size == 0:
            return None

        x_train, x_test = xx[train_idx], xx[test_idx]
        y_train, y_test = yy[train_idx], yy[test_idx]
        x_train, x_test = _standardize_train_test(x_train, x_test)

        n_classes = int(unique.size)
        y_train_oh = np.zeros((y_train.shape[0], n_classes), dtype=np.float64)
        y_train_oh[np.arange(y_train.shape[0]), y_train] = 1.0
        w = _fit_ridge(x_train, y_train_oh, alpha=self.ridge_alpha)
        x_train_aug = np.concatenate(
            [np.ones((x_train.shape[0], 1), dtype=np.float64), x_train], axis=1
        )
        x_test_aug = np.concatenate(
            [np.ones((x_test.shape[0], 1), dtype=np.float64), x_test], axis=1
        )
        logits_train = x_train_aug @ w
        logits_test = x_test_aug @ w
        pred_train = np.argmax(logits_train, axis=1)
        pred_test = np.argmax(logits_test, axis=1)
        accuracy_train = float(np.mean(pred_train == y_train))
        accuracy = float(np.mean(pred_test == y_test))
        return {
            "accuracy": accuracy,
            "accuracy_train": accuracy_train,
            "n_classes": float(n_classes),
            "n_samples": float(xx.shape[0]),
            "n_train": float(x_train.shape[0]),
            "n_test": float(x_test.shape[0]),
        }

    def evaluate(
        self,
        *,
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        embedding_key: str = "graph_projections",
        max_samples: int | None = None,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
        progress_every_batches: int | None = None,
    ) -> dict[str, float]:
        x, targets = self.collect(
            model=model,
            loader=loader,
            device=device,
            embedding_key=embedding_key,
            max_samples=max_samples,
            progress_callback=progress_callback,
            progress_every_batches=progress_every_batches,
        )
        if x.size == 0 or not targets:
            if callable(progress_callback):
                progress_callback("fit_done", {"metrics_count": 0})
            return {}

        if callable(progress_callback):
            progress_callback(
                "fit_start",
                {
                    "num_samples": int(x.shape[0]),
                    "num_features": int(x.shape[1]),
                    "num_targets": int(len(targets)),
                },
            )

        flat_metrics: dict[str, float] = {}
        for prop in self.regression_properties:
            y = targets.get(prop)
            if y is None:
                continue
            if y.dtype.kind not in {"f", "i", "u"}:
                continue
            metrics = self._evaluate_regression_property(x, y.astype(np.float64))
            if not metrics:
                continue
            for key, value in metrics.items():
                flat_metrics[f"{prop}/{key}"] = float(value)

        for prop in self.classification_properties:
            y = targets.get(prop)
            if y is None:
                continue
            metrics = self._evaluate_classification_property(x, y)
            if not metrics:
                continue
            for key, value in metrics.items():
                flat_metrics[f"{prop}/{key}"] = float(value)

        if callable(progress_callback):
            progress_callback("fit_done", {"metrics_count": int(len(flat_metrics))})
        return flat_metrics
