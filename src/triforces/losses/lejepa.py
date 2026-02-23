# Copyright 2025 Triforces Authors

"""LeJEPA loss: JEPA prediction loss with SIGReg regularization."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch import distributed as dist

from .base import BaseLoss


def _get_pred_value(preds: Any, key: str):
    if preds is None:
        return None
    if isinstance(preds, dict):
        return preds.get(key)
    try:
        return preds[key]
    except Exception:
        return getattr(preds, key, None)


def _pred_items(preds: Any):
    if isinstance(preds, dict):
        return preds.items()
    attrs = getattr(preds, "attributes", None)
    if isinstance(attrs, dict):
        return attrs.items()
    return []


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if is_dist_avail_and_initialized():
        return dist.get_world_size()
    return 1


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if is_dist_avail_and_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / get_world_size()
    return tensor


class EppsPulley(nn.Module):
    """Epps-Pulley test for univariate normality.

    Parameters
    ----------
    t_max : float, default=3.0
        Maximum integration point; integration is over ``[0, t_max]``.
    n_points : int, default=17
        Number of integration points (must be odd).
    scale_by_n : bool, default=True
        Whether to scale the statistic by the sample size.
    """

    def __init__(
        self,
        t_max: float = 3.0,
        n_points: int = 17,
        scale_by_n: bool = True,
    ) -> None:
        super().__init__()
        if n_points % 2 != 1:
            raise ValueError("n_points must be odd for proper integration")

        self.scale_by_n = scale_by_n

        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        self.register_buffer("t", t)

        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[0] = dt
        weights[-1] = dt

        phi = torch.exp(-0.5 * t.square())
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Epps-Pulley test statistics.

        Parameters
        ----------
        x : torch.Tensor
            Standardized samples with shape ``(*, N, K)``, where ``N`` is samples
            and ``K`` is the number of slices.

        Returns
        -------
        torch.Tensor
            Test statistic per slice.
        """
        N = x.size(-2)

        t = self.t.to(device=x.device, dtype=x.dtype)
        phi = self.phi.to(device=x.device, dtype=x.dtype)
        weights = self.weights.to(device=x.device, dtype=x.dtype)

        x_t = x.unsqueeze(-1) * t
        cos_vals = torch.cos(x_t)
        sin_vals = torch.sin(x_t)

        cos_mean = cos_vals.mean(dim=-3)
        sin_mean = sin_vals.mean(dim=-3)

        cos_mean = all_reduce_mean(cos_mean)
        sin_mean = all_reduce_mean(sin_mean)

        err = (cos_mean - phi).square() + sin_mean.square()
        stat = err @ weights

        if self.scale_by_n:
            stat = stat * N * get_world_size()

        return stat


class SlicedGaussianRegularizer(nn.Module):
    """SIGReg: Sliced Isotropic Gaussian Regularization.

    Projects embeddings onto random 1D directions and tests each projection
    for Gaussianity using the Epps-Pulley test.

    Parameters
    ----------
    n_slices : int, default=1024
        Number of random projection slices.
    t_max : float, default=3.0
        Maximum integration point for Epps-Pulley.
    n_quadrature_points : int, default=17
        Number of quadrature points for Epps-Pulley.
    reduction : str, default="mean"
        Reduction applied to the slice statistics (``"mean"``, ``"sum"``, or ``"none"``).
    clip_value : float, optional
        Minimum threshold for slice statistics before reduction.
    """

    def __init__(
        self,
        n_slices: int = 1024,
        t_max: float = 3.0,
        n_quadrature_points: int = 17,
        reduction: str = "mean",
        clip_value: float | None = None,
    ) -> None:
        super().__init__()
        self.n_slices = n_slices
        self.reduction = reduction
        self.clip_value = clip_value

        self.univariate_test = EppsPulley(
            t_max=t_max,
            n_points=n_quadrature_points,
            scale_by_n=True,
        )

        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        self._generator: torch.Generator | None = None
        self._generator_device: torch.device | None = None

    def _get_generator(self, device: torch.device, seed: int) -> torch.Generator:
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        n_samples, dim = x.shape

        if n_samples < 2:
            return (
                torch.tensor(0.0, device=x.device, dtype=x.dtype),
                {"sigreg_skipped": 1.0},
            )

        with torch.no_grad():
            if is_dist_avail_and_initialized():
                step_sync = self.global_step.clone().to(x.device)
                dist.all_reduce(step_sync, op=dist.ReduceOp.MAX)
                seed = step_sync.item()
            else:
                seed = self.global_step.item()

            g = self._get_generator(x.device, int(seed))
            proj_shape = (dim, self.n_slices)
            A = torch.randn(proj_shape, device=x.device, dtype=x.dtype, generator=g)
            A = A / A.norm(p=2, dim=0, keepdim=True)
            self.global_step.add_(1)

        projected = x @ A
        stats = self.univariate_test(projected)

        if self.clip_value is not None:
            stats = stats.clamp(min=self.clip_value)

        if self.reduction == "mean":
            loss = stats.mean()
        elif self.reduction == "sum":
            loss = stats.sum()
        else:
            loss = stats

        metrics = {
            "mean": stats.mean().item(),
            "std": stats.std().item(),
            "max": stats.max().item(),
            "min": stats.min().item(),
        }

        return loss, metrics


class LeJEPALoss(BaseLoss):
    """LeJEPA loss combining SIGReg and prediction loss.

    Parameters
    ----------
    embedding_key : str, default="graph_projections"
        Base key for graph-level projections.
    node_embedding_key : str, default="node_projections"
        Base key for node-level projections.
    prediction_weight : float, optional
        Relative weight for prediction loss. If ``lambda_sigreg`` is provided,
        this is ignored and weights are normalized to sum to 1.
    sigreg_weight : float, optional
        Relative weight for SIGReg. If ``lambda_sigreg`` is provided, this is ignored
        and weights are normalized to sum to 1.
    lambda_sigreg : float, optional
        Direct SIGReg mixture weight in ``[0, 1]``. Overrides the weights above.
    sigreg_univariate : str, default="EppsPulley"
        Univariate test name (only ``EppsPulley`` is supported).
    sigreg_num_points : int, default=17
        Quadrature points for Epps-Pulley.
    sigreg_num_slices : int, default=1024
        Number of random projection slices.
    sigreg_t_max : float, default=3.0
        Maximum integration point for Epps-Pulley.
    lambda_node : float, default=0.5
        Weight for node-level loss.
    lambda_graph : float, default=0.5
        Weight for graph-level loss.
    clip_value : float, optional
        Minimum threshold for slice statistics.
    detach_target : bool, default=False
        If True, detach the clean/original view embeddings.

    Notes
    -----
    The total loss is ``w_sigreg * SIGReg + w_pred * L_pred``.
    """

    def __init__(
        self,
        embedding_key: str = "graph_projections",
        node_embedding_key: str = "node_projections",
        prediction_weight: float | None = None,
        sigreg_weight: float | None = None,
        lambda_sigreg: float | None = None,
        sigreg_univariate: str = "EppsPulley",
        sigreg_num_points: int = 17,
        sigreg_num_slices: int = 1024,
        sigreg_t_max: float = 3.0,
        lambda_node: float = 0.5,
        lambda_graph: float = 0.5,
        clip_value: float | None = None,
        detach_target: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.embedding_key = embedding_key
        self.node_embedding_key = node_embedding_key
        self.detach_target = detach_target

        self.lambda_node = float(lambda_node)
        self.lambda_graph = float(lambda_graph)
        total_lambda = self.lambda_node + self.lambda_graph
        if total_lambda > 0:
            self.lambda_node /= total_lambda
            self.lambda_graph /= total_lambda

        if lambda_sigreg is None:
            pred_w = 1.0 if prediction_weight is None else float(prediction_weight)
            sig_w = 1.0 if sigreg_weight is None else float(sigreg_weight)
            total = pred_w + sig_w
            if total > 0:
                self.prediction_weight = pred_w / total
                self.sigreg_weight = sig_w / total
            else:
                self.prediction_weight = 0.0
                self.sigreg_weight = 0.0
        else:
            self.sigreg_weight = float(lambda_sigreg)
            if not 0.0 <= self.sigreg_weight <= 1.0:
                raise ValueError("lambda_sigreg must be in [0, 1]")
            self.prediction_weight = 1.0 - self.sigreg_weight

        self.sigreg = None
        if self.sigreg_weight > 0:
            if (
                sigreg_univariate.lower() != "eppspulley"
                and sigreg_univariate != "EppsPulley"
            ):
                raise ValueError(
                    f"Unsupported sigreg_univariate '{sigreg_univariate}'. Only EppsPulley is available."
                )
            self.sigreg = SlicedGaussianRegularizer(
                n_slices=int(sigreg_num_slices),
                t_max=float(sigreg_t_max),
                n_quadrature_points=int(sigreg_num_points),
                reduction="mean",
                clip_value=clip_value,
            )

    def _find_stream_projections(
        self, preds: Any, base_key: str
    ) -> List[Tuple[str, torch.Tensor]]:
        items = list(_pred_items(preds))
        if not items:
            value = _get_pred_value(preds, base_key)
            return [(base_key, value)] if value is not None else []

        stream_keys = [
            k for k, v in items if k.startswith(f"{base_key}_") and v is not None
        ]
        if stream_keys:
            return [(k, _get_pred_value(preds, k)) for k in sorted(stream_keys)]

        value = _get_pred_value(preds, base_key)
        if value is not None:
            return [(base_key, value)]

        return []

    def _select_projection_group(
        self, preds: Any, keys: List[str]
    ) -> Tuple[str | None, List[Tuple[str, torch.Tensor]]]:
        for key in keys:
            found = self._find_stream_projections(preds, key)
            if found:
                return key, found
        return None, []

    def _compute_prediction_loss(
        self, embeddings1: torch.Tensor, embeddings2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        diff = embeddings1 - embeddings2
        per_pair = 0.25 * diff.pow(2).sum(dim=-1)
        loss = self._reduce(per_pair)
        metrics = {
            "pred_loss": loss.item(),
            "embedding_distance": diff.norm(dim=-1).mean().item(),
        }
        return loss, metrics

    def _lejepa_loss(
        self, embeddings1: torch.Tensor, embeddings2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        metrics: Dict[str, float] = {}

        pred_loss, pred_metrics = self._compute_prediction_loss(
            embeddings1, embeddings2
        )
        metrics.update(pred_metrics)

        sigreg_loss = embeddings1.new_tensor(0.0)
        if self.sigreg is not None and self.sigreg_weight > 0:
            all_embeddings = torch.cat([embeddings1, embeddings2], dim=0)
            sigreg_loss, sigreg_metrics = self.sigreg(all_embeddings)
            metrics.update({f"sigreg_{k}": v for k, v in sigreg_metrics.items()})
        else:
            metrics["sigreg_skipped"] = 1.0

        total_loss = (
            self.prediction_weight * pred_loss + self.sigreg_weight * sigreg_loss
        )

        metrics["sigreg_loss"] = sigreg_loss.item()
        metrics["weighted_sigreg"] = (self.sigreg_weight * sigreg_loss).item()
        metrics["weighted_pred"] = (self.prediction_weight * pred_loss).item()

        return total_loss, metrics

    def _forward_with_pairs(
        self, data: Any, preds: Any
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pair_idx1 = getattr(data, "pair_idx1", None)
        pair_idx2 = getattr(data, "pair_idx2", None)
        if pair_idx1 is None or pair_idx2 is None:
            raise ValueError(
                "LeJEPALoss requires `pair_idx1` and `pair_idx2` from collate."
            )

        if pair_idx1.numel() == 0:
            device = getattr(data, "batch", torch.tensor([], dtype=torch.long)).device
            return torch.tensor(0.0, device=device, requires_grad=True), {
                "skipped": "no_valid_pairs"
            }

        metrics: Dict[str, float] = {}
        pred_losses: List[float] = []
        sig_losses: List[float] = []

        total_loss = None

        # Graph-level loss
        if self.lambda_graph > 0:
            graph_keys = [
                self.embedding_key,
                "graph_projections",
                "graph_features",
                "graph_feats",
            ]
            base_key, graph_projections = self._select_projection_group(
                preds, graph_keys
            )
            if not graph_projections:
                raise ValueError("No graph embeddings found in preds.")

            graph_idx_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
            for key_name, graph_proj in graph_projections:
                device = graph_proj.device
                if device not in graph_idx_cache:
                    graph_idx_cache[device] = (
                        pair_idx1.to(device=device),
                        pair_idx2.to(device=device),
                    )
                idx1, idx2 = graph_idx_cache[device]

                graph_emb1 = graph_proj[idx1]
                graph_emb2 = graph_proj[idx2]

                if self.detach_target:
                    graph_emb1 = graph_emb1.detach()

                graph_loss, graph_metrics = self._lejepa_loss(graph_emb1, graph_emb2)

                total_loss = (
                    graph_loss * self.lambda_graph
                    if total_loss is None
                    else total_loss + graph_loss * self.lambda_graph
                )

                suffix = ""
                if base_key is not None and key_name != base_key:
                    if key_name.startswith(base_key):
                        suffix = key_name[len(base_key) :]

                metrics[f"graph_loss{suffix}"] = graph_loss.item()
                for k, v in graph_metrics.items():
                    metrics[f"graph_{k}{suffix}"] = v

                pred_losses.append(graph_metrics.get("pred_loss", 0.0))
                sig_losses.append(graph_metrics.get("sigreg_loss", 0.0))

        # Node-level loss
        if (
            self.lambda_node > 0
            and hasattr(data, "node_pair_idx1")
            and hasattr(data, "node_pair_idx2")
        ):
            node_idx1 = data.node_pair_idx1
            node_idx2 = data.node_pair_idx2
            if node_idx1.numel() > 0:
                node_keys = [
                    self.node_embedding_key,
                    "node_projections",
                    "node_features",
                    "node_feats",
                ]
                base_key, node_projections = self._select_projection_group(
                    preds, node_keys
                )
                if not node_projections:
                    raise ValueError("No node embeddings found in preds.")

                node_idx_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
                for key_name, node_proj in node_projections:
                    device = node_proj.device
                    if device not in node_idx_cache:
                        node_idx_cache[device] = (
                            node_idx1.to(device=device),
                            node_idx2.to(device=device),
                        )
                    idx1, idx2 = node_idx_cache[device]

                    node_emb1 = node_proj[idx1]
                    node_emb2 = node_proj[idx2]

                    if self.detach_target:
                        node_emb1 = node_emb1.detach()

                    node_loss, node_metrics = self._lejepa_loss(node_emb1, node_emb2)

                    total_loss = (
                        node_loss * self.lambda_node
                        if total_loss is None
                        else total_loss + node_loss * self.lambda_node
                    )

                    suffix = ""
                    if base_key is not None and key_name != base_key:
                        if key_name.startswith(base_key):
                            suffix = key_name[len(base_key) :]

                    metrics[f"node_loss{suffix}"] = node_loss.item()
                    for k, v in node_metrics.items():
                        metrics[f"node_{k}{suffix}"] = v

                    pred_losses.append(node_metrics.get("pred_loss", 0.0))
                    sig_losses.append(node_metrics.get("sigreg_loss", 0.0))

        if total_loss is None:
            device = getattr(data, "batch", torch.tensor([], dtype=torch.long)).device
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        metrics["loss/prediction"] = (
            float(sum(pred_losses) / len(pred_losses)) if pred_losses else 0.0
        )
        metrics["loss/sigreg"] = (
            float(sum(sig_losses) / len(sig_losses)) if sig_losses else 0.0
        )
        metrics["loss/prediction_weight"] = self.prediction_weight
        metrics["loss/sigreg_weight"] = self.sigreg_weight
        metrics["total_loss"] = total_loss.item()

        return total_loss, metrics

    def forward(
        self,
        data: Any | None = None,
        preds: Any | None = None,
        step: int = 0,
        embeddings1: torch.Tensor | None = None,
        embeddings2: torch.Tensor | None = None,
        node_embeddings1: torch.Tensor | None = None,
        node_embeddings2: torch.Tensor | None = None,
        graph_embeddings1: torch.Tensor | None = None,
        graph_embeddings2: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        metrics: Dict[str, float] = {}

        if data is not None and preds is not None:
            return self._forward_with_pairs(data, preds)

        if embeddings1 is not None and embeddings2 is not None:
            return self._lejepa_loss(embeddings1, embeddings2)

        total_loss = None

        if (
            self.lambda_node > 0
            and node_embeddings1 is not None
            and node_embeddings2 is not None
        ):
            node_loss, node_metrics = self._lejepa_loss(
                node_embeddings1, node_embeddings2
            )
            total_loss = self.lambda_node * node_loss
            metrics["node_loss"] = node_loss.item()
            metrics.update({f"node_{k}": v for k, v in node_metrics.items()})

        if (
            self.lambda_graph > 0
            and graph_embeddings1 is not None
            and graph_embeddings2 is not None
        ):
            graph_loss, graph_metrics = self._lejepa_loss(
                graph_embeddings1, graph_embeddings2
            )
            total_loss = (
                self.lambda_graph * graph_loss
                if total_loss is None
                else total_loss + self.lambda_graph * graph_loss
            )
            metrics["graph_loss"] = graph_loss.item()
            metrics.update({f"graph_{k}": v for k, v in graph_metrics.items()})

        if total_loss is None:
            device = (
                node_embeddings1.device
                if node_embeddings1 is not None
                else (
                    graph_embeddings1.device
                    if graph_embeddings1 is not None
                    else torch.device("cpu")
                )
            )
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        metrics["total_loss"] = total_loss.item()

        emb_to_check = (
            graph_embeddings1 if graph_embeddings1 is not None else node_embeddings1
        )
        if emb_to_check is not None and emb_to_check.shape[0] > 1:
            with torch.no_grad():
                emb_std = emb_to_check.std(dim=0).mean().item()
                metrics["embedding_std"] = emb_std
                metrics["collapse_warning"] = 1.0 if emb_std < 0.01 else 0.0

        return total_loss, metrics
