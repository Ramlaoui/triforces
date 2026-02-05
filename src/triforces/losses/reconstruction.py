"""Add an MSE reconstruction term on top of a base loss."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .base import BaseLoss


class ReconstructionLoss(BaseLoss):
    """Combine a base loss with an MSE reconstruction term.

    Parameters
    ----------
    base_loss : nn.Module
        Loss module to wrap. Must return ``(loss, metrics)`` or a scalar loss.
    noise_weight : float, default=1.0
        Weight for the denoising MSE term.
    noise_key : str, default="noise_displacement"
        Key in ``preds`` containing the predicted noise.
    prediction_target : {"noise", "x0"}, default="noise"
        If ``"noise"``, predict the noise itself. If ``"x0"``, predict displacement
        to clean positions (i.e., target = ``-noise``).

    Notes
    -----
    The denoising term expects:
    - ``preds[noise_key]``: predicted noise displacement with shape ``(N, 3)``
    - ``data.noise_displacement``: target noise displacement with shape ``(N, 3)``
    Optionally, ``data.noise_mask`` can be provided as a boolean mask of shape ``(N,)``.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        noise_weight: float = 1.0,
        noise_key: str = "noise_displacement",
        prediction_target: str = "noise",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_loss = base_loss
        self.noise_weight = float(noise_weight)
        self.noise_key = noise_key
        self.prediction_target = prediction_target

    def _call_base_loss(
        self, data: Any, preds: Dict[str, torch.Tensor], step: int
    ) -> Tuple[torch.Tensor, Dict]:
        out = self.base_loss(data, preds, step)
        if isinstance(out, tuple) and len(out) == 2:
            loss, metrics = out
        else:
            loss, metrics = out, {}
        return loss, dict(metrics)

    def _get_pred_value(self, preds: Any, key: str):
        if isinstance(preds, dict):
            return preds.get(key)
        return getattr(preds, key, None)

    def _compute_noise_loss(
        self, data: Any, preds: Any
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        metrics: Dict[str, float] = {}

        noise_pred = self._get_pred_value(preds, self.noise_key)
        if noise_pred is None:
            metrics["loss/noise"] = 0.0
            return self._zero_like_pred(preds), metrics

        target = getattr(data, "noise_displacement", None)
        if target is None:
            metrics["loss/noise"] = 0.0
            return noise_pred.new_tensor(0.0), metrics

        if not torch.is_tensor(target):
            target = torch.as_tensor(
                target, dtype=noise_pred.dtype, device=noise_pred.device
            )
        else:
            target = target.to(device=noise_pred.device, dtype=noise_pred.dtype)

        if target.shape != noise_pred.shape:
            metrics["loss/noise"] = 0.0
            return noise_pred.new_tensor(0.0), metrics

        if self.prediction_target == "x0":
            target = -target

        per_node = (noise_pred - target).pow(2).sum(dim=-1)

        noise_mask = getattr(data, "noise_mask", None)
        if noise_mask is not None:
            noise_mask = noise_mask.to(device=noise_pred.device)
            if noise_mask.shape == per_node.shape:
                per_node = per_node[noise_mask.bool()]

        if per_node.numel() == 0:
            metrics["loss/noise"] = 0.0
            return noise_pred.new_tensor(0.0), metrics

        noise_loss = self._reduce(per_node)

        with torch.no_grad():
            metrics["noise_rmse"] = per_node.mean().sqrt().item()

        metrics["loss/noise"] = noise_loss.item()
        return noise_loss, metrics

    @staticmethod
    def _zero_like_pred(preds: Any) -> torch.Tensor:
        if isinstance(preds, dict):
            values = preds.values()
        else:
            values = (getattr(preds, name) for name in dir(preds))
        for value in values:
            if torch.is_tensor(value):
                return value.new_tensor(0.0)
        return torch.tensor(0.0)

    def forward(
        self, data: Any, preds: Any, step: int = 0
    ) -> Tuple[torch.Tensor, Dict]:
        base_loss, base_metrics = self._call_base_loss(data, preds, step)
        metrics = dict(base_metrics)
        metrics["loss/base"] = base_loss.item()

        total_loss = base_loss

        if self.noise_weight > 0:
            noise_loss, noise_metrics = self._compute_noise_loss(data, preds)
            total_loss = total_loss + self.noise_weight * noise_loss
            metrics.update(noise_metrics)
        else:
            metrics["loss/noise"] = 0.0

        metrics["loss/noise_weight"] = self.noise_weight
        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics
