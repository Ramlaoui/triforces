"""Add an MSE reconstruction term on top of a base loss."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLoss


class ReconstructionLoss(BaseLoss):
    """Combine a base loss with an MSE reconstruction term.

    Parameters
    ----------
    base_loss : nn.Module
        Loss module to wrap. Must return ``(loss, metrics)`` or a scalar loss.
    base_weight : float, default=1.0
        Global weight applied to ``base_loss`` before adding auxiliary terms.
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
        base_weight: float = 1.0,
        noise_weight: float = 1.0,
        noise_key: str = "noise_displacement",
        prediction_target: str = "noise",
        atom_type_weight: float = 0.0,
        atom_type_key: str = "atom_type_logits",
        atom_type_target_key: str = "original_numbers",
        atom_type_mask_key: str = "atom_mask",
        mask_token: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_loss = base_loss
        self.base_weight = float(base_weight)
        self.noise_weight = float(noise_weight)
        self.noise_key = noise_key
        self.prediction_target = prediction_target
        self.atom_type_weight = float(atom_type_weight)
        self.atom_type_key = str(atom_type_key)
        self.atom_type_target_key = str(atom_type_target_key)
        self.atom_type_mask_key = str(atom_type_mask_key)
        self.mask_token = int(mask_token)

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
            metrics["noise_rmse"] = 0.0
            metrics["noise_cosine_similarity"] = 0.0
            return self._zero_like_pred(preds), metrics

        target = getattr(data, "noise_displacement", None)
        if target is None:
            metrics["loss/noise"] = 0.0
            metrics["noise_rmse"] = 0.0
            metrics["noise_cosine_similarity"] = 0.0
            return noise_pred.new_tensor(0.0), metrics

        if not torch.is_tensor(target):
            target = torch.as_tensor(
                target, dtype=noise_pred.dtype, device=noise_pred.device
            )
        else:
            target = target.to(device=noise_pred.device, dtype=noise_pred.dtype)

        if target.shape != noise_pred.shape:
            metrics["loss/noise"] = 0.0
            metrics["noise_rmse"] = 0.0
            metrics["noise_cosine_similarity"] = 0.0
            return noise_pred.new_tensor(0.0), metrics

        if self.prediction_target == "x0":
            target = -target

        per_node = (noise_pred - target).pow(2).sum(dim=-1)

        noise_mask = getattr(data, "noise_mask", None)
        if noise_mask is not None:
            noise_mask = noise_mask.to(device=noise_pred.device)
            if noise_mask.shape == per_node.shape:
                mask = noise_mask.bool()
                per_node = per_node[mask]
                noise_pred = noise_pred[mask]
                target = target[mask]

        if per_node.numel() == 0:
            metrics["loss/noise"] = 0.0
            metrics["noise_rmse"] = 0.0
            metrics["noise_cosine_similarity"] = 0.0
            return noise_pred.new_tensor(0.0), metrics

        noise_loss = self._reduce(per_node)

        with torch.no_grad():
            metrics["noise_rmse"] = per_node.mean().sqrt().item()
            metrics["noise_cosine_similarity"] = (
                F.cosine_similarity(noise_pred, target, dim=-1, eps=1e-8).mean().item()
            )

        metrics["loss/noise"] = noise_loss.item()
        return noise_loss, metrics

    def _compute_atom_type_loss(
        self, data: Any, preds: Any
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        metrics: Dict[str, float] = {}
        logits = self._get_pred_value(preds, self.atom_type_key)
        if logits is None:
            metrics["loss/atom_type"] = 0.0
            metrics["atom_type_accuracy_masked"] = 0.0
            metrics["n_masked_atoms"] = 0.0
            return self._zero_like_pred(preds), metrics

        if logits.ndim != 2:
            metrics["loss/atom_type"] = 0.0
            metrics["atom_type_accuracy_masked"] = 0.0
            metrics["n_masked_atoms"] = 0.0
            return logits.new_tensor(0.0), metrics

        target = getattr(data, self.atom_type_target_key, None)
        if target is None:
            metrics["loss/atom_type"] = 0.0
            metrics["atom_type_accuracy_masked"] = 0.0
            metrics["n_masked_atoms"] = 0.0
            return logits.new_tensor(0.0), metrics
        if not torch.is_tensor(target):
            target = torch.as_tensor(target, device=logits.device, dtype=torch.long)
        else:
            target = target.to(device=logits.device, dtype=torch.long)
        target = target.reshape(-1)

        if target.numel() != logits.size(0):
            metrics["loss/atom_type"] = 0.0
            metrics["atom_type_accuracy_masked"] = 0.0
            metrics["n_masked_atoms"] = 0.0
            return logits.new_tensor(0.0), metrics

        mask = getattr(data, self.atom_type_mask_key, None)
        if mask is not None:
            if not torch.is_tensor(mask):
                mask = torch.as_tensor(mask, device=logits.device, dtype=torch.bool)
            else:
                mask = mask.to(device=logits.device, dtype=torch.bool)
            mask = mask.reshape(-1)
            if mask.numel() != target.numel():
                mask = None
        if mask is None and hasattr(data, "z"):
            z = data.z
            if torch.is_tensor(z):
                mask = z.to(device=logits.device, dtype=torch.long) == self.mask_token

        if mask is None:
            mask = torch.zeros(
                (target.numel(),), device=logits.device, dtype=torch.bool
            )

        valid = (target >= 0) & (target < logits.size(-1))
        mask = mask & valid
        if not mask.any():
            metrics["loss/atom_type"] = 0.0
            metrics["atom_type_accuracy_masked"] = 0.0
            metrics["n_masked_atoms"] = 0.0
            return logits.new_tensor(0.0), metrics

        atom_type_loss = F.cross_entropy(
            logits[mask], target[mask], reduction=self.reduction
        )
        with torch.no_grad():
            preds_argmax = logits[mask].argmax(dim=-1)
            metrics["atom_type_accuracy_masked"] = (
                (preds_argmax == target[mask]).float().mean().item()
            )
            metrics["n_masked_atoms"] = float(mask.sum().item())
        metrics["loss/atom_type"] = float(atom_type_loss.item())
        return atom_type_loss, metrics

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
        metrics: Dict[str, float] = {}
        metrics["loss/base_weight"] = self.base_weight

        if self.base_weight == 0.0:
            total_loss = self._zero_like_pred(preds)
        else:
            base_loss, base_metrics = self._call_base_loss(data, preds, step)
            metrics.update(base_metrics)
            metrics["loss/base"] = base_loss.item()
            total_loss = self.base_weight * base_loss

        if self.noise_weight > 0:
            noise_loss, noise_metrics = self._compute_noise_loss(data, preds)
            total_loss = total_loss + self.noise_weight * noise_loss
            metrics.update(noise_metrics)
        else:
            metrics["loss/noise"] = 0.0

        if self.atom_type_weight > 0:
            atom_type_loss, atom_type_metrics = self._compute_atom_type_loss(
                data, preds
            )
            total_loss = total_loss + self.atom_type_weight * atom_type_loss
            metrics.update(atom_type_metrics)
        else:
            metrics["loss/atom_type"] = 0.0

        metrics["loss/noise_weight"] = self.noise_weight
        metrics["loss/atom_type_weight"] = self.atom_type_weight
        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics
