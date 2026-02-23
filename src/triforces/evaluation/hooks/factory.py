from __future__ import annotations

import torch
from omegaconf import DictConfig

from .base import TrainHook
from .linear_probe import LinearProbeHook


def _linear_probe_cfg(train_cfg: DictConfig | None) -> DictConfig | None:
    if not isinstance(train_cfg, DictConfig):
        return None

    hooks_cfg = train_cfg.get("hooks")
    if isinstance(hooks_cfg, DictConfig):
        probe_cfg = hooks_cfg.get("linear_probe")
        if isinstance(probe_cfg, DictConfig):
            return probe_cfg
    return None


def build_train_hooks(
    *,
    train_cfg: DictConfig | None,
    dataset: object,
    collate_fn: object,
    train_batch_size: int,
    device: torch.device,
    wandb_run: object | None,
) -> list[TrainHook]:
    hooks: list[TrainHook] = []

    probe_cfg = _linear_probe_cfg(train_cfg)
    if isinstance(probe_cfg, DictConfig) and bool(probe_cfg.get("enabled", False)):
        hooks.append(
            LinearProbeHook(
                probe_cfg=probe_cfg,
                dataset=dataset,
                collate_fn=collate_fn,
                train_batch_size=train_batch_size,
                device=device,
                wandb_run=wandb_run,
            )
        )
    return hooks
