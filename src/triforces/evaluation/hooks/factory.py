from __future__ import annotations

import torch
from omegaconf import DictConfig

from .base import TrainHook
from .linear_probe import LinearProbeHook
from .supervised_eval import SupervisedEvalHook


def _hook_cfg(train_cfg: DictConfig | None, hook_name: str) -> DictConfig | None:
    if not isinstance(train_cfg, DictConfig):
        return None

    hooks_cfg = train_cfg.get("hooks")
    if isinstance(hooks_cfg, DictConfig):
        hook_cfg = hooks_cfg.get(hook_name)
        if isinstance(hook_cfg, DictConfig):
            return hook_cfg
    return None


def build_train_hooks(
    *,
    train_cfg: DictConfig | None,
    dataset: object,
    collate_fn: object,
    train_batch_size: int,
    device: torch.device,
    wandb_run: object | None,
    dataset_val: object | None = None,
    loss_fn: torch.nn.Module | None = None,
) -> list[TrainHook]:
    hooks: list[TrainHook] = []

    probe_cfg = _hook_cfg(train_cfg, "linear_probe")
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

    eval_cfg = _hook_cfg(train_cfg, "supervised_eval")
    if isinstance(eval_cfg, DictConfig) and bool(eval_cfg.get("enabled", False)):
        hooks.append(
            SupervisedEvalHook(
                eval_cfg=eval_cfg,
                dataset=dataset,
                dataset_val=dataset_val,
                collate_fn=collate_fn,
                train_batch_size=train_batch_size,
                device=device,
                wandb_run=wandb_run,
                loss_fn=loss_fn,
            )
        )
    return hooks
