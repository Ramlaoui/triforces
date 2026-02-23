from __future__ import annotations

import torch
from omegaconf import OmegaConf

from triforces.evaluation.hooks import (
    LinearProbeHook,
    SupervisedEvalHook,
    build_train_hooks,
)


def test_build_train_hooks_uses_new_hooks_namespace() -> None:
    train_cfg = OmegaConf.create(
        {
            "hooks": {
                "linear_probe": {
                    "enabled": True,
                    "every_n_epochs": 1,
                    "run_on_final_epoch": True,
                    "regression_properties": ["n_atoms"],
                    "classification_properties": [],
                }
            }
        }
    )

    hooks = build_train_hooks(
        train_cfg=train_cfg,
        dataset=[1, 2, 3],
        collate_fn=lambda x: x,
        train_batch_size=2,
        device=torch.device("cpu"),
        wandb_run=None,
    )

    assert len(hooks) == 1
    assert isinstance(hooks[0], LinearProbeHook)


def test_build_train_hooks_requires_hooks_namespace() -> None:
    train_cfg = OmegaConf.create({"linear_probe": {"enabled": True}})

    hooks = build_train_hooks(
        train_cfg=train_cfg,
        dataset=[1, 2, 3],
        collate_fn=lambda x: x,
        train_batch_size=2,
        device=torch.device("cpu"),
        wandb_run=None,
    )

    assert hooks == []


def test_build_train_hooks_disabled_when_linear_probe_hook_disabled() -> None:
    train_cfg = OmegaConf.create(
        {
            "hooks": {
                "linear_probe": {
                    "enabled": False,
                }
            }
        }
    )

    hooks = build_train_hooks(
        train_cfg=train_cfg,
        dataset=[1, 2, 3],
        collate_fn=lambda x: x,
        train_batch_size=2,
        device=torch.device("cpu"),
        wandb_run=None,
    )

    assert hooks == []


def test_build_train_hooks_supports_supervised_eval_hook() -> None:
    train_cfg = OmegaConf.create(
        {
            "hooks": {
                "supervised_eval": {
                    "enabled": True,
                    "val_fraction": 0.5,
                    "every_n_steps": 10,
                }
            }
        }
    )

    hooks = build_train_hooks(
        train_cfg=train_cfg,
        dataset=[1, 2, 3, 4],
        collate_fn=lambda x: x,
        train_batch_size=2,
        device=torch.device("cpu"),
        wandb_run=None,
    )

    assert len(hooks) == 1
    assert isinstance(hooks[0], SupervisedEvalHook)


def test_supervised_eval_prefers_explicit_dataset_val_over_split() -> None:
    train_cfg = OmegaConf.create(
        {
            "hooks": {
                "supervised_eval": {
                    "enabled": True,
                    "val_fraction": 0.1,
                }
            }
        }
    )

    hooks = build_train_hooks(
        train_cfg=train_cfg,
        dataset=list(range(20)),
        dataset_val=list(range(4)),
        collate_fn=lambda x: x,
        train_batch_size=2,
        device=torch.device("cpu"),
        wandb_run=None,
    )

    assert len(hooks) == 1
    hook = hooks[0]
    assert isinstance(hook, SupervisedEvalHook)
    assert hook._val_source == "dataset_val"
    assert hook._val_size == 4
