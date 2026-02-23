from __future__ import annotations

import torch
from omegaconf import OmegaConf

from triforces.train import _build_loss, _get_train_value


def test_get_train_value_uses_train_section_only() -> None:
    train_cfg = OmegaConf.create({"epochs": 3, "lr": None})
    assert _get_train_value(train_cfg, "epochs", 1) == 3
    assert _get_train_value(train_cfg, "lr", 1e-3) == 1e-3
    assert _get_train_value(None, "epochs", 1) == 1


def test_build_loss_ignores_top_level_legacy_loss_fields() -> None:
    cfg = OmegaConf.create(
        {
            "loss": {},
            "temperature_graph": 9.9,
            "lambda_node": 0.9,
        }
    )
    loss = _build_loss(cfg, torch.device("cpu"))
    assert loss.temperature_graph == 0.1
    assert loss.lambda_node == 0.0
