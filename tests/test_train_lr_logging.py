from __future__ import annotations

import torch
import torch.nn as nn

from triforces.train import _optimizer_lr


def test_optimizer_lr_returns_first_param_group_lr() -> None:
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
    optim = torch.optim.AdamW(
        [
            {"params": model[0].parameters(), "lr": 3e-4},
            {"params": model[1].parameters(), "lr": 1e-4},
        ]
    )

    assert _optimizer_lr(optim) == 3e-4


def test_optimizer_lr_reflects_runtime_updates() -> None:
    model = nn.Linear(4, 2)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optim.param_groups[0]["lr"] = 5e-4

    assert _optimizer_lr(optim) == 5e-4
