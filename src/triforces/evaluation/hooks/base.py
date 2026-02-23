from __future__ import annotations

import torch
from typing import Protocol


class TrainHook(Protocol):
    def on_train_start(
        self,
        *,
        start_epoch: int,
        epochs: int,
        global_step: int,
        model: torch.nn.Module,
    ) -> None: ...

    def on_step_end(
        self,
        *,
        epoch: int,
        epochs: int,
        global_step: int,
        model: torch.nn.Module,
    ) -> None: ...

    def on_epoch_end(
        self,
        *,
        epoch: int,
        epochs: int,
        global_step: int,
        model: torch.nn.Module,
    ) -> None: ...

    def on_train_end(
        self,
        *,
        epochs: int,
        global_step: int,
        model: torch.nn.Module,
    ) -> None: ...
