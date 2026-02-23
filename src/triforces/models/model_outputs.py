from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .normalization import NormalizationState

__all__ = ["ModelOutputs"]


@dataclass
class ModelOutputs:
    """Lightweight container for model predictions + metadata.

    This is intentionally minimal: it supports attribute access and dict-like
    access for downstream code that expects ``outputs["energy"]`` etc.
    """

    energy: torch.Tensor | None = None
    forces: torch.Tensor | None = None

    batch: torch.Tensor | None = None
    batch_size: int = 1
    ptr: torch.Tensor | None = None

    attributes: dict[str, Any] = field(default_factory=dict)
    normalization_state: NormalizationState | None = None

    def __getitem__(self, key: str) -> Any:
        if key == "energy":
            return self.energy
        if key == "forces":
            return self.forces
        if key in {"batch", "batch_size", "ptr", "attributes", "normalization_state"}:
            return getattr(self, key)
        return self.attributes[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "energy":
            self.energy = value
            return
        if key == "forces":
            self.forces = value
            return
        if key in {"batch", "batch_size", "ptr", "attributes", "normalization_state"}:
            setattr(self, key, value)
            return
        self.attributes[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default
