from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = ["NormalizationType", "NormalizationTransform", "NormalizationState"]


class NormalizationType(str, Enum):
    MODEL_OUTPUT = "model_output"


@dataclass(frozen=True)
class NormalizationTransform:
    norm_type: NormalizationType
    params: dict[str, Any] = field(default_factory=dict)


def _matches_params(candidate: dict[str, Any], query: dict[str, Any] | None) -> bool:
    if query is None:
        return True
    for k, v in query.items():
        if candidate.get(k) != v:
            return False
    return True


@dataclass
class NormalizationState:
    transforms: list[NormalizationTransform] = field(default_factory=list)

    def add_transform(
        self,
        norm_type: NormalizationType,
        *,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.transforms.append(NormalizationTransform(norm_type, params or {}))

    def remove_transform(
        self,
        norm_type: NormalizationType,
        *,
        key: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        if params is None and key is not None:
            params = {"key": key}

        self.transforms = [
            t
            for t in self.transforms
            if not (t.norm_type == norm_type and _matches_params(t.params, params))
        ]
