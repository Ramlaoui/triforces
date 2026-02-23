from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class BackboneOutputs:
    node_feats: torch.Tensor
    graph_feats: torch.Tensor
    extras: dict[str, Any] = field(default_factory=dict)

