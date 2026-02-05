from __future__ import annotations
from typing import Any, Iterable
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from torch_geometric.data import Batch

import torch


BATCH_INFO_KEYS = (
    "batch",
    "ptr",
    "num_graphs",
    "pair_id",
    "pair_idx1",
    "pair_idx2",
    "node_pair_idx1",
    "node_pair_idx2",
    "node_correspondence",
)


def build_output_batch(
    batch: Any,
    outputs: dict[str, Any],
    *,
    keep_keys: Iterable[str] | None = None,
) -> Any:
    out = Batch()

    keys = list(BATCH_INFO_KEYS)
    if keep_keys:
        keys.extend(list(keep_keys))

    for key in keys:
        if hasattr(batch, key):
            setattr(out, key, getattr(batch, key))

    if not hasattr(out, "num_graphs") or out.num_graphs is None:
        if hasattr(out, "batch") and out.batch is not None:
            num_graphs = int(out.batch.max().item()) + 1 if out.batch.numel() else 0
            out.num_graphs = num_graphs

    if not hasattr(out, "ptr") and hasattr(out, "batch") and out.batch is not None:
        num_graphs = getattr(out, "num_graphs", None)
        if num_graphs is None:
            num_graphs = int(out.batch.max().item()) + 1 if out.batch.numel() else 0
            out.num_graphs = num_graphs
        counts = torch.bincount(out.batch, minlength=int(num_graphs))
        ptr = torch.zeros(
            int(num_graphs) + 1, device=out.batch.device, dtype=torch.long
        )
        ptr[1:] = torch.cumsum(counts, dim=0)
        out.ptr = ptr

    for key, value in outputs.items():
        setattr(out, key, value)

    return out


def stress_to_voigt_6(stress: torch.Tensor | None) -> torch.Tensor | None:
    if stress is None:
        return None

    if stress.shape[-1] == 6:
        return stress

    if stress.shape[-1] == 9:
        stress = stress.reshape(-1, 3, 3)

    if stress.shape[-2:] != (3, 3):
        raise ValueError(
            "Input stress tensor must have shape (..., 3, 3) or (..., 6). "
            f"Got shape {stress.shape}"
        )

    batch_shape = stress.shape[:-2]
    voigt = torch.empty((*batch_shape, 6), dtype=stress.dtype, device=stress.device)

    voigt[..., 0] = stress[..., 0, 0]
    voigt[..., 1] = stress[..., 1, 1]
    voigt[..., 2] = stress[..., 2, 2]
    voigt[..., 3] = (stress[..., 1, 2] + stress[..., 2, 1]) * 0.5
    voigt[..., 4] = (stress[..., 2, 0] + stress[..., 0, 2]) * 0.5
    voigt[..., 5] = (stress[..., 0, 1] + stress[..., 1, 0]) * 0.5

    return voigt


def voigt_6_to_stress(voigt: torch.Tensor | None) -> torch.Tensor | None:
    if voigt is None:
        return None

    if voigt.shape[-1] == 9:
        return voigt.reshape(-1, 3, 3)
    if voigt.shape[-2:] == (3, 3):
        return voigt

    if voigt.shape[-1] != 6:
        raise ValueError(
            f"Input voigt tensor must have shape (..., 6). Got shape {voigt.shape}"
        )

    batch_shape = voigt.shape[:-1]
    stress = torch.empty((*batch_shape, 3, 3), dtype=voigt.dtype, device=voigt.device)

    stress[..., 0, 0] = voigt[..., 0]
    stress[..., 1, 1] = voigt[..., 1]
    stress[..., 2, 2] = voigt[..., 2]

    stress[..., 1, 2] = voigt[..., 3]
    stress[..., 2, 1] = voigt[..., 3]

    stress[..., 2, 0] = voigt[..., 4]
    stress[..., 0, 2] = voigt[..., 4]

    stress[..., 0, 1] = voigt[..., 5]
    stress[..., 1, 0] = voigt[..., 5]

    return stress


@dataclass(init=False)
class BackboneOutputs:
    node_feats: torch.Tensor
    graph_feats: torch.Tensor
    extras: Dict[str, Any]

    def __init__(
        self,
        *,
        node_feats: Optional[torch.Tensor] = None,
        graph_feats: Optional[torch.Tensor] = None,
        node_features: Optional[torch.Tensor] = None,
        graph_features: Optional[torch.Tensor] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        if node_feats is None:
            node_feats = node_features
        if graph_feats is None:
            graph_feats = graph_features
        if node_feats is None or graph_feats is None:
            missing = []
            if node_feats is None:
                missing.append("node_feats/node_features")
            if graph_feats is None:
                missing.append("graph_feats/graph_features")
            raise ValueError(f"BackboneOutputs missing {', '.join(missing)}")
        self.node_feats = node_feats
        self.graph_feats = graph_feats
        self.extras = extras or {}

    @property
    def node_features(self) -> torch.Tensor:
        return self.node_feats

    @property
    def graph_features(self) -> torch.Tensor:
        return self.graph_feats
