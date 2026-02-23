from __future__ import annotations

from typing import Any, Iterable

import torch
from torch_geometric.data import Batch

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
