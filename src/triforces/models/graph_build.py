from __future__ import annotations

import torch
from torch_geometric.nn import radius_graph as pyg_radius_graph

__all__ = ["radius_graph"]


def radius_graph(
    *,
    pos: torch.Tensor,
    batch: torch.Tensor | None,
    r: float,
    max_num_neighbors: int | None = 32,
    loop: bool = False,
) -> torch.Tensor:
    """Build a radius graph with torch-cluster."""
    max_n = pos.size(0) if max_num_neighbors is None else int(max_num_neighbors)
    return pyg_radius_graph(
        pos,
        r=float(r),
        batch=batch,
        loop=bool(loop),
        max_num_neighbors=max_n,
    )
