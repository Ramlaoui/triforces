from __future__ import annotations

from types import SimpleNamespace

import torch
from torch_geometric.data import Batch, Data

from triforces.train import _extract_batch_stats


def test_extract_batch_stats_from_pyg_batch() -> None:
    g1 = Data(
        z=torch.tensor([1, 6], dtype=torch.long),
        pos=torch.randn(2, 3),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    g2 = Data(
        z=torch.tensor([8, 1, 1], dtype=torch.long),
        pos=torch.randn(3, 3),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
    )
    batch = Batch.from_data_list([g1, g2])

    stats = _extract_batch_stats(batch)

    assert stats["batch_size"] == 2.0
    assert stats["batch_num_nodes"] == 5.0
    assert stats["batch_num_edges"] == 6.0
    assert stats["batch_nodes_per_graph"] == 2.5
    assert stats["batch_edges_per_graph"] == 3.0
    assert stats["batch_avg_degree"] == 12.0 / 5.0


def test_extract_batch_stats_fallback_without_num_graphs() -> None:
    batch = SimpleNamespace(
        batch=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        edge_index=torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.long),
    )

    stats = _extract_batch_stats(batch)

    assert stats["batch_size"] == 2.0
    assert stats["batch_num_nodes"] == 4.0
    assert stats["batch_num_edges"] == 3.0
