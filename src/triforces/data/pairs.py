from __future__ import annotations

from collections import defaultdict

import torch


@torch.no_grad()
def compute_pair_indices(pair_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build paired graph indices from ``pair_id`` values."""
    pair_id = torch.as_tensor(pair_id)
    if pair_id.ndim != 1:
        raise ValueError(f"Expected pair_id as 1D tensor, got {tuple(pair_id.shape)}")

    buckets: dict[int, list[int]] = defaultdict(list)
    for i, pid in enumerate(pair_id.tolist()):
        buckets[int(pid)].append(i)

    idx1_list: list[int] = []
    idx2_list: list[int] = []
    for indices in buckets.values():
        if len(indices) < 2:
            continue
        idx1_list.append(indices[0])
        idx2_list.append(indices[1])

    device = pair_id.device
    if not idx1_list:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty

    return (
        torch.tensor(idx1_list, dtype=torch.long, device=device),
        torch.tensor(idx2_list, dtype=torch.long, device=device),
    )


@torch.no_grad()
def match_nodes_by_correspondence(
    *,
    node_correspondence: torch.Tensor,
    ptr: torch.Tensor,
    idx1: torch.Tensor,
    idx2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match node indices across paired graphs using correspondence IDs."""
    node_correspondence = torch.as_tensor(node_correspondence)
    ptr = torch.as_tensor(ptr)

    device = node_correspondence.device
    node_pair_1: list[torch.Tensor] = []
    node_pair_2: list[torch.Tensor] = []

    for g1, g2 in zip(idx1.tolist(), idx2.tolist()):
        s1, e1 = int(ptr[g1].item()), int(ptr[g1 + 1].item())
        s2, e2 = int(ptr[g2].item()), int(ptr[g2 + 1].item())

        corr1 = node_correspondence[s1:e1]
        corr2 = node_correspondence[s2:e2]

        valid1 = corr1 >= 0
        valid2 = corr2 >= 0
        if not valid1.any() or not valid2.any():
            continue

        corr1v = corr1[valid1]
        corr2v = corr2[valid2]

        idx1v = (torch.arange(s1, e1, device=device, dtype=torch.long))[valid1]
        idx2v = (torch.arange(s2, e2, device=device, dtype=torch.long))[valid2]

        order2 = torch.argsort(corr2v)
        corr2s = corr2v[order2]
        idx2s = idx2v[order2]

        pos = torch.searchsorted(corr2s, corr1v)
        in_bounds = pos < corr2s.numel()
        pos = pos[in_bounds]
        corr1q = corr1v[in_bounds]
        idx1q = idx1v[in_bounds]

        if pos.numel() == 0:
            continue

        matches = corr2s[pos] == corr1q
        if not matches.any():
            continue

        node_pair_1.append(idx1q[matches])
        node_pair_2.append(idx2s[pos[matches]])

    if not node_pair_1:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty

    return torch.cat(node_pair_1, dim=0), torch.cat(node_pair_2, dim=0)
