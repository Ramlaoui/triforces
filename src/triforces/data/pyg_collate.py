from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence
from torch_geometric.data import Batch

import torch


@torch.no_grad()
def _compute_pair_indices(pair_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build paired indices for a batch.

    Parameters
    ----------
    pair_id : torch.Tensor
        Pair identifiers with shape ``(B,)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(idx1, idx2)`` tensors of matched graph indices.

    Notes
    -----
    Assumes the common contrastive setup where each ``pair_id`` appears exactly twice,
    but skips IDs with fewer than two occurrences.
    """
    pair_id = torch.as_tensor(pair_id)
    if pair_id.ndim != 1:
        raise ValueError(f"Expected pair_id as 1D tensor, got {tuple(pair_id.shape)}")

    buckets: dict[int, list[int]] = defaultdict(list)
    for i, pid in enumerate(pair_id.tolist()):
        buckets[int(pid)].append(i)

    idx1_list: list[int] = []
    idx2_list: list[int] = []
    for pid, indices in buckets.items():
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
def _match_nodes_by_correspondence(
    *,
    node_correspondence: torch.Tensor,
    ptr: torch.Tensor,
    idx1: torch.Tensor,
    idx2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match node indices across paired graphs using correspondence IDs.

    Parameters
    ----------
    node_correspondence : torch.Tensor
        Per-node correspondence IDs.
    ptr : torch.Tensor
        Batch pointer array for nodes.
    idx1 : torch.Tensor
        Graph indices for first view in each pair.
    idx2 : torch.Tensor
        Graph indices for second view in each pair.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Matched node index pairs ``(node_pair_idx1, node_pair_idx2)``.
    """
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

        # Sort corr2 and search for corr1 values.
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


def pyg_contrastive_collate(data_list: Sequence[object]):
    """Collate function for contrastive learning with PyG.

    Parameters
    ----------
    data_list : Sequence[object]
        Batch items, possibly nested as pairs/tuples.

    Returns
    -------
    Batch
        Collated PyG batch with ``pair_idx1/pair_idx2`` and optional node pairs.

    Notes
    -----
    Adds the following batch-level fields expected by some losses:
    - ``pair_idx1``, ``pair_idx2``: graph indices for paired views
    - ``node_pair_idx1``, ``node_pair_idx2``: matched node indices across paired views
    Each graph item must define ``pair_id``; ``node_correspondence`` is optional.
    """
    flat_list: list[object] = []
    for item in data_list:
        if isinstance(item, (list, tuple)):
            flat_list.extend(item)
        else:
            flat_list.append(item)

    batch = Batch.from_data_list(list(flat_list))

    if not hasattr(batch, "pair_id"):
        raise ValueError("Each item must define `pair_id` for contrastive batching.")

    idx1, idx2 = _compute_pair_indices(batch.pair_id)
    batch.pair_idx1 = idx1
    batch.pair_idx2 = idx2

    if hasattr(batch, "node_correspondence") and hasattr(batch, "ptr"):
        n1, n2 = _match_nodes_by_correspondence(
            node_correspondence=batch.node_correspondence,
            ptr=batch.ptr,
            idx1=idx1,
            idx2=idx2,
        )
        batch.node_pair_idx1 = n1
        batch.node_pair_idx2 = n2

    return batch
