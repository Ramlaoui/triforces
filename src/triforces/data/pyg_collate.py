from __future__ import annotations

from typing import Sequence

from torch_geometric.data import Batch

from triforces.data.pairs import compute_pair_indices, match_nodes_by_correspondence


def _flatten_items(data_list: Sequence[object]) -> list[object]:
    flat_list: list[object] = []
    for item in data_list:
        if isinstance(item, (list, tuple)):
            flat_list.extend(item)
        else:
            flat_list.append(item)
    return flat_list


def pyg_supervised_collate(data_list: Sequence[object]):
    """Collate PyG items without requiring contrastive pair metadata."""
    return Batch.from_data_list(_flatten_items(data_list))


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
    batch = pyg_supervised_collate(data_list)

    if not hasattr(batch, "pair_id"):
        raise ValueError("Each item must define `pair_id` for contrastive batching.")

    idx1, idx2 = compute_pair_indices(batch.pair_id)
    batch.pair_idx1 = idx1
    batch.pair_idx2 = idx2

    if hasattr(batch, "node_correspondence") and hasattr(batch, "ptr"):
        n1, n2 = match_nodes_by_correspondence(
            node_correspondence=batch.node_correspondence,
            ptr=batch.ptr,
            idx1=idx1,
            idx2=idx2,
        )
        batch.node_pair_idx1 = n1
        batch.node_pair_idx2 = n2

    return batch
