from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RetrievalMetrics:
    r_at_1: float
    r_at_5: float
    r_at_10: float
    mrr: float


def _as_tensor_1d(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x)
    if x.ndim != 1:
        raise ValueError(f"Expected a 1D tensor, got shape {tuple(x.shape)}")
    return x


@torch.no_grad()
def compute_retrieval_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    ks: Iterable[int] = (1, 5, 10),
    normalize: bool = True,
) -> RetrievalMetrics:
    """Compute retrieval metrics for self-supervised embeddings.

    Parameters
    ----------
    embeddings : torch.Tensor
        Embeddings with shape ``(N, D)``.
    labels : torch.Tensor
        Labels with shape ``(N,)`` used to define relevance.
    ks : Iterable[int], default=(1, 5, 10)
        Recall cutoff values.
    normalize : bool, default=True
        Whether to L2-normalize embeddings before computing similarity.

    Returns
    -------
    RetrievalMetrics
        Recall@K and mean reciprocal rank metrics.

    Notes
    -----
    Relevance is defined by matching labels (excluding self-match). This is useful
    when labels are pair IDs or class IDs.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings [N, D], got {tuple(embeddings.shape)}")

    labels = _as_tensor_1d(labels).to(device=embeddings.device)
    if labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels and embeddings must have the same first dimension")

    if normalize:
        embeddings = F.normalize(embeddings, dim=-1, p=2)

    sim = embeddings @ embeddings.t()
    sim.fill_diagonal_(-float("inf"))

    sorted_idx = torch.argsort(sim, dim=1, descending=True)
    sorted_labels = labels[sorted_idx]

    relevant = sorted_labels == labels.unsqueeze(1)
    first_rel = relevant.float().argmax(dim=1)
    has_rel = relevant.any(dim=1)

    ranks = first_rel[has_rel] + 1  # 1-based
    if ranks.numel() == 0:
        return RetrievalMetrics(r_at_1=0.0, r_at_5=0.0, r_at_10=0.0, mrr=0.0)

    ks = sorted(set(int(k) for k in ks if int(k) > 0))
    recalls = {}
    for k in ks:
        recalls[k] = relevant[:, :k].any(dim=1)[has_rel].float().mean().item()

    r1 = recalls.get(1, 0.0)
    r5 = recalls.get(5, 0.0)
    r10 = recalls.get(10, 0.0)
    mrr = (1.0 / ranks.float()).mean().item()
    return RetrievalMetrics(r_at_1=r1, r_at_5=r5, r_at_10=r10, mrr=mrr)


@torch.no_grad()
def compute_contrastive_metrics(
    graph_projections: torch.Tensor,
    pair_ids: torch.Tensor,
    *,
    temperature: float = 0.1,
    normalize: bool = True,
) -> dict[str, float]:
    """Compute basic contrastive diagnostics for graph-level projections.

    Parameters
    ----------
    graph_projections : torch.Tensor
        Graph-level projections with shape ``(B, D)``.
    pair_ids : torch.Tensor
        Pair IDs with shape ``(B,)`` identifying positives.
    temperature : float, default=0.1
        Temperature for similarity scaling.
    normalize : bool, default=True
        Whether to L2-normalize projections before similarity computation.

    Returns
    -------
    dict[str, float]
        Dictionary with ``acc@1``, ``pos_sim_mean``, and ``neg_sim_mean``.
    """
    if graph_projections.ndim != 2:
        raise ValueError(
            f"Expected graph_projections [B, D], got {tuple(graph_projections.shape)}"
        )

    pair_ids = _as_tensor_1d(pair_ids).to(device=graph_projections.device)
    if pair_ids.shape[0] != graph_projections.shape[0]:
        raise ValueError("pair_ids and graph_projections must share batch size")

    z = graph_projections
    if normalize:
        z = F.normalize(z, dim=-1, p=2)

    sim = (z @ z.t()) / float(temperature)
    sim.fill_diagonal_(-float("inf"))

    same = pair_ids.unsqueeze(0) == pair_ids.unsqueeze(1)
    same.fill_diagonal_(False)

    top1 = sim.argmax(dim=1)
    acc1 = same.gather(1, top1.unsqueeze(1)).squeeze(1).float().mean().item()

    pos_sims = sim.masked_select(same)
    neg_sims = sim.masked_select(~same)

    pos_mean = pos_sims.mean().item() if pos_sims.numel() else 0.0
    neg_mean = neg_sims.mean().item() if neg_sims.numel() else 0.0

    return {"acc@1": acc1, "pos_sim_mean": pos_mean, "neg_sim_mean": neg_mean}
