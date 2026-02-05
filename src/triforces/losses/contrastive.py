"""Contrastive losses without explicit negative sampling heuristics."""

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from .base import BaseLoss


def _get_pred_value(preds: Any, key: str):
    if isinstance(preds, dict):
        return preds.get(key)
    return getattr(preds, key, None)


class ContrastiveLoss(BaseLoss):
    """Simplified contrastive loss using full similarity matrices.

    Parameters
    ----------
    temperature_node : float, default=0.07
        Temperature for node-level similarities.
    temperature_graph : float, default=0.1
        Temperature for graph-level similarities.
    lambda_node : float, default=0.5
        Weight for the node-level loss.
    lambda_graph : float, default=0.5
        Weight for the graph-level loss.
    max_negatives : int, default=1024
        Maximum number of negatives to include for node-level loss.
    similarity_metric : str, default="cosine"
        Similarity metric to use (currently supports ``"cosine"``).
    **kwargs : Any
        Additional keyword arguments forwarded to ``BaseLoss``.
    """

    def __init__(
        self,
        temperature_node: float = 0.07,
        temperature_graph: float = 0.1,
        lambda_node: float = 0.5,
        lambda_graph: float = 0.5,
        max_negatives: int = 1024,
        similarity_metric: str = "cosine",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.temperature_node = temperature_node
        self.temperature_graph = temperature_graph
        self.lambda_node = lambda_node
        self.lambda_graph = lambda_graph
        self.max_negatives = max_negatives
        self.similarity_metric = similarity_metric

        total_lambda = lambda_node + lambda_graph
        if total_lambda > 0:
            self.lambda_node /= total_lambda
            self.lambda_graph /= total_lambda

    def compute_node_level_loss(
        self,
        node_embeddings: torch.Tensor,
        node_correspondence: torch.Tensor,
        batch_idx: torch.Tensor,
        pair_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        device = node_embeddings.device
        metrics = {}

        if self.similarity_metric == "cosine":
            node_embeddings = F.normalize(node_embeddings, dim=-1, p=2)

        valid_mask = node_correspondence >= 0
        valid_indices = torch.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), {
                "node_accuracy": 0,
                "n_pairs": 0,
            }

        valid_corrs = node_correspondence[valid_indices]
        valid_batches = batch_idx[valid_indices]
        node_pair_ids = pair_ids[batch_idx]
        valid_pair_ids = node_pair_ids[valid_indices]

        corr_match = valid_corrs.unsqueeze(0) == valid_corrs.unsqueeze(1)
        same_pair = valid_pair_ids.unsqueeze(0) == valid_pair_ids.unsqueeze(1)
        diff_graph = valid_batches.unsqueeze(0) != valid_batches.unsqueeze(1)

        positive_mask = corr_match & same_pair & diff_graph

        # Get unique anchor-positive pairs
        anchor_idx, positive_idx = torch.where(positive_mask)
        mask = anchor_idx < positive_idx  # Avoid duplicates
        anchor_idx = anchor_idx[mask]
        positive_idx = positive_idx[mask]

        if len(anchor_idx) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), {
                "node_accuracy": 0,
                "n_pairs": 0,
            }

        # Map back to original indices
        anchor_indices = valid_indices[anchor_idx]
        positive_indices = valid_indices[positive_idx]

        n_total = len(node_embeddings)
        n_anchors = len(anchor_indices)

        if n_total <= self.max_negatives:
            all_sims = (
                torch.mm(node_embeddings[anchor_indices], node_embeddings.t())
                / self.temperature_node
            )

            all_sims[torch.arange(n_anchors), anchor_indices] = -float("inf")

            positive_mask = torch.zeros_like(all_sims, dtype=torch.bool)
            positive_mask[torch.arange(n_anchors), positive_indices] = True

        else:
            # Sample subset of negatives
            essential_indices = torch.unique(
                torch.cat([anchor_indices, positive_indices])
            )
            n_essential = len(essential_indices)

            all_indices = torch.arange(n_total, device=device)
            mask = torch.ones(n_total, dtype=torch.bool, device=device)
            mask[essential_indices] = False
            available_indices = all_indices[mask]

            n_to_sample = min(self.max_negatives - n_essential, len(available_indices))
            if n_to_sample > 0:
                sampled = available_indices[
                    torch.randperm(len(available_indices))[:n_to_sample]
                ]
                keep_indices = torch.cat([essential_indices, sampled])
            else:
                keep_indices = essential_indices

            keep_indices, _ = torch.sort(keep_indices)

            all_sims = (
                torch.mm(
                    node_embeddings[anchor_indices], node_embeddings[keep_indices].t()
                )
                / self.temperature_node
            )

            anchor_positions = (
                keep_indices.unsqueeze(0) == anchor_indices.unsqueeze(1)
            ).nonzero()[:, 1]
            positive_positions = (
                keep_indices.unsqueeze(0) == positive_indices.unsqueeze(1)
            ).nonzero()[:, 1]

            all_sims[torch.arange(n_anchors), anchor_positions] = -float("inf")

            positive_mask = torch.zeros_like(all_sims, dtype=torch.bool)
            positive_mask[torch.arange(n_anchors), positive_positions] = True

        pos_sims = all_sims.masked_fill(~positive_mask, -float("inf"))
        pos_logsumexp = torch.logsumexp(pos_sims, dim=1)
        all_logsumexp = torch.logsumexp(all_sims, dim=1)

        loss = (-pos_logsumexp + all_logsumexp).mean()

        with torch.no_grad():
            predictions = all_sims.argmax(dim=1)
            correct = positive_mask.gather(1, predictions.unsqueeze(1)).squeeze()
            accuracy = correct.float().mean().item()

        metrics["node_accuracy"] = accuracy
        metrics["n_pairs"] = n_anchors

        return loss, metrics

    def compute_graph_level_loss(
        self, graph_embeddings: torch.Tensor, pair_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        device = graph_embeddings.device
        metrics = {}

        # Normalize embeddings if using cosine similarity
        if self.similarity_metric == "cosine":
            graph_embeddings = F.normalize(graph_embeddings, dim=-1, p=2)

        n_graphs = len(graph_embeddings)

        # Compute similarity matrix
        sim_matrix = (
            torch.mm(graph_embeddings, graph_embeddings.t()) / self.temperature_graph
        )

        # Mask out self-similarities
        sim_matrix.fill_diagonal_(-float("inf"))

        # Create positive mask based on pair IDs
        positive_mask = pair_ids.unsqueeze(0) == pair_ids.unsqueeze(1)
        positive_mask.fill_diagonal_(False)

        # Find graphs with at least one positive
        has_positive = positive_mask.any(dim=1)
        valid_indices = torch.where(has_positive)[0]

        if len(valid_indices) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), {
                "graph_accuracy": 0,
                "n_graphs": 0,
            }

        # Compute loss only for valid graphs
        valid_sims = sim_matrix[valid_indices]
        valid_pos_mask = positive_mask[valid_indices]

        # Compute InfoNCE loss
        pos_sims = valid_sims.masked_fill(~valid_pos_mask, -float("inf"))
        pos_logsumexp = torch.logsumexp(pos_sims, dim=1)
        all_logsumexp = torch.logsumexp(valid_sims, dim=1)

        loss = (-pos_logsumexp + all_logsumexp).mean()

        # Compute accuracy
        with torch.no_grad():
            predictions = valid_sims.argmax(dim=1)
            correct = valid_pos_mask.gather(1, predictions.unsqueeze(1)).squeeze()
            accuracy = correct.float().mean().item()

        metrics["graph_accuracy"] = accuracy
        metrics["n_graphs"] = len(valid_indices)

        return loss, metrics

    def forward(
        self, data: Any, preds: Any, step: int = 0
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute the contrastive loss for a batch.

        Parameters
        ----------
        data : Any
            Batch containing ``pair_id`` and optionally ``batch`` and
            ``node_correspondence`` attributes (typically a PyG ``Data``/``Batch``).
        preds : Any
            Model predictions containing ``node_projections`` and/or
            ``graph_projections``.
        step : int, default=0
            Training step (unused).

        Returns
        -------
        torch.Tensor
            Total contrastive loss.
        dict
            Metrics dictionary for logging.
        """
        metrics = {}

        device = getattr(data, "batch", torch.tensor([], dtype=torch.long)).device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        if not hasattr(data, "pair_id"):
            raise ValueError("Data must contain 'pair_id' field")

        pair_ids = data.pair_id

        node_projections = _get_pred_value(preds, "node_projections")
        graph_projections = _get_pred_value(preds, "graph_projections")

        if (
            self.lambda_node > 0
            and node_projections is not None
            and hasattr(data, "node_correspondence")
            and hasattr(data, "batch")
        ):
            node_loss, node_metrics = self.compute_node_level_loss(
                node_projections,
                data.node_correspondence,
                data.batch,
                pair_ids,
            )

            total_loss = total_loss + self.lambda_node * node_loss
            metrics["node_loss"] = node_loss.item()
            metrics.update({f"node_{k}": v for k, v in node_metrics.items()})

        if self.lambda_graph > 0 and graph_projections is not None:
            graph_loss, graph_metrics = self.compute_graph_level_loss(
                graph_projections, pair_ids
            )

            total_loss = total_loss + self.lambda_graph * graph_loss
            metrics["graph_loss"] = graph_loss.item()
            metrics.update({f"graph_{k}": v for k, v in graph_metrics.items()})

        if not isinstance(total_loss, torch.Tensor):
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        metrics["total_loss"] = total_loss.item()

        return total_loss, metrics
