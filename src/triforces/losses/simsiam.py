"""SimSiam loss for self-supervised learning without negative pairs.

Based on "Exploring Simple Siamese Representation Learning" (Chen & He, 2020),
SimSiam uses a stop-gradient operation to prevent collapse without relying on
negative pairs, large batches, or momentum encoders.
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimSiamLoss(nn.Module):
    """SimSiam loss using cosine similarity and stop-gradient.

    The loss is computed as:
    ``0.5 * (D(p1, z2) + D(p2, z1))``, where ``D`` is negative cosine similarity.

    Parameters
    ----------
    lambda_node : float, default=0.5
        Weight for node-level loss.
    lambda_graph : float, default=0.5
        Weight for graph-level loss.
    symmetrize : bool, default=True
        Whether to compute symmetric loss (both directions).
    """

    def __init__(
        self,
        lambda_node: float = 0.5,
        lambda_graph: float = 0.5,
        symmetrize: bool = True,
    ):
        super().__init__()

        self.lambda_node = lambda_node
        self.lambda_graph = lambda_graph
        self.symmetrize = symmetrize

        total_lambda = lambda_node + lambda_graph
        if total_lambda > 0:
            self.lambda_node /= total_lambda
            self.lambda_graph /= total_lambda

    def _forward_with_pairs(
        self, data: Any, preds: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute loss when the batch contains paired indices.

        Parameters
        ----------
        data : Any
            Batch with ``pair_id`` and optionally ``batch`` attributes.
        preds : dict[str, torch.Tensor]
            Dictionary containing projections and predictions.

        Returns
        -------
        torch.Tensor
            Total loss value.
        dict
            Metrics dictionary.
        """
        if not hasattr(data, "pair_idx1") or not hasattr(data, "pair_idx2"):
            raise ValueError(
                "SimSiamLoss requires `pair_idx1` and `pair_idx2` from collate."
            )

        device = getattr(data, "batch", torch.tensor([], dtype=torch.long)).device
        idx1 = data.pair_idx1.to(device)
        idx2 = data.pair_idx2.to(device)
        if idx1.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), {
                "skipped": "no_valid_pairs"
            }

        kwargs = {}
        if "graph_predictions" in preds and "graph_projections" in preds:
            kwargs["graph_predictions1"] = preds["graph_predictions"][idx1]
            kwargs["graph_projections1"] = preds["graph_projections"][idx1]
            kwargs["graph_predictions2"] = preds["graph_predictions"][idx2]
            kwargs["graph_projections2"] = preds["graph_projections"][idx2]

        if (
            self.lambda_node > 0
            and "node_predictions" in preds
            and "node_projections" in preds
        ):
            if not hasattr(data, "node_pair_idx1") or not hasattr(
                data, "node_pair_idx2"
            ):
                raise ValueError(
                    "SimSiamLoss node loss requires `node_pair_idx1` and "
                    "`node_pair_idx2` from collate."
                )
            node_idx1 = data.node_pair_idx1.to(device)
            node_idx2 = data.node_pair_idx2.to(device)
            if node_idx1.numel() > 0:
                kwargs["node_predictions1"] = preds["node_predictions"][node_idx1]
                kwargs["node_projections1"] = preds["node_projections"][node_idx1]
                kwargs["node_predictions2"] = preds["node_predictions"][node_idx2]
                kwargs["node_projections2"] = preds["node_projections"][node_idx2]

        if not kwargs:
            return torch.tensor(0.0, device=device, requires_grad=True), {
                "skipped": "no_valid_pairs"
            }

        return self._compute_from_views(**kwargs)

    def negative_cosine_similarity(
        self, p: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Compute negative cosine similarity between predictions and projections.

        Parameters
        ----------
        p : torch.Tensor
            Predictions from the predictor MLP (not normalized).
        z : torch.Tensor
            Projections with stop-gradient (already L2-normalized).

        Returns
        -------
        torch.Tensor
            Negative cosine similarity loss.
        """
        # Only normalize predictions; projections are already normalized
        p = F.normalize(p, dim=-1, p=2)

        return -(p * z).sum(dim=-1).mean()

    def _compute_from_views(
        self,
        *,
        node_predictions1: torch.Tensor | None = None,
        node_projections1: torch.Tensor | None = None,
        node_predictions2: torch.Tensor | None = None,
        node_projections2: torch.Tensor | None = None,
        graph_predictions1: torch.Tensor | None = None,
        graph_projections1: torch.Tensor | None = None,
        graph_predictions2: torch.Tensor | None = None,
        graph_projections2: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict]:
        metrics = {}
        total_loss = 0

        # Node-level loss
        if (
            self.lambda_node > 0
            and node_predictions1 is not None
            and node_projections1 is not None
            and node_predictions2 is not None
            and node_projections2 is not None
        ):
            z1_sg = node_projections1.detach()
            z2_sg = node_projections2.detach()

            node_loss = self.negative_cosine_similarity(node_predictions1, z2_sg)
            if self.symmetrize:
                node_loss = 0.5 * (
                    node_loss
                    + self.negative_cosine_similarity(node_predictions2, z1_sg)
                )

            total_loss = total_loss + self.lambda_node * node_loss
            metrics["node_loss"] = node_loss.item()
            metrics["node_cos_sim"] = -node_loss.item()

        # Graph-level loss
        if (
            self.lambda_graph > 0
            and graph_predictions1 is not None
            and graph_projections1 is not None
            and graph_predictions2 is not None
            and graph_projections2 is not None
        ):
            z1_sg = graph_projections1.detach()
            z2_sg = graph_projections2.detach()

            graph_loss = self.negative_cosine_similarity(graph_predictions1, z2_sg)
            if self.symmetrize:
                graph_loss = 0.5 * (
                    graph_loss
                    + self.negative_cosine_similarity(graph_predictions2, z1_sg)
                )

            total_loss = total_loss + self.lambda_graph * graph_loss
            metrics["graph_loss"] = graph_loss.item()
            metrics["graph_cos_sim"] = -graph_loss.item()

        metrics["total_loss"] = (
            total_loss.item() if isinstance(total_loss, torch.Tensor) else 0
        )

        # Add comprehensive collapse monitoring metrics
        with torch.no_grad():
            # Monitor projections
            proj = None
            if graph_projections1 is not None:
                proj = graph_projections1
            elif node_projections1 is not None:
                proj = node_projections1

            if proj is not None and len(proj) > 1:
                # Compute embedding statistics
                proj_std = proj.std(dim=0)
                metrics["proj_std_mean"] = proj_std.mean().item()
                metrics["proj_std_min"] = proj_std.min().item()

                # Check for dimension collapse (some dims going to zero)
                collapsed_dims = (proj_std < 0.001).sum().item()
                metrics["collapsed_dims"] = collapsed_dims
                metrics["active_dims"] = len(proj_std) - collapsed_dims

                # Compute pairwise similarities
                sim_matrix = torch.mm(proj, proj.t())
                mask = ~torch.eye(len(proj), dtype=bool, device=proj.device)
                off_diag_sims = sim_matrix[mask]

                metrics["proj_avg_similarity"] = off_diag_sims.mean().item()
                metrics["proj_max_similarity"] = off_diag_sims.max().item()
                metrics["proj_sim_std"] = off_diag_sims.std().item()

            # Monitor predictions
            pred = None
            if graph_predictions1 is not None:
                pred = graph_predictions1
            elif node_predictions1 is not None:
                pred = node_predictions1

            if pred is not None and len(pred) > 1:
                # Normalize predictions for similarity computation
                pred_norm = F.normalize(pred, dim=-1, p=2)

                # Compute prediction statistics
                pred_std = pred.std(dim=0)
                metrics["pred_std_mean"] = pred_std.mean().item()
                metrics["pred_std_min"] = pred_std.min().item()

                # Prediction similarities
                pred_sim_matrix = torch.mm(pred_norm, pred_norm.t())
                pred_mask = ~torch.eye(
                    len(pred_norm), dtype=bool, device=pred_norm.device
                )
                pred_off_diag_sims = pred_sim_matrix[pred_mask]

                metrics["pred_avg_similarity"] = pred_off_diag_sims.mean().item()
                metrics["pred_max_similarity"] = pred_off_diag_sims.max().item()

            # Collapse detection
            collapse_indicators = []
            if "proj_std_mean" in metrics:
                collapse_indicators.append(metrics["proj_std_mean"] < 0.01)
            if "proj_avg_similarity" in metrics:
                collapse_indicators.append(metrics["proj_avg_similarity"] > 0.95)
            if "collapsed_dims" in metrics and proj is not None:
                collapse_indicators.append(
                    metrics["collapsed_dims"] > proj.shape[1] * 0.1
                )

            metrics["collapse_warning"] = (
                float(any(collapse_indicators)) if collapse_indicators else 0.0
            )

        return total_loss, metrics

    def forward(
        self,
        data: Any | None = None,
        preds: Dict[str, torch.Tensor] | None = None,
        step: int = 0,
    ) -> Tuple[torch.Tensor, Dict]:
        _ = step
        if data is None or preds is None:
            raise ValueError("SimSiamLoss expects `data` and `preds` inputs.")
        return self._forward_with_pairs(data, preds)
