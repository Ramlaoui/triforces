from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn as nn


def _get_pred_value(preds: Any, key: str):
    if isinstance(preds, dict):
        return preds.get(key)
    return getattr(preds, key, None)


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Return a flattened view of off-diagonal elements.

    Parameters
    ----------
    x : torch.Tensor
        Square matrix with shape ``(N, N)``.

    Returns
    -------
    torch.Tensor
        Flattened off-diagonal elements.
    """
    n, m = x.shape
    assert n == m, "Matrix must be square"
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwinsLoss(nn.Module):
    """Barlow Twins loss using the cross-correlation matrix.

    The loss encourages:
    - Invariance: diagonal correlation entries approach 1.
    - Redundancy reduction: off-diagonal entries approach 0.

    Parameters
    ----------
    lambda_param : float, default=0.005
        Weight for the redundancy reduction term (off-diagonal).
    lambda_node : float, default=0.5
        Weight for node-level loss.
    lambda_graph : float, default=0.5
        Weight for graph-level loss.
    use_batch_norm : bool, default=True
        Whether to use batch normalization before computing correlations.
    use_node_correspondence : bool, default=False
        Whether to use node correspondence when matching nodes between views.
    auto_scale_lambda : bool, default=True
        Whether to scale ``lambda_param`` with the projection dimension.
    reference_dim : int, default=8192
        Reference dimension for lambda scaling.

    Notes
    -----
    The loss is computed as:
    ``sum((1 - C_ii)^2) + lambda * sum(C_ij^2 for i != j)``.
    """

    def __init__(
        self,
        lambda_param: float = 0.005,
        lambda_node: float = 0.5,
        lambda_graph: float = 0.5,
        use_batch_norm: bool = True,
        use_node_correspondence: bool = False,
        auto_scale_lambda: bool = True,
        reference_dim: int = 8192,
    ):
        super().__init__()

        self.lambda_param = lambda_param
        self.lambda_node = lambda_node
        self.lambda_graph = lambda_graph
        self.use_batch_norm = use_batch_norm
        self.use_node_correspondence = use_node_correspondence
        self.auto_scale_lambda = auto_scale_lambda
        self.reference_dim = reference_dim

        total_lambda = lambda_node + lambda_graph
        if total_lambda > 0:
            self.lambda_node /= total_lambda
            self.lambda_graph /= total_lambda

        # BatchNorm layers will be created dynamically based on embedding dimensions
        self.bn_node = None
        self.bn_graph = None

    def _forward_with_pairs(self, data: Any, preds: Any) -> Tuple[torch.Tensor, Dict]:
        """Compute loss when batch contains paired indices.

        Parameters
        ----------
        data : Any
            Batch with ``pair_id`` and optionally ``batch`` attributes. If provided,
            ``pair_idx1`` and ``pair_idx2`` from the collate function are used.
        preds : dict[str, torch.Tensor]
            Dictionary containing projections.

        Returns
        -------
        torch.Tensor
            Total loss value.
        dict
            Metrics dictionary.

        Notes
        -----
        Uses precomputed pair indices when available to avoid CUDA synchronization.
        """
        if not hasattr(data, "pair_id"):
            raise ValueError("Data must contain 'pair_id' field")

        device = getattr(data, "batch", torch.tensor([], dtype=torch.long)).device

        node_projections = _get_pred_value(preds, "node_projections")
        graph_projections = _get_pred_value(preds, "graph_projections")

        # FAST PATH: Use precomputed pair indices (no CUDA sync!)
        if hasattr(data, "pair_idx1") and hasattr(data, "pair_idx2"):
            idx1 = data.pair_idx1.to(device)
            idx2 = data.pair_idx2.to(device)

            # Check if we have valid pairs without syncing
            if idx1.numel() == 0:
                return torch.tensor(0.0, device=device, requires_grad=True), {
                    "skipped": "no_valid_pairs"
                }

            kwargs = {}

            # Graph-level: Simple vectorized indexing (no loops, no sync)
            if graph_projections is not None:
                kwargs["graph_embeddings1"] = graph_projections[idx1]
                kwargs["graph_embeddings2"] = graph_projections[idx2]

            # Node-level: Use precomputed node indices (ULTRA FAST!)
            if (
                node_projections is not None
                and hasattr(data, "node_pair_idx1")
                and hasattr(data, "node_pair_idx2")
            ):
                node_idx1 = data.node_pair_idx1.to(device)
                node_idx2 = data.node_pair_idx2.to(device)

                if node_idx1.numel() > 0:
                    # Single vectorized indexing - no loop, no CUDA sync!
                    kwargs["node_embeddings1"] = node_projections[node_idx1]
                    kwargs["node_embeddings2"] = node_projections[node_idx2]
            elif node_projections is not None and hasattr(data, "batch"):
                # FALLBACK: Old node-level path (slower, with loop)
                node_emb1_list, node_emb2_list = [], []

                # This loop is over valid pairs (already filtered), not raw data
                # Much faster than before, but still some indexing
                for i in range(len(idx1)):
                    graph_idx1 = idx1[i]
                    graph_idx2 = idx2[i]

                    node_mask1 = data.batch == graph_idx1
                    node_mask2 = data.batch == graph_idx2

                    # Extract embeddings
                    node_emb1 = node_projections[node_mask1]
                    node_emb2 = node_projections[node_mask2]

                    # Handle node correspondence for augmentations that change node count
                    if self.use_node_correspondence and hasattr(
                        data, "node_correspondence"
                    ):
                        corr = data.node_correspondence[node_mask1]
                        valid_mask1 = corr >= 0

                        if valid_mask1.any():
                            mask2_local_to_global = torch.where(node_mask2)[0]
                            corr_valid = corr[valid_mask1]

                            # Vectorized matching
                            matches_mask = corr_valid.unsqueeze(
                                1
                            ) == mask2_local_to_global.unsqueeze(0)
                            has_match = matches_mask.any(dim=1)

                            if has_match.any():
                                match_positions = matches_mask.long().argmax(dim=1)
                                valid_indices = torch.where(valid_mask1)[0]
                                view1_indices = valid_indices[has_match]
                                view2_indices = match_positions[has_match]

                                node_emb1_list.append(node_emb1[view1_indices])
                                node_emb2_list.append(node_emb2[view2_indices])
                    else:
                        # No correspondence tracking - require same node count
                        if node_emb1.shape[0] == node_emb2.shape[0]:
                            node_emb1_list.append(node_emb1)
                            node_emb2_list.append(node_emb2)

                if node_emb1_list:
                    kwargs["node_embeddings1"] = torch.cat(node_emb1_list, dim=0)
                    kwargs["node_embeddings2"] = torch.cat(node_emb2_list, dim=0)

            return self.forward(**kwargs)

        # FALLBACK: Old slow path for backward compatibility
        # This triggers CUDA synchronization and should be avoided
        pair_ids = data.pair_id
        unique_pairs = torch.unique(pair_ids)

        node_emb1_list, node_emb2_list = [], []
        graph_emb1_list, graph_emb2_list = [], []

        for pair_id in unique_pairs:
            mask = pair_ids == pair_id
            indices = torch.where(mask)[0]

            if len(indices) != 2:
                continue

            idx1, idx2 = indices[0], indices[1]

            if node_projections is not None and hasattr(data, "batch"):
                node_mask1 = data.batch == idx1
                node_mask2 = data.batch == idx2

                if node_mask1.any() and node_mask2.any():
                    node_emb1 = node_projections[node_mask1]
                    node_emb2 = node_projections[node_mask2]

                    if self.use_node_correspondence and hasattr(
                        data, "node_correspondence"
                    ):
                        corr = data.node_correspondence[node_mask1]
                        valid_mask1 = corr >= 0

                        if valid_mask1.any():
                            node2_global_indices = torch.where(node_mask2)[0]
                            valid_corr = corr[valid_mask1]
                            mask2_local_to_global = torch.where(node_mask2)[0]

                            matches = []
                            view1_indices = []
                            for i, (is_valid, corr_idx) in enumerate(
                                zip(valid_mask1, corr)
                            ):
                                if is_valid:
                                    local_idx2 = (
                                        mask2_local_to_global == corr_idx
                                    ).nonzero(as_tuple=True)[0]
                                    if len(local_idx2) > 0:
                                        view1_indices.append(i)
                                        matches.append(local_idx2[0].item())

                            if len(matches) > 0:
                                view1_indices = torch.tensor(
                                    view1_indices, device=node_emb1.device
                                )
                                matches = torch.tensor(matches, device=node_emb2.device)
                                node_emb1_list.append(node_emb1[view1_indices])
                                node_emb2_list.append(node_emb2[matches])
                    else:
                        if node_emb1.shape[0] == node_emb2.shape[0]:
                            node_emb1_list.append(node_emb1)
                            node_emb2_list.append(node_emb2)

            if graph_projections is not None:
                graph_emb1_list.append(graph_projections[idx1 : idx1 + 1])
                graph_emb2_list.append(graph_projections[idx2 : idx2 + 1])

        kwargs = {}
        if node_emb1_list:
            kwargs["node_embeddings1"] = torch.cat(node_emb1_list, dim=0)
            kwargs["node_embeddings2"] = torch.cat(node_emb2_list, dim=0)

        if graph_emb1_list:
            kwargs["graph_embeddings1"] = torch.cat(graph_emb1_list, dim=0)
            kwargs["graph_embeddings2"] = torch.cat(graph_emb2_list, dim=0)

        if not kwargs:
            return torch.tensor(0.0, device=device, requires_grad=True), {
                "skipped": "no_valid_pairs"
            }

        return self.forward(**kwargs)

    def compute_correlation_matrix(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-correlation between two embedding sets.

        Parameters
        ----------
        z1 : torch.Tensor
            First embedding set with shape ``(batch_size, embedding_dim)``.
        z2 : torch.Tensor
            Second embedding set with shape ``(batch_size, embedding_dim)``.

        Returns
        -------
        torch.Tensor
            Cross-correlation matrix with shape ``(embedding_dim, embedding_dim)``.
        """
        batch_size = z1.size(0)

        if self.use_batch_norm:
            # CRITICAL FIX: Normalize BOTH views together as in original Barlow Twins paper
            # Concatenate z1 and z2, apply batch norm, then split
            embedding_dim = z1.size(1)
            if not hasattr(self, "_bn") or self._bn.num_features != embedding_dim:
                self._bn = nn.BatchNorm1d(
                    embedding_dim, affine=False, eps=1e-5, momentum=0.1
                ).to(z1.device)

            # Concatenate both views along batch dimension
            z_combined = torch.cat([z1, z2], dim=0)
            z_combined_norm = self._bn(z_combined)

            # Split back into two views
            z1_norm = z_combined_norm[:batch_size]
            z2_norm = z_combined_norm[batch_size:]
        else:
            z1_centered = z1 - z1.mean(dim=0, keepdim=True)
            z2_centered = z2 - z2.mean(dim=0, keepdim=True)

            z1_std = torch.sqrt(
                torch.var(z1_centered, dim=0, keepdim=True, unbiased=False) + 1e-6
            )
            z2_std = torch.sqrt(
                torch.var(z2_centered, dim=0, keepdim=True, unbiased=False) + 1e-6
            )

            z1_norm = z1_centered / z1_std
            z2_norm = z2_centered / z2_std

        # # Compute cross-correlation matrix
        # c = torch.mm(z1_norm.T, z2_norm) / batch_size

        # Einsum is smarter about memory layout and can avoid the transpose copy
        c = torch.einsum("bd,be->de", z1_norm, z2_norm) / batch_size

        return c

    def barlow_twins_loss(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute Barlow Twins loss for a pair of embeddings.

        Parameters
        ----------
        z1 : torch.Tensor
            First view embeddings with shape ``(batch_size, embedding_dim)``.
        z2 : torch.Tensor
            Second view embeddings with shape ``(batch_size, embedding_dim)``.

        Returns
        -------
        torch.Tensor
            Loss value.
        dict
            Metrics dictionary.
        """
        # Check for NaN in input embeddings and filter them out
        nan_mask = torch.isnan(z1).any(dim=-1) | torch.isnan(z2).any(dim=-1)

        if nan_mask.any():
            valid_mask = ~nan_mask
            z1_filtered = z1[valid_mask]
            z2_filtered = z2[valid_mask]

            # If all samples have NaN, return zero loss
            if z1_filtered.size(0) == 0:
                device = z1.device
                return torch.tensor(0.0, device=device, requires_grad=True), {
                    "nan_samples_filtered": nan_mask.sum().item(),
                    "skipped": "all_nan",
                }

            z1 = z1_filtered
            z2 = z2_filtered
            nan_count = nan_mask.sum().item()
        else:
            nan_count = 0

        c = self.compute_correlation_matrix(z1, z2)

        # Compute effective lambda based on embedding dimension
        embedding_dim = z1.size(1)
        if self.auto_scale_lambda:
            # Scale lambda inversely with dimension to maintain balance
            # λ_eff = λ × (reference_dim / actual_dim)
            effective_lambda = self.lambda_param * (self.reference_dim / embedding_dim)
        else:
            effective_lambda = self.lambda_param

        # Invariance loss: diagonal elements should be 1
        on_diag = torch.diagonal(c).add(-1).pow(2).sum()

        # Redundancy reduction: off-diagonal elements should be 0
        off_diag = off_diagonal(c).pow(2).sum()

        loss = on_diag + effective_lambda * off_diag

        metrics = {
            "on_diag_loss": on_diag.item(),
            "off_diag_loss": off_diag.item(),
            "correlation_mean": c.mean().item(),
            "correlation_std": c.std().item(),
            "diagonal_mean": torch.diagonal(c).mean().item(),
            "effective_lambda": effective_lambda,
            "embedding_dim": embedding_dim,
        }

        if nan_count > 0:
            metrics["nan_samples_filtered"] = nan_count

        return loss, metrics

    def forward(
        self,
        data: Optional[Any] = None,
        preds: Optional[Any] = None,
        step: int = 0,
        # direct tensor inputs
        z1: Optional[torch.Tensor] = None,
        z2: Optional[torch.Tensor] = None,
        node_embeddings1: Optional[torch.Tensor] = None,
        node_embeddings2: Optional[torch.Tensor] = None,
        graph_embeddings1: Optional[torch.Tensor] = None,
        graph_embeddings2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute Barlow Twins loss from data or direct embeddings.

        Parameters
        ----------
        data : Any, optional
            Batch containing ``pair_id`` and optionally ``batch`` attributes.
        preds : dict[str, torch.Tensor], optional
            Dictionary containing ``node_projections`` and/or ``graph_projections``.
        step : int, default=0
            Training step (unused).
        z1, z2 : torch.Tensor, optional
            Direct embeddings for a simple forward pass.
        node_embeddings1, node_embeddings2 : torch.Tensor, optional
            Node-level embeddings from two views.
        graph_embeddings1, graph_embeddings2 : torch.Tensor, optional
            Graph-level embeddings from two views.

        Returns
        -------
        torch.Tensor
            Total loss value.
        dict
            Metrics dictionary.
        """
        metrics = {}
        total_loss = 0

        # New interface: handle data with pair_ids
        if data is not None and preds is not None:
            return self._forward_with_pairs(data, preds)

        # Simple forward pass with z1, z2
        if z1 is not None and z2 is not None:
            loss, bt_metrics = self.barlow_twins_loss(z1, z2)
            return loss, bt_metrics

        # Node-level loss
        if (
            self.lambda_node > 0
            and node_embeddings1 is not None
            and node_embeddings2 is not None
        ):
            node_loss, node_metrics = self.barlow_twins_loss(
                node_embeddings1, node_embeddings2
            )
            total_loss = total_loss + self.lambda_node * node_loss
            metrics["node_loss"] = node_loss.item()
            metrics.update({f"node_{k}": v for k, v in node_metrics.items()})

        # Graph-level loss
        if (
            self.lambda_graph > 0
            and graph_embeddings1 is not None
            and graph_embeddings2 is not None
        ):
            graph_loss, graph_metrics = self.barlow_twins_loss(
                graph_embeddings1, graph_embeddings2
            )
            total_loss = total_loss + self.lambda_graph * graph_loss
            metrics["graph_loss"] = graph_loss.item()
            metrics.update({f"graph_{k}": v for k, v in graph_metrics.items()})

        metrics["total_loss"] = (
            total_loss.item() if isinstance(total_loss, torch.Tensor) else 0
        )

        # Add collapse monitoring (Barlow Twins is more robust, but still monitor)
        if graph_embeddings1 is not None:
            with torch.no_grad():
                proj = graph_embeddings1
                if len(proj) > 1:
                    # Check embedding diversity
                    proj_std = proj.std(dim=0).mean().item()
                    metrics["proj_std"] = proj_std

                    # Check if embeddings are collapsing
                    if proj_std < 0.01:
                        metrics["collapse_warning"] = 1.0
                    else:
                        metrics["collapse_warning"] = 0.0

        return total_loss, metrics
