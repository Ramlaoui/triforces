"""iBOT/DINOv2 losses for self-supervised masked prediction."""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLoss


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic regularizer for representation spread.

    Notes
    -----
    Based on Sablayrolles et al. (2018), this loss encourages each sample to be
    far from its nearest neighbor, helping prevent representation collapse. It is
    used in DINOv2 as an entropy regularizer.
    """

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """Compute nearest neighbors for L2-normalized vectors.

        Parameters
        ----------
        x : torch.Tensor
            L2-normalized vectors with shape ``(N, D)``.

        Returns
        -------
        torch.Tensor
            Indices of nearest neighbors with shape ``(N,)``.
        """
        # Pairwise dot products (= inverse distance for normalized vectors)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, student_output, eps=1e-8):
        """Compute KoLeo loss for a batch of embeddings.

        Parameters
        ----------
        student_output : torch.Tensor
            Student embeddings with shape ``(batch_size, embedding_dim)``.
        eps : float, default=1e-8
            Numerical stability epsilon.

        Returns
        -------
        torch.Tensor
            Scalar KoLeo loss.
        """
        with torch.cuda.amp.autocast(enabled=False):
            # Normalize to unit sphere
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
            # Find nearest neighbor for each sample
            I = self.pairwise_NNs_inner(student_output)  # noqa: E741
            # Compute distances to nearest neighbors
            distances = self.pdist(student_output, student_output[I])
            # Loss: negative log distance (encourages spreading)
            loss = -torch.log(distances + eps).mean()
        return loss


class iBOTLoss(BaseLoss):
    """iBOT loss combining global distillation and masked token prediction.

    This implements iBOT (Image BERT pre-training with Online Tokenizer), also used
    in DINOv2. It combines two components:

    - Global distillation between student and teacher features (DINO-style).
    - Masked token prediction using ``iBOTPatchLoss``.

    Parameters
    ----------
    patch_out_dim : int
        Output dimension for patch predictions.
    student_temp : float, default=0.1
        Temperature for student softmax in patch loss.
    teacher_temp : float, default=0.04
        Temperature for teacher softmax (lower is sharper).
    center_momentum : float, default=0.9
        EMA momentum for teacher centers (used when ``centering="center"``).
    lambda_ibot : float, default=0.5
        Weight for masked token prediction loss.
    lambda_dino : float, default=0.5
        Weight for global distillation loss.
    use_masked_loss : bool, default=True
        Whether to apply masked token prediction.
    koleo_loss_weight : float, default=0.0
        Weight for KoLeo entropy regularization loss. Set to 0 to disable.
    centering : str, default="center"
        Teacher centering method, ``"center"`` or ``"sinkhorn"``.
    n_sinkhorn_iterations : int, default=3
        Sinkhorn-Knopp iterations (used when ``centering="sinkhorn"``).
    **kwargs : Any
        Additional arguments forwarded to ``BaseLoss``.
    """

    def __init__(
        self,
        patch_out_dim: int,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
        lambda_ibot: float = 0.5,
        lambda_dino: float = 0.5,
        use_masked_loss: bool = True,
        koleo_loss_weight: float = 0.0,
        centering: str = "center",
        n_sinkhorn_iterations: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.lambda_ibot = lambda_ibot
        self.lambda_dino = lambda_dino
        self.use_masked_loss = use_masked_loss
        self.koleo_loss_weight = koleo_loss_weight

        # Centering method: "center" for standard centering, "sinkhorn" for Sinkhorn-Knopp
        assert centering in [
            "center",
            "sinkhorn",
        ], f"centering must be 'center' or 'sinkhorn', got {centering}"
        self.centering = centering
        self.n_sinkhorn_iterations = n_sinkhorn_iterations

        # KoLeo loss for preventing collapse
        if koleo_loss_weight > 0:
            self.koleo_loss = KoLeoLoss()
        else:
            self.koleo_loss = None

        # Centers for global loss (similar to DINO)
        # Only used when centering == "center"
        self.register_buffer("center_ibot", torch.zeros(1, patch_out_dim))
        self.register_buffer("center_dino", torch.zeros(1, patch_out_dim))
        self.center_momentum = center_momentum

    @torch.no_grad()
    def update_center(
        self,
        teacher_node_output: torch.Tensor | None = None,
        teacher_graph_output: torch.Tensor | None = None,
    ):
        """Update teacher output centers for centering.

        Parameters
        ----------
        teacher_node_output : torch.Tensor, optional
            Teacher node-level outputs.
        teacher_graph_output : torch.Tensor, optional
            Teacher graph-level outputs.

        Returns
        -------
        None
        """
        if teacher_node_output is not None:
            batch_center = torch.mean(teacher_node_output, dim=0, keepdim=True)
            # Ensure center is on same device as input
            if self.center_ibot.device != teacher_node_output.device:
                self.center_ibot = self.center_ibot.to(teacher_node_output.device)
            self.center_ibot = (
                self.center_ibot * self.center_momentum
                + batch_center * (1 - self.center_momentum)
            )

        if teacher_graph_output is not None:
            batch_center = torch.mean(teacher_graph_output, dim=0, keepdim=True)
            # Ensure center is on same device as input
            if self.center_dino.device != teacher_graph_output.device:
                self.center_dino = self.center_dino.to(teacher_graph_output.device)
            self.center_dino = (
                self.center_dino * self.center_momentum
                + batch_center * (1 - self.center_momentum)
            )

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        """Apply Sinkhorn-Knopp normalization to teacher outputs.

        Parameters
        ----------
        teacher_output : torch.Tensor
            Teacher outputs with shape ``(B, K)``.
        teacher_temp : float
            Temperature for sharpening.
        n_iterations : int, default=3
            Number of Sinkhorn-Knopp iterations.

        Returns
        -------
        torch.Tensor
            Normalized assignment matrix with shape ``(B, K)``.
        """
        teacher_output = teacher_output.float()

        # Handle distributed training
        try:
            import torch.distributed as dist

            world_size = dist.get_world_size() if dist.is_initialized() else 1
        except (ImportError, RuntimeError):
            world_size = 1

        Q = torch.exp(
            teacher_output / teacher_temp
        ).t()  # Q is K-by-B for consistency with notations
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes (embedding dimensions)

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if world_size > 1:
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if world_size > 1:
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def global_loss(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        center: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the DINO-style global distillation loss.

        Parameters
        ----------
        student_output : torch.Tensor
            Student network outputs.
        teacher_output : torch.Tensor
            Teacher network outputs.
        center : torch.Tensor, optional
            Center for teacher centering (used when ``centering="center"``).

        Returns
        -------
        torch.Tensor
            Cross-entropy loss value.
        """
        # Get teacher probabilities based on centering method
        if self.centering == "sinkhorn":
            # Sinkhorn-Knopp normalization
            teacher_probs = self.sinkhorn_knopp_teacher(
                teacher_output, self.teacher_temp, self.n_sinkhorn_iterations
            )
        else:
            # Standard centering and sharpening
            teacher_centered = (teacher_output - center) / self.teacher_temp
            teacher_probs = F.softmax(teacher_centered, dim=-1)

        # Student output
        student_log_probs = F.log_softmax(student_output / self.student_temp, dim=-1)

        # Cross-entropy loss
        loss = -torch.sum(teacher_probs * student_log_probs, dim=-1).mean()

        return loss

    def forward(
        self,
        data: Any | None = None,
        preds: Dict[str, torch.Tensor] | None = None,
        step: int = 0,
        # Direct tensor inputs
        student_node_projections: torch.Tensor | None = None,
        teacher_node_projections: torch.Tensor | None = None,
        student_graph_projections: torch.Tensor | None = None,
        teacher_graph_projections: torch.Tensor | None = None,
        student_patch_predictions: torch.Tensor | None = None,
        teacher_patch_predictions: torch.Tensor | None = None,
        masked_indices: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute iBOT loss with multi-crop filtering.

        Parameters
        ----------
        data : Any, optional
            Batch containing ``pair_id`` and optionally ``batch`` and ``crop_type``.
        preds : dict[str, torch.Tensor], optional
            Dictionary containing student and teacher outputs.
        step : int, default=0
            Training step (unused).
        student_node_projections : torch.Tensor, optional
            Student node projections.
        teacher_node_projections : torch.Tensor, optional
            Teacher node projections.
        student_graph_projections : torch.Tensor, optional
            Student graph projections.
        teacher_graph_projections : torch.Tensor, optional
            Teacher graph projections.
        student_patch_predictions : torch.Tensor, optional
            Student patch predictions for masked tokens.
        teacher_patch_predictions : torch.Tensor, optional
            Teacher patch predictions for masked tokens.
        masked_indices : torch.Tensor, optional
            Boolean mask for masked tokens.

        Returns
        -------
        torch.Tensor
            Total loss value.
        dict
            Metrics dictionary.
        """
        metrics = {}
        total_loss = 0

        # Extract from preds if provided
        if preds is not None:
            student_graph_projections = preds.get("student_graph_projections")
            teacher_graph_projections = preds.get("teacher_graph_projections")
            # Use explicit None check to avoid tensor boolean evaluation error
            student_patch_predictions = preds.get("student_patch_predictions")
            if student_patch_predictions is None:
                student_patch_predictions = preds.get("node_patch_predictions")
            teacher_patch_predictions = preds.get("teacher_patch_predictions")
            masked_indices = preds.get("masked_indices")

        # Extract crop type information for multi-crop DINOv2
        # If crop_type is present, filter teacher to only global crops
        # crop_type encoding: 0 = global, 1 = local
        crop_type = None
        global_crop_mask = None
        batch_attr = None

        if data is not None and hasattr(data, "crop_type"):
            crop_type = data.crop_type
            # Create mask for global crops (teacher only sees these)
            # PyG batches integer attributes as tensors
            if isinstance(crop_type, torch.Tensor):
                global_crop_mask = crop_type == 0  # 0 = global
            elif isinstance(crop_type, (list, tuple)):
                # List of integers
                global_crop_mask = torch.tensor(
                    [ct == 0 for ct in crop_type],
                    dtype=torch.bool,
                    device=data.batch.device if hasattr(data, "batch") else "cpu",
                )
            else:
                # Single value (shouldn't happen in batched mode)
                global_crop_mask = None

            # Get batch attribute for mapping graphs to nodes
            if hasattr(data, "batch"):
                batch_attr = data.batch

            if global_crop_mask is not None:
                metrics["n_global_crops"] = global_crop_mask.sum().item()
                metrics["n_total_crops"] = (
                    len(crop_type)
                    if isinstance(crop_type, (list, tuple))
                    else crop_type.shape[0]
                )

        # Graph-level global loss (DINO loss)
        # For multi-crop DINOv2:
        # - Student sees ALL crops (global + local)
        # - Teacher sees ONLY global crops (already filtered in trainer)
        # This is the key asymmetry in DINOv2
        if (
            self.lambda_dino > 0
            and student_graph_projections is not None
            and teacher_graph_projections is not None
        ):
            # Teacher outputs are already filtered to global crops by trainer
            # No need to filter again here
            # Note: Projections are L2-normalized in the head
            teacher_graph_proj_filtered = teacher_graph_projections

            metrics["n_teacher_graphs"] = teacher_graph_proj_filtered.shape[0]
            metrics["n_student_graphs"] = student_graph_projections.shape[0]

            if self.centering == "center":
                self.center_dino = self.center_dino.to(
                    teacher_graph_proj_filtered.device
                )

            # Compute DINO loss following Meta's implementation
            # Meta's approach: Crop-wise comparison across batch
            #
            # Crops are organized by crop index, not by structure:
            # With batch_size=12, n_global=2, n_local=8:
            # - Local crop 0: [struct_0, struct_1, ..., struct_11] (12 samples)
            # - Local crop 1: [struct_0, struct_1, ..., struct_11] (12 samples)
            # - ...
            # - Global crop 0: [struct_0, struct_1, ..., struct_11] (12 samples)
            # - Global crop 1: [struct_0, struct_1, ..., struct_11] (12 samples)
            #
            # Meta chunks by crop index: student_output_list.chunk(n_crops)
            # Then compares each student crop chunk to each teacher crop chunk
            #
            # Example: student local crop 0 (12 samples) compared to:
            #   - teacher global crop 0 (12 samples)
            #   - teacher global crop 1 (12 samples)

            # Get teacher probabilities based on centering method
            if self.centering == "sinkhorn":
                # Sinkhorn-Knopp normalization
                teacher_probs = self.sinkhorn_knopp_teacher(
                    teacher_graph_proj_filtered,
                    self.teacher_temp,
                    self.n_sinkhorn_iterations,
                )  # [batch_size * n_global, projection_dim]
            else:
                # Teacher centering and sharpening (once for all teacher outputs)
                teacher_centered = (
                    teacher_graph_proj_filtered - self.center_dino
                ) / self.teacher_temp
                teacher_probs = F.softmax(
                    teacher_centered, dim=-1
                )  # [batch_size * n_global, projection_dim]

            # Student log probabilities (once for all student outputs)
            student_log_probs = F.log_softmax(
                student_graph_projections / self.student_temp, dim=-1
            )  # [batch_size * (n_global + n_local), projection_dim]

            # Log teacher/student output statistics for debugging
            metrics["teacher_proj_mean"] = teacher_graph_proj_filtered.mean().item()
            metrics["teacher_proj_std"] = teacher_graph_proj_filtered.std().item()
            metrics["teacher_proj_norm"] = (
                teacher_graph_proj_filtered.norm(dim=-1).mean().item()
            )
            metrics["teacher_proj_min"] = teacher_graph_proj_filtered.min().item()
            metrics["teacher_proj_max"] = teacher_graph_proj_filtered.max().item()

            metrics["student_proj_mean"] = student_graph_projections.mean().item()
            metrics["student_proj_std"] = student_graph_projections.std().item()
            metrics["student_proj_norm"] = (
                student_graph_projections.norm(dim=-1).mean().item()
            )
            metrics["student_proj_min"] = student_graph_projections.min().item()
            metrics["student_proj_max"] = student_graph_projections.max().item()

            # Log teacher probability statistics
            metrics["teacher_probs_entropy"] = (
                -(teacher_probs * teacher_probs.log()).sum(dim=-1).mean().item()
            )
            metrics["teacher_probs_max"] = teacher_probs.max(dim=-1)[0].mean().item()
            metrics["teacher_probs_std_per_sample"] = (
                teacher_probs.std(dim=-1).mean().item()
            )

            # Infer crop structure from shapes
            # Teacher has batch_size * n_global crops (only global)
            # Student has batch_size * (n_global + n_local) crops (all)
            n_teacher = teacher_probs.shape[0]
            n_student = student_log_probs.shape[0]

            # Determine batch_size from pair_id if available
            batch_size = None
            n_global = None

            if data is not None and hasattr(data, "pair_id"):
                # Use unique pair_ids to determine batch size
                pair_ids = data.pair_id
                if isinstance(pair_ids, torch.Tensor):
                    batch_size = torch.unique(pair_ids).shape[0]
                else:
                    batch_size = len(set(pair_ids))

                # Infer n_global from teacher and batch_size
                if n_teacher % batch_size == 0:
                    n_global = n_teacher // batch_size
                    # Verify it makes sense for student too
                    if n_student % batch_size != 0:
                        # Something's wrong, fall back to heuristic
                        batch_size = None
                        n_global = None

            # Reshape teacher: [n_global, batch_size, projection_dim]
            teacher_probs_chunked = teacher_probs.reshape(n_global, batch_size, -1)

            # Reshape student: [n_crops, batch_size, projection_dim]
            n_crops_per_structure = n_student // batch_size
            student_log_probs_chunked = student_log_probs.reshape(
                n_crops_per_structure, batch_size, -1
            )

            # Meta's DINOv2 separates local and global student crops
            # Crop order from collate_fn: global crops first, then local crops
            # n_global global crops, then (n_crops_per_structure - n_global) local crops
            n_local = n_crops_per_structure - n_global

            # Split student crops into global and local
            student_global_crops = student_log_probs_chunked[
                :n_global
            ]  # [n_global, batch_size, dim]
            student_local_crops = student_log_probs_chunked[
                n_global:
            ]  # [n_local, batch_size, dim]

            # Meta's approach:
            # 1. Local student crops vs global teacher crops: n_local * n_global comparisons
            # 2. Global student crops vs global teacher crops: (n_global - 1) * n_global comparisons
            #    (excluding self-comparisons where student crop i == teacher crop i)

            graph_loss = 0.0

            # 1. Local student crops vs all global teacher crops
            if n_local > 0:
                local_loss = 0.0
                for student_crop in student_local_crops:  # [batch_size, dim]
                    for teacher_crop in teacher_probs_chunked:  # [batch_size, dim]
                        loss = torch.sum(
                            teacher_crop * student_crop, dim=-1
                        )  # [batch_size]
                        local_loss -= loss.mean()  # Average over batch dimension

                # Normalize by number of local crop comparisons per sample
                n_local_loss_terms = n_local * n_global
                local_loss = local_loss / n_local_loss_terms
                graph_loss += local_loss
                metrics["local_student_loss"] = local_loss.item()
                metrics["n_local_crops"] = n_local

            # 2. Global student crops vs global teacher crops (excluding self-comparisons)
            global_loss = 0.0
            for i, student_crop in enumerate(student_global_crops):  # [batch_size, dim]
                for j, teacher_crop in enumerate(
                    teacher_probs_chunked
                ):  # [batch_size, dim]
                    if i != j:  # Exclude self-comparison
                        loss = torch.sum(
                            teacher_crop * student_crop, dim=-1
                        )  # [batch_size]
                        global_loss -= loss.mean()  # Average over batch dimension

            # Normalize by number of global crop comparisons per sample
            # (n_global - 1) comparisons per global crop * n_global crops
            n_global_loss_terms = (n_global - 1) * n_global
            if n_global_loss_terms > 0:
                global_loss = global_loss / n_global_loss_terms
                graph_loss += global_loss
                metrics["global_student_loss"] = global_loss.item()

            metrics["n_crops_per_structure"] = n_crops_per_structure
            metrics["batch_size_inferred"] = batch_size

            # Log center statistics for debugging (only for standard centering)
            if self.centering == "center":
                metrics["dino_center_mean"] = self.center_dino.mean().item()
                metrics["dino_center_std"] = self.center_dino.std().item()
                metrics["dino_center_norm"] = self.center_dino.norm().item()

                # Update center after computing loss
                with torch.no_grad():
                    self.update_center(teacher_graph_output=teacher_graph_proj_filtered)

            total_loss = total_loss + self.lambda_dino * graph_loss
            metrics["graph_global_loss"] = graph_loss.item()

        # Masked token prediction loss (iBOT loss)
        # For multi-crop DINOv2: ONLY applied to global crops (not local crops)
        # Both teacher and student predictions are already filtered to global crops:
        # - Teacher: filtered in get_teacher_outputs (only sees global crops)
        # - Student: uses same masked_indices which only has masks for global crops
        # So no additional filtering needed here!
        if (
            self.use_masked_loss
            and student_patch_predictions is not None
            and teacher_patch_predictions is not None
        ):
            # Get teacher probabilities based on centering method
            if self.centering == "sinkhorn":
                # Sinkhorn-Knopp normalization
                teacher_patch_probs = self.sinkhorn_knopp_teacher(
                    teacher_patch_predictions,
                    self.teacher_temp,
                    self.n_sinkhorn_iterations,
                )
            else:
                # Standard centering
                self.center_ibot = self.center_ibot.to(
                    teacher_graph_proj_filtered.device
                )
                teacher_centered = (
                    teacher_patch_predictions - self.center_ibot
                ) / self.teacher_temp
                teacher_patch_probs = F.softmax(teacher_centered, dim=-1)

                # Update patch loss center
                with torch.no_grad():
                    self.update_center(teacher_node_output=teacher_patch_predictions)

            # Compute per-token cross-entropy loss
            token_losses = -torch.sum(
                teacher_patch_probs
                * F.log_softmax(student_patch_predictions / self.student_temp, dim=-1),
                dim=-1,
            )  # [n_masked_tokens]

            # Apply per-sample weighting: each sample contributes equally
            # regardless of how many tokens were masked
            # Get batch indices for masked tokens only
            masked_batch_indices = batch_attr[masked_indices]  # [n_masked_tokens]

            # Count masked tokens per sample
            n_masked_per_sample = torch.bincount(
                masked_batch_indices,
                minlength=batch_attr.max().item() + 1,
            )  # [n_samples_in_batch]

            # Compute per-token weights: 1 / n_masked_in_sample
            sample_weights = 1.0 / n_masked_per_sample.clamp(min=1.0)  # [n_samples]
            token_weights = sample_weights[masked_batch_indices]  # [n_masked_tokens]

            # Apply weights and normalize by batch size
            # This ensures each sample contributes equally
            batch_size = batch_attr.max().item() + 1
            patch_loss = (token_losses * token_weights).sum() / batch_size

            metrics["n_samples_with_masks"] = (n_masked_per_sample > 0).sum().item()

            # Log iBOT center statistics for debugging (only for standard centering)
            if self.centering == "center":
                metrics["ibot_center_mean"] = self.center_ibot.mean().item()
                metrics["ibot_center_std"] = self.center_ibot.std().item()
                metrics["ibot_center_norm"] = self.center_ibot.norm().item()

            # Log additional diagnostics
            metrics["token_loss_mean"] = token_losses.mean().item()
            metrics["token_loss_std"] = token_losses.std().item()

            total_loss = total_loss + self.lambda_ibot * patch_loss
            metrics["patch_loss"] = patch_loss.item()
            metrics["n_masked_tokens"] = student_patch_predictions.shape[0]

        # KoLeo loss (entropy regularization to prevent collapse)
        if self.koleo_loss_weight > 0 and self.koleo_loss is not None:
            # Apply KoLeo to student graph projections (like DINOv2)
            # Note: DINOv2 applies this to each global crop separately to avoid
            # computing KoLeo loss between different views of the same sample
            if student_graph_projections is not None:
                # Split into views if we have multiple (typically 2 global crops)
                # Assuming batch contains pairs, chunk into views
                n_views = 2  # Default assumption for contrastive learning
                if student_graph_projections.shape[0] % n_views == 0:
                    view_size = student_graph_projections.shape[0] // n_views
                    koleo_loss = 0
                    for view_idx in range(n_views):
                        view_start = view_idx * view_size
                        view_end = (view_idx + 1) * view_size
                        view_features = student_graph_projections[view_start:view_end]
                        koleo_loss = koleo_loss + self.koleo_loss(view_features)
                    koleo_loss = koleo_loss / n_views  # Average across views
                else:
                    # Single view or odd batch size - apply to all
                    koleo_loss = self.koleo_loss(student_graph_projections)

                total_loss = total_loss + self.koleo_loss_weight * koleo_loss
                metrics["koleo_loss"] = koleo_loss.item()

        metrics["total_loss"] = (
            total_loss.item() if isinstance(total_loss, torch.Tensor) else 0
        )

        return total_loss, metrics
