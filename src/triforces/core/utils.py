"""Utility functions for crystal contrastive learning."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that falls back to running stats for tiny batches."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.training and input.size(0) < 2:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )
        return super().forward(input)


def create_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    use_batch_norm: bool = False,
    dropout: float = 0.0,
    activation: str = "relu",
    final_activation: bool = False,
) -> nn.Sequential:
    layers = []
    dims = [input_dim] + hidden_dims + [output_dim]

    activation_fn = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(0.2),
    }.get(activation.lower(), nn.ReLU())

    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))

        # Add batch norm and activation for all but last layer
        if i < len(dims) - 2 or final_activation:
            if use_batch_norm:
                layers.append(SafeBatchNorm1d(dims[i + 1]))
            layers.append(activation_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


def compute_alignment_uniformity(
    embeddings: torch.Tensor,
    pair_ids: torch.Tensor,
    temperature: float = 2.0,
) -> Tuple[float, float]:
    embeddings = F.normalize(embeddings, dim=-1, p=2)

    # Compute alignment: mean distance between positive pairs
    positive_mask = pair_ids.unsqueeze(0) == pair_ids.unsqueeze(1)
    positive_mask.fill_diagonal_(False)

    if positive_mask.any():
        distances = 2 - 2 * torch.mm(embeddings, embeddings.t())
        positive_distances = distances[positive_mask]
        alignment = positive_distances.mean().item()
    else:
        alignment = 0.0

    # Compute uniformity: log mean pairwise Gaussian potential
    n = len(embeddings)
    if n > 1:
        # Sample subset for efficiency if too many embeddings
        if n > 1000:
            indices = torch.randperm(n)[:1000]
            embeddings = embeddings[indices]

        pairwise_distances = torch.cdist(embeddings, embeddings, p=2) ** 2
        mask = torch.ones_like(pairwise_distances, dtype=torch.bool)
        mask.fill_diagonal_(False)

        uniformity = (
            torch.exp(-temperature * pairwise_distances[mask]).mean().log().item()
        )
    else:
        uniformity = 0.0

    return alignment, uniformity


def gather_from_all_ranks(tensor: torch.Tensor) -> torch.Tensor:
    if not torch.distributed.is_initialized():
        return tensor

    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return tensor

    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


def cosine_similarity_matrix(
    x: torch.Tensor, y: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x = F.normalize(x, dim=-1, p=2)
    if y is None:
        return torch.mm(x, x.t())
    else:
        y = F.normalize(y, dim=-1, p=2)
        return torch.mm(x, y.t())


def sample_negatives(
    embeddings: torch.Tensor,
    anchor_idx: int,
    positive_idx: int,
    n_negatives: int,
    exclude_indices: Optional[List[int]] = None,
) -> torch.Tensor:
    n_total = len(embeddings)
    device = embeddings.device

    # Create mask of valid negative indices
    valid_mask = torch.ones(n_total, dtype=torch.bool, device=device)
    valid_mask[anchor_idx] = False
    valid_mask[positive_idx] = False

    if exclude_indices:
        valid_mask[torch.tensor(exclude_indices, device=device)] = False

    valid_indices = torch.where(valid_mask)[0]

    if len(valid_indices) <= n_negatives:
        return valid_indices

    # Sample without replacement
    perm = torch.randperm(len(valid_indices), device=device)
    return valid_indices[perm[:n_negatives]]
