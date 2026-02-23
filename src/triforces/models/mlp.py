from __future__ import annotations

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
    hidden_dims: list[int],
    output_dim: int,
    use_batch_norm: bool = False,
    dropout: float = 0.0,
    activation: str = "relu",
    final_activation: bool = False,
) -> nn.Sequential:
    """Create an MLP as an ``nn.Sequential``."""
    layers = []
    dims = [input_dim, *hidden_dims, output_dim]

    activation_fn = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(0.2),
    }.get(activation.lower(), nn.ReLU())

    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or final_activation:
            if use_batch_norm:
                layers.append(SafeBatchNorm1d(dims[i + 1]))
            layers.append(activation_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)
