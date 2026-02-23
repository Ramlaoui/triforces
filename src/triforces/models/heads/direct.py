"""Direct vector heads."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from triforces.models.mlp import create_mlp
from triforces.models.outputs import BackboneOutputs


class DirectVectorHead(nn.Module):
    """Predict per-node vectors from invariant node features."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 3,
        hidden_dims: list[int] | None = None,
        output_key: str = "noise_displacement",
        activation: str = "relu",
        use_batch_norm: bool = False,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        _ = kwargs
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be > 0, got {self.output_dim}")

        self.hidden_dims = list(hidden_dims or [self.input_dim, self.input_dim])
        self.output_key = str(output_key)
        self.predictor = create_mlp(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            activation=activation,
            use_batch_norm=bool(use_batch_norm),
            dropout=float(dropout),
            final_activation=False,
        )

    @classmethod
    def build_from_backbone_info(
        cls, backbone_info: Dict[str, Any], **kwargs: Any
    ) -> "DirectVectorHead":
        kwargs = dict(kwargs)
        if kwargs.get("input_dim") is None:
            output_dim = backbone_info.get("output_dim")
            if output_dim is None:
                raise ValueError(
                    "DirectVectorHead requires `input_dim` or "
                    "backbone_info['output_dim']."
                )
            kwargs["input_dim"] = int(output_dim)
        return cls(**kwargs)

    def forward(
        self,
        backbone_outputs: BackboneOutputs,
        batch: Any,
        outputs: Dict[str, torch.Tensor] | None = None,
        training: bool = False,
        transform: Any = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        _ = batch, outputs, training, transform, kwargs
        return {self.output_key: self.predictor(backbone_outputs.node_feats)}
