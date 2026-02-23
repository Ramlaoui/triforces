"""Generic classification heads."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from triforces.models.mlp import create_mlp
from triforces.models.outputs import BackboneOutputs


class ClassificationHead(nn.Module):
    """Predict per-node class logits from backbone node features."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int] | None = None,
        output_key: str = "logits",
        activation: str = "relu",
        use_batch_norm: bool = False,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        _ = kwargs
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.hidden_dims = list(hidden_dims or [self.input_dim, self.input_dim])
        self.output_key = str(output_key)

        self.predictor = create_mlp(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.num_classes,
            activation=activation,
            use_batch_norm=bool(use_batch_norm),
            dropout=float(dropout),
            final_activation=False,
        )

    @classmethod
    def build_from_backbone_info(
        cls, backbone_info: Dict[str, Any], **kwargs: Any
    ) -> "ClassificationHead":
        kwargs = dict(kwargs)
        if kwargs.get("input_dim") is None:
            output_dim = backbone_info.get("output_dim")
            if output_dim is None:
                raise ValueError(
                    "ClassificationHead requires `input_dim` or "
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
