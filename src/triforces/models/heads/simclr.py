"""Projection head for contrastive learning."""

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..mlp import create_mlp
from ..outputs import BackboneOutputs


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    node_projection_dim : int, optional
        Output dimension for node projections.
    graph_projection_dim : int, optional
        Output dimension for graph projections.
    compute_node_level : bool, default=True
        Whether to compute node-level projections.
    compute_graph_level : bool, default=True
        Whether to compute graph-level projections.
    reduce : str, default="mean"
        Pooling reduction for graph features.
    normalize_output : bool, default=False
        Whether to L2-normalize projections.
    dropout : float, default=0.0
        Dropout probability in the MLP.
    activation : str, default="relu"
        Activation function name.
    use_batch_norm : bool, default=True
        Whether to use batch normalization.
    projection_hidden_dims : list[int], optional
        Hidden layer sizes for the projection MLP.
    **kwargs : Any
        Additional unused arguments.
    """

    def __init__(
        self,
        input_dim: int,
        node_projection_dim: int | None = 128,
        graph_projection_dim: int | None = 128,
        compute_node_level: bool = True,
        compute_graph_level: bool = True,
        reduce: str = "mean",
        normalize_output: bool = False,
        dropout: float = 0.0,
        activation: str = "relu",
        use_batch_norm: bool = True,
        projection_hidden_dims: List[int] | None = None,
        **kwargs,
    ):
        super().__init__()

        hidden_dims = projection_hidden_dims
        if hidden_dims is None:
            hidden_dims = [256]
        elif isinstance(hidden_dims, dict):
            hidden_dims = [256]

        if not compute_node_level:
            node_projection_dim = None
        if not compute_graph_level:
            graph_projection_dim = None

        self.input_dim = input_dim
        self.node_projection_dim = node_projection_dim
        self.graph_projection_dim = graph_projection_dim
        self.pooling_method = reduce
        self.normalize_output = normalize_output

        self.hidden_dims = hidden_dims
        self.use_batch_norm = use_batch_norm
        self.dropout = dropout
        self.activation = activation

        if self.node_projection_dim is not None:
            self.node_projector = create_mlp(
                input_dim=input_dim,
                hidden_dims=self.hidden_dims,
                output_dim=self.node_projection_dim,
                use_batch_norm=self.use_batch_norm,
                dropout=self.dropout,
                activation=self.activation,
                final_activation=False,
            )
        else:
            self.node_projector = None

        if self.graph_projection_dim is not None:
            self.pooling = self.pooling_method

            self.graph_projector = create_mlp(
                input_dim=input_dim,
                hidden_dims=self.hidden_dims,
                output_dim=self.graph_projection_dim,
                use_batch_norm=self.use_batch_norm,
                dropout=self.dropout,
                activation=self.activation,
                final_activation=False,
            )
        else:
            self.graph_projector = None

    @classmethod
    def build_from_backbone_info(
        cls, backbone_info: Dict[str, Any], **kwargs: Any
    ) -> "ProjectionHead":
        kwargs = dict(kwargs)
        if kwargs.get("input_dim") is None:
            output_dim = backbone_info.get("output_dim")
            if output_dim is None:
                raise ValueError(
                    "ProjectionHead requires `input_dim` or backbone_info['output_dim']."
                )
            kwargs["input_dim"] = int(output_dim)
        return cls(**kwargs)

    def get_head_build_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        if isinstance(self.graph_projection_dim, int) and self.graph_projection_dim > 0:
            info["projection_dim"] = int(self.graph_projection_dim)
        elif isinstance(self.node_projection_dim, int) and self.node_projection_dim > 0:
            info["projection_dim"] = int(self.node_projection_dim)
        return info

    def forward(
        self,
        backbone_outputs: BackboneOutputs,
        batch: Any,
        outputs: Dict[str, torch.Tensor] | None = None,
        training: bool = False,
        transform: Any = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        node_features = backbone_outputs.node_feats
        graph_features = backbone_outputs.graph_feats

        batch_idx = getattr(batch, "batch", None)
        num_graphs = getattr(batch, "num_graphs", None)
        if num_graphs is None and batch_idx is not None and batch_idx.numel():
            num_graphs = int(batch_idx.max().item()) + 1

        out: Dict[str, torch.Tensor] = {}

        if self.node_projector is not None:
            node_proj = self.node_projector(node_features)
            if self.normalize_output:
                node_proj = F.normalize(node_proj, dim=-1, p=2)
            out["node_projections"] = node_proj
            out["node_feats"] = node_features

        if self.graph_projector is not None:
            if graph_features is None and batch_idx is not None:
                graph_features = self._simple_pool(node_features, batch_idx, num_graphs)
            if graph_features is not None:
                graph_proj = self.graph_projector(graph_features)
                if self.normalize_output:
                    graph_proj = F.normalize(graph_proj, dim=-1, p=2)
                out["graph_projections"] = graph_proj
                out["graph_features"] = graph_features

        return out

    def _simple_pool(
        self, x: torch.Tensor, batch: torch.Tensor, num_graphs: int | None = None
    ) -> torch.Tensor:
        if num_graphs is None:
            num_graphs = batch.max().item() + 1

        output = torch.zeros((num_graphs, x.size(1)), device=x.device, dtype=x.dtype)

        if self.pooling_method == "mean":
            output.index_add_(0, batch, x)
            count = torch.bincount(batch, minlength=num_graphs).float()
            output = output / count.unsqueeze(-1).clamp_min(1)
        elif self.pooling_method == "sum":
            output.index_add_(0, batch, x)
        elif self.pooling_method == "max":
            output.index_reduce_(0, batch, x, "amax")
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

        return output
