"""BYOL projection and predictor heads for self-supervised learning."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..outputs import BackboneOutputs


class BYOLProjectionHead(nn.Module):
    """
    BYOL projection head (shared by online and target networks).

    Projects backbone features to a lower-dimensional space where
    contrastive learning happens.
    """

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 256,
        hidden_dim: int = 4096,
        compute_node_level: bool = True,
        compute_graph_level: bool = True,
        use_bn: bool = True,
        reduce: str = "mean",
        **kwargs,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.compute_node_level = compute_node_level
        self.compute_graph_level = compute_graph_level
        self.reduce = reduce
        self.use_bn = use_bn

        # Build projectors (2-layer MLP as in BYOL paper)
        if self.compute_node_level:
            self.node_projector = self._build_projector(
                input_dim, hidden_dim, projection_dim, use_bn=use_bn
            )
        else:
            self.node_projector = None

        if self.compute_graph_level:
            self.graph_projector = self._build_projector(
                input_dim, hidden_dim, projection_dim, use_bn=use_bn
            )
        else:
            self.graph_projector = None

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                if m.affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def _build_projector(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_bn: bool = True,
    ) -> nn.Module:
        """Build 2-layer projection MLP as in BYOL paper."""
        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(hidden_dim, output_dim, bias=False))

        return nn.Sequential(*layers)

    def forward(
        self,
        backbone_outputs: BackboneOutputs,
        batch: Any,
        outputs: Optional[Dict[str, torch.Tensor]] = None,
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
        out["node_feats"] = node_features

        if self.node_projector is not None:
            out["node_projections"] = self.node_projector(node_features)

        if graph_features is None and batch_idx is not None:
            graph_features = torch.zeros(
                (num_graphs, node_features.size(1)),
                device=node_features.device,
                dtype=node_features.dtype,
            )
            graph_features.index_add_(0, batch_idx, node_features)

            if self.reduce == "mean":
                count = torch.bincount(batch_idx, minlength=num_graphs).float()
                count = torch.clamp(count, min=1)
                graph_features = graph_features / count.unsqueeze(1)

        if graph_features is not None:
            out["graph_features"] = graph_features
            if self.graph_projector is not None:
                out["graph_projections"] = self.graph_projector(graph_features)

        return out


class BYOLPredictorHead(nn.Module):
    """
    BYOL predictor head (only in online network).

    Predicts target network projections from online network projections.
    This asymmetry is crucial for preventing collapse in BYOL.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 4096,
        output_dim: Optional[int] = None,
        compute_node_level: bool = True,
        compute_graph_level: bool = True,
        use_bn: bool = True,
        **kwargs,
    ):
        super().__init__()

        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.compute_node_level = compute_node_level
        self.compute_graph_level = compute_graph_level
        self.use_bn = use_bn

        # Build predictors (2-layer MLP as in BYOL paper)
        if self.compute_node_level:
            self.node_predictor = self._build_predictor(
                input_dim, hidden_dim, output_dim, use_bn=use_bn
            )
        else:
            self.node_predictor = None

        if self.compute_graph_level:
            self.graph_predictor = self._build_predictor(
                input_dim, hidden_dim, output_dim, use_bn=use_bn
            )
        else:
            self.graph_predictor = None

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                if m.affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def _build_predictor(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_bn: bool = True,
    ) -> nn.Module:
        """Build 2-layer predictor MLP as in BYOL paper."""
        layers = []

        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(hidden_dim, output_dim, bias=False))

        return nn.Sequential(*layers)

    def forward(
        self,
        backbone_outputs: BackboneOutputs,
        batch: Any,
        outputs: Optional[Dict[str, torch.Tensor]] = None,
        training: bool = False,
        transform: Any = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply predictor to projections.

        Parameters
        ----------
        projections : Dict[str, torch.Tensor]
            Dictionary with 'node_projections' and/or 'graph_projections'

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with 'node_predictions' and/or 'graph_predictions'
        """
        if outputs is None:
            raise ValueError("BYOLPredictorHead requires outputs with projections.")

        projections = outputs
        out: Dict[str, torch.Tensor] = {}

        if self.node_predictor is not None and "node_projections" in projections:
            out["node_predictions"] = self.node_predictor(
                projections["node_projections"]
            )

        if self.graph_predictor is not None and "graph_projections" in projections:
            out["graph_predictions"] = self.graph_predictor(
                projections["graph_projections"]
            )

        return out


class BYOLCombinedHead(nn.Module):
    """
    Combined BYOL head with projection and prediction.

    This combines the projection and predictor heads for the online network.
    The target network only uses the projection head.
    """

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 256,
        hidden_dim: int = 4096,
        predictor_hidden_dim: Optional[int] = None,
        compute_node_level: bool = True,
        compute_graph_level: bool = True,
        use_bn: bool = True,
        reduce: str = "mean",
        **kwargs,
    ):
        super().__init__()

        if predictor_hidden_dim is None:
            predictor_hidden_dim = hidden_dim

        self.projection_head = BYOLProjectionHead(
            input_dim=input_dim,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            compute_node_level=compute_node_level,
            compute_graph_level=compute_graph_level,
            use_bn=use_bn,
            reduce=reduce,
        )

        self.predictor_head = BYOLPredictorHead(
            input_dim=projection_dim,
            hidden_dim=predictor_hidden_dim,
            output_dim=projection_dim,
            compute_node_level=compute_node_level,
            compute_graph_level=compute_graph_level,
            use_bn=use_bn,
        )

    def forward(
        self,
        backbone_outputs: BackboneOutputs,
        batch: Any,
        outputs: Optional[Dict[str, torch.Tensor]] = None,
        training: bool = False,
        transform: Any = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through projection and predictor.

        Parameters
        ----------
        node_features : torch.Tensor
            Input node features
        batch : Optional[torch.Tensor]
            Batch indices for graph pooling
        num_graphs : Optional[int]
            Number of graphs in batch
        return_predictions : bool
            Whether to return predictions (set False for target network)

        Returns
        -------
        Dict[str, torch.Tensor]
            Projections and optionally predictions
        """
        return_predictions = kwargs.get("return_predictions", True)

        outputs = self.projection_head(
            backbone_outputs,
            batch,
            outputs=outputs,
            training=training,
            transform=transform,
        )

        if return_predictions and self.predictor_head is not None:
            predictions = self.predictor_head(
                backbone_outputs,
                batch,
                outputs=outputs,
                training=training,
                transform=transform,
            )
            outputs.update(predictions)

        return outputs
