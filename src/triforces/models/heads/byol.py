"""BYOL projection and predictor heads for self-supervised learning."""

from typing import Any, Dict

import torch
import torch.nn as nn

from ..outputs import BackboneOutputs
from .simclr import ProjectionHead


class BYOLProjectionHead(ProjectionHead):
    """Projection head with BYOL defaults.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    projection_dim : int, default=256
        Output projection dimension.
    hidden_dim : int, default=4096
        Hidden dimension for the MLP.
    compute_node_level : bool, default=True
        Whether to compute node-level projections.
    compute_graph_level : bool, default=True
        Whether to compute graph-level projections.
    use_bn : bool, default=True
        Whether to use batch normalization.
    reduce : str, default="mean"
        Reduction method for graph pooling.
    **kwargs : Any
        Additional unused arguments.
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
    ) -> None:
        self.projection_dim = int(projection_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_bn = bool(use_bn)
        super().__init__(
            input_dim=input_dim,
            node_projection_dim=projection_dim if compute_node_level else None,
            graph_projection_dim=projection_dim if compute_graph_level else None,
            compute_node_level=compute_node_level,
            compute_graph_level=compute_graph_level,
            reduce=reduce,
            normalize_output=False,
            dropout=0.0,
            activation="relu",
            use_batch_norm=use_bn,
            projection_hidden_dims=[hidden_dim],
            **kwargs,
        )

    @classmethod
    def build_from_backbone_info(
        cls, backbone_info: Dict[str, Any], **kwargs: Any
    ) -> "BYOLProjectionHead":
        kwargs = dict(kwargs)
        if kwargs.get("input_dim") is None:
            output_dim = backbone_info.get("output_dim")
            if output_dim is None:
                raise ValueError(
                    "BYOLProjectionHead requires `input_dim` or backbone_info['output_dim']."
                )
            kwargs["input_dim"] = int(output_dim)
        return cls(**kwargs)


class BYOLPredictorHead(nn.Module):
    """BYOL predictor head used only in the online network.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dim : int, default=4096
        Hidden dimension for the MLP.
    output_dim : int, optional
        Output feature dimension. Defaults to ``input_dim``.
    compute_node_level : bool, default=True
        Whether to compute node-level predictions.
    compute_graph_level : bool, default=True
        Whether to compute graph-level predictions.
    use_bn : bool, default=True
        Whether to use batch normalization.
    **kwargs : Any
        Additional unused arguments.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 4096,
        output_dim: int | None = None,
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

    @classmethod
    def build_from_backbone_info(
        cls, backbone_info: Dict[str, Any], **kwargs: Any
    ) -> "BYOLPredictorHead":
        kwargs = dict(kwargs)
        if kwargs.get("input_dim") is None:
            candidate = backbone_info.get("projection_dim")
            if candidate is None:
                candidate = backbone_info.get("output_dim")
            if candidate is None:
                raise ValueError(
                    "BYOLPredictorHead requires `input_dim` or "
                    "backbone_info['projection_dim']/['output_dim']."
                )
            kwargs["input_dim"] = int(candidate)
        return cls(**kwargs)

    def get_head_build_info(self) -> Dict[str, Any]:
        return {"projection_dim": int(self.output_dim)}

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
        """Build a 2-layer predictor MLP as in BYOL.

        Parameters
        ----------
        input_dim : int
            Input feature dimension.
        hidden_dim : int
            Hidden layer dimension.
        output_dim : int
            Output feature dimension.
        use_bn : bool, default=True
            Whether to use batch normalization.

        Returns
        -------
        nn.Module
            Predictor MLP.
        """
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
        outputs: Dict[str, torch.Tensor] | None = None,
        training: bool = False,
        transform: Any = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Apply the predictor to existing projections.

        Parameters
        ----------
        backbone_outputs : BackboneOutputs
            Backbone outputs (unused; included for interface consistency).
        batch : Any
            Batch object (unused).
        outputs : dict[str, torch.Tensor], optional
            Dictionary containing ``node_projections`` and/or ``graph_projections``.
        training : bool, default=False
            Training mode flag.
        transform : Any, optional
            Optional transform (unused).
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with ``node_predictions`` and/or ``graph_predictions``.
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
    """Combined BYOL head with projection and prediction.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    projection_dim : int, default=256
        Output projection dimension.
    hidden_dim : int, default=4096
        Hidden dimension for projection MLP.
    predictor_hidden_dim : int, optional
        Hidden dimension for predictor MLP. Defaults to ``hidden_dim``.
    compute_node_level : bool, default=True
        Whether to compute node-level projections/predictions.
    compute_graph_level : bool, default=True
        Whether to compute graph-level projections/predictions.
    use_bn : bool, default=True
        Whether to use batch normalization.
    reduce : str, default="mean"
        Reduction method for graph pooling.
    **kwargs : Any
        Additional unused arguments.
    """

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 256,
        hidden_dim: int = 4096,
        predictor_hidden_dim: int | None = None,
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

    @classmethod
    def build_from_backbone_info(
        cls, backbone_info: Dict[str, Any], **kwargs: Any
    ) -> "BYOLCombinedHead":
        kwargs = dict(kwargs)
        if kwargs.get("input_dim") is None:
            output_dim = backbone_info.get("output_dim")
            if output_dim is None:
                raise ValueError(
                    "BYOLCombinedHead requires `input_dim` or backbone_info['output_dim']."
                )
            kwargs["input_dim"] = int(output_dim)
        return cls(**kwargs)

    def get_head_build_info(self) -> Dict[str, Any]:
        return {"projection_dim": int(self.projection_head.projection_dim)}

    def forward(
        self,
        backbone_outputs: BackboneOutputs,
        batch: Any,
        outputs: Dict[str, torch.Tensor] | None = None,
        training: bool = False,
        transform: Any = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through projection and predictor heads.

        Parameters
        ----------
        backbone_outputs : BackboneOutputs
            Backbone outputs with node and graph features.
        batch : Any
            Batch object.
        outputs : dict[str, torch.Tensor], optional
            Existing outputs dict.
        training : bool, default=False
            Training mode flag.
        transform : Any, optional
            Optional transform (unused).
        **kwargs : Any
            Additional arguments. Use ``return_predictions=False`` for target network.

        Returns
        -------
        dict[str, torch.Tensor]
            Projections and, optionally, predictions.
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
