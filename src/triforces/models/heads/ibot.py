"""iBOT/DINOv2 projection head for self-supervised learning with masked prediction."""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..outputs import BackboneOutputs


class L2Normalize(nn.Module):
    """L2 normalization layer for DINO/iBOT projection heads."""

    def forward(self, x):
        """Normalize input to unit norm.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            L2-normalized tensor.
        """
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        return F.normalize(x, dim=-1, p=2, eps=eps)


class WeightNormLinear(nn.Module):
    """Linear layer with weight normalization that's deepcopy-safe.

    Parameters
    ----------
    in_features : int
        Input feature dimension.
    out_features : int
        Output feature dimension.
    bias : bool, default=False
        Whether to include a bias term.

    Notes
    -----
    Unlike ``nn.utils.weight_norm``, this stores ``weight_g`` and ``weight_v`` as
    regular parameters, making it compatible with ``copy.deepcopy``.
    """

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Store direction (v) and magnitude (g) separately
        self.weight_v = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_g = nn.Parameter(torch.ones(out_features, 1))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Initialize
        nn.init.kaiming_uniform_(
            self.weight_v, a=torch.nn.init.calculate_gain("linear")
        )

    def forward(self, x):
        # Normalize weight_v and scale by weight_g
        # ||w|| = g, w = g * v / ||v||
        w = self.weight_v * (self.weight_g / self.weight_v.norm(dim=1, keepdim=True))
        return F.linear(x, w, self.bias)


class iBOTProjectionHead(nn.Module):
    """iBOT projection head for masked token prediction.

    Parameters
    ----------
    input_dim : int
        Input feature dimension from the backbone.
    projection_dim : int, default=8192
        Output dimension for graph projections.
    patch_out_dim : int, default=8192
        Output dimension for patch predictions (must match ``projection_dim`` when
        ``shared_head=True``).
    hidden_dim : int, default=2048
        Hidden dimension in the projection MLP.
    bottleneck_dim : int, default=256
        Bottleneck dimension in the 3-layer MLP.
    use_bn : bool, default=False
        Whether to use batch normalization in the projection MLP.
    norm_last_layer : bool, default=True
        Whether to normalize the last layer weights.
    do_ibot : bool, default=True
        Whether to compute iBOT patch predictions.
    reduce : str, default="mean"
        Reduction method for graph-level pooling (``"mean"`` or ``"sum"``).
    use_gelu : bool, default=True
        Whether to use GELU activation (True for DINOv2, False for ReLU).
    shared_head : bool, default=True
        Whether to use a single shared head (Meta's DINOv2) or separate heads.

    Notes
    -----
    With ``shared_head=True`` (Meta's DINOv2 approach), graph and node predictions
    share a single projection head and therefore the same feature space.
    """

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 8192,
        patch_out_dim: int = 8192,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        use_bn: bool = False,
        norm_last_layer: bool = True,
        do_ibot: bool = True,
        reduce: str = "mean",
        use_gelu: bool = True,
        shared_head: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.patch_out_dim = patch_out_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.reduce = reduce
        self.use_bn = use_bn
        self.norm_last_layer = norm_last_layer
        self.use_gelu = use_gelu
        self.shared_head = shared_head
        self.do_ibot = do_ibot

        # Build projection heads
        if shared_head:
            # Meta's DINOv2 approach: single shared head
            # Both graph-level and node-level predictions use the same head
            # This ensures they live in the same feature space
            if projection_dim != patch_out_dim:
                raise ValueError(
                    f"projection_dim ({projection_dim}) must equal patch_out_dim ({patch_out_dim}) "
                    "for shared head architecture. Meta's DINOv2 uses the same projection space "
                    "for both graph-level (DINO) and node-level (iBOT) predictions."
                )

            self.head = self._build_projector(
                input_dim,
                hidden_dim,
                bottleneck_dim,
                projection_dim,
                use_bn=use_bn,
                norm_last_layer=norm_last_layer,
                use_gelu=use_gelu,
            )
            self.node_projector = None
            self.node_patch_predictor = None
            self.graph_projector = None
        else:
            self.dino_head = self._build_projector(
                input_dim,
                hidden_dim,
                bottleneck_dim,
                projection_dim,
                use_bn=use_bn,
                norm_last_layer=norm_last_layer,
                use_gelu=use_gelu,
            )

            if self.do_ibot:
                self.ibot_head = self._build_projector(
                    input_dim,
                    hidden_dim,
                    bottleneck_dim,
                    patch_out_dim,
                    use_bn=use_bn,
                    norm_last_layer=norm_last_layer,
                    use_gelu=use_gelu,
                )
            else:
                self.ibot_head = None

            self.head = None

        self._initialize_weights()

    @classmethod
    def build_from_backbone_info(
        cls, backbone_info: Dict[str, Any], **kwargs: Any
    ) -> "iBOTProjectionHead":
        kwargs = dict(kwargs)
        if kwargs.get("input_dim") is None:
            output_dim = backbone_info.get("output_dim")
            if output_dim is None:
                raise ValueError(
                    "iBOTProjectionHead requires `input_dim` or backbone_info['output_dim']."
                )
            kwargs["input_dim"] = int(output_dim)
        return cls(**kwargs)

    def get_head_build_info(self) -> Dict[str, Any]:
        return {"projection_dim": int(self.projection_dim)}

    def _initialize_weights(self):
        """Initialize weights following DINOv2 initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Handle weight-normalized layers (last layer when norm_last_layer=True)
                if hasattr(m, "weight_v"):
                    # Weight is already split into weight_v and weight_g
                    nn.init.trunc_normal_(m.weight_v, std=0.02)
                elif hasattr(m, "weight"):
                    # Standard linear layer
                    nn.init.trunc_normal_(m.weight, std=0.02)

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
        bottleneck_dim: int,
        output_dim: int,
        use_bn: bool = False,
        norm_last_layer: bool = True,
        use_gelu: bool = True,
    ) -> nn.Module:
        """Build a 3-layer projection MLP as in DINOv2.

        Parameters
        ----------
        input_dim : int
            Input feature dimension.
        hidden_dim : int
            Hidden layer dimension.
        bottleneck_dim : int
            Bottleneck layer dimension.
        output_dim : int
            Output feature dimension.
        use_bn : bool, default=False
            Whether to use batch normalization.
        norm_last_layer : bool, default=True
            Whether to normalize the last layer weights.
        use_gelu : bool, default=True
            Whether to use GELU activation.

        Returns
        -------
        nn.Module
            Projection MLP.
        """
        layers = []
        activation = nn.GELU() if use_gelu else nn.ReLU(inplace=True)

        # First layer: input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(activation)

        # Second layer: hidden_dim -> bottleneck_dim
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm1d(bottleneck_dim))
        # NOTE: No activation here! L2 norm is applied directly after linear layer

        # L2 normalization layer (critical for DINO/iBOT)
        # Normalizes bottleneck features before the last layer
        layers.append(L2Normalize())

        # Third layer: bottleneck_dim -> output_dim
        if norm_last_layer:
            # Use custom WeightNormLinear that's deepcopy-safe
            last_layer = WeightNormLinear(bottleneck_dim, output_dim, bias=False)
        else:
            last_layer = nn.Linear(bottleneck_dim, output_dim, bias=False)

        layers.append(last_layer)

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
        """Forward pass through the iBOT projection head.

        Parameters
        ----------
        backbone_outputs : BackboneOutputs
            Backbone outputs with node and graph features.
        batch : Any
            Batch object providing ``batch`` and ``num_graphs`` metadata.
        outputs : dict[str, torch.Tensor], optional
            Existing outputs dict to read optional fields from.
        training : bool, default=False
            Training mode flag.
        transform : Any, optional
            Optional transform (unused).
        **kwargs : Any
            Additional arguments (e.g., ``masked_indices``).

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing graph and node projections and optional masks.
        """
        node_features = backbone_outputs.node_feats
        batch_idx = getattr(batch, "batch", None)
        num_graphs = getattr(batch, "num_graphs", None)
        if num_graphs is None and batch_idx is not None and batch_idx.numel():
            num_graphs = int(batch_idx.max().item()) + 1

        masked_indices = kwargs.get("masked_indices")
        if masked_indices is None and outputs is not None:
            masked_indices = outputs.get("masked_indices")

        if self.shared_head:
            return self._forward_shared(
                node_features, batch_idx, num_graphs, masked_indices
            )
        else:
            return self._forward_separate(
                node_features, batch_idx, num_graphs, masked_indices
            )

    def _forward_shared(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor | None,
        num_graphs: int | None,
        masked_indices: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass using the shared-head approach.

        Parameters
        ----------
        node_features : torch.Tensor
            Node features with shape ``(N, D)``.
        batch : torch.Tensor, optional
            Batch indices for graph pooling.
        num_graphs : int, optional
            Number of graphs in the batch.
        masked_indices : torch.Tensor, optional
            Boolean mask for masked tokens.

        Returns
        -------
        dict[str, torch.Tensor]
            Outputs containing graph projections and optional patch predictions.
        """
        outputs = {}
        outputs["node_feats"] = node_features

        num_graphs = batch.max().item() + 1

        graph_features = torch.zeros(
            (num_graphs, node_features.size(1)),
            device=node_features.device,
            dtype=node_features.dtype,
        )
        graph_features.index_add_(0, batch, node_features)

        if self.reduce == "mean":
            count = torch.bincount(batch, minlength=num_graphs).float()
            count = torch.clamp(count, min=1)
            graph_features = graph_features / count.unsqueeze(1)

        outputs["graph_features"] = graph_features

        # Step 2: Extract masked node features
        masked_features = None

        if self.do_ibot and masked_indices is not None:
            masked_features = node_features[masked_indices]
            outputs["masked_indices"] = masked_indices

        # Step 3:
        # Concatenate [graph_features; masked_node_features] -> single forward -> split

        if self.do_ibot:
            # Both graph and node predictions
            # Buffer layout: [graph_features; masked_node_features]
            buffer_tensor = torch.cat([graph_features, masked_features], dim=0)

            # Single forward pass through shared head
            # Note: L2 normalization is done inside the head before the last layer
            projections = self.head(buffer_tensor)

            # Split output
            outputs["graph_projections"] = projections[:num_graphs]
            outputs["node_patch_predictions"] = projections[num_graphs:]

        else:
            # Note: L2 normalization is done inside the head before the last layer
            graph_proj = self.head(graph_features)
            outputs["graph_projections"] = graph_proj

        return outputs

    def _forward_separate(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor | None,
        num_graphs: int | None,
        masked_indices: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass using separate heads for graph and node predictions.

        Parameters
        ----------
        node_features : torch.Tensor
            Node features with shape ``(N, D)``.
        batch : torch.Tensor, optional
            Batch indices for graph pooling.
        num_graphs : int, optional
            Number of graphs in the batch.
        masked_indices : torch.Tensor, optional
            Boolean mask for masked tokens.

        Returns
        -------
        dict[str, torch.Tensor]
            Outputs containing graph and node projections and optional patch predictions.
        """
        outputs = {}
        outputs["node_feats"] = node_features

        # Node-level projections (all tokens)
        if self.node_projector is not None:
            node_proj = self.node_projector(node_features)
            outputs["node_projections"] = node_proj

        # Patch predictions (masked tokens only)
        if self.node_patch_predictor is not None:
            if masked_indices is not None:
                # Only compute predictions for masked tokens (efficiency)
                masked_features = node_features[masked_indices]
                patch_pred = self.node_patch_predictor(masked_features)
                outputs["node_patch_predictions"] = patch_pred
                outputs["masked_indices"] = masked_indices
            else:
                # Compute for all tokens if no mask provided
                patch_pred = self.node_patch_predictor(node_features)
                outputs["node_patch_predictions"] = patch_pred

        # Graph-level projections (pooled features)
        if batch is not None:
            if num_graphs is None:
                num_graphs = batch.max().item() + 1

            graph_features = torch.zeros(
                (num_graphs, node_features.size(1)),
                device=node_features.device,
                dtype=node_features.dtype,
            )
            graph_features.index_add_(0, batch, node_features)

            if self.reduce == "mean":
                count = torch.bincount(batch, minlength=num_graphs).float()
                count = torch.clamp(count, min=1)
                graph_features = graph_features / count.unsqueeze(1)

            outputs["graph_features"] = graph_features

            # Note: L2 normalization is done inside the projector before the last layer
            if self.graph_projector is not None:
                outputs["graph_projections"] = self.graph_projector(graph_features)

        return outputs


class iBOTCombinedHead(nn.Module):
    """Combined iBOT head for both student and teacher networks.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    projection_dim : int, default=8192
        Output dimension for graph projections.
    patch_out_dim : int, default=8192
        Output dimension for patch predictions.
    hidden_dim : int, default=2048
        Hidden dimension in projection MLP.
    bottleneck_dim : int, default=256
        Bottleneck dimension in projection MLP.
    do_ibot : bool, default=True
        Whether to compute iBOT patch predictions.
    use_bn : bool, default=False
        Whether to use batch normalization.
    norm_last_layer : bool, default=True
        Whether to normalize the last layer weights.
    reduce : str, default="mean"
        Reduction method for graph pooling.
    use_gelu : bool, default=True
        Whether to use GELU activation.
    """

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 8192,
        patch_out_dim: int = 8192,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        do_ibot: bool = True,
        use_bn: bool = False,
        norm_last_layer: bool = True,
        reduce: str = "mean",
        use_gelu: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.projection_head = iBOTProjectionHead(
            input_dim=input_dim,
            projection_dim=projection_dim,
            patch_out_dim=patch_out_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            do_ibot=do_ibot,
            use_bn=use_bn,
            norm_last_layer=norm_last_layer,
            reduce=reduce,
            use_gelu=use_gelu,
        )

    @classmethod
    def build_from_backbone_info(
        cls, backbone_info: Dict[str, Any], **kwargs: Any
    ) -> "iBOTCombinedHead":
        kwargs = dict(kwargs)
        if kwargs.get("input_dim") is None:
            output_dim = backbone_info.get("output_dim")
            if output_dim is None:
                raise ValueError(
                    "iBOTCombinedHead requires `input_dim` or backbone_info['output_dim']."
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
        """Forward pass that delegates to the projection head.

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
            Optional transform.
        **kwargs : Any
            Additional arguments forwarded to the projection head.

        Returns
        -------
        dict[str, torch.Tensor]
            Projection outputs.
        """
        return self.projection_head(
            backbone_outputs,
            batch,
            outputs=outputs,
            training=training,
            transform=transform,
            **kwargs,
        )
