"""Barlow Twins projection head for self-supervised learning."""

from typing import Any, Dict, List, Union

import torch
import torch.nn as nn

from ..outputs import BackboneOutputs


def maybe_import_e3nn():
    """Conditionally import e3nn if available."""
    try:
        # e3nn<=0.4 reads constants with torch.load. Torch>=2.6 defaults to
        # weights_only=True, which requires allowlisting builtins like `slice`.
        try:
            from torch.serialization import add_safe_globals

            add_safe_globals([slice])
        except Exception:
            pass

        from e3nn import o3

        return o3
    except Exception:
        return None


o3 = maybe_import_e3nn()


def needs_invariance(irreps: Union[str, int, None]) -> bool:
    """Check if the input irreps contain any non-scalar (l > 0) components."""
    if irreps is None:
        return False

    try:
        irreps = int(irreps)
        return False
    except (ValueError, TypeError):
        pass  # not an integer, so it's not an irrep

    if o3 is None:
        return False

    return any(mul > 0 and _l > 0 for mul, ir in o3.Irreps(irreps) for _l in [ir.l])


def get_o3_irreps(irreps: str, hidden_dim: int) -> nn.Module:
    """Create e3nn linear layer to project non-scalar irreps to scalars."""
    if o3 is None:
        return None

    input_irreps = o3.Irreps(irreps)
    if needs_invariance(irreps):
        return o3.Linear(
            input_irreps,
            o3.Irreps(f"{hidden_dim}x0e"),
        )
    else:
        return None


class GraphAggregator(nn.Module):
    """Reusable per-graph aggregator for ``"mean"`` and ``"sum"`` reductions.

    Parameters
    ----------
    reduce : str, default="mean"
        Reduction method, ``"mean"`` or ``"sum"``.
    """

    def __init__(self, reduce: str = "mean"):
        super().__init__()
        assert reduce in {"mean", "sum"}, "GraphAggregator supports 'mean' or 'sum'."
        self.reduce = reduce

    @torch.no_grad()
    def _num_graphs(self, batch: torch.Tensor, num_graphs: int | None) -> int:
        return int(batch.max().item()) + 1 if num_graphs is None else int(num_graphs)

    def forward(
        self,
        x: torch.Tensor,  # [num_nodes, feat_dim]
        batch: torch.Tensor,  # [num_nodes] graph id per node (0..G-1)
        num_graphs: int | None = None,
    ) -> torch.Tensor:  # [G, feat_dim]
        G = self._num_graphs(batch, num_graphs)
        out = x.new_zeros((G, x.size(1)))
        out.index_add_(0, batch, x)
        if self.reduce == "mean":
            count = torch.bincount(batch, minlength=G).clamp_min(1).to(out.dtype)
            out = out / count.unsqueeze(1)
        return out


class BarlowTwinsProjectionHead(nn.Module):
    """Barlow Twins projection head with symmetric architecture.

    Parameters
    ----------
    input_dim : int, optional
        Input feature dimension (required when ``input_irreps`` is not provided).
    input_irreps : str, optional
        e3nn irreps specification for equivariant inputs.
    projection_dim : int or str, default=8192
        Output projection dimension or e3nn irreps string.
    compute_node_level : bool, default=True
        Whether to compute node-level projections.
    compute_graph_level : bool, default=True
        Whether to compute graph-level projections.
    hidden_dims : list[int], optional
        Hidden layer dimensions for the MLP projector.
    use_bn : bool, default=True
        Whether to use batch normalization.
    reduce : str, default="mean"
        Reduction method for graph pooling.
    dropout : float, default=0.0
        Dropout probability.
    separate_projectors : bool, default=False
        Whether to use separate projectors per view.

    Notes
    -----
    With ``separate_projectors=False``, both views share the same projector,
    enforcing invariance. With ``separate_projectors=True``, each view uses a
    different projector to allow equivariant representations.
    """

    def __init__(
        self,
        input_dim: int | None = None,
        input_irreps: str | None = None,
        projection_dim: Union[int, str] = 8192,
        compute_node_level: bool = True,
        compute_graph_level: bool = True,
        hidden_dims: List[int] | None = None,
        use_bn: bool = True,
        reduce: str = "mean",  # "mean" | "sum"
        dropout: float = 0.0,
        separate_projectors: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.input_irreps = input_irreps
        self.projection_dim = projection_dim
        self.compute_node_level = compute_node_level
        self.compute_graph_level = compute_graph_level
        self.reduce = reduce
        self.dropout = dropout
        self.use_bn = use_bn
        self.separate_projectors = separate_projectors

        if hidden_dims is None:
            hidden_dims = [8192, 8192, 8192]
        self.hidden_dims = hidden_dims

        # Handle input irreps (extract scalars or project to invariants)
        if input_irreps is not None:
            if o3 is None:
                raise ImportError(
                    "e3nn is required when using input_irreps. "
                    "Install with: pip install e3nn"
                )
            self.requires_invariance = needs_invariance(input_irreps)
            self.to_invariant = get_o3_irreps(input_irreps, hidden_dims[0])

            # Determine actual input dimension after invariance conversion
            if self.requires_invariance:
                # Will project l>0 to scalars using e3nn
                self.input_dim = hidden_dims[0]
            else:
                # Already all scalars, just extract them
                self.input_dim = sum(
                    mul for mul, ir in o3.Irreps(input_irreps) if ir.l == 0
                )
        else:
            # Standard scalar input
            if input_dim is None:
                raise ValueError("Must provide either input_dim or input_irreps")
            self.input_dim = input_dim
            self.requires_invariance = False
            self.to_invariant = None

        # Determine if output uses e3nn irreps
        self.use_output_irreps = isinstance(projection_dim, str)
        if self.use_output_irreps and o3 is None:
            raise ImportError(
                "e3nn is required for irreps-based projection_dim. "
                "Install with: pip install e3nn"
            )

        # Store actual output dimension for tensor initialization
        if self.use_output_irreps:
            self._output_dim = o3.Irreps(projection_dim).dim
        else:
            self._output_dim = projection_dim

        self.attention = None
        self.aggregator = GraphAggregator(reduce=self.reduce)

        # Projectors (view 0 / shared projector)
        self.node_projector = (
            self._build_projector(
                self.input_dim, hidden_dims, projection_dim, use_bn=use_bn
            )
            if compute_node_level
            else None
        )
        self.graph_projector = (
            self._build_projector(
                self.input_dim, hidden_dims, projection_dim, use_bn=use_bn
            )
            if compute_graph_level
            else None
        )

        # Second set of projectors for equivariant mode (view 1)
        if self.separate_projectors:
            self.node_projector_v1 = (
                self._build_projector(
                    self.input_dim, hidden_dims, projection_dim, use_bn=use_bn
                )
                if compute_node_level
                else None
            )
            self.graph_projector_v1 = (
                self._build_projector(
                    self.input_dim, hidden_dims, projection_dim, use_bn=use_bn
                )
                if compute_graph_level
                else None
            )
        else:
            self.node_projector_v1 = None
            self.graph_projector_v1 = None

        self._initialize_weights()

    @classmethod
    def build_from_backbone_info(
        cls, backbone_info: Dict[str, Any], **kwargs: Any
    ) -> "BarlowTwinsProjectionHead":
        kwargs = dict(kwargs)
        if kwargs.get("input_irreps") is None and kwargs.get("input_dim") is None:
            output_dim = backbone_info.get("output_dim")
            if output_dim is None:
                raise ValueError(
                    "BarlowTwinsProjectionHead requires `input_dim`/`input_irreps` "
                    "or backbone_info['output_dim']."
                )
            kwargs["input_dim"] = int(output_dim)
        return cls(**kwargs)

    def get_head_build_info(self) -> Dict[str, Any]:
        if isinstance(self.projection_dim, int) and self.projection_dim > 0:
            return {"projection_dim": int(self.projection_dim)}
        return {}

    def _initialize_weights(self):
        """Initialize weights to prevent initial collapse."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                if m.affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
            elif o3 is not None and isinstance(m, o3.Linear):
                # e3nn layers have their own initialization
                # but we can optionally normalize them here
                pass

    def _build_projector(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: Union[int, str],
        use_bn: bool = True,
    ) -> nn.Module:
        """Build the projection MLP.

        Parameters
        ----------
        input_dim : int
            Input feature dimension.
        hidden_dims : list[int]
            Hidden layer dimensions.
        output_dim : int or str
            Output dimension. When a string is provided, it is interpreted as an
            e3nn irreps specification (e.g., ``"64x0e+6x1e"``).
        use_bn : bool, default=True
            Whether to use batch normalization.

        Returns
        -------
        nn.Module
            Projector module.
        """
        if self.use_output_irreps:
            # E3NN equivariant projection
            return self._build_e3nn_projector(
                input_dim, hidden_dims, output_dim, use_bn
            )
        else:
            # Standard scalar projection
            return self._build_standard_projector(
                input_dim, hidden_dims, output_dim, use_bn
            )

    def _build_standard_projector(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        use_bn: bool = True,
    ) -> nn.Module:
        """Build standard scalar projection MLP."""
        layers: List[nn.Module] = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=not use_bn))
            if use_bn:
                layers.append(
                    nn.BatchNorm1d(hidden_dim, affine=True, eps=1e-5, momentum=0.1)
                )
            if i < len(hidden_dims) - 1:
                layers.append(nn.ReLU(inplace=True))
                if self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim

        if prev_dim != output_dim:
            layers.append(nn.Linear(prev_dim, output_dim, bias=True))

        return nn.Sequential(*layers)

    def _build_e3nn_projector(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_irreps: str,
        use_bn: bool = True,
    ) -> nn.Module:
        """Build an e3nn equivariant projector.

        Parameters
        ----------
        input_dim : int
            Input feature dimension.
        hidden_dims : list[int]
            Hidden layer dimensions.
        output_irreps : str
            Output irreps specification.
        use_bn : bool, default=True
            Whether to use batch normalization.

        Returns
        -------
        nn.Module
            Projector mapping scalar features to the requested irreps.
        """
        layers: List[nn.Module] = []
        prev_dim = input_dim

        # Build scalar hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=not use_bn))
            if use_bn:
                layers.append(
                    nn.BatchNorm1d(hidden_dim, affine=True, eps=1e-5, momentum=0.1)
                )
            if i < len(hidden_dims) - 1:
                layers.append(nn.ReLU(inplace=True))
                if self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim

        # Final layer: scalar -> irreps using e3nn
        input_irreps = o3.Irreps(f"{prev_dim}x0e")
        output_irreps_obj = o3.Irreps(output_irreps)
        layers.append(o3.Linear(input_irreps, output_irreps_obj, biases=True))

        return nn.Sequential(*layers)

    def _pool_graph(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int | None,
    ) -> torch.Tensor:
        """Pool node features to graph level using the selected reduction."""
        if self.attention is not None:
            return self.attention(node_features, batch, num_graphs)
        else:
            return self.aggregator(node_features, batch, num_graphs)

    def forward(
        self,
        backbone_outputs: BackboneOutputs,
        batch: Any,
        outputs: Dict[str, torch.Tensor] | None = None,
        training: bool = False,
        transform: Any = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Compute node and graph projections for Barlow Twins.

        Parameters
        ----------
        backbone_outputs : BackboneOutputs
            Backbone outputs containing node and optional graph features.
        batch : Any
            Batch object providing ``batch`` (node-to-graph indices) and
            optionally ``num_graphs``.
        outputs : dict[str, torch.Tensor], optional
            Existing outputs dict (unused).
        training : bool, default=False
            Training mode flag (unused; behavior depends on module mode).
        transform : Any, optional
            Optional transform (unused).
        **kwargs : Any
            Additional arguments.

            - graph_features : torch.Tensor, optional
              Pre-pooled graph features with shape ``(G, D)`` to use instead of
              pooling from node features.
            - view_index : torch.Tensor, optional
              Per-graph view index (0 or 1) used to route samples to different
              projectors when ``separate_projectors=True``. Shape ``(G,)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Outputs dictionary containing ``node_feats`` and optional node/graph
            projections and pooled features.
        """
        outputs: Dict[str, torch.Tensor] = {}

        node_features = backbone_outputs.node_feats
        graph_features = backbone_outputs.graph_feats

        batch_idx = getattr(batch, "batch", None)
        num_graphs = getattr(batch, "num_graphs", None)
        if num_graphs is None and batch_idx is not None and batch_idx.numel():
            num_graphs = int(batch_idx.max().item()) + 1

        if "graph_features" in kwargs and kwargs["graph_features"] is not None:
            graph_features = kwargs["graph_features"]
        view_index = kwargs.get("view_index")

        # Check for NaN in input
        if torch.isnan(node_features).any():
            import warnings

            warnings.warn(
                f"NaN detected in input node_features! Count: {torch.isnan(node_features).sum().item()}"
            )

        outputs["node_feats"] = node_features

        # Convert to invariant features if needed
        if self.requires_invariance and self.to_invariant is not None:
            # Project l>0 components to scalars using e3nn
            features = self.to_invariant(node_features)
        elif self.input_irreps is not None and not self.requires_invariance:
            # Extract only scalar (l=0) components
            features = node_features[..., : self.input_dim]
        else:
            # Already scalar features
            features = node_features

        # Node-level projection head
        if self.node_projector is not None:
            if (
                self.separate_projectors
                and view_index is not None
                and batch_idx is not None
            ):
                # Route nodes to different projectors based on their graph's view_index
                node_view_index = view_index[batch_idx]  # [N] view index per node
                mask_v0 = node_view_index == 0
                mask_v1 = ~mask_v0

                # Process views through their respective projectors
                # Compute v0 first to get the output dtype (handles mixed precision)
                proj_v0 = (
                    self.node_projector(features[mask_v0]) if mask_v0.any() else None
                )
                proj_v1 = (
                    self.node_projector_v1(features[mask_v1])
                    if mask_v1.any() and self.node_projector_v1 is not None
                    else None
                )

                # Reconstruct in original order - use output dtype from projector
                out_dtype = proj_v0.dtype if proj_v0 is not None else proj_v1.dtype
                node_proj = torch.empty(
                    features.size(0),
                    self._output_dim,
                    dtype=out_dtype,
                    device=features.device,
                )
                if proj_v0 is not None:
                    node_proj[mask_v0] = proj_v0
                if proj_v1 is not None:
                    node_proj[mask_v1] = proj_v1
            else:
                # Standard mode: all nodes through same projector
                node_proj = self.node_projector(features)

            if torch.isnan(node_proj).any():
                import warnings

                warnings.warn(
                    f"NaN detected in node_projections after projection! Count: {torch.isnan(node_proj).sum().item()}"
                )
            outputs["node_projections"] = node_proj

        # Graph-level paths
        # Use pre-pooled graph_features if provided, otherwise pool from node features
        if graph_features is not None:
            # Use pre-pooled graph features directly (e.g., CLS token outputs)
            if torch.isnan(graph_features).any():
                import warnings

                warnings.warn(
                    f"NaN detected in input graph_features! Count: {torch.isnan(graph_features).sum().item()}"
                )
            outputs["graph_features"] = graph_features

            if self.graph_projector is not None:
                if self.separate_projectors and view_index is not None:
                    # Route graphs to different projectors based on view_index
                    mask_v0 = view_index == 0
                    mask_v1 = ~mask_v0

                    # Process views through their respective projectors
                    proj_v0 = (
                        self.graph_projector(graph_features[mask_v0])
                        if mask_v0.any()
                        else None
                    )
                    proj_v1 = (
                        self.graph_projector_v1(graph_features[mask_v1])
                        if mask_v1.any() and self.graph_projector_v1 is not None
                        else None
                    )

                    # Reconstruct in original order - use output dtype from projector
                    out_dtype = proj_v0.dtype if proj_v0 is not None else proj_v1.dtype
                    graph_proj = torch.empty(
                        graph_features.size(0),
                        self._output_dim,
                        dtype=out_dtype,
                        device=graph_features.device,
                    )
                    if proj_v0 is not None:
                        graph_proj[mask_v0] = proj_v0
                    if proj_v1 is not None:
                        graph_proj[mask_v1] = proj_v1
                else:
                    # Standard mode: all graphs through same projector
                    graph_proj = self.graph_projector(graph_features)

                if torch.isnan(graph_proj).any():
                    import warnings

                    warnings.warn(
                        f"NaN detected in graph_projections after projection! Count: {torch.isnan(graph_proj).sum().item()}"
                    )
                outputs["graph_projections"] = graph_proj
        elif batch_idx is not None:
            # Pool node features to graph level
            pooled_graph_features = self._pool_graph(features, batch_idx, num_graphs)
            if torch.isnan(pooled_graph_features).any():
                import warnings

                warnings.warn(
                    f"NaN detected in graph_features after pooling! Count: {torch.isnan(pooled_graph_features).sum().item()}"
                )
            outputs["graph_features"] = pooled_graph_features

            if self.graph_projector is not None:
                if self.separate_projectors and view_index is not None:
                    # Route graphs to different projectors based on view_index
                    mask_v0 = view_index == 0
                    mask_v1 = ~mask_v0

                    # Process views through their respective projectors
                    proj_v0 = (
                        self.graph_projector(pooled_graph_features[mask_v0])
                        if mask_v0.any()
                        else None
                    )
                    proj_v1 = (
                        self.graph_projector_v1(pooled_graph_features[mask_v1])
                        if mask_v1.any() and self.graph_projector_v1 is not None
                        else None
                    )

                    # Reconstruct in original order - use output dtype from projector
                    out_dtype = proj_v0.dtype if proj_v0 is not None else proj_v1.dtype
                    graph_proj = torch.empty(
                        pooled_graph_features.size(0),
                        self._output_dim,
                        dtype=out_dtype,
                        device=pooled_graph_features.device,
                    )
                    if proj_v0 is not None:
                        graph_proj[mask_v0] = proj_v0
                    if proj_v1 is not None:
                        graph_proj[mask_v1] = proj_v1
                else:
                    # Standard mode: all graphs through same projector
                    graph_proj = self.graph_projector(pooled_graph_features)

                if torch.isnan(graph_proj).any():
                    import warnings

                    warnings.warn(
                        f"NaN detected in graph_projections after projection! Count: {torch.isnan(graph_proj).sum().item()}"
                    )
                outputs["graph_projections"] = graph_proj

        return outputs
