"""Split Barlow Twins projection head with dual-branch MLP projectors."""

import warnings
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from ..outputs import BackboneOutputs
from .barlow_twins import (
    GraphAggregator,
    get_o3_irreps,
    maybe_import_e3nn,
    needs_invariance,
)

o3 = maybe_import_e3nn()


class DualBranchProjector(nn.Module):
    """Two independent MLPs that operate on split feature halves.

    Parameters
    ----------
    left_in_dim, right_in_dim : int
        Size of each feature half (must sum to total input dim).
    hidden_dims : List[int]
        Hidden dimensions for each branch.
    branch_output_dim : int
        Output dimension per branch (concatenated output doubles this).
    use_bn : bool, default=True
        Apply BatchNorm1d after each hidden linear layer.
    dropout : float, default=0.0
        Dropout probability applied after ReLU activations.
    """

    def __init__(
        self,
        left_in_dim: int,
        right_in_dim: int,
        hidden_dims: List[int],
        branch_output_dim: int,
        use_bn: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if left_in_dim <= 0 or right_in_dim <= 0:
            raise ValueError(
                "DualBranchProjector requires both split sizes to be > 0; "
                f"got {left_in_dim} and {right_in_dim}."
            )
        self.left_in_dim = left_in_dim
        self.right_in_dim = right_in_dim
        self.branch_output_dim = branch_output_dim
        self.use_bn = use_bn
        self.dropout = dropout
        self.hidden_dims = hidden_dims or []

        self.left_branch = self._build_branch(left_in_dim)
        self.right_branch = self._build_branch(right_in_dim)

    def _build_branch(self, input_dim: int) -> nn.Module:
        layers: List[nn.Module] = []
        prev_dim = input_dim
        hidden_dims = self.hidden_dims

        if not hidden_dims:
            layers.append(nn.Linear(prev_dim, self.branch_output_dim, bias=True))
            return nn.Sequential(*layers)

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=not self.use_bn))
            if self.use_bn:
                layers.append(
                    nn.BatchNorm1d(hidden_dim, affine=True, eps=1e-5, momentum=0.1)
                )
            if i < len(hidden_dims) - 1:
                layers.append(nn.ReLU(inplace=True))
                if self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim

        if prev_dim != self.branch_output_dim:
            layers.append(nn.Linear(prev_dim, self.branch_output_dim, bias=True))

        return nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left, right = torch.split(x, [self.left_in_dim, self.right_in_dim], dim=-1)
        left_out = self.left_branch(left)
        right_out = self.right_branch(right)
        concat = torch.cat([left_out, right_out], dim=-1)
        return concat, left_out, right_out


class SplitBarlowTwinsProjectionHead(nn.Module):
    """Barlow Twins head that enforces separate projectors per feature half.

    Parameters
    ----------
    input_dim : int, optional
        Input feature dimension (required when ``input_irreps`` is not provided).
    input_irreps : str, optional
        e3nn irreps specification for equivariant inputs.
    projection_dim : int, default=8192
        Output projection dimension (concatenated across branches).
    compute_node_level : bool, default=True
        Whether to compute node-level projections.
    compute_graph_level : bool, default=True
        Whether to compute graph-level projections.
    hidden_dims : list[int], optional
        Hidden layer dimensions for each branch MLP.
    use_bn : bool, default=True
        Whether to use batch normalization.
    reduce : str, default="mean"
        Reduction method for graph pooling.
    dropout : float, default=0.0
        Dropout probability.
    split_dim : int, optional
        Dimension where the input is split into two branches.
    branch_projection_dim : int, optional
        Output dimension per branch.
    branch_names : tuple[str, str], default=("comp", "geom")
        Names for the two branches.
    **kwargs : Any
        Additional unused arguments.

    Notes
    -----
    This head is useful when the latent is conceptually split into two parts
    (e.g., composition vs geometry). Each half is processed by its own MLP and
    the resulting projections are concatenated, while also being returned
    individually for custom losses.
    """

    def __init__(
        self,
        input_dim: int | None = None,
        input_irreps: str | None = None,
        projection_dim: int = 8192,
        compute_node_level: bool = True,
        compute_graph_level: bool = True,
        hidden_dims: List[int] | None = None,
        use_bn: bool = True,
        reduce: str = "mean",
        dropout: float = 0.0,
        split_dim: int | None = None,
        branch_projection_dim: int | None = None,
        branch_names: Tuple[str, str] = ("comp", "geom"),
        **kwargs,
    ) -> None:
        super().__init__()

        self.input_irreps = input_irreps
        self.compute_node_level = compute_node_level
        self.compute_graph_level = compute_graph_level
        self.use_bn = use_bn
        self.reduce = reduce
        self.dropout = dropout
        self.branch_names = branch_names

        if hidden_dims is None:
            hidden_dims = [8192, 8192, 8192]
        self.hidden_dims = hidden_dims

        # Handle input irreps (mirrors BarlowTwinsProjectionHead logic)
        if input_irreps is not None:
            if o3 is None:
                raise ImportError(
                    "e3nn is required when using input_irreps. Install with: pip install e3nn"
                )
            self.requires_invariance = needs_invariance(input_irreps)
            self.to_invariant = get_o3_irreps(input_irreps, hidden_dims[0])

            if self.requires_invariance:
                self.input_dim = hidden_dims[0]
            else:
                self.input_dim = sum(
                    mul for mul, ir in o3.Irreps(input_irreps) if ir.l == 0
                )
        else:
            if input_dim is None:
                raise ValueError("Must provide either input_dim or input_irreps")
            self.requires_invariance = False
            self.to_invariant = None
            self.input_dim = input_dim

        if isinstance(projection_dim, str):
            raise ValueError(
                "SplitBarlowTwinsProjectionHead currently supports only integer projection_dim"
            )

        if branch_projection_dim is None:
            if projection_dim % 2 != 0:
                raise ValueError(
                    f"projection_dim ({projection_dim}) must be even when branch_projection_dim is None"
                )
            branch_projection_dim = projection_dim // 2

        if branch_projection_dim <= 0:
            raise ValueError(
                f"branch_projection_dim must be positive, got {branch_projection_dim}"
            )

        self.branch_projection_dim = branch_projection_dim
        self.projection_dim = branch_projection_dim * 2

        if split_dim is None:
            split_dim = self.input_dim // 2
        if split_dim <= 0 or split_dim >= self.input_dim:
            raise ValueError(
                f"split_dim must lie in (0, {self.input_dim}), got {split_dim}."
            )

        if len(branch_names) != 2:
            raise ValueError(
                f"branch_names must contain exactly two entries, got {branch_names}"
            )

        self.left_in_dim = split_dim
        self.right_in_dim = self.input_dim - split_dim

        self.attention = None
        self.aggregator = GraphAggregator(reduce=self.reduce)

        self.node_projector = (
            DualBranchProjector(
                self.left_in_dim,
                self.right_in_dim,
                hidden_dims,
                branch_projection_dim,
                use_bn=use_bn,
                dropout=dropout,
            )
            if compute_node_level
            else None
        )
        self.graph_projector = (
            DualBranchProjector(
                self.left_in_dim,
                self.right_in_dim,
                hidden_dims,
                branch_projection_dim,
                use_bn=use_bn,
                dropout=dropout,
            )
            if compute_graph_level
            else None
        )

        self._initialize_weights()

    @classmethod
    def build_from_backbone_info(
        cls, backbone_info: Dict[str, Any], **kwargs: Any
    ) -> "SplitBarlowTwinsProjectionHead":
        kwargs = dict(kwargs)
        if kwargs.get("input_irreps") is None and kwargs.get("input_dim") is None:
            output_dim = backbone_info.get("output_dim")
            if output_dim is None:
                raise ValueError(
                    "SplitBarlowTwinsProjectionHead requires `input_dim`/`input_irreps` "
                    "or backbone_info['output_dim']."
                )
            kwargs["input_dim"] = int(output_dim)
        return cls(**kwargs)

    def get_head_build_info(self) -> Dict[str, Any]:
        return {"projection_dim": int(self.projection_dim)}

    def _initialize_weights(self) -> None:
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
                pass

    def _pool_graph(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int | None,
    ) -> torch.Tensor:
        if self.attention is not None:
            return self.attention(node_features, batch, num_graphs)
        return self.aggregator(node_features, batch, num_graphs)

    def _project_split(
        self,
        projector: DualBranchProjector,
        features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        concat, left, right = projector(features)
        out: Dict[str, torch.Tensor] = {
            "concat": concat,
            f"branch_{self.branch_names[0]}": left,
            f"branch_{self.branch_names[1]}": right,
        }
        return out

    def forward(
        self,
        backbone_outputs: BackboneOutputs,
        batch: Any,
        outputs: Dict[str, torch.Tensor] | None = None,
        training: bool = False,
        transform: Any = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for the split projection head.

        Parameters
        ----------
        backbone_outputs : BackboneOutputs
            Backbone outputs with node and graph features.
        batch : Any
            Batch object providing ``batch`` and ``num_graphs`` metadata.
        outputs : dict[str, torch.Tensor], optional
            Existing outputs dict.
        training : bool, default=False
            Training mode flag.
        transform : Any, optional
            Optional transform (unused).
        **kwargs : Any
            Additional arguments (e.g., ``graph_features`` override).

        Returns
        -------
        dict[str, torch.Tensor]
            Outputs including node/graph projections and branch splits.
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

        if torch.isnan(node_features).any():
            warnings.warn(
                f"NaN detected in input node_features! Count: {torch.isnan(node_features).sum().item()}"
            )

        outputs["node_feats"] = node_features

        if self.requires_invariance and self.to_invariant is not None:
            features = self.to_invariant(node_features)
        elif self.input_irreps is not None and not self.requires_invariance:
            features = node_features[..., : self.input_dim]
        else:
            features = node_features

        if self.node_projector is not None:
            proj_dict = self._project_split(self.node_projector, features)
            if torch.isnan(proj_dict["concat"]).any():
                warnings.warn(
                    "NaN detected in node_projections after split projector! "
                    f"Count: {torch.isnan(proj_dict['concat']).sum().item()}"
                )
            outputs["node_projections"] = proj_dict["concat"]
            outputs[f"node_projections_{self.branch_names[0]}"] = proj_dict[
                f"branch_{self.branch_names[0]}"
            ]
            outputs[f"node_projections_{self.branch_names[1]}"] = proj_dict[
                f"branch_{self.branch_names[1]}"
            ]

        # Graph-level paths
        # Use pre-pooled graph_features if provided, otherwise pool from node features
        if graph_features is not None:
            # Use pre-pooled graph features directly (e.g., CLS token outputs)
            if torch.isnan(graph_features).any():
                warnings.warn(
                    f"NaN detected in input graph_features! Count: {torch.isnan(graph_features).sum().item()}"
                )
            outputs["graph_features"] = graph_features

            if self.graph_projector is not None:
                proj_dict = self._project_split(self.graph_projector, graph_features)
                if torch.isnan(proj_dict["concat"]).any():
                    warnings.warn(
                        "NaN detected in graph_projections after split projector! "
                        f"Count: {torch.isnan(proj_dict['concat']).sum().item()}"
                    )
                outputs["graph_projections"] = proj_dict["concat"]
                outputs[f"graph_projections_{self.branch_names[0]}"] = proj_dict[
                    f"branch_{self.branch_names[0]}"
                ]
                outputs[f"graph_projections_{self.branch_names[1]}"] = proj_dict[
                    f"branch_{self.branch_names[1]}"
                ]
        elif batch_idx is not None:
            # Pool node features to graph level
            pooled_graph_features = self._pool_graph(features, batch_idx, num_graphs)
            if torch.isnan(pooled_graph_features).any():
                warnings.warn(
                    f"NaN detected in graph_features after pooling! Count: {torch.isnan(pooled_graph_features).sum().item()}"
                )
            outputs["graph_features"] = pooled_graph_features

            if self.graph_projector is not None:
                proj_dict = self._project_split(
                    self.graph_projector, pooled_graph_features
                )
                if torch.isnan(proj_dict["concat"]).any():
                    warnings.warn(
                        "NaN detected in graph_projections after split projector! "
                        f"Count: {torch.isnan(proj_dict['concat']).sum().item()}"
                    )
                outputs["graph_projections"] = proj_dict["concat"]
                outputs[f"graph_projections_{self.branch_names[0]}"] = proj_dict[
                    f"branch_{self.branch_names[0]}"
                ]
                outputs[f"graph_projections_{self.branch_names[1]}"] = proj_dict[
                    f"branch_{self.branch_names[1]}"
                ]

        return outputs
