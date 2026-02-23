from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .composition_stream import CompositionStream
from .outputs import BackboneOutputs
from .structural_stream import StructuralStreamPowerSpectrum


def _get_first_attr(batch: object, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if hasattr(batch, name):
            value = getattr(batch, name)
            if value is not None:
                return value
    return default


def _mean_pool(
    node_feats: torch.Tensor, batch: torch.Tensor, num_graphs: int
) -> torch.Tensor:
    out = node_feats.new_zeros((num_graphs, node_feats.size(-1)))
    out.index_add_(0, batch, node_feats)
    count = torch.bincount(batch, minlength=num_graphs).clamp_min(1).to(out.dtype)
    return out / count.unsqueeze(1)


class TriForcesModel(nn.Module):
    """TriForces multi-stream model with interaction, composition, and structural streams.

    Parameters
    ----------
    interaction : nn.Module
        Interaction backbone returning node and graph features.
    interaction_dim : int, optional
        Dimension of interaction features. If ``None``, inferred from the
        interaction backbone when possible.
    interaction_name : str, default="interaction"
        Name for the interaction stream.
    enable_composition : bool, default=True
        Whether to include the composition stream.
    enable_structural : bool, default=True
        Whether to include the structural stream.
    composition_dim : int, default=256
        Composition stream embedding dimension.
    structural_dim : int, default=256
        Structural stream embedding dimension.
    num_heads : int, default=8
        Number of attention heads for composition stream.
    num_comp_layers : int, default=4
        Number of composition stream layers.
    ffn_dim : int, optional
        Feed-forward hidden dimension. Defaults to ``4 * embed_dim``.
    dropout : float, default=0.1
        Dropout probability.
    force_no_dropout : bool, default=True
        If True, override dropout to zero.
    cutoff : float, default=6.0
        Cutoff radius for structural stream.
    num_elements : int, default=100
        Number of elements for composition embeddings.
    num_radial : int, default=8
        Number of radial basis functions.
    num_radial_out : int, default=8
        Output radial dimension after mixing.
    l_max : int, default=4
        Maximum angular momentum.
    radial_type : str, default="bessel"
        Radial basis type for the structural stream.
    structural_num_layers : int, default=3
        Number of structural stream MLP blocks.
    use_lattice : bool, default=False
        Whether to include lattice features.
    disable_lattice : bool, default=False
        If True, zero out lattice features while keeping the pathway.
    stoichiometry_mode : str, default="none"
        Stoichiometry handling mode for composition stream.
    fusion_mode : str, default="concat"
        Stream fusion mode: ``"concat"``, ``"add"``, or ``"gated"``.
    use_final_mlp : bool, default=False
        Whether to apply a final MLP after fusion.
    output_dim : int, optional
        Output dimension when ``use_final_mlp=True``.
    """

    def __init__(
        self,
        *,
        interaction: nn.Module,
        interaction_dim: int | None = None,
        interaction_name: str = "interaction",
        enable_composition: bool = True,
        enable_structural: bool = True,
        composition_dim: int = 256,
        structural_dim: int = 256,
        num_heads: int = 8,
        num_comp_layers: int = 4,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        force_no_dropout: bool = True,
        cutoff: float = 6.0,
        num_elements: int = 100,
        num_radial: int = 8,
        num_radial_out: int = 8,
        l_max: int = 4,
        radial_type: str = "bessel",
        structural_num_layers: int = 3,
        use_lattice: bool = False,
        disable_lattice: bool = False,
        stoichiometry_mode: str = "none",
        fusion_mode: str = "concat",
        use_final_mlp: bool = False,
        output_dim: int | None = None,
    ):
        super().__init__()

        if force_no_dropout:
            dropout = 0.0

        self.interaction = interaction
        self.interaction_dim = int(
            interaction_dim
            if interaction_dim is not None
            else self._infer_interaction_dim(interaction)
        )
        self.interaction_name = interaction_name

        self.enable_composition = enable_composition
        self.enable_structural = enable_structural
        self.fusion_mode = fusion_mode
        self.use_final_mlp = use_final_mlp
        self.cutoff = cutoff
        self.use_lattice = use_lattice
        self.disable_lattice = disable_lattice
        self.stoichiometry_mode = stoichiometry_mode

        stream_dims = [self.interaction_dim]
        self.num_streams = 1

        if enable_composition:
            self.composition_stream = CompositionStream(
                num_elements=num_elements,
                embed_dim=composition_dim,
                num_heads=num_heads,
                num_layers=num_comp_layers,
                ffn_dim=ffn_dim,
                dropout=dropout,
                use_final_norm=True,
                use_cls=True,
                stoichiometry_mode=stoichiometry_mode,
            )
            self.num_streams += 1
            stream_dims.append(composition_dim)
            self.composition_dim = composition_dim
        else:
            self.composition_stream = None
            self.composition_dim = 0

        if enable_structural:
            self.structural_stream = StructuralStreamPowerSpectrum(
                embed_dim=structural_dim,
                num_radial=num_radial,
                num_radial_out=num_radial_out,
                l_max=l_max,
                cutoff=cutoff,
                radial_type=radial_type,
                num_layers=structural_num_layers,
                ffn_dim=ffn_dim,
                dropout=dropout,
                use_lattice=use_lattice,
                disable_lattice=disable_lattice,
            )
            self.num_streams += 1
            stream_dims.append(structural_dim)
            self.structural_dim = structural_dim
        else:
            self.structural_stream = None
            self.structural_dim = 0

        self.concat_dim = sum(stream_dims)

        if fusion_mode == "concat":
            self.fused_dim = self.concat_dim
        elif fusion_mode == "add":
            if not all(d == stream_dims[0] for d in stream_dims):
                raise ValueError(
                    f"All streams must share dimension for add fusion: {stream_dims}"
                )
            self.fused_dim = stream_dims[0]
        elif fusion_mode == "gated":
            self.gate = nn.Sequential(
                nn.Linear(self.concat_dim, self.num_streams),
                nn.Softmax(dim=-1),
            )
            self.interaction_proj = nn.Linear(
                self.interaction_dim, self.interaction_dim
            )
            if enable_composition:
                self.comp_proj = nn.Linear(composition_dim, self.interaction_dim)
            if enable_structural:
                self.struct_proj = nn.Linear(structural_dim, self.interaction_dim)
            self.fused_dim = self.interaction_dim
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        if use_final_mlp:
            final_output_dim = output_dim if output_dim is not None else self.fused_dim
            final_ffn_dim = ffn_dim if ffn_dim is not None else final_output_dim * 4
            self.final_mlp = nn.Sequential(
                nn.Linear(self.fused_dim, final_ffn_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(final_ffn_dim, final_output_dim),
            )
            self.output_dim = final_output_dim
        else:
            self.final_mlp = None
            self.output_dim = self.fused_dim

    def _infer_interaction_dim(self, interaction: nn.Module) -> int:
        for attr in ("output_dim", "embed_dim", "hidden_dim"):
            value = getattr(interaction, attr, None)
            if isinstance(value, int) and value > 0:
                return int(value)

        get_info = getattr(interaction, "get_head_build_info", None)
        if callable(get_info):
            info = get_info()
            value = info.get("output_dim") if isinstance(info, dict) else None
            if isinstance(value, int) and value > 0:
                return int(value)

        get_readout = getattr(interaction, "get_readout", None)
        if callable(get_readout):
            try:
                readout = get_readout()
            except Exception:
                readout = None
            if (
                isinstance(readout, tuple)
                and len(readout) >= 2
                and readout[1] is not None
            ):
                dim = readout[1]
                if isinstance(dim, int) and dim > 0:
                    return int(dim)
                dim_attr: Any = getattr(dim, "dim", None)
                if isinstance(dim_attr, int) and dim_attr > 0:
                    return int(dim_attr)

        raise ValueError(
            "Could not infer interaction_dim from interaction backbone. "
            "Provide `interaction_dim` explicitly."
        )

    def _resolve_structural_edges(
        self,
        *,
        batch: object,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_index = _get_first_attr(batch, ("edge_index",))
        if edge_index is None:
            raise ValueError("Batch must provide `edge_index` for structural stream.")

        edge_vec = _get_first_attr(batch, ("edge_vec", "edge_distance_vec", "vectors"))
        if edge_vec is None:
            pos = _get_first_attr(batch, ("pos", "positions"))
            if pos is None:
                raise ValueError(
                    "Batch must provide `edge_vec`/`edge_distance_vec`/`vectors` "
                    "or (`pos` and `edge_index`) for structural stream."
                )
            edge_index_t = torch.as_tensor(edge_index, device=device, dtype=torch.long)
            src, dst = edge_index_t[0], edge_index_t[1]
            pos_t = torch.as_tensor(pos, device=device, dtype=dtype)
            edge_vec = pos_t[src] - pos_t[dst]

            cell_offsets = _get_first_attr(batch, ("cell_offsets",))
            cell = _get_first_attr(batch, ("cell", "lattice"))
            batch_idx = _get_first_attr(batch, ("batch",))
            cell_offsets_t = (
                torch.as_tensor(cell_offsets, device=device, dtype=dtype)
                if cell_offsets is not None
                else None
            )
            if (
                cell_offsets_t is not None
                and cell is not None
                and batch_idx is not None
                and cell_offsets_t.numel() > 0
            ):
                cell_t = torch.as_tensor(cell, device=device, dtype=dtype).view(
                    -1, 3, 3
                )
                edge_batch = torch.as_tensor(
                    batch_idx, device=device, dtype=torch.long
                )[src]
                edge_cell = cell_t[edge_batch]
                offset_vec = torch.einsum(
                    "ei,eij->ej",
                    cell_offsets_t,
                    edge_cell,
                )
                edge_vec = edge_vec + offset_vec

        edge_dist = _get_first_attr(batch, ("edge_dist", "edge_distance", "distances"))
        if edge_dist is None:
            edge_dist = torch.as_tensor(edge_vec, device=device, dtype=dtype).norm(
                dim=-1
            )

        return (
            torch.as_tensor(edge_index, device=device, dtype=torch.long),
            torch.as_tensor(edge_vec, device=device, dtype=dtype),
            torch.as_tensor(edge_dist, device=device, dtype=dtype).reshape(-1),
        )

    def _run_interaction(
        self,
        *,
        batch: object,
        training: bool,
        transform: object | None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        out = self.interaction(batch, training=training, transform=transform)
        if not isinstance(out, BackboneOutputs):
            raise TypeError("Interaction backbone must return BackboneOutputs.")
        interaction_extras = dict(out.extras) if isinstance(out.extras, dict) else {}
        return out.node_feats, out.graph_feats, interaction_extras

    def forward(
        self,
        batch: object,
        training: bool = False,
        transform: object | None = None,
    ) -> BackboneOutputs:
        z = _get_first_attr(batch, ("z", "atomic_numbers"))
        pos = _get_first_attr(batch, ("pos", "positions"))
        batch_idx = getattr(batch, "batch", None)

        if z is None or pos is None or batch_idx is None:
            raise ValueError("Batch must provide z, pos, and batch attributes.")

        # Keep backward compatibility for components that expect canonical names.
        if getattr(batch, "z", None) is None:
            setattr(batch, "z", z)
        if getattr(batch, "pos", None) is None:
            setattr(batch, "pos", pos)

        num_graphs = getattr(batch, "num_graphs", None)
        if num_graphs is None:
            num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() else 0
        num_graphs = int(num_graphs)

        cell = _get_first_attr(batch, ("cell", "lattice"))

        N = z.size(0)
        device = z.device

        # If a graph batch is already provided, transforms are already applied upstream.
        model_transform = None if hasattr(batch, "edge_index") else transform
        interaction_node, interaction_graph, interaction_extras = self._run_interaction(
            batch=batch,
            training=training,
            transform=model_transform,
        )

        stream_node_feats: Dict[str, torch.Tensor] = {
            self.interaction_name: interaction_node
        }
        stream_graph_feats: Dict[str, torch.Tensor] = {
            self.interaction_name: interaction_graph
        }
        # Provide a stable alias for downstream code/configs.
        if self.interaction_name != "interaction":
            stream_node_feats["interaction"] = interaction_node
            stream_graph_feats["interaction"] = interaction_graph
        node_outputs = [interaction_node]

        if self.composition_stream is not None:
            comp_node, comp_graph = self.composition_stream(
                batch,
                training=training,
                transform=model_transform,
            )
            stream_node_feats["composition"] = comp_node
            stream_graph_feats["composition"] = comp_graph
            node_outputs.append(comp_node)

        if self.structural_stream is not None:
            edge_index, edge_vec, edge_dist = self._resolve_structural_edges(
                batch=batch,
                device=device,
                dtype=pos.dtype,
            )
            if edge_index.numel() == 0:
                struct_node = torch.zeros(
                    (N, self.structural_dim), device=device, dtype=pos.dtype
                )
                struct_graph = _mean_pool(struct_node, batch_idx, num_graphs)
            else:
                num_atoms_per_graph = None
                if hasattr(batch, "num_atoms_per_graph"):
                    num_atoms_per_graph = torch.as_tensor(
                        getattr(batch, "num_atoms_per_graph"),
                        device=device,
                        dtype=torch.long,
                    )
                if cell is not None:
                    if num_atoms_per_graph is None:
                        num_atoms_per_graph = torch.zeros(
                            num_graphs, device=device, dtype=torch.long
                        )
                        num_atoms_per_graph.scatter_add_(
                            0,
                            batch_idx,
                            torch.ones(N, device=device, dtype=torch.long),
                        )

                struct_node, struct_graph = self.structural_stream(
                    pos=pos,
                    edge_index=edge_index,
                    edge_vec=edge_vec,
                    edge_dist=edge_dist,
                    batch_idx=batch_idx,
                    B=num_graphs,
                    cell=cell,
                    num_atoms_per_graph=num_atoms_per_graph,
                )
            stream_node_feats["structural"] = struct_node
            stream_graph_feats["structural"] = struct_graph
            node_outputs.append(struct_node)

        if self.fusion_mode == "concat":
            node_feats = torch.cat(node_outputs, dim=-1)
        elif self.fusion_mode == "add":
            node_feats = sum(node_outputs)
        else:  # gated
            concat_feats = torch.cat(node_outputs, dim=-1)
            gates = self.gate(concat_feats)
            projected = [self.interaction_proj(interaction_node)]
            if self.composition_stream is not None:
                projected.append(self.comp_proj(stream_node_feats["composition"]))
            if self.structural_stream is not None:
                projected.append(self.struct_proj(stream_node_feats["structural"]))
            node_feats = sum(
                gates[:, i : i + 1] * proj for i, proj in enumerate(projected)
            )

        if self.final_mlp is not None:
            node_feats = self.final_mlp(node_feats)

        graph_feats = _mean_pool(node_feats, batch_idx, num_graphs)

        extras: Dict[str, Any] = dict(interaction_extras)
        extras["stream_node_feats"] = stream_node_feats
        extras["stream_graph_feats"] = stream_graph_feats
        return BackboneOutputs(
            node_feats=node_feats,
            graph_feats=graph_feats,
            extras=extras,
        )

    def get_head_build_info(self) -> dict[str, object]:
        stream_dims: dict[str, int] = {"interaction": int(self.interaction_dim)}
        if self.composition_stream is not None and self.composition_dim > 0:
            stream_dims["composition"] = int(self.composition_dim)
        if self.structural_stream is not None and self.structural_dim > 0:
            stream_dims["structural"] = int(self.structural_dim)
        return {
            "output_dim": int(self.output_dim),
            "stream_dims": stream_dims,
        }
