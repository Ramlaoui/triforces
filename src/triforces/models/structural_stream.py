"""Learnable structural stream for materials science models.

Self-contained copy of the EntalOracle power-spectrum structural stream.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class CosineCutoff(nn.Module):
    """Cosine cutoff function for smooth transition to zero.

    Parameters
    ----------
    cutoff_upper : float, default=6.0
        Upper cutoff distance.
    """

    def __init__(self, cutoff_upper: float = 6.0):
        super().__init__()
        self.cutoff_upper = cutoff_upper

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff_upper).to(distances.dtype)
        return cutoffs


class BesselBasis(nn.Module):
    """Spherical Bessel basis functions for radial expansion.

    Parameters
    ----------
    num_basis : int, default=8
        Number of radial basis functions.
    cutoff : float, default=6.0
        Upper cutoff distance.
    """

    def __init__(self, num_basis: int = 8, cutoff: float = 6.0):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff

        freqs = math.pi * torch.arange(1, num_basis + 1) / cutoff
        self.register_buffer("freqs", freqs)
        self.register_buffer("norm", torch.tensor(math.sqrt(2.0 / cutoff)))

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        d = distances.unsqueeze(-1).clamp(min=1e-8)
        return self.norm * torch.sin(self.freqs * d) / d


class GaussianRBF(nn.Module):
    """Gaussian radial basis functions.

    Parameters
    ----------
    num_basis : int, default=16
        Number of radial basis functions.
    cutoff : float, default=6.0
        Upper cutoff distance.
    learnable : bool, default=False
        Whether centers and widths are learnable.
    """

    def __init__(
        self,
        num_basis: int = 16,
        cutoff: float = 6.0,
        learnable: bool = False,
    ):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff

        centers = torch.linspace(0, cutoff, num_basis)
        width = cutoff / (num_basis - 1) if num_basis > 1 else cutoff

        if learnable:
            self.centers = nn.Parameter(centers)
            self.log_width = nn.Parameter(torch.tensor(math.log(width)))
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("log_width", torch.tensor(math.log(width)))

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        d = distances.unsqueeze(-1)
        width = torch.exp(self.log_width)
        return torch.exp(-((d - self.centers) ** 2) / (2 * width**2))


@torch.jit.script
def _spherical_harmonics_l0_to_l4(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    sh_0_0 = torch.ones_like(x) * 0.28209479177387814

    sh_1_m1 = 0.4886025119029199 * y
    sh_1_0 = 0.4886025119029199 * z
    sh_1_p1 = 0.4886025119029199 * x

    sh_2_m2 = 1.0925484305920792 * x * y
    sh_2_m1 = 1.0925484305920792 * y * z
    sh_2_0 = 0.31539156525252005 * (3 * z * z - 1)
    sh_2_p1 = 1.0925484305920792 * x * z
    sh_2_p2 = 0.5462742152960396 * (x * x - y * y)

    sh_3_m3 = 0.5900435899266435 * y * (3 * x * x - y * y)
    sh_3_m2 = 2.890611442640554 * x * y * z
    sh_3_m1 = 0.4570457994644658 * y * (5 * z * z - 1)
    sh_3_0 = 0.3731763325901154 * z * (5 * z * z - 3)
    sh_3_p1 = 0.4570457994644658 * x * (5 * z * z - 1)
    sh_3_p2 = 1.445305721320277 * z * (x * x - y * y)
    sh_3_p3 = 0.5900435899266435 * x * (x * x - 3 * y * y)

    x2, y2, z2 = x * x, y * y, z * z
    sh_4_m4 = 2.5033429417967046 * x * y * (x2 - y2)
    sh_4_m3 = 1.7701307697799304 * y * z * (3 * x2 - y2)
    sh_4_m2 = 0.9461746957575601 * x * y * (7 * z2 - 1)
    sh_4_m1 = 0.6690465435572892 * y * z * (7 * z2 - 3)
    sh_4_0 = 0.10578554691520431 * (35 * z2 * z2 - 30 * z2 + 3)
    sh_4_p1 = 0.6690465435572892 * x * z * (7 * z2 - 3)
    sh_4_p2 = 0.47308734787878004 * (x2 - y2) * (7 * z2 - 1)
    sh_4_p3 = 1.7701307697799304 * x * z * (x2 - 3 * y2)
    sh_4_p4 = 0.6258357354491761 * (x2 * (x2 - 3 * y2) - y2 * (3 * x2 - y2))

    return torch.stack(
        [
            sh_0_0,
            sh_1_m1,
            sh_1_0,
            sh_1_p1,
            sh_2_m2,
            sh_2_m1,
            sh_2_0,
            sh_2_p1,
            sh_2_p2,
            sh_3_m3,
            sh_3_m2,
            sh_3_m1,
            sh_3_0,
            sh_3_p1,
            sh_3_p2,
            sh_3_p3,
            sh_4_m4,
            sh_4_m3,
            sh_4_m2,
            sh_4_m1,
            sh_4_0,
            sh_4_p1,
            sh_4_p2,
            sh_4_p3,
            sh_4_p4,
        ],
        dim=-1,
    )


class SphericalHarmonics(nn.Module):
    """Real spherical harmonics from direction vectors.

    Parameters
    ----------
    l_max : int, default=4
        Maximum angular momentum.
    """

    def __init__(self, l_max: int = 4):
        super().__init__()
        if l_max > 4:
            raise ValueError(f"l_max > 4 not implemented, got {l_max}")
        self.l_max = l_max
        self.num_coeffs = (l_max + 1) ** 2

    def forward(self, directions: torch.Tensor) -> torch.Tensor:
        x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]
        sh = _spherical_harmonics_l0_to_l4(x, y, z)
        return sh[..., : self.num_coeffs]


class LearnablePowerSpectrumDescriptor(nn.Module):
    """Learnable local environment descriptor based on power spectrum invariants.

    Parameters
    ----------
    num_radial : int, default=8
        Number of radial basis functions.
    num_radial_out : int, default=8
        Output radial dimension after mixing.
    l_max : int, default=4
        Maximum angular momentum.
    cutoff : float, default=6.0
        Upper cutoff distance.
    radial_type : str, default="bessel"
        Radial basis type, ``"bessel"`` or ``"gaussian"``.
    learnable_radial : bool, default=True
        Whether Gaussian radial parameters are learnable.
    multi_scale_cutoffs : list[float], optional
        Cutoff radii for multi-scale aggregation.
    """

    def __init__(
        self,
        num_radial: int = 8,
        num_radial_out: int = 8,
        l_max: int = 4,
        cutoff: float = 6.0,
        radial_type: str = "bessel",
        learnable_radial: bool = True,
        multi_scale_cutoffs: list[float] | None = None,
    ):
        super().__init__()
        self.num_radial = num_radial
        self.num_radial_out = num_radial_out
        self.l_max = l_max
        self.cutoff = cutoff

        if multi_scale_cutoffs is None:
            self.multi_scale_cutoffs = [cutoff * 0.5, cutoff * 0.75, cutoff]
        else:
            self.multi_scale_cutoffs = multi_scale_cutoffs
        self.num_scales = len(self.multi_scale_cutoffs)

        if radial_type == "bessel":
            self.radial_basis = BesselBasis(num_radial, cutoff)
        else:
            self.radial_basis = GaussianRBF(
                num_basis=num_radial, cutoff=cutoff, learnable=learnable_radial
            )

        self.spherical_harmonics = SphericalHarmonics(l_max)
        self.cutoff_fn = CosineCutoff(cutoff_upper=cutoff)

        self.radial_mixing = nn.Parameter(torch.randn(num_radial_out, num_radial) * 0.1)

        self.num_l = l_max + 1
        self.num_m = (l_max + 1) ** 2
        self.power_dim = num_radial_out * num_radial_out * self.num_l

    def compute_multi_scale_radial(self, distances: torch.Tensor):
        edge_radials = []
        edge_weights = []
        for cutoff in self.multi_scale_cutoffs:
            weight = CosineCutoff(cutoff_upper=cutoff)(distances)
            rad = self.radial_basis(distances)
            edge_radials.append(rad * weight.unsqueeze(-1))
            edge_weights.append(weight)
        edge_radial_multi = torch.cat(edge_radials, dim=-1)
        edge_weights = torch.stack(edge_weights, dim=-1).sum(dim=-1).clamp(min=1e-8)
        edge_weights = edge_weights / edge_weights.max()
        return edge_radial_multi, edge_weights

    def forward_with_edge_feats(
        self,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_dist: torch.Tensor,
        num_nodes: int,
        edge_radial_multi: torch.Tensor,
    ):
        src, dst = edge_index
        edge_dir = edge_vec / edge_dist.unsqueeze(-1).clamp(min=1e-8)
        sh = self.spherical_harmonics(edge_dir)

        radial = edge_radial_multi[:, : self.num_radial]
        mixed = torch.einsum("ek,ak->ea", radial, self.radial_mixing)

        c = mixed.unsqueeze(-1) * sh.unsqueeze(1)
        c_agg = torch.zeros(
            (num_nodes, self.num_radial_out, self.num_m),
            device=edge_vec.device,
            dtype=edge_vec.dtype,
        )
        c_agg.index_add_(0, dst, c)

        power = []
        idx = 0
        for l in range(self.num_l):
            n_coeffs = 2 * l + 1
            c_l = c_agg[:, :, idx : idx + n_coeffs]
            power_l = torch.einsum("nia,nja->nij", c_l, c_l)
            power.append(power_l.reshape(num_nodes, -1))
            idx += n_coeffs
        power_spectrum = torch.cat(power, dim=-1)

        return c_agg, power_spectrum


class LatticeEncoder(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(6, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        cell: torch.Tensor,
        batch_idx: torch.Tensor,
        num_atoms_per_graph: torch.Tensor,
    ) -> torch.Tensor:
        cell = cell.view(-1, 3, 3)
        a = torch.norm(cell[:, 0], dim=-1)
        b = torch.norm(cell[:, 1], dim=-1)
        c = torch.norm(cell[:, 2], dim=-1)
        alpha = torch.acos(
            torch.clamp(
                torch.sum(cell[:, 1] * cell[:, 2], dim=-1) / (b * c + 1e-8),
                -1.0,
                1.0,
            )
        )
        beta = torch.acos(
            torch.clamp(
                torch.sum(cell[:, 0] * cell[:, 2], dim=-1) / (a * c + 1e-8),
                -1.0,
                1.0,
            )
        )
        gamma = torch.acos(
            torch.clamp(
                torch.sum(cell[:, 0] * cell[:, 1], dim=-1) / (a * b + 1e-8),
                -1.0,
                1.0,
            )
        )

        lattice_params = torch.stack([a, b, c, alpha, beta, gamma], dim=-1)
        lattice_emb = self.mlp(lattice_params)
        return lattice_emb[batch_idx]


class MessagePassingLayer(nn.Module):
    def __init__(self, embed_dim: int, num_radial: int, dropout: float = 0.1):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim + num_radial, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_radial: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h
        src, dst = edge_index
        m_in = torch.cat([h[src], h[dst], edge_radial], dim=-1)
        m = self.edge_mlp(m_in)
        agg = h.new_zeros(h.shape)
        agg.index_add_(0, dst, m)
        out = self.node_mlp(torch.cat([h, agg], dim=-1))
        return h + out


class StructuralStreamPowerSpectrum(nn.Module):
    """Structural stream using learnable power spectrum descriptors.

    Parameters
    ----------
    embed_dim : int, default=256
        Embedding dimension for node and graph features.
    num_radial : int, default=8
        Number of radial basis functions.
    num_radial_out : int, default=8
        Output radial dimension after mixing.
    l_max : int, default=4
        Maximum angular momentum.
    cutoff : float, default=6.0
        Upper cutoff distance.
    radial_type : str, default="bessel"
        Radial basis type, ``"bessel"`` or ``"gaussian"``.
    num_layers : int, default=3
        Number of MLP blocks.
    num_mp_layers : int, default=2
        Number of message passing layers.
    ffn_dim : int, optional
        Feed-forward hidden dimension. Defaults to ``4 * embed_dim``.
    dropout : float, default=0.1
        Dropout probability.
    use_lattice : bool, default=True
        Whether to include lattice features.
    disable_lattice : bool, default=False
        If True, zero out lattice features while keeping the pathway.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_radial: int = 8,
        num_radial_out: int = 8,
        l_max: int = 4,
        cutoff: float = 6.0,
        radial_type: str = "bessel",
        num_layers: int = 3,
        num_mp_layers: int = 2,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        use_lattice: bool = True,
        disable_lattice: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_lattice = use_lattice
        self.disable_lattice = disable_lattice
        self.num_mp_layers = num_mp_layers

        self.descriptor = LearnablePowerSpectrumDescriptor(
            num_radial=num_radial,
            num_radial_out=num_radial_out,
            l_max=l_max,
            cutoff=cutoff,
            radial_type=radial_type,
            learnable_radial=True,
        )

        descriptor_dim = self.descriptor.power_dim
        self.ps_norm = nn.LayerNorm(descriptor_dim)

        if use_lattice:
            self.lattice_encoder = LatticeEncoder(embed_dim, dropout=dropout)
            input_dim = descriptor_dim + embed_dim
            self.lattice_skip_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            nn.init.orthogonal_(self.lattice_skip_proj.weight)
        else:
            self.lattice_encoder = None
            self.lattice_skip_proj = None
            input_dim = descriptor_dim

        ffn_dim = ffn_dim or embed_dim * 4
        layers = [
            nn.Linear(input_dim, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        ]

        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(embed_dim, ffn_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, embed_dim),
                ]
            )

        self.mlp = nn.Sequential(*layers)

        self.ps_skip_proj = nn.Linear(descriptor_dim, embed_dim, bias=False)
        nn.init.orthogonal_(self.ps_skip_proj.weight)
        with torch.no_grad():
            self.ps_skip_proj.weight.mul_(1.0)

        num_scales = self.descriptor.num_scales
        self.mp_radial_dim = num_radial
        self.mp_radial_proj = nn.Linear(
            num_radial * num_scales, self.mp_radial_dim, bias=False
        )

        self.mp_layers = nn.ModuleList(
            [
                MessagePassingLayer(
                    embed_dim=embed_dim,
                    num_radial=self.mp_radial_dim,
                    dropout=dropout,
                )
                for _ in range(num_mp_layers)
            ]
        )

    def forward(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_dist: torch.Tensor,
        batch_idx: torch.Tensor,
        B: int,
        cell: Optional[torch.Tensor] = None,
        num_atoms_per_graph: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = pos.device
        N = pos.size(0)

        edge_radial_multi, edge_weights = self.descriptor.compute_multi_scale_radial(
            edge_dist
        )
        edge_radial_mp = self.mp_radial_proj(edge_radial_multi)

        _, power_spectrum = self.descriptor.forward_with_edge_feats(
            edge_index, edge_vec, edge_dist, N, edge_radial_multi
        )

        power_spectrum_normed = self.ps_norm(power_spectrum)

        if self.use_lattice and self.lattice_encoder is not None and cell is not None:
            if num_atoms_per_graph is None:
                num_atoms_per_graph = torch.zeros(B, device=device, dtype=torch.long)
                num_atoms_per_graph.scatter_add_(
                    0, batch_idx, torch.ones(N, device=device, dtype=torch.long)
                )
            lattice_feats = self.lattice_encoder(cell, batch_idx, num_atoms_per_graph)
            if self.disable_lattice:
                lattice_feats = torch.zeros_like(lattice_feats)
            node_input = torch.cat([power_spectrum_normed, lattice_feats], dim=-1)
        else:
            lattice_feats = None
            node_input = power_spectrum_normed

        node_feats = self.mlp(node_input)
        ps_skip = self.ps_skip_proj(power_spectrum_normed)
        node_feats = node_feats + ps_skip

        if lattice_feats is not None and self.lattice_skip_proj is not None:
            node_feats = node_feats + self.lattice_skip_proj(lattice_feats)

        for mp in self.mp_layers:
            node_feats = mp(node_feats, edge_index, edge_radial_mp)

        graph_feats = torch.zeros(
            B, node_feats.size(-1), device=device, dtype=node_feats.dtype
        )
        graph_feats.scatter_add_(
            0, batch_idx.unsqueeze(-1).expand(-1, node_feats.size(-1)), node_feats
        )
        counts = torch.zeros(B, device=device, dtype=node_feats.dtype)
        counts.scatter_add_(
            0, batch_idx, torch.ones(N, device=device, dtype=node_feats.dtype)
        )
        graph_feats = graph_feats / counts.unsqueeze(-1).clamp(min=1.0)

        return node_feats, graph_feats


__all__ = [
    "CosineCutoff",
    "BesselBasis",
    "GaussianRBF",
    "SphericalHarmonics",
    "LearnablePowerSpectrumDescriptor",
    "LatticeEncoder",
    "MessagePassingLayer",
    "StructuralStreamPowerSpectrum",
]
