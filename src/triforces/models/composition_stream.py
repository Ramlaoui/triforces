from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CountWeightedTransformerBlock(nn.Module):
    """Transformer block with count-weighted attention for composition tokens.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.
    num_heads : int, default=8
        Number of attention heads.
    ffn_dim : int, optional
        Feed-forward hidden dimension. Defaults to ``4 * embed_dim``.
    dropout : float, default=0.1
        Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        ffn_dim = ffn_dim or embed_dim * 4

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        """Compute count-weighted self-attention for composition tokens.

        Parameters
        ----------
        x : torch.Tensor
            (B, N, D) token features (CLS + unique elements).
        mask : torch.Tensor
            (B, N) True for valid tokens.
        counts : torch.Tensor
            (B, N) multiplicity of each token (1 for CLS, element counts for others).

        Returns
        -------
        torch.Tensor
            Updated token features with shape (B, N, D).
        """
        B, N, _ = x.shape
        H, D = self.num_heads, self.head_dim

        x_normed = self.norm1(x)
        qkv = self.qkv_proj(x_normed).view(B, N, 3, H, D)
        q, k, v = qkv.unbind(dim=2)

        q = q.transpose(1, 2)  # (B, H, N, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        log_counts = torch.log(counts.clamp(min=1)).unsqueeze(1).unsqueeze(1)
        padding_mask = torch.zeros(B, 1, 1, N, device=x.device, dtype=x.dtype)
        padding_mask = padding_mask.masked_fill(
            ~mask.unsqueeze(1).unsqueeze(2), float("-inf")
        )
        attn_mask = log_counts + padding_mask

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.scale,
        )

        out = out.transpose(1, 2).reshape(B, N, self.embed_dim)
        out = self.out_proj(out)

        x = x + self.dropout(out)
        x = x + self.ffn(self.norm2(x))
        return x


class CompositionStream(nn.Module):
    """Composition stream modeling element patterns without geometry.

    Parameters
    ----------
    num_elements : int
        Number of elements for the embedding table.
    embed_dim : int
        Embedding dimension.
    num_heads : int, default=8
        Number of attention heads.
    num_layers : int, default=6
        Number of transformer blocks.
    ffn_dim : int, optional
        Feed-forward hidden dimension. Defaults to ``4 * embed_dim``.
    dropout : float, default=0.1
        Dropout probability.
    use_final_norm : bool, default=False
        Whether to apply a final layer normalization.
    max_unique_elements : int, default=100
        Maximum number of unique elements per graph.
    use_cls : bool, default=True
        Whether to use a CLS token.
    normalize_stoichiometry : bool, default=False
        Whether to normalize stoichiometry by GCD (legacy flag).
    stoichiometry_mode : str, optional
        Stoichiometry handling mode: ``"none"``, ``"gcd"``, or
        ``"fraction_embedding"``.
    """

    def __init__(
        self,
        num_elements: int,
        embed_dim: int,
        num_heads: int = 8,
        num_layers: int = 6,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        use_final_norm: bool = False,
        max_unique_elements: int = 100,
        use_cls: bool = True,
        normalize_stoichiometry: bool = False,
        stoichiometry_mode: str | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_final_norm = use_final_norm
        self.max_unique_elements = max_unique_elements
        self.use_cls = use_cls

        if stoichiometry_mode is not None:
            self.stoichiometry_mode = stoichiometry_mode
        elif normalize_stoichiometry:
            self.stoichiometry_mode = "gcd"
        else:
            self.stoichiometry_mode = "none"

        valid_modes = ("none", "gcd", "fraction_embedding")
        if self.stoichiometry_mode not in valid_modes:
            raise ValueError(
                f"stoichiometry_mode must be one of {valid_modes}, "
                f"got {self.stoichiometry_mode}"
            )

        self.num_heads = min(num_heads, embed_dim)
        while embed_dim % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1

        self.atom_embedding = nn.Embedding(num_elements, embed_dim)

        if use_cls:
            self.cls_token = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
        else:
            self.cls_token = None

        if self.stoichiometry_mode == "fraction_embedding":
            self.frac_embedding = nn.Sequential(
                nn.Linear(1, embed_dim // 2),
                nn.SiLU(),
                nn.Linear(embed_dim // 2, embed_dim),
            )
        else:
            self.frac_embedding = None

        self.layers = nn.ModuleList(
            [
                CountWeightedTransformerBlock(
                    embed_dim, self.num_heads, ffn_dim, dropout
                )
                for _ in range(num_layers)
            ]
        )

        if use_final_norm:
            self.norm = nn.LayerNorm(embed_dim)
        else:
            self.norm = None

    def _get_composition_tokens(
        self,
        atomic_numbers: torch.Tensor,
        ptr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = ptr.size(0) - 1
        device = atomic_numbers.device

        unique_elements = torch.zeros(
            B, self.max_unique_elements, dtype=torch.long, device=device
        )
        counts = torch.zeros(
            B, self.max_unique_elements, dtype=torch.long, device=device
        )
        type_mask = torch.zeros(
            B, self.max_unique_elements, dtype=torch.bool, device=device
        )
        atom_to_type_idx = torch.zeros_like(atomic_numbers, dtype=torch.long)

        for b in range(B):
            start, end = ptr[b], ptr[b + 1]
            atoms_in_graph = atomic_numbers[start:end]

            uniq, inverse, cnts = torch.unique(
                atoms_in_graph, sorted=True, return_inverse=True, return_counts=True
            )
            num_unique = len(uniq)

            unique_elements[b, :num_unique] = uniq
            counts[b, :num_unique] = cnts
            type_mask[b, :num_unique] = True
            atom_to_type_idx[start:end] = inverse

        return unique_elements, counts, atom_to_type_idx, type_mask

    def _batch_gcd(self, counts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B = counts.size(0)
        device = counts.device
        gcd_result = torch.ones(B, dtype=counts.dtype, device=device)

        for b in range(B):
            valid_counts = counts[b][mask[b]]
            if len(valid_counts) == 0:
                continue
            g = valid_counts[0].item()
            for c in valid_counts[1:]:
                c_val = c.item()
                while c_val:
                    g, c_val = c_val, g % c_val
            gcd_result[b] = max(g, 1)

        return gcd_result

    def forward(
        self,
        batch: object,
        training: bool = False,
        transform: Optional[object] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if transform is not None:
            batch = transform(batch)

        atomic_numbers = getattr(batch, "z", None)
        batch_idx = getattr(batch, "batch", None)
        if atomic_numbers is None or batch_idx is None:
            raise ValueError(
                "Batch must provide z/atomic_numbers and batch attributes."
            )

        ptr = getattr(batch, "ptr", None)
        if ptr is None:
            num_graphs = getattr(batch, "num_graphs", None)
            if num_graphs is None:
                num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() else 0
            counts = torch.bincount(batch_idx, minlength=int(num_graphs))
            ptr = torch.zeros(
                int(num_graphs) + 1, device=batch_idx.device, dtype=torch.long
            )
            ptr[1:] = torch.cumsum(counts, dim=0)

        device = atomic_numbers.device
        B = ptr.size(0) - 1

        unique_elements, counts, atom_to_type_idx, type_mask = (
            self._get_composition_tokens(atomic_numbers, ptr)
        )

        elem_feats = self.atom_embedding(unique_elements)
        counts_float = counts.float()

        if self.stoichiometry_mode == "gcd":
            gcd = self._batch_gcd(counts, type_mask)
            counts_float = counts_float / gcd.unsqueeze(1).clamp(min=1.0)
        elif self.stoichiometry_mode == "fraction_embedding":
            counts_sum = counts_float.sum(dim=1, keepdim=True).clamp(min=1.0)
            fractions = counts_float / counts_sum
            frac_feats = self.frac_embedding(fractions.unsqueeze(-1))
            elem_feats = elem_feats + frac_feats
            counts_float = torch.ones_like(counts_float)
            counts_float = counts_float.masked_fill(~type_mask, 0.0)

        if self.use_cls:
            max_types = unique_elements.size(1)
            x = torch.zeros(
                B, max_types + 1, self.embed_dim, device=device, dtype=elem_feats.dtype
            )
            x[:, 0] = self.cls_token.expand(B, -1)
            x[:, 1:] = elem_feats

            full_mask = torch.zeros(B, max_types + 1, device=device, dtype=torch.bool)
            full_mask[:, 0] = True
            full_mask[:, 1:] = type_mask

            full_counts = torch.ones(B, max_types + 1, device=device, dtype=torch.float)
            full_counts[:, 1:] = counts_float

            for layer in self.layers:
                x = layer(x, full_mask, full_counts)

            graph_feats = x[:, 0]
            type_feats = x[:, 1:]
            node_feats = type_feats[batch_idx, atom_to_type_idx]
        else:
            x = elem_feats
            for layer in self.layers:
                x = layer(x, type_mask, counts_float)
            node_feats = x[batch_idx, atom_to_type_idx]

            counts_for_avg = counts.float()
            counts_for_avg = counts_for_avg.masked_fill(~type_mask, 0.0)
            total_counts = counts_for_avg.sum(dim=1, keepdim=True).clamp(min=1.0)
            weights = counts_for_avg / total_counts
            graph_feats = (x * weights.unsqueeze(-1)).sum(dim=1)

        if self.norm is not None:
            node_feats = self.norm(node_feats)
            graph_feats = self.norm(graph_feats)

        return node_feats, graph_feats
