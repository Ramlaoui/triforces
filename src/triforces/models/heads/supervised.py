"""Supervised heads for property prediction."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .barlow_twins import maybe_import_e3nn
from triforces.models.mlp import create_mlp
from triforces.models.outputs import BackboneOutputs
from triforces.utils.stress import stress_to_voigt_6


def _mean_pool(
    node_feats: torch.Tensor, batch: torch.Tensor, num_graphs: int
) -> torch.Tensor:
    out = node_feats.new_zeros((num_graphs, node_feats.size(-1)))
    out.index_add_(0, batch, node_feats)
    count = torch.bincount(batch, minlength=num_graphs).clamp_min(1).to(out.dtype)
    return out / count.unsqueeze(1)


class DirectSupervisedHead(nn.Module):
    """Direct supervised property head on top of node/graph features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "silu",
        use_batch_norm: bool = False,
        dropout: float = 0.0,
        predict_energy: bool = True,
        predict_forces: bool = True,
        predict_stress: bool = False,
        stress_representation: str = "voigt",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims = list(hidden_dims or [self.input_dim, self.input_dim])
        self.activation = activation
        self.use_batch_norm = bool(use_batch_norm)
        self.dropout = float(dropout)
        self.predict_energy = bool(predict_energy)
        self.predict_forces = bool(predict_forces)
        self.predict_stress = bool(predict_stress)
        self.stress_representation = str(stress_representation)
        if self.stress_representation not in {"voigt", "matrix"}:
            raise ValueError(
                "stress_representation must be 'voigt' or 'matrix', got "
                f"{self.stress_representation!r}"
            )

        self.energy_head = (
            create_mlp(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                output_dim=1,
                use_batch_norm=self.use_batch_norm,
                dropout=self.dropout,
                activation=self.activation,
                final_activation=False,
            )
            if self.predict_energy
            else None
        )
        self.force_head = (
            create_mlp(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                output_dim=3,
                use_batch_norm=self.use_batch_norm,
                dropout=self.dropout,
                activation=self.activation,
                final_activation=False,
            )
            if self.predict_forces
            else None
        )
        stress_out = 6 if self.stress_representation == "voigt" else 9
        self.stress_head = (
            create_mlp(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                output_dim=stress_out,
                use_batch_norm=self.use_batch_norm,
                dropout=self.dropout,
                activation=self.activation,
                final_activation=False,
            )
            if self.predict_stress
            else None
        )

    @classmethod
    def build_from_backbone_info(
        cls, backbone_info: Dict[str, Any], **kwargs: Any
    ) -> "DirectSupervisedHead":
        kwargs = dict(kwargs)
        if kwargs.get("input_dim") is None:
            output_dim = backbone_info.get("output_dim")
            if output_dim is None:
                raise ValueError(
                    "DirectSupervisedHead requires `input_dim` or "
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
        node_features = backbone_outputs.node_feats
        graph_features = backbone_outputs.graph_feats
        if graph_features is None:
            batch_idx = getattr(batch, "batch", None)
            num_graphs = getattr(batch, "num_graphs", None)
            if batch_idx is None or num_graphs is None:
                raise ValueError(
                    "DirectSupervisedHead requires graph_feats or batch/batch_size info."
                )
            graph_features = _mean_pool(node_features, batch_idx, int(num_graphs))

        out: Dict[str, torch.Tensor] = {}
        if self.energy_head is not None:
            out["energy"] = self.energy_head(graph_features).squeeze(-1)
        if self.force_head is not None:
            out["forces"] = self.force_head(node_features)
        if self.stress_head is not None:
            stress = self.stress_head(graph_features)
            if self.stress_representation == "matrix":
                stress = stress.reshape(-1, 3, 3)
            out["stress"] = stress
        return out


class EnergyConservingHead(nn.Module):
    """Predict energy and derive conservative forces via autograd."""

    requires_grad_for_inference = True

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "silu",
        use_batch_norm: bool = False,
        dropout: float = 0.0,
        predict_forces: bool = True,
        predict_stress: bool = False,
        forces_output_key: str = "forces",
        include_energy: bool = True,
        stress_representation: str = "voigt",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims = list(hidden_dims or [self.input_dim, self.input_dim])
        self.activation = activation
        self.use_batch_norm = bool(use_batch_norm)
        self.dropout = float(dropout)
        self.predict_forces = bool(predict_forces)
        self.predict_stress = bool(predict_stress)
        self.forces_output_key = str(forces_output_key)
        self.include_energy = bool(include_energy)
        self.stress_representation = str(stress_representation)
        if self.stress_representation not in {"voigt", "matrix"}:
            raise ValueError(
                "stress_representation must be 'voigt' or 'matrix', got "
                f"{self.stress_representation!r}"
            )

        self.energy_head = create_mlp(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=1,
            use_batch_norm=self.use_batch_norm,
            dropout=self.dropout,
            activation=self.activation,
            final_activation=False,
        )

    @classmethod
    def build_from_backbone_info(
        cls, backbone_info: Dict[str, Any], **kwargs: Any
    ) -> "EnergyConservingHead":
        kwargs = dict(kwargs)
        if kwargs.get("input_dim") is None:
            output_dim = backbone_info.get("output_dim")
            if output_dim is None:
                raise ValueError(
                    "EnergyConservingHead requires `input_dim` or "
                    "backbone_info['output_dim']."
                )
            kwargs["input_dim"] = int(output_dim)
        return cls(**kwargs)

    def _resolve_positions(
        self,
        *,
        batch: Any,
        outputs: Dict[str, torch.Tensor] | None,
        backbone_outputs: BackboneOutputs,
    ) -> torch.Tensor:
        if outputs is not None and torch.is_tensor(outputs.get("pos")):
            return outputs["pos"]
        if torch.is_tensor(backbone_outputs.extras.get("pos")):
            return backbone_outputs.extras["pos"]
        if hasattr(batch, "pos") and torch.is_tensor(batch.pos):
            return batch.pos
        raise ValueError(
            "EnergyConservingHead requires `pos` in batch/backbone extras/outputs."
        )

    def _resolve_stress_from_displacement(
        self,
        *,
        total_energy: torch.Tensor,
        outputs: Dict[str, torch.Tensor] | None,
        backbone_outputs: BackboneOutputs,
        create_graph: bool,
    ) -> torch.Tensor | None:
        displacement = None
        if outputs is not None and torch.is_tensor(outputs.get("displacement")):
            displacement = outputs["displacement"]
        elif torch.is_tensor(backbone_outputs.extras.get("displacement")):
            displacement = backbone_outputs.extras["displacement"]
        if displacement is None:
            return None
        stress = torch.autograd.grad(
            total_energy,
            displacement,
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=True,
        )[0]
        return stress

    def _stress_from_virial(
        self,
        *,
        pos: torch.Tensor,
        forces: torch.Tensor,
        batch: Any,
    ) -> torch.Tensor | None:
        if not hasattr(batch, "cell") or not hasattr(batch, "batch"):
            return None
        cell = torch.as_tensor(batch.cell, device=pos.device, dtype=pos.dtype)
        if cell.ndim != 3 or cell.shape[-2:] != (3, 3):
            return None
        batch_idx = torch.as_tensor(batch.batch, device=pos.device, dtype=torch.long)
        num_graphs = int(getattr(batch, "num_graphs", int(batch_idx.max().item()) + 1))
        virial = pos.new_zeros((num_graphs, 3, 3))
        virial.index_add_(0, batch_idx, torch.einsum("ni,nj->nij", pos, forces))
        volume = torch.abs(torch.linalg.det(cell)).clamp_min(1e-8).unsqueeze(-1).unsqueeze(-1)
        return -virial / volume

    def forward(
        self,
        backbone_outputs: BackboneOutputs,
        batch: Any,
        outputs: Dict[str, torch.Tensor] | None = None,
        training: bool = False,
        transform: Any = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        graph_features = backbone_outputs.graph_feats
        if graph_features is None:
            raise ValueError("EnergyConservingHead requires graph-level features.")

        energy = self.energy_head(graph_features).squeeze(-1)
        out: Dict[str, torch.Tensor] = {}
        if self.include_energy:
            out["energy"] = energy
        if not (self.predict_forces or self.predict_stress):
            return out

        pos = self._resolve_positions(
            batch=batch,
            outputs=outputs,
            backbone_outputs=backbone_outputs,
        )
        total_energy = energy.sum()
        create_graph = bool(training)

        force_grad = torch.autograd.grad(
            total_energy,
            pos,
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if force_grad is None:
            raise ValueError(
                "EnergyConservingHead could not compute dE/dpos. "
                "Ensure backbone outputs depend on differentiable positions."
            )
        forces = -force_grad
        if self.predict_forces:
            out[self.forces_output_key] = forces

        if self.predict_stress:
            stress = self._resolve_stress_from_displacement(
                total_energy=total_energy,
                outputs=outputs,
                backbone_outputs=backbone_outputs,
                create_graph=create_graph,
            )
            if stress is None:
                stress = self._stress_from_virial(pos=pos, forces=forces, batch=batch)
            if stress is None:
                raise ValueError(
                    "EnergyConservingHead could not compute stress. "
                    "Provide differentiable displacement in backbone extras or "
                    "cell+batch in the input batch."
                )
            if self.stress_representation == "voigt":
                stress = stress_to_voigt_6(stress)
            out["stress"] = stress

        return out


class EquivariantVectorHead(nn.Module):
    """Equivariant vector head using scalar-gated vector features."""

    def __init__(
        self,
        scalar_dim: int | None = None,
        vector_channels: int | None = None,
        output_dim: int = 3,
        equivariant_key: str = "node_feats_equivariant",
        output_key: str = "noise_displacement",
        use_tensor_product: bool = False,
        hidden_dims: list[int] | None = None,
        activation: str = "silu",
        use_batch_norm: bool = False,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.scalar_dim = scalar_dim
        self.vector_channels = vector_channels
        self.output_dim = int(output_dim)
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be > 0, got {self.output_dim}")

        self.equivariant_key = equivariant_key
        self.output_key = str(output_key)
        self.use_tensor_product = bool(use_tensor_product)
        self.hidden_dims = list(hidden_dims or [])
        self.activation = activation
        self.use_batch_norm = bool(use_batch_norm)
        self.dropout = float(dropout)
        self.gate: nn.Module | None = None
        self.tensor_product: nn.Module | None = None
        self._tensor_product_layout: tuple[int, int] | None = None

        if not self.use_tensor_product and self.output_dim != 3:
            raise ValueError(
                "EquivariantVectorHead without tensor product only supports output_dim=3. "
                f"Got output_dim={self.output_dim}."
            )
        if self.use_tensor_product and self.output_dim % 3 != 0:
            raise ValueError(
                "EquivariantVectorHead with tensor product requires output_dim % 3 == 0. "
                f"Got output_dim={self.output_dim}."
            )

    @classmethod
    def build_from_backbone_info(
        cls, backbone_info: Dict[str, Any], **kwargs: Any
    ) -> "EquivariantVectorHead":
        return cls(**kwargs)

    def _resolve_equivariant_features(
        self,
        *,
        backbone_outputs: BackboneOutputs,
        outputs: Dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if outputs is not None and torch.is_tensor(outputs.get(self.equivariant_key)):
            return outputs[self.equivariant_key]
        if torch.is_tensor(backbone_outputs.extras.get(self.equivariant_key)):
            return backbone_outputs.extras[self.equivariant_key]
        for alt_key in ("node_feats_equivariant", "node_feats_l1"):
            if outputs is not None and torch.is_tensor(outputs.get(alt_key)):
                return outputs[alt_key]
            if torch.is_tensor(backbone_outputs.extras.get(alt_key)):
                return backbone_outputs.extras[alt_key]
        raise ValueError(
            "EquivariantVectorHead requires equivariant node features in outputs/"
            f"backbone extras under `{self.equivariant_key}`."
        )

    def _infer_layout(self, x: torch.Tensor) -> tuple[int, int]:
        d = int(x.size(-1))
        if self.scalar_dim is not None:
            scalar_dim = int(self.scalar_dim)
            vec_dim = d - scalar_dim
            if vec_dim < 0 or vec_dim % 3 != 0:
                raise ValueError(
                    "Invalid scalar_dim for equivariant features: "
                    f"total={d}, scalar_dim={scalar_dim}"
                )
            vector_channels = vec_dim // 3
            return scalar_dim, vector_channels
        if d % 4 == 0:
            c = d // 4
            return c, c
        if d % 3 == 0:
            return 0, d // 3
        raise ValueError(
            "Could not infer equivariant layout. Expected dim = S + 3*V, "
            f"got {d}."
        )

    def _get_gate(
        self,
        scalar_dim: int,
        vector_channels: int,
        reference: torch.Tensor | None = None,
    ) -> nn.Module | None:
        if scalar_dim == 0:
            return None
        if self.gate is None:
            self.gate = create_mlp(
                input_dim=scalar_dim,
                hidden_dims=self.hidden_dims,
                output_dim=vector_channels,
                use_batch_norm=self.use_batch_norm,
                dropout=self.dropout,
                activation=self.activation,
                final_activation=False,
            )
        if reference is not None:
            self.gate = self.gate.to(device=reference.device)
        return self.gate

    def _get_tensor_product(
        self,
        scalar_dim: int,
        vector_channels: int,
        reference: torch.Tensor,
    ) -> nn.Module:
        if scalar_dim <= 0:
            raise ValueError(
                "EquivariantVectorHead with tensor product requires scalar features."
            )

        layout = (scalar_dim, vector_channels)
        if self.tensor_product is None or self._tensor_product_layout != layout:
            o3 = maybe_import_e3nn()
            if o3 is None:
                raise ImportError(
                    "e3nn is required for EquivariantVectorHead(use_tensor_product=True). "
                    "Install with: pip install e3nn"
                )

            out_vectors = self.output_dim // 3
            self.tensor_product = o3.FullyConnectedTensorProduct(
                o3.Irreps(f"{scalar_dim}x0e"),
                o3.Irreps(f"{vector_channels}x1o"),
                o3.Irreps(f"{out_vectors}x1o"),
            )
            self._tensor_product_layout = layout

        self.tensor_product = self.tensor_product.to(
            device=reference.device, dtype=reference.dtype
        )
        return self.tensor_product

    def forward(
        self,
        backbone_outputs: BackboneOutputs,
        batch: Any,
        outputs: Dict[str, torch.Tensor] | None = None,
        training: bool = False,
        transform: Any = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        equivariant = self._resolve_equivariant_features(
            backbone_outputs=backbone_outputs, outputs=outputs
        )
        scalar_dim, vector_channels = self._infer_layout(equivariant)
        scalar = equivariant[:, :scalar_dim] if scalar_dim > 0 else None
        vectors = equivariant[:, scalar_dim:].reshape(-1, vector_channels, 3)

        if self.use_tensor_product:
            if scalar is None:
                raise ValueError(
                    "EquivariantVectorHead(use_tensor_product=True) requires scalar features."
                )
            tensor_product = self._get_tensor_product(
                scalar_dim=scalar_dim,
                vector_channels=vector_channels,
                reference=equivariant,
            )
            vector_pred = tensor_product(scalar, vectors.reshape(vectors.size(0), -1))
        else:
            gate = self._get_gate(scalar_dim, vector_channels, reference=scalar)
            if gate is None:
                weights = torch.ones(
                    vectors.size(0),
                    vector_channels,
                    device=vectors.device,
                    dtype=vectors.dtype,
                )
            else:
                assert scalar is not None
                weights = torch.sigmoid(gate(scalar))
            vector_pred = torch.sum(weights.unsqueeze(-1) * vectors, dim=1)

        return {self.output_key: vector_pred}
