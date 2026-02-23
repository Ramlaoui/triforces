from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

__all__ = ["EnergyReferenceNormalizer", "StandardizationNormalizer"]


def _to_float_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().clone().to(dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


@dataclass
class EnergyReferenceNormalizer:
    """Subtract/add per-element reference energies at graph level."""

    references: torch.Tensor

    def __post_init__(self) -> None:
        self.references = _to_float_tensor(self.references).reshape(-1)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"references": self.references.detach().clone().cpu()}

    @classmethod
    def from_state(cls, state: Any) -> "EnergyReferenceNormalizer":
        if isinstance(state, Mapping):
            if not state:
                refs = torch.zeros(1, dtype=torch.float32)
            else:
                max_z = max(int(k) for k in state.keys())
                refs = torch.zeros(max_z + 1, dtype=torch.float32)
                for key, value in state.items():
                    refs[int(key)] = float(value)
            return cls(references=refs)
        return cls(references=_to_float_tensor(state))

    def _graph_offsets(
        self,
        *,
        energy: torch.Tensor,
        atomic_numbers: Any,
        batch: Any,
    ) -> torch.Tensor:
        z = torch.as_tensor(
            atomic_numbers,
            dtype=torch.long,
            device=energy.device,
        ).reshape(-1)
        batch_idx = torch.as_tensor(batch, dtype=torch.long, device=energy.device).reshape(
            -1
        )
        if z.shape != batch_idx.shape:
            raise ValueError(
                "`atomic_numbers` and `batch` must have the same shape, got "
                f"{tuple(z.shape)} and {tuple(batch_idx.shape)}"
            )

        refs = self.references.to(device=energy.device, dtype=energy.dtype)
        if z.numel() and int(z.max().item()) >= refs.numel():
            raise ValueError(
                "Found atomic number outside energy reference table. "
                f"max_z={int(z.max().item())}, num_refs={refs.numel()}"
            )

        per_atom_offsets = refs[z]
        num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() else 0
        graph_offsets = energy.new_zeros((num_graphs,))
        graph_offsets.index_add_(0, batch_idx, per_atom_offsets)
        return graph_offsets

    def normalize(
        self,
        energy: Any,
        *,
        atomic_numbers: Any,
        batch: Any,
    ) -> torch.Tensor:
        energy_t = torch.as_tensor(energy)
        flat_energy = energy_t.reshape(-1)
        graph_offsets = self._graph_offsets(
            energy=flat_energy,
            atomic_numbers=atomic_numbers,
            batch=batch,
        )
        if flat_energy.shape != graph_offsets.shape:
            raise ValueError(
                "Energy tensor must be graph-level with one value per graph, got "
                f"{tuple(flat_energy.shape)} and expected {tuple(graph_offsets.shape)}"
            )
        return (flat_energy - graph_offsets).reshape_as(energy_t)

    def denormalize(
        self,
        energy: Any,
        *,
        atomic_numbers: Any,
        batch: Any,
    ) -> torch.Tensor:
        energy_t = torch.as_tensor(energy)
        flat_energy = energy_t.reshape(-1)
        graph_offsets = self._graph_offsets(
            energy=flat_energy,
            atomic_numbers=atomic_numbers,
            batch=batch,
        )
        if flat_energy.shape != graph_offsets.shape:
            raise ValueError(
                "Energy tensor must be graph-level with one value per graph, got "
                f"{tuple(flat_energy.shape)} and expected {tuple(graph_offsets.shape)}"
            )
        return (flat_energy + graph_offsets).reshape_as(energy_t)


@dataclass
class StandardizationNormalizer:
    """Apply fixed affine standardization: (x - mean) / std."""

    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1e-12

    def __post_init__(self) -> None:
        self.mean = _to_float_tensor(self.mean)
        self.std = _to_float_tensor(self.std)
        self.eps = float(self.eps)

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.detach().clone().cpu(),
            "std": self.std.detach().clone().cpu(),
            "eps": float(self.eps),
        }

    @classmethod
    def from_state(cls, state: Any) -> "StandardizationNormalizer":
        if isinstance(state, StandardizationNormalizer):
            return cls(mean=state.mean, std=state.std, eps=state.eps)
        if not isinstance(state, Mapping):
            raise TypeError(
                "Standardization normalizer state must be a mapping with 'mean' and "
                "'std' (or 'rmsd')."
            )
        if "mean" not in state:
            raise KeyError("Missing 'mean' in standardization state.")
        if "std" not in state and "rmsd" not in state:
            raise KeyError("Missing 'std' (or 'rmsd') in standardization state.")
        std_value = state["std"] if "std" in state else state["rmsd"]
        return cls(
            mean=state["mean"],
            std=std_value,
            eps=float(state.get("eps", 1e-12)),
        )

    def _scale(self, ref: torch.Tensor) -> torch.Tensor:
        std = self.std.to(device=ref.device, dtype=ref.dtype)
        return std.clamp_min(self.eps)

    def normalize(self, value: Any) -> torch.Tensor:
        value_t = torch.as_tensor(value)
        mean = self.mean.to(device=value_t.device, dtype=value_t.dtype)
        return (value_t - mean) / self._scale(value_t)

    def denormalize(self, value: Any) -> torch.Tensor:
        value_t = torch.as_tensor(value)
        mean = self.mean.to(device=value_t.device, dtype=value_t.dtype)
        return value_t * self._scale(value_t) + mean
