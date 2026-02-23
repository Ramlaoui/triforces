"""Supervised losses for atomistic property prediction."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from triforces.normalizers import EnergyReferenceNormalizer, StandardizationNormalizer
from triforces.utils.stress import stress_to_voigt_6

from .base import BaseLoss


def _get_pred_value(preds: Any, key: str):
    if isinstance(preds, dict):
        return preds.get(key)
    return getattr(preds, key, None)


class SupervisedLoss(BaseLoss):
    """Weighted supervised loss for energy, forces, and stress."""

    def __init__(
        self,
        energy_weight: float = 1.0,
        forces_weight: float = 0.0,
        stress_weight: float = 0.0,
        energy_key: str = "energy",
        forces_key: str = "forces",
        stress_key: str = "stress",
        energy_loss: str = "mse",
        forces_loss: str = "mse",
        stress_loss: str = "mse",
        energy_huber_delta: float = 1.0,
        forces_huber_delta: float = 1.0,
        stress_huber_delta: float = 1.0,
        energy_per_atom: bool = False,
        prediction_space: str = "normalized",
        energy_references: Any | None = None,
        standardization: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.energy_weight = float(energy_weight)
        self.forces_weight = float(forces_weight)
        self.stress_weight = float(stress_weight)
        self.energy_key = str(energy_key)
        self.forces_key = str(forces_key)
        self.stress_key = str(stress_key)
        self.energy_loss = str(energy_loss).strip().lower()
        self.forces_loss = str(forces_loss).strip().lower()
        self.stress_loss = str(stress_loss).strip().lower()
        self.energy_huber_delta = float(energy_huber_delta)
        self.forces_huber_delta = float(forces_huber_delta)
        self.stress_huber_delta = float(stress_huber_delta)
        self.energy_per_atom = bool(energy_per_atom)
        for name, mode in (
            ("energy_loss", self.energy_loss),
            ("forces_loss", self.forces_loss),
            ("stress_loss", self.stress_loss),
        ):
            if mode not in {"mse", "mae", "huber"}:
                raise ValueError(
                    f"Unsupported {name}={mode!r}. Use one of 'mse', 'mae', 'huber'."
                )
        for name, delta in (
            ("energy_huber_delta", self.energy_huber_delta),
            ("forces_huber_delta", self.forces_huber_delta),
            ("stress_huber_delta", self.stress_huber_delta),
        ):
            if delta <= 0:
                raise ValueError(f"{name} must be > 0, got {delta}.")
        self.prediction_space = str(prediction_space).strip().lower()
        if self.prediction_space not in {"normalized", "raw"}:
            raise ValueError(
                "prediction_space must be 'normalized' or 'raw', got "
                f"{prediction_space!r}."
            )
        self.energy_reference_normalizer = (
            None
            if energy_references is None
            else EnergyReferenceNormalizer.from_state(energy_references)
        )
        self.standardizers: dict[str, StandardizationNormalizer] = {}
        for key, state in (standardization or {}).items():
            self.standardizers[str(key)] = StandardizationNormalizer.from_state(state)

    def get_checkpoint_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "prediction_space": self.prediction_space,
            "energy_per_atom": self.energy_per_atom,
            "huber_delta": {
                "energy": self.energy_huber_delta,
                "forces": self.forces_huber_delta,
                "stress": self.stress_huber_delta,
            },
        }
        if self.energy_reference_normalizer is not None:
            state["energy_references"] = self.energy_reference_normalizer.state_dict()
        if self.standardizers:
            state["standardization"] = {
                key: normalizer.state_dict()
                for key, normalizer in self.standardizers.items()
            }
        return state

    def load_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        state = state or {}
        prediction_space = state.get("prediction_space")
        if prediction_space is not None:
            parsed = str(prediction_space).strip().lower()
            if parsed not in {"normalized", "raw"}:
                raise ValueError(
                    "Invalid checkpoint `prediction_space`. Expected 'normalized' or "
                    f"'raw', got {prediction_space!r}."
                )
            self.prediction_space = parsed
        energy_per_atom = state.get("energy_per_atom")
        if energy_per_atom is not None:
            self.energy_per_atom = bool(energy_per_atom)
        huber_state = state.get("huber_delta")
        if isinstance(huber_state, dict):
            if "energy" in huber_state:
                self.energy_huber_delta = float(huber_state["energy"])
            if "forces" in huber_state:
                self.forces_huber_delta = float(huber_state["forces"])
            if "stress" in huber_state:
                self.stress_huber_delta = float(huber_state["stress"])
        energy_ref_state = state.get("energy_references")
        if energy_ref_state is None:
            self.energy_reference_normalizer = None
        else:
            if isinstance(energy_ref_state, dict) and "references" in energy_ref_state:
                energy_ref_state = energy_ref_state["references"]
            self.energy_reference_normalizer = EnergyReferenceNormalizer.from_state(
                energy_ref_state
            )

        standardization_state = state.get("standardization") or {}
        self.standardizers = {}
        for key, norm_state in standardization_state.items():
            self.standardizers[str(key)] = StandardizationNormalizer.from_state(
                norm_state
            )

    @staticmethod
    def _pointwise_error(
        pred: torch.Tensor, target: torch.Tensor, mode: str, *, huber_delta: float = 1.0
    ) -> torch.Tensor:
        if mode == "mse":
            return (pred - target).pow(2)
        if mode == "mae":
            return (pred - target).abs()
        if mode == "huber":
            return torch.nn.functional.huber_loss(
                pred,
                target,
                delta=float(huber_delta),
                reduction="none",
            )
        raise ValueError(
            f"Unsupported loss mode {mode!r}. Use one of 'mse', 'mae', 'huber'."
        )

    @staticmethod
    def _to_tensor_like(value: Any, ref: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(device=ref.device, dtype=ref.dtype)
        return torch.as_tensor(value, device=ref.device, dtype=ref.dtype)

    @staticmethod
    def _zero_from_preds(preds: Any) -> torch.Tensor:
        if isinstance(preds, dict):
            values = preds.values()
        else:
            values = (getattr(preds, name) for name in dir(preds))
        for value in values:
            if torch.is_tensor(value):
                return value.new_tensor(0.0)
        return torch.tensor(0.0)

    @staticmethod
    def _get_data_attr(data: Any, names: tuple[str, ...]) -> Any | None:
        for name in names:
            if hasattr(data, name):
                value = getattr(data, name)
                if value is not None:
                    return value
        return None

    def _resolve_num_atoms(self, data: Any, *, ref: torch.Tensor) -> torch.Tensor:
        num_graphs = int(ref.reshape(-1).shape[0])
        for name in ("natoms", "num_atoms_per_graph", "n_atoms"):
            value = self._get_data_attr(data, (name,))
            if value is None:
                continue
            num_atoms = torch.as_tensor(
                value, device=ref.device, dtype=ref.dtype
            ).reshape(-1)
            if num_atoms.numel() == num_graphs:
                return num_atoms.clamp_min(1.0)
        batch = self._get_data_attr(data, ("batch",))
        if batch is not None:
            batch_t = torch.as_tensor(
                batch, device=ref.device, dtype=torch.long
            ).reshape(-1)
            num_atoms = torch.bincount(batch_t, minlength=num_graphs).to(
                dtype=ref.dtype
            )
            return num_atoms.clamp_min(1.0)
        raise ValueError(
            "energy_per_atom=True requires `natoms`, `num_atoms_per_graph`, "
            "`n_atoms`, or `batch` on the input batch."
        )

    def _apply_standardization(
        self,
        key: str,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalizer = self.standardizers.get(key)
        if normalizer is None:
            return pred, target
        return normalizer.normalize(pred), normalizer.normalize(target)

    def _standardize(self, key: str, value: torch.Tensor) -> torch.Tensor:
        normalizer = self.standardizers.get(key)
        if normalizer is None:
            return value
        return normalizer.normalize(value)

    def _destandardize(self, key: str, value: torch.Tensor) -> torch.Tensor:
        normalizer = self.standardizers.get(key)
        if normalizer is None:
            return value
        return normalizer.denormalize(value)

    def _normalize_energy_target(self, data: Any, value: torch.Tensor) -> torch.Tensor:
        out = value
        if self.energy_reference_normalizer is not None:
            atomic_numbers = self._get_data_attr(data, ("atomic_numbers", "z"))
            batch = self._get_data_attr(data, ("batch",))
            if atomic_numbers is None or batch is None:
                raise ValueError(
                    "Energy reference normalization requires `atomic_numbers` (or `z`) "
                    "and `batch` on the input batch."
                )
            out = self.energy_reference_normalizer.normalize(
                out, atomic_numbers=atomic_numbers, batch=batch
            )
        if self.energy_per_atom:
            out = out / self._resolve_num_atoms(data, ref=out)
        return self._standardize("energy", out)

    def _denormalize_energy_prediction(
        self, data: Any, value: torch.Tensor
    ) -> torch.Tensor:
        out = self._destandardize("energy", value)
        if self.energy_per_atom:
            out = out * self._resolve_num_atoms(data, ref=out)
        if self.energy_reference_normalizer is not None:
            atomic_numbers = self._get_data_attr(data, ("atomic_numbers", "z"))
            batch = self._get_data_attr(data, ("batch",))
            if atomic_numbers is None or batch is None:
                raise ValueError(
                    "Energy denormalization requires `atomic_numbers` (or `z`) and "
                    "`batch` on the input batch."
                )
            out = self.energy_reference_normalizer.denormalize(
                out, atomic_numbers=atomic_numbers, batch=batch
            )
        return out

    def normalize_target(
        self, key: str, value: torch.Tensor, *, data: Any
    ) -> torch.Tensor:
        if key == "energy":
            return self._normalize_energy_target(data, value)
        return self._standardize(key, value)

    def denormalize_prediction(
        self, key: str, value: torch.Tensor, *, data: Any
    ) -> torch.Tensor:
        if key == "energy":
            return self._denormalize_energy_prediction(data, value)
        return self._destandardize(key, value)

    def _energy_term(
        self, data: Any, preds: Any
    ) -> tuple[torch.Tensor | None, Dict[str, float]]:
        pred = _get_pred_value(preds, "energy")
        target = getattr(data, self.energy_key, None)
        if pred is None or target is None:
            return None, {"n_energy": 0.0}
        pred = torch.as_tensor(pred).reshape(-1)
        target = self._to_tensor_like(target, pred).reshape(-1)
        if pred.shape != target.shape:
            raise ValueError(
                "Energy prediction/target shape mismatch: "
                f"{tuple(pred.shape)} vs {tuple(target.shape)}"
            )
        if self.energy_per_atom and self.prediction_space == "raw":
            n_atoms = self._resolve_num_atoms(data, ref=pred)
            pred = pred / n_atoms
            target = target / n_atoms
        if self.prediction_space == "normalized":
            target = self.normalize_target("energy", target, data=data)
        elif self.prediction_space == "raw":
            pred, target = self._apply_standardization("energy", pred, target)
        mask = torch.isfinite(target)
        if not mask.any():
            return pred.new_tensor(0.0), {"n_energy": 0.0}
        err = self._pointwise_error(
            pred[mask],
            target[mask],
            self.energy_loss,
            huber_delta=self.energy_huber_delta,
        )
        loss = self._reduce(err)
        return loss, {
            "n_energy": float(mask.sum().item()),
            "loss/energy": float(loss.item()),
        }

    def _forces_term(
        self, data: Any, preds: Any
    ) -> tuple[torch.Tensor | None, Dict[str, float]]:
        pred = _get_pred_value(preds, "forces")
        target = getattr(data, self.forces_key, None)
        if pred is None or target is None:
            return None, {"n_forces": 0.0}
        pred = torch.as_tensor(pred)
        target = self._to_tensor_like(target, pred)
        if pred.shape != target.shape:
            raise ValueError(
                "Forces prediction/target shape mismatch: "
                f"{tuple(pred.shape)} vs {tuple(target.shape)}"
            )
        if pred.ndim != 2 or pred.size(-1) != 3:
            raise ValueError(f"Expected forces shape (N, 3), got {tuple(pred.shape)}")
        if self.prediction_space == "normalized":
            target = self.normalize_target("forces", target, data=data)
        elif self.prediction_space == "raw":
            pred, target = self._apply_standardization("forces", pred, target)
        mask = torch.isfinite(target).all(dim=-1)
        if not mask.any():
            return pred.new_tensor(0.0), {"n_forces": 0.0}
        err = self._pointwise_error(
            pred[mask],
            target[mask],
            self.forces_loss,
            huber_delta=self.forces_huber_delta,
        ).mean(dim=-1)
        loss = self._reduce(err)
        return loss, {
            "n_forces": float(mask.sum().item()),
            "loss/forces": float(loss.item()),
        }

    def _as_voigt6(self, stress: torch.Tensor) -> torch.Tensor:
        if stress.ndim >= 2 and stress.shape[-2:] == (3, 3):
            out = stress_to_voigt_6(stress)
            assert out is not None
            return out
        if stress.shape[-1] == 9:
            out = stress_to_voigt_6(stress.reshape(*stress.shape[:-1], 3, 3))
            assert out is not None
            return out
        if stress.shape[-1] == 6:
            return stress
        raise ValueError(
            "Stress tensor must be (..., 6), (..., 9), or (..., 3, 3), got "
            f"{tuple(stress.shape)}"
        )

    def _stress_term(
        self, data: Any, preds: Any
    ) -> tuple[torch.Tensor | None, Dict[str, float]]:
        pred = _get_pred_value(preds, "stress")
        target = getattr(data, self.stress_key, None)
        if pred is None or target is None:
            return None, {"n_stress": 0.0}
        pred = self._as_voigt6(torch.as_tensor(pred))
        target = self._as_voigt6(self._to_tensor_like(target, pred))
        pred = pred.reshape(-1, 6)
        target = target.reshape(-1, 6)
        if pred.shape != target.shape:
            raise ValueError(
                "Stress prediction/target shape mismatch: "
                f"{tuple(pred.shape)} vs {tuple(target.shape)}"
            )
        if self.prediction_space == "normalized":
            target = self.normalize_target("stress", target, data=data)
        elif self.prediction_space == "raw":
            pred, target = self._apply_standardization("stress", pred, target)
        mask = torch.isfinite(target).all(dim=-1)
        if not mask.any():
            return pred.new_tensor(0.0), {"n_stress": 0.0}
        err = self._pointwise_error(
            pred[mask],
            target[mask],
            self.stress_loss,
            huber_delta=self.stress_huber_delta,
        ).mean(dim=-1)
        loss = self._reduce(err)
        return loss, {
            "n_stress": float(mask.sum().item()),
            "loss/stress": float(loss.item()),
        }

    def forward(
        self, data: Any, preds: Any, step: int = 0
    ) -> Tuple[torch.Tensor, Dict]:
        _ = step
        metrics: Dict[str, float] = {}
        base = self._zero_from_preds(preds)
        total = base.clone()

        energy_term, energy_metrics = self._energy_term(data, preds)
        metrics.update(energy_metrics)
        if energy_term is not None and self.energy_weight > 0:
            total = total + self.energy_weight * energy_term

        forces_term, forces_metrics = self._forces_term(data, preds)
        metrics.update(forces_metrics)
        if forces_term is not None and self.forces_weight > 0:
            total = total + self.forces_weight * forces_term

        stress_term, stress_metrics = self._stress_term(data, preds)
        metrics.update(stress_metrics)
        if stress_term is not None and self.stress_weight > 0:
            total = total + self.stress_weight * stress_term

        metrics["loss/energy_weight"] = self.energy_weight
        metrics["loss/forces_weight"] = self.forces_weight
        metrics["loss/stress_weight"] = self.stress_weight
        metrics["total_loss"] = float(total.item())
        return total, metrics
