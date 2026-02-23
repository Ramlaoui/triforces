from __future__ import annotations

import numpy as np
import torch

__all__ = ["stress_array_to_voigt_6", "stress_to_voigt_6", "voigt_6_to_stress"]


def stress_array_to_voigt_6(stress: np.ndarray) -> np.ndarray:
    """Convert a stress array to Voigt-6 ordering."""
    stress = np.asarray(stress, dtype=np.float32)
    if stress.shape == (6,):
        return stress
    if stress.shape != (3, 3):
        stress = stress.reshape(3, 3)
    return np.array(
        [
            stress[0, 0],
            stress[1, 1],
            stress[2, 2],
            stress[1, 2],
            stress[0, 2],
            stress[0, 1],
        ],
        dtype=np.float32,
    )


def stress_to_voigt_6(stress: torch.Tensor | None) -> torch.Tensor | None:
    """Convert stress tensors to Voigt-6 ordering."""
    if stress is None:
        return None

    if stress.shape[-1] == 6:
        return stress

    if stress.shape[-1] == 9:
        stress = stress.reshape(*stress.shape[:-1], 3, 3)

    if stress.shape[-2:] != (3, 3):
        raise ValueError(
            "Input stress tensor must have shape (..., 3, 3) or (..., 6). "
            f"Got shape {tuple(stress.shape)}"
        )

    batch_shape = stress.shape[:-2]
    voigt = torch.empty((*batch_shape, 6), dtype=stress.dtype, device=stress.device)

    voigt[..., 0] = stress[..., 0, 0]
    voigt[..., 1] = stress[..., 1, 1]
    voigt[..., 2] = stress[..., 2, 2]
    voigt[..., 3] = (stress[..., 1, 2] + stress[..., 2, 1]) * 0.5
    voigt[..., 4] = (stress[..., 2, 0] + stress[..., 0, 2]) * 0.5
    voigt[..., 5] = (stress[..., 0, 1] + stress[..., 1, 0]) * 0.5

    return voigt


def voigt_6_to_stress(voigt: torch.Tensor | None) -> torch.Tensor | None:
    """Convert Voigt-6 tensors to full 3x3 stress tensors."""
    if voigt is None:
        return None

    if voigt.shape[-1] == 9:
        return voigt.reshape(*voigt.shape[:-1], 3, 3)
    if voigt.shape[-2:] == (3, 3):
        return voigt

    if voigt.shape[-1] != 6:
        raise ValueError(
            "Input voigt tensor must have shape (..., 6). "
            f"Got shape {tuple(voigt.shape)}"
        )

    batch_shape = voigt.shape[:-1]
    stress = torch.empty((*batch_shape, 3, 3), dtype=voigt.dtype, device=voigt.device)

    stress[..., 0, 0] = voigt[..., 0]
    stress[..., 1, 1] = voigt[..., 1]
    stress[..., 2, 2] = voigt[..., 2]

    stress[..., 1, 2] = voigt[..., 3]
    stress[..., 2, 1] = voigt[..., 3]

    stress[..., 2, 0] = voigt[..., 4]
    stress[..., 0, 2] = voigt[..., 4]

    stress[..., 0, 1] = voigt[..., 5]
    stress[..., 1, 0] = voigt[..., 5]

    return stress
