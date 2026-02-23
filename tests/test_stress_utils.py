import numpy as np
import torch

from triforces.utils.stress import (
    stress_array_to_voigt_6,
    stress_to_voigt_6,
    voigt_6_to_stress,
)


def test_stress_array_to_voigt_6():
    stress = np.array(
        [[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]], dtype=np.float32
    )
    voigt = stress_array_to_voigt_6(stress)
    assert voigt.shape == (6,)
    assert np.allclose(voigt, np.array([1.0, 2.0, 3.0, 0.3, 0.2, 0.1]))


def test_torch_stress_voigt_roundtrip():
    stress = torch.tensor(
        [[[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]]], dtype=torch.float32
    )
    voigt = stress_to_voigt_6(stress)
    back = voigt_6_to_stress(voigt)
    assert voigt is not None
    assert back is not None
    assert voigt.shape == (1, 6)
    assert back.shape == (1, 3, 3)
    assert torch.allclose(back, stress)
