"""Projection heads for contrastive learning."""

from .barlow_twins import BarlowTwinsProjectionHead
from .byol import BYOLCombinedHead, BYOLPredictorHead, BYOLProjectionHead
from .classification import ClassificationHead
from .direct import DirectVectorHead
from .ibot import iBOTCombinedHead, iBOTProjectionHead
from .simclr import ProjectionHead
from .split_barlow_twins import SplitBarlowTwinsProjectionHead
from .supervised import (
    DirectSupervisedHead,
    EnergyConservingHead,
    EquivariantVectorHead,
)

__all__ = [
    "ClassificationHead",
    "ProjectionHead",
    "BarlowTwinsProjectionHead",
    "SplitBarlowTwinsProjectionHead",
    "BYOLProjectionHead",
    "BYOLPredictorHead",
    "BYOLCombinedHead",
    "DirectVectorHead",
    "iBOTProjectionHead",
    "iBOTCombinedHead",
    "DirectSupervisedHead",
    "EnergyConservingHead",
    "EquivariantVectorHead",
]
