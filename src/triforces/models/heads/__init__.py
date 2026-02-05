"""
Projection heads for contrastive learning.
"""

from .simclr import ProjectionHead
from .barlow_twins import BarlowTwinsProjectionHead
from .split_barlow_twins import SplitBarlowTwinsProjectionHead
from .multi_stream_barlow_twins import MultiStreamBarlowTwinsProjectionHead
from .byol import BYOLProjectionHead, BYOLPredictorHead, BYOLCombinedHead
from .ibot import iBOTProjectionHead, iBOTCombinedHead

__all__ = [
    "ProjectionHead",
    "BarlowTwinsProjectionHead",
    "SplitBarlowTwinsProjectionHead",
    "MultiStreamBarlowTwinsProjectionHead",
    "BYOLProjectionHead",
    "BYOLPredictorHead",
    "BYOLCombinedHead",
    "iBOTProjectionHead",
    "iBOTCombinedHead",
]
