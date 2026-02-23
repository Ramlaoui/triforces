"""Loss functions for contrastive and self-supervised training."""

from .barlow_twins import BarlowTwinsLoss
from .base import BaseLoss
from .contrastive import ContrastiveLoss
from .dino import KoLeoLoss, iBOTLoss
from .ibot import iBOTPatchLoss
from .lejepa import LeJEPALoss
from .reconstruction import ReconstructionLoss
from .supervised import SupervisedLoss

__all__ = [
    "BaseLoss",
    "ContrastiveLoss",
    "SupervisedLoss",
    "BarlowTwinsLoss",
    "LeJEPALoss",
    "ReconstructionLoss",
    "iBOTPatchLoss",
    "iBOTLoss",
    "KoLeoLoss",
]
