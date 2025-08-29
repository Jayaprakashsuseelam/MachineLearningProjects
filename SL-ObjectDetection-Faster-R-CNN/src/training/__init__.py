"""
Training components for Faster R-CNN
"""

from .trainer import Trainer
from .optimizer import get_optimizer, get_scheduler

__all__ = [
    "Trainer",
    "get_optimizer",
    "get_scheduler",
]
