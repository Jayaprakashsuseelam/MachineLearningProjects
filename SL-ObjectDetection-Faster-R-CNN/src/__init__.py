"""
SL-ObjectDetection-Faster-R-CNN - A comprehensive implementation of Faster R-CNN
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Main imports for easy access
from .models import FasterRCNN, RPN, RoIPooling
from .data import VOCDataset, get_transform
from .training import Trainer
from .utils import visualize_predictions, calculate_map

__all__ = [
    "FasterRCNN",
    "RPN", 
    "RoIPooling",
    "VOCDataset",
    "get_transform",
    "Trainer",
    "visualize_predictions",
    "calculate_map",
]
