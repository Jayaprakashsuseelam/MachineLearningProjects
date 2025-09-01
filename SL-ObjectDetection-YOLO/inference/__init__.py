"""
Inference module for YOLO object detection
"""

from .detector import YOLODetector, load_detector

__all__ = ['YOLODetector', 'load_detector']
