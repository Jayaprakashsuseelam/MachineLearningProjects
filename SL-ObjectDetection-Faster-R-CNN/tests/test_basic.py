"""
Basic tests for the Faster R-CNN implementation
"""
import sys
import os
import torch
import unittest
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))


class TestBasicImports(unittest.TestCase):
    """Test basic imports and model creation"""
    
    def test_imports(self):
        """Test that all modules can be imported"""
        try:
            from models import FasterRCNN, faster_rcnn_resnet50
            from data import VOCDataset, get_transform
            from training import Trainer
            from utils import calculate_map
            from config import get_config
            print("✓ All modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import modules: {e}")
    
    def test_model_creation(self):
        """Test that models can be created"""
        try:
            model = faster_rcnn_resnet50(num_classes=21, pretrained=False)
            self.assertIsInstance(model, FasterRCNN)
            print("✓ Model created successfully")
        except Exception as e:
            self.fail(f"Failed to create model: {e}")
    
    def test_config_loading(self):
        """Test configuration loading"""
        try:
            config = get_config()
            self.assertIsNotNone(config)
            print("✓ Configuration loaded successfully")
        except Exception as e:
            self.fail(f"Failed to load configuration: {e}")
    
    def test_device_availability(self):
        """Test device availability"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.assertIsInstance(device, torch.device)
        print(f"✓ Device available: {device}")


if __name__ == '__main__':
    unittest.main()
