#!/usr/bin/env python3
"""
Basic tests for YOLO implementation
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.yolo_v5 import get_yolov5_model
from config.yolo_config import COCO_CONFIG, YOLO_V5_SMALL
from training.loss import YOLOLoss

def test_model_creation():
    """Test YOLO model creation"""
    print("Testing model creation...")
    
    config = COCO_CONFIG.copy()
    config.update(YOLO_V5_SMALL)
    
    # Test different model sizes
    model_sizes = ['yolov5n', 'yolov5s', 'yolov5m']
    
    for size in model_sizes:
        model = get_yolov5_model(size, config)
        assert model is not None, f"Failed to create {size} model"
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ {size} model created and forward pass successful")
    
    print("✓ Model creation tests passed")

def test_loss_function():
    """Test YOLO loss function"""
    print("Testing loss function...")
    
    config = COCO_CONFIG.copy()
    config.update({
        'lambda_coord': 5.0,
        'lambda_noobj': 0.5,
        'lambda_obj': 1.0,
        'lambda_class': 1.0,
        'num_classes': 80
    })
    
    loss_fn = YOLOLoss(config)
    
    # Create dummy predictions and targets
    predictions = [
        torch.randn(1, 3 * (80 + 5), 80, 80),  # P3
        torch.randn(1, 3 * (80 + 5), 40, 40),  # P4
        torch.randn(1, 3 * (80 + 5), 20, 20)   # P5
    ]
    
    targets = [
        torch.randn(1, 3 * (80 + 5), 80, 80),
        torch.randn(1, 3 * (80 + 5), 40, 40),
        torch.randn(1, 3 * (80 + 5), 20, 20)
    ]
    
    # Test loss calculation
    loss_dict = loss_fn(predictions, targets)
    
    assert 'total_loss' in loss_dict, "Loss dictionary missing total_loss"
    assert loss_dict['total_loss'].item() >= 0, "Loss should be non-negative"
    
    print("✓ Loss function tests passed")

def test_config():
    """Test configuration loading"""
    print("Testing configuration...")
    
    # Test COCO config
    assert 'num_classes' in COCO_CONFIG, "COCO config missing num_classes"
    assert 'class_names' in COCO_CONFIG, "COCO config missing class_names"
    assert len(COCO_CONFIG['class_names']) == COCO_CONFIG['num_classes'], "Class names count mismatch"
    
    # Test YOLO config
    assert 'input_size' in YOLO_V5_SMALL, "YOLO config missing input_size"
    assert 'depth_multiple' in YOLO_V5_SMALL, "YOLO config missing depth_multiple"
    assert 'width_multiple' in YOLO_V5_SMALL, "YOLO config missing width_multiple"
    
    print("✓ Configuration tests passed")

def main():
    """Run all tests"""
    print("Running YOLO implementation tests...")
    print("=" * 50)
    
    try:
        test_config()
        test_model_creation()
        test_loss_function()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        print("YOLO implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
