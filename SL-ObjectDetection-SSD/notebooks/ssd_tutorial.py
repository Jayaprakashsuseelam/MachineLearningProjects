#!/usr/bin/env python3
"""
SSD Object Detection Tutorial Script

This script provides a comprehensive tutorial on Single Shot Detector (SSD) 
object detection, including theoretical background, implementation, and practical examples.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json

def main():
    """Main tutorial function"""
    print("SSD Object Detection Tutorial")
    print("=" * 50)
    
    # Check environment
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    print("\n1. Introduction to SSD")
    print("-" * 30)
    print("Single Shot Detector (SSD) is a state-of-the-art object detection")
    print("algorithm that achieves real-time performance while maintaining high accuracy.")
    print("Key features:")
    print("- Single Pass Detection")
    print("- Multi-scale Feature Maps") 
    print("- Default Boxes (Anchor Boxes)")
    print("- Real-time Performance (30+ FPS)")
    
    print("\n2. Architecture Overview")
    print("-" * 30)
    print("SSD uses a VGG16 backbone with additional feature extraction layers")
    print("to create multi-scale feature maps for detecting objects at different scales.")
    
    # Load configuration
    try:
        with open('configs/ssd300_config.json', 'r') as f:
            config = json.load(f)
        
        print("\n3. Model Configuration")
        print("-" * 30)
        print("SSD Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    except FileNotFoundError:
        print("Configuration file not found. Please run the setup first.")
        return
    
    print("\n4. Implementation Demo")
    print("-" * 30)
    print("The implementation includes:")
    print("- SSDNetwork: Main network architecture")
    print("- SSDLoss: Loss function implementation")
    print("- SSDDetector: High-level detection interface")
    print("- Training framework for custom datasets")
    
    print("\n5. Usage Examples")
    print("-" * 30)
    print("Basic usage:")
    print("  from models.ssd_detector import SSDDetector")
    print("  detector = SSDDetector()")
    print("  detections = detector.detect('image.jpg')")
    
    print("\nReal-time detection:")
    print("  detector.detect_video(source=0)  # Webcam")
    
    print("\n6. Performance Characteristics")
    print("-" * 30)
    print("Typical performance on modern hardware:")
    print("- CPU: 5-15 FPS")
    print("- GPU: 30-60 FPS")
    print("- Model size: ~100MB")
    print("- Memory usage: 2-4GB")
    
    print("\n7. Applications")
    print("-" * 30)
    print("Common applications:")
    print("- Real-time object detection")
    print("- Traffic monitoring")
    print("- Security surveillance")
    print("- Autonomous vehicles")
    print("- Robotics")
    
    print("\n8. Next Steps")
    print("-" * 30)
    print("To get started:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Download models: python download_models.py")
    print("3. Run examples: python examples/basic_detection.py")
    print("4. Try real-time: python examples/real_time_detection.py")
    print("5. Explore traffic monitoring: python examples/traffic_monitoring.py")
    
    print("\n" + "=" * 50)
    print("Tutorial completed! Check the examples directory for practical usage.")
    print("=" * 50)

if __name__ == "__main__":
    main()
