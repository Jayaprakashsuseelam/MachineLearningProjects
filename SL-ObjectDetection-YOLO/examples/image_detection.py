"""
Image Detection Example
Demonstrates how to use the YOLO detector for image object detection
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from inference.detector import YOLODetector
from config.yolo_config import COCO_CONFIG
import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='YOLO Image Detection Example')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default=None, help='Path to model weights')
    parser.add_argument('--output', type=str, default='output.jpg', help='Output image path')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file {args.image} does not exist")
        return
    
    print("Initializing YOLO detector...")
    
    # Initialize detector
    detector = YOLODetector(
        model_path=args.model,
        device=args.device,
        confidence_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold
    )
    
    print(f"Detecting objects in {args.image}...")
    
    # Perform detection
    detections, visualization = detector.detect_and_visualize(
        args.image, 
        save_path=args.output
    )
    
    # Print results
    print(f"\nDetection Results:")
    print(f"Found {len(detections)} objects")
    
    for i, detection in enumerate(detections):
        bbox = detection['bbox']
        class_name = detection['class_name']
        confidence = detection['confidence']
        
        print(f"  {i+1}. {class_name}: {confidence:.3f} at {bbox}")
    
    # Print statistics
    stats = detector.get_statistics()
    if stats:
        print(f"\nInference Statistics:")
        print(f"  Average FPS: {stats['fps']:.2f}")
        print(f"  Average inference time: {stats['mean_inference_time']*1000:.2f} ms")
    
    print(f"\nVisualization saved to {args.output}")

def demo_with_sample_image():
    """Demo function with a sample image"""
    print("Running YOLO detection demo...")
    
    # Create a sample image (you can replace this with a real image)
    sample_image = create_sample_image()
    
    # Initialize detector
    detector = YOLODetector(
        model_path=None,  # Will create model from scratch
        device='auto',
        confidence_threshold=0.25
    )
    
    # Perform detection
    detections = detector.detect_image(sample_image)
    
    print(f"Demo completed. Found {len(detections)} objects.")
    return detections

def create_sample_image():
    """Create a sample image for demo purposes"""
    # Create a simple image with some geometric shapes
    img = np.ones((640, 640, 3), dtype=np.uint8) * 255
    
    # Draw some rectangles (simulating objects)
    cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), -1)  # Green rectangle
    cv2.rectangle(img, (300, 300), (400, 400), (255, 0, 0), -1)  # Blue rectangle
    cv2.rectangle(img, (500, 150), (600, 250), (0, 0, 255), -1)  # Red rectangle
    
    return img

if __name__ == "__main__":
    # Check if arguments are provided
    if len(sys.argv) > 1:
        main()
    else:
        print("No arguments provided. Running demo...")
        demo_with_sample_image()
