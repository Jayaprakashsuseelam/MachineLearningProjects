#!/usr/bin/env python3
"""
Basic Object Detection Example using SSD

This example demonstrates how to use the SSD detector for basic object detection
on images and videos.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.ssd_detector import SSDDetector
from utils.visualization import draw_detections, plot_detection_statistics


def basic_image_detection():
    """Demonstrate basic image detection"""
    print("=" * 50)
    print("BASIC IMAGE DETECTION")
    print("=" * 50)
    
    # Initialize detector
    detector = SSDDetector(
        config_path='configs/ssd300_config.json',
        device='auto'
    )
    
    # Print model information
    model_info = detector.get_model_info()
    print(f"Model Information:")
    print(f"  Input Size: {model_info['input_size']}")
    print(f"  Number of Classes: {model_info['num_classes']}")
    print(f"  Total Parameters: {model_info['total_parameters']:,}")
    print(f"  Device: {model_info['device']}")
    
    # Example image path (you would replace this with your own image)
    image_path = "sample_images/cat.jpg"
    
    # Check if image exists, if not create a dummy image
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Creating a dummy image for demonstration...")
        
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        os.makedirs("sample_images", exist_ok=True)
        cv2.imwrite(image_path, dummy_image)
    
    # Perform detection
    print(f"\nDetecting objects in: {image_path}")
    detections = detector.detect(image_path, confidence_threshold=0.5)
    
    # Print detection results
    print(f"Found {len(detections)} objects:")
    for i, detection in enumerate(detections):
        bbox = detection['bbox']
        label = detection['label']
        score = detection['score']
        print(f"  {i+1}. Class {label}: {score:.3f} at {bbox}")
    
    # Visualize detections
    print("\nVisualizing detections...")
    annotated_image = detector.visualize_detections(
        image_path, detections, save_path="outputs/detection_result.jpg"
    )
    
    # Create detection statistics plot
    if detections:
        fig = plot_detection_statistics(detections)
        fig.savefig("outputs/detection_statistics.png")
        plt.close(fig)
        print("Detection statistics saved to: outputs/detection_statistics.png")
    
    return detector, detections


def batch_detection_example():
    """Demonstrate batch detection on multiple images"""
    print("\n" + "=" * 50)
    print("BATCH DETECTION EXAMPLE")
    print("=" * 50)
    
    # Initialize detector
    detector = SSDDetector(config_path='configs/ssd300_config.json')
    
    # Create sample images directory
    sample_dir = "sample_images"
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create dummy images for demonstration
    image_paths = []
    for i in range(3):
        # Create a dummy image with different content
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        image_path = os.path.join(sample_dir, f"sample_{i+1}.jpg")
        cv2.imwrite(image_path, dummy_image)
        image_paths.append(image_path)
    
    print(f"Created {len(image_paths)} sample images")
    
    # Perform batch detection
    print("Performing batch detection...")
    batch_detections = detector.detect_batch(image_paths, confidence_threshold=0.5)
    
    # Print results
    total_detections = sum(len(detections) for detections in batch_detections)
    print(f"Total detections across all images: {total_detections}")
    
    for i, (image_path, detections) in enumerate(zip(image_paths, batch_detections)):
        print(f"  Image {i+1}: {len(detections)} detections")
        
        # Save annotated image
        output_path = f"outputs/batch_detection_{i+1}.jpg"
        detector.visualize_detections(image_path, detections, save_path=output_path)
    
    return batch_detections


def performance_benchmark():
    """Benchmark detector performance"""
    print("\n" + "=" * 50)
    print("PERFORMANCE BENCHMARK")
    print("=" * 50)
    
    # Initialize detector
    detector = SSDDetector(config_path='configs/ssd300_config.json')
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_image_path = "sample_images/test_image.jpg"
    cv2.imwrite(test_image_path, test_image)
    
    # Warm up
    print("Warming up detector...")
    for _ in range(10):
        detector.detect(test_image_path)
    
    # Benchmark
    print("Running performance benchmark...")
    num_runs = 100
    inference_times = []
    
    for i in range(num_runs):
        detections = detector.detect(test_image_path)
        # Note: actual inference time is tracked internally by the detector
    
    # Get performance metrics
    avg_fps = detector.get_average_fps()
    model_info = detector.get_model_info()
    
    print(f"Performance Results:")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Average Inference Time: {1000/avg_fps:.2f} ms")
    print(f"  Model Parameters: {model_info['total_parameters']:,}")
    print(f"  Device: {model_info['device']}")


def confidence_threshold_comparison():
    """Compare detection results with different confidence thresholds"""
    print("\n" + "=" * 50)
    print("CONFIDENCE THRESHOLD COMPARISON")
    print("=" * 50)
    
    # Initialize detector
    detector = SSDDetector(config_path='configs/ssd300_config.json')
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_image_path = "sample_images/confidence_test.jpg"
    cv2.imwrite(test_image_path, test_image)
    
    # Test different confidence thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    print("Comparing detection results with different confidence thresholds:")
    print(f"{'Threshold':<12} {'Detections':<12} {'Avg Score':<12}")
    print("-" * 36)
    
    for threshold in thresholds:
        detections = detector.detect(test_image_path, confidence_threshold=threshold)
        num_detections = len(detections)
        avg_score = np.mean([d['score'] for d in detections]) if detections else 0
        
        print(f"{threshold:<12} {num_detections:<12} {avg_score:<12.3f}")
        
        # Save annotated image
        output_path = f"outputs/confidence_{threshold}.jpg"
        detector.visualize_detections(test_image_path, detections, save_path=output_path)


def main():
    """Main function to run all examples"""
    print("SSD Object Detection Examples")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    try:
        # Run basic image detection
        detector, detections = basic_image_detection()
        
        # Run batch detection
        batch_detections = batch_detection_example()
        
        # Run performance benchmark
        performance_benchmark()
        
        # Run confidence threshold comparison
        confidence_threshold_comparison()
        
        print("\n" + "=" * 50)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("Check the 'outputs' directory for generated images and results.")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
