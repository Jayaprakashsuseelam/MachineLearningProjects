#!/usr/bin/env python3
"""
Demo script for Faster R-CNN inference
"""
import os
import sys
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import faster_rcnn_resnet50
from data import VOCDataset
from utils import visualize_predictions


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint"""
    # Create model
    model = faster_rcnn_resnet50(num_classes=21, pretrained=False)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    model.to(device)
    
    return model


def preprocess_image(image_path: str, target_size: tuple = (800, 800)):
    """Preprocess image for inference"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Resize image
    image = image.resize(target_size, Image.BILINEAR)
    
    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    
    # Normalize with ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    return image_tensor, original_size


def postprocess_predictions(predictions: dict, original_size: tuple, target_size: tuple):
    """Postprocess predictions to original image coordinates"""
    # Scale bounding boxes back to original image size
    scale_w = original_size[0] / target_size[0]
    scale_h = original_size[1] / target_size[1]
    
    if predictions['boxes'].numel() > 0:
        scaled_boxes = predictions['boxes'].clone()
        scaled_boxes[:, [0, 2]] *= scale_w  # x coordinates
        scaled_boxes[:, [1, 3]] *= scale_h  # y coordinates
        
        # Clip to image boundaries
        scaled_boxes[:, [0, 2]] = torch.clamp(scaled_boxes[:, [0, 2]], 0, original_size[0])
        scaled_boxes[:, [1, 3]] = torch.clamp(scaled_boxes[:, [1, 3]], 0, original_size[1])
        
        predictions['boxes'] = scaled_boxes
    
    return predictions


def visualize_results(image_path: str, predictions: dict, class_names: list, 
                     confidence_threshold: float = 0.5, save_path: str = None):
    """Visualize detection results"""
    # Load original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    # VOC class names
    if class_names is None:
        class_names = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
            'train', 'tvmonitor'
        ]
    
    # Draw bounding boxes
    if predictions['boxes'].numel() > 0:
        for i in range(len(predictions['boxes'])):
            box = predictions['boxes'][i]
            score = predictions['scores'][i]
            label = predictions['labels'][i]
            
            # Filter by confidence
            if score < confidence_threshold:
                continue
            
            # Get box coordinates
            x1, y1, x2, y2 = box.tolist()
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            class_name = class_names[label] if label < len(class_names) else f'class_{label}'
            ax.text(x1, y1 - 5, f'{class_name}: {score:.2f}',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                   fontsize=8, color='white')
    
    ax.set_title('Faster R-CNN Detection Results')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results saved to {save_path}")
    
    plt.show()


def run_inference(model, image_path: str, device: torch.device, 
                 confidence_threshold: float = 0.5):
    """Run inference on a single image"""
    print(f"Processing image: {image_path}")
    
    # Preprocess image
    image_tensor, original_size = preprocess_image(image_path)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        predictions = model(image_tensor)
        predictions = predictions['detections'][0]
    
    # Postprocess predictions
    predictions = postprocess_predictions(predictions, original_size, (800, 800))
    
    # Filter by confidence
    if predictions['scores'].numel() > 0:
        keep_mask = predictions['scores'] >= confidence_threshold
        predictions = {
            'boxes': predictions['boxes'][keep_mask],
            'labels': predictions['labels'][keep_mask],
            'scores': predictions['scores'][keep_mask]
        }
    
    # Print results
    print(f"Found {len(predictions['boxes'])} detections")
    if len(predictions['boxes']) > 0:
        for i in range(len(predictions['boxes'])):
            box = predictions['boxes'][i]
            score = predictions['scores'][i]
            label = predictions['labels'][i]
            print(f"  Detection {i+1}: Class {label}, Score {score:.3f}, Box {box.tolist()}")
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='Faster R-CNN Demo')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for detections')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output image')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Check if files exist
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, device)
    
    # VOC class names
    class_names = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
        'train', 'tvmonitor'
    ]
    
    # Run inference
    predictions = run_inference(model, args.image, device, args.confidence)
    
    # Visualize results
    visualize_results(args.image, predictions, class_names, 
                     args.confidence, args.output)
    
    print("Demo completed!")


if __name__ == '__main__':
    main()
