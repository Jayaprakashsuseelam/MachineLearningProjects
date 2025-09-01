"""
Data Utilities for YOLO
Comprehensive data processing utilities for YOLO training and inference
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import json
import yaml
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

class YOLODataset(Dataset):
    """YOLO Dataset class"""
    
    def __init__(self, 
                 images_dir: str,
                 labels_dir: str,
                 class_names: List[str],
                 transform: Optional[A.Compose] = None,
                 input_size: int = 640,
                 augment: bool = True):
        """
        Initialize YOLO dataset
        
        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing YOLO format labels
            class_names: List of class names
            transform: Albumentations transform pipeline
            input_size: Input image size
            augment: Whether to apply augmentation
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.class_names = class_names
        self.input_size = input_size
        self.augment = augment
        
        # Get image and label files
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')) + 
                                 list(self.images_dir.glob('*.jpeg')) + 
                                 list(self.images_dir.glob('*.png')))
        
        # Create class name to index mapping
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        # Default transforms
        if transform is None:
            self.transform = self._get_default_transforms()
        else:
            self.transform = transform
    
    def _get_default_transforms(self) -> A.Compose:
        """Get default transforms"""
        if self.augment:
            return A.Compose([
                A.RandomResizedCrop(
                    height=self.input_size,
                    width=self.input_size,
                    scale=(0.8, 1.0),
                    ratio=(0.8, 1.2),
                    p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.RandomBrightnessContrast(p=0.2),
                A.HueSaturationValue(p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            return A.Compose([
                A.Resize(height=self.input_size, width=self.input_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get a sample from the dataset"""
        # Load image
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels
        label_path = self.labels_dir / (image_path.stem + '.txt')
        bboxes = []
        class_labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    values = line.strip().split()
                    if len(values) >= 5:
                        class_idx = int(values[0])
                        x_center = float(values[1])
                        y_center = float(values[2])
                        width = float(values[3])
                        height = float(values[4])
                        
                        bboxes.append([x_center, y_center, width, height])
                        class_labels.append(class_idx)
        
        # Apply transforms
        if len(bboxes) > 0:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            image = transformed['image']
            bboxes = transformed['bboxes']
            class_labels = transformed['class_labels']
        else:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Create target dictionary
        target = {
            'image_path': str(image_path),
            'bboxes': bboxes,
            'class_labels': class_labels,
            'image_size': (self.input_size, self.input_size)
        }
        
        return image, target

class YOLODataLoader:
    """YOLO Data Loader with collate function"""
    
    def __init__(self, dataset: YOLODataset, batch_size: int = 16, shuffle: bool = True, 
                 num_workers: int = 4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
    
    def get_loader(self) -> DataLoader:
        """Get PyTorch DataLoader"""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )
    
    def _collate_fn(self, batch: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Custom collate function for YOLO data"""
        images = torch.stack([item[0] for item in batch])
        targets = [item[1] for item in batch]
        return images, targets

def prepare_dataset(source_dir: str, 
                   output_dir: str, 
                   classes: List[str],
                   train_split: float = 0.8,
                   val_split: float = 0.2) -> Dict[str, str]:
    """
    Prepare YOLO format dataset from various formats
    
    Args:
        source_dir: Source dataset directory
        output_dir: Output directory for YOLO format
        classes: List of class names
        train_split: Training split ratio
        val_split: Validation split ratio
        
    Returns:
        Dictionary with dataset paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create directory structure
    (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    # Create data.yaml
    data_yaml = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(classes),
        'names': classes
    }
    
    with open(output_path / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    return {
        'data_yaml': str(output_path / 'data.yaml'),
        'train_images': str(output_path / 'images' / 'train'),
        'train_labels': str(output_path / 'labels' / 'train'),
        'val_images': str(output_path / 'images' / 'val'),
        'val_labels': str(output_path / 'labels' / 'val')
    }

def convert_bbox_format(bbox: List[float], 
                       from_format: str, 
                       to_format: str,
                       image_size: Tuple[int, int]) -> List[float]:
    """
    Convert bounding box between different formats
    
    Args:
        bbox: Bounding box coordinates
        from_format: Source format ('xyxy', 'xywh', 'yolo', 'center')
        to_format: Target format ('xyxy', 'xywh', 'yolo', 'center')
        image_size: Image size (width, height)
        
    Returns:
        Converted bounding box
    """
    img_w, img_h = image_size
    
    # Convert to absolute coordinates first
    if from_format == 'yolo':
        # YOLO format: (x_center, y_center, width, height) - normalized
        x_center, y_center, width, height = bbox
        x_center *= img_w
        y_center *= img_h
        width *= img_w
        height *= img_h
    elif from_format == 'xyxy':
        # (x1, y1, x2, y2) - absolute
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
    elif from_format == 'xywh':
        # (x1, y1, width, height) - absolute
        x1, y1, width, height = bbox
        x_center = x1 + width / 2
        y_center = y1 + height / 2
    elif from_format == 'center':
        # (x_center, y_center, width, height) - absolute
        x_center, y_center, width, height = bbox
    else:
        raise ValueError(f"Unsupported format: {from_format}")
    
    # Convert to target format
    if to_format == 'yolo':
        # Normalize
        return [x_center / img_w, y_center / img_h, width / img_w, height / img_h]
    elif to_format == 'xyxy':
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return [x1, y1, x2, y2]
    elif to_format == 'xywh':
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        return [x1, y1, width, height]
    elif to_format == 'center':
        return [x_center, y_center, width, height]
    else:
        raise ValueError(f"Unsupported format: {to_format}")

def create_anchor_boxes(dataset_path: str, 
                       num_anchors: int = 9,
                       input_size: int = 640) -> List[List[Tuple[int, int]]]:
    """
    Create anchor boxes using k-means clustering on dataset
    
    Args:
        dataset_path: Path to dataset
        num_anchors: Number of anchor boxes to create
        input_size: Input image size
        
    Returns:
        List of anchor boxes [(width, height), ...]
    """
    from sklearn.cluster import KMeans
    
    # Collect all bounding boxes
    bboxes = []
    dataset_path = Path(dataset_path)
    labels_dir = dataset_path / 'labels'
    
    for label_file in labels_dir.rglob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                values = line.strip().split()
                if len(values) >= 5:
                    # YOLO format: class x_center y_center width height
                    width = float(values[3]) * input_size
                    height = float(values[4]) * input_size
                    bboxes.append([width, height])
    
    if len(bboxes) == 0:
        # Default COCO anchors
        return [
            [(10, 13), (16, 30), (33, 23)],
            [(30, 61), (62, 45), (59, 119)],
            [(116, 90), (156, 198), (373, 326)]
        ]
    
    # Convert to numpy array
    bboxes = np.array(bboxes)
    
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=num_anchors, random_state=42)
    kmeans.fit(bboxes)
    
    # Get cluster centers
    anchors = kmeans.cluster_centers_
    
    # Sort by area
    areas = anchors[:, 0] * anchors[:, 1]
    sorted_indices = np.argsort(areas)
    anchors = anchors[sorted_indices]
    
    # Convert to list of tuples
    anchor_list = [(int(w), int(h)) for w, h in anchors]
    
    # Group into 3 scales (for 3 detection heads)
    anchors_per_scale = num_anchors // 3
    grouped_anchors = []
    for i in range(0, num_anchors, anchors_per_scale):
        grouped_anchors.append(anchor_list[i:i + anchors_per_scale])
    
    return grouped_anchors

def visualize_dataset(dataset: YOLODataset, 
                     num_samples: int = 5,
                     save_path: Optional[str] = None):
    """
    Visualize dataset samples
    
    Args:
        dataset: YOLO dataset
        num_samples: Number of samples to visualize
        save_path: Path to save visualization
    """
    fig, axes = plt.subplots(1, num_samples, figsize=(5 * num_samples, 5))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        image, target = dataset[i]
        
        # Convert tensor to numpy
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = image * std + mean
            image = np.clip(image, 0, 1)
        
        axes[i].imshow(image)
        axes[i].set_title(f'Sample {i+1}')
        axes[i].axis('off')
        
        # Draw bounding boxes
        bboxes = target['bboxes']
        class_labels = target['class_labels']
        
        for bbox, class_label in zip(bboxes, class_labels):
            x_center, y_center, width, height = bbox
            x1 = (x_center - width / 2) * dataset.input_size
            y1 = (y_center - height / 2) * dataset.input_size
            x2 = (x_center + width / 2) * dataset.input_size
            y2 = (y_center + height / 2) * dataset.input_size
            
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            axes[i].add_patch(rect)
            
            # Add class label
            class_name = dataset.class_names[class_label]
            axes[i].text(x1, y1 - 5, class_name, color='red', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Dataset visualization saved to {save_path}")
    else:
        plt.show()

def get_dataset_statistics(dataset_path: str) -> Dict[str, Any]:
    """
    Get dataset statistics
    
    Args:
        dataset_path: Path to dataset
        
    Returns:
        Dictionary with dataset statistics
    """
    dataset_path = Path(dataset_path)
    labels_dir = dataset_path / 'labels'
    
    total_images = 0
    total_objects = 0
    class_counts = {}
    bbox_sizes = []
    
    for label_file in labels_dir.rglob('*.txt'):
        total_images += 1
        with open(label_file, 'r') as f:
            for line in f:
                values = line.strip().split()
                if len(values) >= 5:
                    class_idx = int(values[0])
                    width = float(values[3])
                    height = float(values[4])
                    
                    total_objects += 1
                    class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
                    bbox_sizes.append([width, height])
    
    bbox_sizes = np.array(bbox_sizes)
    
    return {
        'total_images': total_images,
        'total_objects': total_objects,
        'class_counts': class_counts,
        'avg_bbox_width': float(np.mean(bbox_sizes[:, 0])),
        'avg_bbox_height': float(np.mean(bbox_sizes[:, 1])),
        'min_bbox_width': float(np.min(bbox_sizes[:, 0])),
        'min_bbox_height': float(np.min(bbox_sizes[:, 1])),
        'max_bbox_width': float(np.max(bbox_sizes[:, 0])),
        'max_bbox_height': float(np.max(bbox_sizes[:, 1]))
    }

# Convenience functions
def create_yolo_dataset(images_dir: str,
                       labels_dir: str,
                       class_names: List[str],
                       **kwargs) -> YOLODataset:
    """Create YOLO dataset"""
    return YOLODataset(images_dir, labels_dir, class_names, **kwargs)

def create_yolo_dataloader(dataset: YOLODataset, **kwargs) -> DataLoader:
    """Create YOLO data loader"""
    loader = YOLODataLoader(dataset, **kwargs)
    return loader.get_loader()
