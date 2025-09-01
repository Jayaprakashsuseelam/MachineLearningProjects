"""
Custom collate function for Faster R-CNN datasets
"""
import torch
from torch.utils.data import default_collate
from typing import List, Dict, Any, Tuple
import numpy as np


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for Faster R-CNN datasets.
    
    This function handles variable-sized images and bounding boxes by:
    1. Padding images to the same size
    2. Stacking bounding boxes and labels
    3. Handling variable numbers of objects per image
    
    Args:
        batch: List of dictionaries containing image, boxes, and labels
        
    Returns:
        Dictionary with batched tensors
    """
    if not batch:
        return {}
    
    # Separate images, boxes, and labels
    images = []
    boxes = []
    labels = []
    image_ids = []
    
    # Find maximum dimensions for padding
    max_height = max(item['image'].shape[1] for item in batch)
    max_width = max(item['image'].shape[2] for item in batch)
    
    for item in batch:
        # Pad image to max dimensions
        img = item['image']
        pad_height = max_height - img.shape[1]
        pad_width = max_width - img.shape[2]
        
        if pad_height > 0 or pad_width > 0:
            # Pad with zeros (black)
            img = torch.nn.functional.pad(
                img, 
                (0, pad_width, 0, pad_height), 
                mode='constant', 
                value=0
            )
        
        images.append(img)
        
        # Handle boxes and labels
        if 'boxes' in item and len(item['boxes']) > 0:
            boxes.append(item['boxes'])
            labels.append(item['labels'])
        else:
            # Empty image - add dummy box and label
            boxes.append(torch.zeros((1, 4), dtype=torch.float32))
            labels.append(torch.zeros((1,), dtype=torch.long))
        
        # Add image ID if available
        if 'image_id' in item:
            image_ids.append(item['image_id'])
    
    # Stack images
    images = torch.stack(images, dim=0)
    
    # Handle variable number of boxes per image
    max_boxes = max(len(box) for box in boxes)
    
    padded_boxes = []
    padded_labels = []
    
    for box, label in zip(boxes, labels):
        if len(box) < max_boxes:
            # Pad with dummy boxes
            pad_size = max_boxes - len(box)
            padded_box = torch.cat([
                box,
                torch.zeros((pad_size, 4), dtype=torch.float32)
            ], dim=0)
            padded_label = torch.cat([
                label,
                torch.zeros((pad_size,), dtype=torch.long)
            ], dim=0)
        else:
            padded_box = box
            padded_label = label
        
        padded_boxes.append(padded_box)
        padded_labels.append(padded_label)
    
    boxes = torch.stack(padded_boxes, dim=0)
    labels = torch.stack(padded_labels, dim=0)
    
    result = {
        'images': images,
        'boxes': boxes,
        'labels': labels,
    }
    
    # Add image IDs if available
    if image_ids:
        result['image_ids'] = image_ids
    
    # Add other metadata if present
    if 'area' in batch[0]:
        areas = [item['area'] for item in batch]
        max_objects = max(len(area) for area in areas)
        
        padded_areas = []
        for area in areas:
            if len(area) < max_objects:
                pad_size = max_objects - len(area)
                padded_area = torch.cat([
                    area,
                    torch.zeros((pad_size,), dtype=torch.float32)
                ], dim=0)
            else:
                padded_area = area
            padded_areas.append(padded_area)
        
        result['areas'] = torch.stack(padded_areas, dim=0)
    
    if 'iscrowd' in batch[0]:
        iscrowds = [item['iscrowd'] for item in batch]
        max_objects = max(len(iscrowd) for iscrowd in iscrowds)
        
        padded_iscrowds = []
        for iscrowd in iscrowds:
            if len(iscrowd) < max_objects:
                pad_size = max_objects - len(iscrowd)
                padded_iscrowd = torch.cat([
                    iscrowd,
                    torch.zeros((pad_size,), dtype=torch.bool)
                ], dim=0)
            else:
                padded_iscrowd = iscrowd
            padded_iscrowds.append(padded_iscrowd)
        
        result['iscrowds'] = torch.stack(padded_iscrowds, dim=0)
    
    return result


def collate_fn_simple(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Simplified collate function for basic use cases.
    
    Args:
        batch: List of dictionaries containing image, boxes, and labels
        
    Returns:
        Dictionary with batched tensors
    """
    if not batch:
        return {}
    
    # Simple stacking for fixed-size inputs
    images = torch.stack([item['image'] for item in batch], dim=0)
    
    # Handle boxes and labels
    boxes = []
    labels = []
    
    for item in batch:
        if 'boxes' in item and len(item['boxes']) > 0:
            boxes.append(item['boxes'])
            labels.append(item['labels'])
        else:
            # Empty image
            boxes.append(torch.zeros((1, 4), dtype=torch.float32))
            labels.append(torch.zeros((1,), dtype=torch.long))
    
    # Find max boxes for padding
    max_boxes = max(len(box) for box in boxes)
    
    padded_boxes = []
    padded_labels = []
    
    for box, label in zip(boxes, labels):
        if len(box) < max_boxes:
            pad_size = max_boxes - len(box)
            padded_box = torch.cat([
                box,
                torch.zeros((pad_size, 4), dtype=torch.float32)
            ], dim=0)
            padded_label = torch.cat([
                label,
                torch.zeros((pad_size,), dtype=torch.long)
            ], dim=0)
        else:
            padded_box = box
            padded_label = label
        
        padded_boxes.append(padded_box)
        padded_labels.append(padded_label)
    
    boxes = torch.stack(padded_boxes, dim=0)
    labels = torch.stack(padded_labels, dim=0)
    
    return {
        'images': images,
        'boxes': boxes,
        'labels': labels,
    }


def collate_fn_with_metadata(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function that preserves all metadata.
    
    Args:
        batch: List of dictionaries containing image, boxes, and labels
        
    Returns:
        Dictionary with batched tensors and metadata
    """
    if not batch:
        return {}
    
    # Use the main collate function
    result = collate_fn(batch)
    
    # Add all other metadata
    metadata_keys = set()
    for item in batch:
        metadata_keys.update(item.keys())
    
    # Remove keys already handled by collate_fn
    handled_keys = {'image', 'boxes', 'labels', 'area', 'iscrowd'}
    metadata_keys = metadata_keys - handled_keys
    
    for key in metadata_keys:
        if key in batch[0]:
            values = [item.get(key, None) for item in batch]
            # Only add if all values are the same type
            if all(v is not None for v in values):
                try:
                    if isinstance(values[0], torch.Tensor):
                        result[key] = torch.stack(values, dim=0)
                    elif isinstance(values[0], (int, float)):
                        result[key] = torch.tensor(values)
                    else:
                        result[key] = values
                except:
                    # If stacking fails, keep as list
                    result[key] = values
    
    return result
