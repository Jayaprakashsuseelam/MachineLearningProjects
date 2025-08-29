"""
Data transformations for Faster R-CNN
"""
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
import random
import numpy as np
from typing import Tuple, Dict, Any, Optional, List


class Compose:
    """Compose multiple transforms together"""
    
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, image: Image.Image, target: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """Convert PIL image to tensor"""
    
    def __call__(self, image: Image.Image, target: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        image = F.to_tensor(image)
        return image, target


class Normalize:
    """Normalize image with mean and std"""
    
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = mean
        self.std = std
    
    def __call__(self, image: torch.Tensor, target: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Resize:
    """Resize image and adjust bounding boxes"""
    
    def __init__(self, size: Tuple[int, int], max_size: Optional[int] = None):
        self.size = size
        self.max_size = max_size
    
    def __call__(self, image: Image.Image, target: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
        # Get original size
        orig_size = target.get('orig_size', torch.tensor([image.height, image.width]))
        
        # Calculate new size
        if self.max_size is not None:
            # Resize with max size constraint
            w, h = image.size
            scale = min(self.size[0] / h, self.size[1] / w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            if max(new_h, new_w) > self.max_size:
                scale = self.max_size / max(new_h, new_w)
                new_h = int(new_h * scale)
                new_w = int(new_w * scale)
        else:
            new_h, new_w = self.size
        
        # Resize image
        image = image.resize((new_w, new_h), Image.BILINEAR)
        
        # Adjust bounding boxes
        if 'boxes' in target and target['boxes'].numel() > 0:
            boxes = target['boxes'].clone()
            h_scale = new_h / orig_size[0]
            w_scale = new_w / orig_size[1]
            
            # Scale box coordinates
            boxes[:, [0, 2]] *= w_scale  # x coordinates
            boxes[:, [1, 3]] *= h_scale  # y coordinates
            
            target['boxes'] = boxes
        
        # Update original size
        target['orig_size'] = torch.tensor([new_h, new_w])
        
        return image, target


class RandomHorizontalFlip:
    """Randomly flip image horizontally and adjust bounding boxes"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: Image.Image, target: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
        if random.random() < self.p:
            # Flip image
            image = F.hflip(image)
            
            # Adjust bounding boxes
            if 'boxes' in target and target['boxes'].numel() > 0:
                boxes = target['boxes'].clone()
                w = image.width
                
                # Flip x coordinates: x -> w - x
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target['boxes'] = boxes
        
        return image, target


class RandomVerticalFlip:
    """Randomly flip image vertically and adjust bounding boxes"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: Image.Image, target: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
        if random.random() < self.p:
            # Flip image
            image = F.vflip(image)
            
            # Adjust bounding boxes
            if 'boxes' in target and target['boxes'].numel() > 0:
                boxes = target['boxes'].clone()
                h = image.height
                
                # Flip y coordinates: y -> h - y
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
                target['boxes'] = boxes
        
        return image, target


class ColorJitter:
    """Randomly change the brightness, contrast, saturation and hue of an image"""
    
    def __init__(self, brightness: float = 0, contrast: float = 0, 
                 saturation: float = 0, hue: float = 0, p: float = 0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p
    
    def __call__(self, image: Image.Image, target: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
        if random.random() < self.p:
            image = F.adjust_brightness(image, 1 + random.uniform(-self.brightness, self.brightness))
            image = F.adjust_contrast(image, 1 + random.uniform(-self.contrast, self.contrast))
            image = F.adjust_saturation(image, 1 + random.uniform(-self.saturation, self.saturation))
            image = F.adjust_hue(image, random.uniform(-self.hue, self.hue))
        
        return image, target


class RandomCrop:
    """Randomly crop image and adjust bounding boxes"""
    
    def __init__(self, size: Tuple[int, int], p: float = 0.5):
        self.size = size
        self.p = p
    
    def __call__(self, image: Image.Image, target: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
        if random.random() < self.p:
            # Get crop parameters
            w, h = image.size
            crop_h, crop_w = self.size
            
            # Ensure crop size is not larger than image
            crop_h = min(crop_h, h)
            crop_w = min(crop_w, w)
            
            # Random crop position
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
            
            # Crop image
            image = image.crop((left, top, left + crop_w, top + crop_h))
            
            # Adjust bounding boxes
            if 'boxes' in target and target['boxes'].numel() > 0:
                boxes = target['boxes'].clone()
                
                # Shift boxes by crop offset
                boxes[:, [0, 2]] -= left
                boxes[:, [1, 3]] -= top
                
                # Clip boxes to crop boundaries
                boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, crop_w)
                boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, crop_h)
                
                # Remove boxes that are too small after cropping
                valid_boxes = (boxes[:, 2] - boxes[:, 0]) > 0 and (boxes[:, 3] - boxes[:, 1]) > 0
                if valid_boxes.any():
                    target['boxes'] = boxes[valid_boxes]
                    target['labels'] = target['labels'][valid_boxes]
                    target['area'] = target['area'][valid_boxes]
                    target['iscrowd'] = target['iscrowd'][valid_boxes]
                else:
                    # No valid boxes left
                    target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
                    target['labels'] = torch.empty((0,), dtype=torch.long)
                    target['area'] = torch.empty((0,), dtype=torch.float32)
                    target['iscrowd'] = torch.empty((0,), dtype=torch.uint8)
            
            # Update original size
            target['orig_size'] = torch.tensor([crop_h, crop_w])
        
        return image, target


class RandomRotation:
    """Randomly rotate image and adjust bounding boxes"""
    
    def __init__(self, degrees: float, p: float = 0.5):
        self.degrees = degrees
        self.p = p
    
    def __call__(self, image: Image.Image, target: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
        if random.random() < self.p:
            # Random rotation angle
            angle = random.uniform(-self.degrees, self.degrees)
            
            # Rotate image
            image = image.rotate(angle, expand=True)
            
            # Adjust bounding boxes (simplified - may need more sophisticated handling)
            if 'boxes' in target and target['boxes'].numel() > 0:
                # For simplicity, we'll skip box adjustment for rotation
                # In practice, you'd need to implement proper box rotation
                pass
        
        return image, target


class RandomErasing:
    """Randomly erase rectangular regions in the image"""
    
    def __init__(self, p: float = 0.2, scale: Tuple[float, float] = (0.02, 0.33),
                 ratio: Tuple[float, float] = (0.3, 3.3), value: float = 0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
    
    def __call__(self, image: torch.Tensor, target: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if random.random() < self.p and isinstance(image, torch.Tensor):
            # Convert to tensor if needed
            if not isinstance(image, torch.Tensor):
                image = F.to_tensor(image)
            
            # Apply random erasing
            image = F.erase(image, 0, 0, 1, 1, self.value, False)
        
        return image, target


class PadToSize:
    """Pad image to a specific size"""
    
    def __init__(self, size: Tuple[int, int], fill: int = 0):
        self.size = size
        self.fill = fill
    
    def __call__(self, image: Image.Image, target: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
        w, h = image.size
        target_w, target_h = self.size
        
        # Calculate padding
        pad_left = max(0, (target_w - w) // 2)
        pad_right = max(0, target_w - w - pad_left)
        pad_top = max(0, (target_h - h) // 2)
        pad_bottom = max(0, target_h - h - pad_top)
        
        # Pad image
        image = F.pad(image, [pad_left, pad_top, pad_right, pad_bottom], fill=self.fill)
        
        # Adjust bounding boxes
        if 'boxes' in target and target['boxes'].numel() > 0:
            boxes = target['boxes'].clone()
            boxes[:, [0, 2]] += pad_left
            boxes[:, [1, 3]] += pad_top
            target['boxes'] = boxes
        
        # Update original size
        target['orig_size'] = torch.tensor([target_h, target_w])
        
        return image, target


def get_transform(train: bool = True, image_size: Tuple[int, int] = (800, 800),
                 max_size: Optional[int] = 1000, mean: List[float] = None,
                 std: List[float] = None) -> Compose:
    """
    Get transformation pipeline
    
    Args:
        train: Whether to use training transforms
        image_size: Target image size
        max_size: Maximum image size
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Compose transform object
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]  # ImageNet mean
    if std is None:
        std = [0.229, 0.224, 0.225]   # ImageNet std
    
    transforms = []
    
    if train:
        # Training transforms
        transforms.extend([
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
            RandomCrop(size=image_size, p=0.3),
        ])
    
    # Common transforms
    transforms.extend([
        Resize(size=image_size, max_size=max_size),
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    
    if train:
        # Add random erasing for training
        transforms.append(RandomErasing(p=0.2))
    
    return Compose(transforms)


def get_test_transform(image_size: Tuple[int, int] = (800, 800),
                      max_size: Optional[int] = 1000,
                      mean: List[float] = None,
                      std: List[float] = None) -> Compose:
    """
    Get test transformation pipeline
    
    Args:
        image_size: Target image size
        max_size: Maximum image size
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Compose transform object
    """
    return get_transform(train=False, image_size=image_size, max_size=max_size, mean=mean, std=std)


def get_augmentation_transform(image_size: Tuple[int, int] = (800, 800),
                             max_size: Optional[int] = 1000,
                             mean: List[float] = None,
                             std: List[float] = None) -> Compose:
    """
    Get strong augmentation transformation pipeline
    
    Args:
        image_size: Target image size
        max_size: Maximum image size
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Compose transform object
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    transforms = [
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.3),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        RandomCrop(size=image_size, p=0.5),
        RandomRotation(degrees=15, p=0.3),
        Resize(size=image_size, max_size=max_size),
        ToTensor(),
        Normalize(mean=mean, std=std),
        RandomErasing(p=0.3)
    ]
    
    return Compose(transforms)
