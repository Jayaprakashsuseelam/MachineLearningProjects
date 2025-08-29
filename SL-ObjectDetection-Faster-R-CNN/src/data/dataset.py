"""
Dataset implementations for Faster R-CNN
"""
import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import json


class VOCDataset(Dataset):
    """PASCAL VOC Dataset for object detection"""
    
    def __init__(self, root: str, year: str = "2012", image_set: str = "train",
                 transform: Optional[Any] = None, target_transform: Optional[Any] = None,
                 download: bool = False):
        """
        Initialize PASCAL VOC dataset
        
        Args:
            root: Root directory path
            year: Dataset year (2007, 2008, 2009, 2010, 2011, 2012)
            image_set: Image set (train, trainval, val, test)
            transform: Optional transform to be applied on images
            target_transform: Optional transform to be applied on targets
            download: Whether to download the dataset
        """
        self.root = root
        self.year = year
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        
        # VOC classes (20 classes + background)
        self.classes = [
            'background',  # Index 0 is background
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        
        # Dataset paths
        self.image_dir = os.path.join(root, 'VOC' + year, 'JPEGImages')
        self.annotation_dir = os.path.join(root, 'VOC' + year, 'Annotations')
        self.image_set_file = os.path.join(root, 'VOC' + year, 'ImageSets', 'Main', f'{image_set}.txt')
        
        # Check if dataset exists
        if not os.path.exists(self.image_dir):
            if download:
                self._download_dataset()
            else:
                raise RuntimeError(
                    f"Dataset not found at {root}. "
                    "Use download=True to download it automatically."
                )
        
        # Load image list
        self.images = self._load_image_list()
        
        # Validate dataset
        self._validate_dataset()
    
    def _load_image_list(self) -> List[str]:
        """Load list of image filenames"""
        if not os.path.exists(self.image_set_file):
            raise RuntimeError(f"Image set file not found: {self.image_set_file}")
        
        with open(self.image_set_file, 'r') as f:
            images = [line.strip() for line in f.readlines()]
        
        return images
    
    def _validate_dataset(self):
        """Validate that all images and annotations exist"""
        missing_files = []
        
        for img_id in self.images:
            img_path = os.path.join(self.image_dir, f'{img_id}.jpg')
            ann_path = os.path.join(self.annotation_dir, f'{img_id}.xml')
            
            if not os.path.exists(img_path):
                missing_files.append(img_path)
            if not os.path.exists(ann_path):
                missing_files.append(ann_path)
        
        if missing_files:
            print(f"Warning: {len(missing_files)} files are missing from the dataset")
            print("First few missing files:", missing_files[:5])
    
    def _download_dataset(self):
        """Download PASCAL VOC dataset"""
        import urllib.request
        import tarfile
        
        print("Downloading PASCAL VOC dataset...")
        
        # VOC 2012 download URL
        if self.year == "2012":
            url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
            filename = "VOCtrainval_11-May-2012.tar"
        else:
            raise ValueError(f"Download not supported for year {self.year}")
        
        # Download file
        if not os.path.exists(filename):
            print(f"Downloading {url}...")
            urllib.request.urlretrieve(url, filename)
        
        # Extract file
        print("Extracting dataset...")
        with tarfile.open(filename, 'r') as tar:
            tar.extractall(self.root)
        
        print("Dataset downloaded and extracted successfully!")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get a single sample from the dataset
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (image, target) where target is a dictionary containing:
                - boxes: Bounding boxes [N, 4] in (x1, y1, x2, y2) format
                - labels: Class labels [N]
                - image_id: Image identifier
                - area: Box areas [N]
                - iscrowd: Crowd flag [N]
        """
        img_id = self.images[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, f'{img_id}.jpg')
        image = Image.open(img_path).convert('RGB')
        
        # Load annotations
        ann_path = os.path.join(self.annotation_dir, f'{img_id}.xml')
        target = self._load_annotations(ann_path)
        
        # Apply transforms
        if self.transform is not None:
            image, target = self.transform(image, target)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target
    
    def _load_annotations(self, ann_path: str) -> Dict[str, Any]:
        """Load annotations from XML file"""
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        # Parse objects
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for obj in root.findall('object'):
            # Get class name
            class_name = obj.find('n').text
            if class_name not in self.class_to_idx:
                continue  # Skip unknown classes
            
            # Get bounding box
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Validate coordinates
            xmin = max(0, min(xmin, width))
            ymin = max(0, min(ymin, height))
            xmax = max(0, min(xmax, width))
            ymax = max(0, min(ymax, height))
            
            # Skip invalid boxes
            if xmax <= xmin or ymax <= ymin:
                continue
            
            # Add box
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[class_name])
            
            # Calculate area
            area = (xmax - xmin) * (ymax - ymin)
            areas.append(area)
            
            # Check if crowd (usually false for VOC)
            iscrowd.append(False)
        
        # Convert to tensors
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.long)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.uint8)
        else:
            # No objects in image
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.long)
            areas = torch.empty((0,), dtype=torch.float32)
            iscrowd = torch.empty((0,), dtype=torch.uint8)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': areas,
            'iscrowd': iscrowd,
            'orig_size': torch.tensor([height, width])
        }
        
        return target
    
    def get_class_names(self) -> List[str]:
        """Get list of class names"""
        return self.classes[1:]  # Exclude background
    
    def get_class_to_idx(self) -> Dict[str, int]:
        """Get class name to index mapping"""
        return {cls_name: idx for cls_name, idx in self.class_to_idx.items() if cls_name != 'background'}


class CustomDataset(Dataset):
    """Base class for custom object detection datasets"""
    
    def __init__(self, root: str, transform: Optional[Any] = None, 
                 target_transform: Optional[Any] = None):
        """
        Initialize custom dataset
        
        Args:
            root: Root directory path
            transform: Optional transform to be applied on images
            target_transform: Optional transform to be applied on targets
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.images = []
        self.annotations = []
        
        # Override in subclass
        self.classes = []
        self.class_to_idx = {}
        self.num_classes = 0
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get a single sample from the dataset
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (image, target)
        """
        raise NotImplementedError("Subclass must implement __getitem__")
    
    def _load_image(self, img_path: str) -> Image.Image:
        """Load image from path"""
        return Image.open(img_path).convert('RGB')
    
    def _validate_annotation(self, annotation: Dict[str, Any]) -> bool:
        """Validate annotation format"""
        required_keys = ['boxes', 'labels']
        return all(key in annotation for key in required_keys)
    
    def get_class_names(self) -> List[str]:
        """Get list of class names"""
        return self.classes
    
    def get_class_to_idx(self) -> Dict[str, int]:
        """Get class name to index mapping"""
        return self.class_to_idx


class COCODataset(CustomDataset):
    """COCO format dataset"""
    
    def __init__(self, root: str, ann_file: str, transform: Optional[Any] = None,
                 target_transform: Optional[Any] = None):
        """
        Initialize COCO format dataset
        
        Args:
            root: Root directory path
            ann_file: Path to annotation file (JSON)
            transform: Optional transform to be applied on images
            target_transform: Optional transform to be applied on targets
        """
        super().__init__(root, transform, target_transform)
        
        # Load annotations
        self._load_annotations(ann_file)
    
    def _load_annotations(self, ann_file: str):
        """Load COCO format annotations"""
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # Load categories
        self.classes = ['background'] + [cat['name'] for cat in data['categories']]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        
        # Load images and annotations
        self.images = data['images']
        self.annotations = data['annotations']
        
        # Create image to annotation mapping
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get a single sample"""
        img_info = self.images[idx]
        img_id = img_info['id']
        
        # Load image
        img_path = os.path.join(self.root, img_info['file_name'])
        image = self._load_image(img_path)
        
        # Load annotations
        anns = self.img_to_anns.get(img_id, [])
        target = self._load_coco_annotations(anns, img_info)
        
        # Apply transforms
        if self.transform is not None:
            image, target = self.transform(image, target)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target
    
    def _load_coco_annotations(self, anns: List[Dict], img_info: Dict) -> Dict[str, Any]:
        """Load COCO format annotations"""
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            # Get bounding box (COCO format: [x, y, width, height])
            bbox = ann['bbox']
            x, y, w, h = bbox
            
            # Convert to (x1, y1, x2, y2) format
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            # Get category
            cat_id = ann['category_id']
            label = cat_id + 1  # +1 because 0 is background
            
            boxes.append([x1, y1, x2, y2])
            labels.append(label)
            areas.append(ann['area'])
            iscrowd.append(ann['iscrowd'])
        
        # Convert to tensors
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.long)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.uint8)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.long)
            areas = torch.empty((0,), dtype=torch.float32)
            iscrowd = torch.empty((0,), dtype=torch.uint8)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_info['id']]),
            'area': areas,
            'iscrowd': iscrowd,
            'orig_size': torch.tensor([img_info['height'], img_info['width']])
        }
        
        return target


def get_dataset(dataset_type: str, root: str, **kwargs) -> Dataset:
    """Factory function to get dataset"""
    if dataset_type.lower() == 'voc':
        return VOCDataset(root, **kwargs)
    elif dataset_type.lower() == 'coco':
        return COCODataset(root, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
