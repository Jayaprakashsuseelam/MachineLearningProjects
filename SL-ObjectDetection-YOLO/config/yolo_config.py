"""
YOLO Configuration File
Contains all configuration parameters for different YOLO versions
"""

import os
from typing import Dict, List, Tuple, Any

class YOLOConfig:
    """Base configuration class for YOLO models"""
    
    def __init__(self, version: str = "v5"):
        self.version = version
        self.config = self._get_config()
    
    def _get_config(self) -> Dict[str, Any]:
        """Get configuration based on YOLO version"""
        if self.version == "v3":
            return self._get_yolov3_config()
        elif self.version == "v4":
            return self._get_yolov4_config()
        elif self.version == "v5":
            return self._get_yolov5_config()
        else:
            raise ValueError(f"Unsupported YOLO version: {self.version}")
    
    def _get_yolov3_config(self) -> Dict[str, Any]:
        """YOLOv3 configuration"""
        return {
            # Model Architecture
            'backbone': 'darknet53',
            'input_size': 416,
            'num_classes': 80,
            'anchors': [
                [(10, 13), (16, 30), (33, 23)],  # P3/8
                [(30, 61), (62, 45), (59, 119)],  # P4/16
                [(116, 90), (156, 198), (373, 326)]  # P5/32
            ],
            
            # Training Parameters
            'batch_size': 16,
            'epochs': 300,
            'learning_rate': 0.001,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            
            # Loss Weights
            'lambda_coord': 5.0,
            'lambda_noobj': 0.5,
            'lambda_obj': 1.0,
            'lambda_class': 1.0,
            
            # Data Augmentation
            'hsv_h': 0.015,  # HSV-Hue augmentation
            'hsv_s': 0.7,    # HSV-Saturation augmentation
            'hsv_v': 0.4,    # HSV-Value augmentation
            'degrees': 0.0,  # Image rotation
            'translate': 0.1,  # Image translation
            'scale': 0.5,    # Image scaling
            'shear': 0.0,    # Image shear
            'perspective': 0.0,  # Image perspective
            'flipud': 0.0,   # Vertical flip
            'fliplr': 0.5,   # Horizontal flip
            'mosaic': 0.0,   # Mosaic augmentation
            'mixup': 0.0,    # Mixup augmentation
            
            # Inference Parameters
            'conf_threshold': 0.25,
            'iou_threshold': 0.45,
            'max_det': 300,
            
            # Optimization
            'optimizer': 'SGD',
            'scheduler': 'cosine',
            'amp': False,  # Automatic Mixed Precision
        }
    
    def _get_yolov4_config(self) -> Dict[str, Any]:
        """YOLOv4 configuration"""
        config = self._get_yolov3_config()
        config.update({
            # Model Architecture
            'backbone': 'cspdarknet53',
            'neck': 'panet',
            'input_size': 608,
            
            # YOLOv4 specific improvements
            'mosaic': 1.0,   # Mosaic augmentation
            'mixup': 0.1,    # Mixup augmentation
            'copy_paste': 0.1,  # Copy-paste augmentation
            
            # Bag of Freebies
            'label_smoothing': 0.01,
            'dropout': 0.0,
            
            # Bag of Specials
            'activation': 'mish',
            'attention': 'sam',
            
            # Training Parameters
            'learning_rate': 0.00261,
            'momentum': 0.949,
            'weight_decay': 0.0005,
        })
        return config
    
    def _get_yolov5_config(self) -> Dict[str, Any]:
        """YOLOv5 configuration"""
        config = self._get_yolov4_config()
        config.update({
            # Model Architecture
            'backbone': 'cspdarknet',
            'input_size': 640,
            
            # YOLOv5 specific features
            'auto_anchor': True,
            'rect': False,
            'multi_scale': True,
            
            # Training Parameters
            'batch_size': 16,
            'epochs': 300,
            'learning_rate': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Data Augmentation
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.1,
            'copy_paste': 0.1,
            
            # Optimization
            'optimizer': 'SGD',
            'scheduler': 'cosine',
            'amp': True,  # Automatic Mixed Precision
            'sync_bn': False,
            'workers': 8,
            
            # Model Export
            'export_format': ['torchscript', 'onnx', 'coreml'],
        })
        return config
    
    def get_model_config(self, model_size: str = "s") -> Dict[str, Any]:
        """Get configuration for specific model size"""
        size_configs = {
            'n': {  # nano
                'depth_multiple': 0.33,
                'width_multiple': 0.25,
                'input_size': 640,
            },
            's': {  # small
                'depth_multiple': 0.33,
                'width_multiple': 0.50,
                'input_size': 640,
            },
            'm': {  # medium
                'depth_multiple': 0.67,
                'width_multiple': 0.75,
                'input_size': 640,
            },
            'l': {  # large
                'depth_multiple': 1.0,
                'width_multiple': 1.0,
                'input_size': 640,
            },
            'x': {  # extra large
                'depth_multiple': 1.33,
                'width_multiple': 1.25,
                'input_size': 640,
            }
        }
        
        config = self.config.copy()
        config.update(size_configs.get(model_size, size_configs['s']))
        return config

# Predefined configurations
YOLO_V3_CONFIG = YOLOConfig("v3").config
YOLO_V4_CONFIG = YOLOConfig("v4").config
YOLO_V5_CONFIG = YOLOConfig("v5").config

# Model size configurations
YOLO_V5_NANO = YOLOConfig("v5").get_model_config("n")
YOLO_V5_SMALL = YOLOConfig("v5").get_model_config("s")
YOLO_V5_MEDIUM = YOLOConfig("v5").get_model_config("m")
YOLO_V5_LARGE = YOLOConfig("v5").get_model_config("l")
YOLO_V5_XLARGE = YOLOConfig("v5").get_model_config("x")

# Dataset configurations
COCO_CONFIG = {
    'num_classes': 80,
    'class_names': [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
}

VOC_CONFIG = {
    'num_classes': 20,
    'class_names': [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
        'train', 'tvmonitor'
    ]
}

# Training configurations
TRAINING_CONFIGS = {
    'default': {
        'epochs': 300,
        'batch_size': 16,
        'learning_rate': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'patience': 50,
        'save_period': 10,
        'eval_period': 10,
    },
    'fast': {
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 1,
        'patience': 20,
        'save_period': 5,
        'eval_period': 5,
    },
    'accurate': {
        'epochs': 500,
        'batch_size': 8,
        'learning_rate': 0.005,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5,
        'patience': 100,
        'save_period': 20,
        'eval_period': 20,
    }
}

# Inference configurations
INFERENCE_CONFIGS = {
    'fast': {
        'conf_threshold': 0.3,
        'iou_threshold': 0.5,
        'max_det': 100,
        'input_size': 416,
    },
    'balanced': {
        'conf_threshold': 0.25,
        'iou_threshold': 0.45,
        'max_det': 300,
        'input_size': 640,
    },
    'accurate': {
        'conf_threshold': 0.1,
        'iou_threshold': 0.4,
        'max_det': 500,
        'input_size': 832,
    }
}
