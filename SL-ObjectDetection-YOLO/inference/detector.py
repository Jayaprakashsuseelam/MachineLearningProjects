"""
YOLO Detector for Inference
Comprehensive inference module with preprocessing, detection, and visualization
"""

import torch
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
import time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

from models.yolo_v5 import get_yolov5_model
from config.yolo_config import COCO_CONFIG, INFERENCE_CONFIGS

class YOLODetector:
    """YOLO Object Detector for inference"""
    
    def __init__(self, model_path: Optional[str] = None, 
                 model_name: str = 'yolov5s',
                 config: Optional[Dict[str, Any]] = None,
                 device: str = 'auto',
                 confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.45):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to pre-trained model weights
            model_name: Name of YOLO model (yolov5n, yolov5s, yolov5m, yolov5l, yolov5x)
            config: Model configuration dictionary
            device: Device to run inference on ('cpu', 'cuda', 'auto')
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = model_path
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Set device
        self.device = self._set_device(device)
        
        # Load configuration
        self.config = config or COCO_CONFIG.copy()
        self.config.update(INFERENCE_CONFIGS['balanced'])
        
        # Load model
        self.model = self._load_model()
        
        # Class names
        self.class_names = self.config.get('class_names', COCO_CONFIG['class_names'])
        self.num_classes = len(self.class_names)
        
        # Color palette for visualization
        self.colors = self._generate_colors()
        
        # Statistics
        self.inference_times = []
    
    def _set_device(self, device: str) -> torch.device:
        """Set the device for inference"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = 'cpu'
        
        return torch.device(device)
    
    def _load_model(self) -> torch.nn.Module:
        """Load YOLO model"""
        if self.model_path and Path(self.model_path).exists():
            # Load pre-trained weights
            print(f"Loading model from {self.model_path}")
            model = torch.load(self.model_path, map_location=self.device)
            if isinstance(model, dict):
                model = model['model'] if 'model' in model else model
        else:
            # Create model from scratch
            print(f"Creating {self.model_name} model")
            from models.yolo_v5 import get_yolov5_model
            model = get_yolov5_model(self.model_name, self.config)
        
        model.to(self.device)
        model.eval()
        return model
    
    def _generate_colors(self) -> List[Tuple[int, int, int]]:
        """Generate color palette for visualization"""
        np.random.seed(42)
        colors = []
        for _ in range(self.num_classes):
            color = tuple(map(int, np.random.randint(0, 255, 3)))
            colors.append(color)
        return colors
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for inference
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            
        Returns:
            Preprocessed tensor and original image size
        """
        # Load image
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            img = image.copy()
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Store original size
        original_size = img.shape[:2]  # (height, width)
        
        # Resize image
        input_size = self.config.get('input_size', 640)
        img_resized = cv2.resize(img, (input_size, input_size))
        
        # Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        return img_tensor, original_size
    
    def postprocess_detections(self, predictions: torch.Tensor, 
                              original_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Post-process model predictions
        
        Args:
            predictions: Raw model predictions
            original_size: Original image size (height, width)
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        # Apply confidence threshold
        conf_mask = predictions[..., 4] > self.confidence_threshold
        predictions = predictions[conf_mask]
        
        if len(predictions) == 0:
            return detections
        
        # Get class probabilities
        class_probs = predictions[..., 5:]
        class_ids = torch.argmax(class_probs, dim=1)
        class_scores = torch.max(class_probs, dim=1)[0]
        
        # Combine confidence and class scores
        scores = predictions[..., 4] * class_scores
        
        # Apply score threshold
        score_mask = scores > self.confidence_threshold
        predictions = predictions[score_mask]
        class_ids = class_ids[score_mask]
        scores = scores[score_mask]
        
        if len(predictions) == 0:
            return detections
        
        # Convert to numpy
        boxes = predictions[..., :4].cpu().numpy()
        class_ids = class_ids.cpu().numpy()
        scores = scores.cpu().numpy()
        
        # Scale boxes to original image size
        input_size = self.config.get('input_size', 640)
        scale_x = original_size[1] / input_size
        scale_y = original_size[0] / input_size
        
        boxes[:, [0, 2]] *= scale_x  # x coordinates
        boxes[:, [1, 3]] *= scale_y  # y coordinates
        
        # Convert from center format to corner format
        boxes_corners = np.zeros_like(boxes)
        boxes_corners[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_corners[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_corners[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_corners[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        
        # Apply NMS
        keep_indices = self._non_max_suppression(boxes_corners, scores, self.iou_threshold)
        
        # Create detection results
        for idx in keep_indices:
            detection = {
                'bbox': boxes_corners[idx].tolist(),
                'class_id': int(class_ids[idx]),
                'class_name': self.class_names[int(class_ids[idx])],
                'confidence': float(scores[idx])
            }
            detections.append(detection)
        
        return detections
    
    def _non_max_suppression(self, boxes: np.ndarray, scores: np.ndarray, 
                           iou_threshold: float) -> np.ndarray:
        """
        Apply Non-Maximum Suppression
        
        Args:
            boxes: Bounding boxes in corner format (x1, y1, x2, y2)
            scores: Confidence scores
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Indices of kept boxes
        """
        if len(boxes) == 0:
            return np.array([])
        
        # Sort by scores
        sorted_indices = np.argsort(scores)[::-1]
        keep_indices = []
        
        while len(sorted_indices) > 0:
            # Keep the box with highest score
            current_idx = sorted_indices[0]
            keep_indices.append(current_idx)
            
            if len(sorted_indices) == 1:
                break
            
            # Remove current box from consideration
            sorted_indices = sorted_indices[1:]
            
            # Calculate IoU with remaining boxes
            current_box = boxes[current_idx]
            remaining_boxes = boxes[sorted_indices]
            
            # Calculate intersection
            x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
            y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
            x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
            y2 = np.minimum(current_box[3], remaining_boxes[:, 3])
            
            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            
            # Calculate union
            current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            remaining_areas = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * \
                            (remaining_boxes[:, 3] - remaining_boxes[:, 1])
            union = current_area + remaining_areas - intersection
            
            # Calculate IoU
            iou = intersection / (union + 1e-6)
            
            # Keep boxes with IoU below threshold
            sorted_indices = sorted_indices[iou < iou_threshold]
        
        return np.array(keep_indices)
    
    def detect_image(self, image: Union[str, np.ndarray, Image.Image]) -> List[Dict[str, Any]]:
        """
        Detect objects in an image
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            
        Returns:
            List of detection dictionaries
        """
        # Preprocess
        img_tensor, original_size = self.preprocess_image(image)
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            predictions = self.model(img_tensor)
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Post-process
        detections = self.postprocess_detections(predictions, original_size)
        
        return detections
    
    def visualize_results(self, image: Union[str, np.ndarray, Image.Image], 
                         detections: List[Dict[str, Any]], 
                         save_path: Optional[str] = None,
                         show_labels: bool = True,
                         show_scores: bool = True) -> np.ndarray:
        """
        Visualize detection results
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            save_path: Path to save visualization
            show_labels: Whether to show class labels
            show_scores: Whether to show confidence scores
            
        Returns:
            Visualization image
        """
        # Load image
        if isinstance(image, str):
            img = cv2.imread(image)
        elif isinstance(image, np.ndarray):
            img = image.copy()
        elif isinstance(image, Image.Image):
            img = np.array(image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Draw detections
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            class_id = detection['class_id']
            
            # Get color
            color = self.colors[class_id]
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = ""
            if show_labels:
                label += class_name
            if show_scores:
                if label:
                    label += " "
                label += f"{confidence:.2f}"
            
            if label:
                # Calculate text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # Draw label background
                cv2.rectangle(img, (x1, y1 - text_height - baseline - 5), 
                            (x1 + text_width, y1), color, -1)
                
                # Draw label text
                cv2.putText(img, label, (x1, y1 - baseline - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save if requested
        if save_path:
            cv2.imwrite(save_path, img)
            print(f"Visualization saved to {save_path}")
        
        return img
    
    def detect_and_visualize(self, image: Union[str, np.ndarray, Image.Image],
                           save_path: Optional[str] = None) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Detect objects and visualize results in one step
        
        Args:
            image: Input image
            save_path: Path to save visualization
            
        Returns:
            Tuple of (detections, visualization_image)
        """
        detections = self.detect_image(image)
        visualization = self.visualize_results(image, detections, save_path)
        return detections, visualization
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get inference statistics"""
        if not self.inference_times:
            return {}
        
        times = np.array(self.inference_times)
        return {
            'total_inferences': len(times),
            'mean_inference_time': float(np.mean(times)),
            'std_inference_time': float(np.std(times)),
            'min_inference_time': float(np.min(times)),
            'max_inference_time': float(np.max(times)),
            'fps': float(1.0 / np.mean(times))
        }
    
    def reset_statistics(self):
        """Reset inference statistics"""
        self.inference_times = []

# Convenience functions
def load_detector(model_path: str, **kwargs) -> YOLODetector:
    """Load a YOLO detector from pre-trained weights"""
    return YOLODetector(model_path=model_path, **kwargs)

def detect_objects(image_path: str, model_path: str, **kwargs) -> List[Dict[str, Any]]:
    """Quick function to detect objects in an image"""
    detector = load_detector(model_path, **kwargs)
    return detector.detect_image(image_path)
