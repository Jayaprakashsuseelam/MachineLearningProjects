import torch
import torch.nn as nn
import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Optional
import time

from models.ssd_network import SSDNetwork, DefaultBoxGenerator
from models.ssd_loss import BoxEncoder
from utils.visualization import draw_detections
from utils.data_utils import preprocess_image, postprocess_image


class SSDDetector:
    """
    Single Shot Detector for real-time object detection
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 config_path: str = 'configs/ssd300_config.json',
                 device: str = 'auto'):
        """
        Initialize SSD Detector
        
        Args:
            model_path: Path to pre-trained model weights
            config_path: Path to model configuration file
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
        """
        self.device = self._get_device(device)
        self.config = self._load_config(config_path)
        
        # Initialize network
        self.network = SSDNetwork(self.config).to(self.device)
        
        # Generate default boxes
        self.default_boxes = self._generate_default_boxes()
        
        # Initialize box encoder
        self.box_encoder = BoxEncoder(self.default_boxes, self.config['variance'])
        
        # Load pre-trained weights if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Set network to evaluation mode
        self.network.eval()
        
        # Performance tracking
        self.fps_counter = []
        
    def _get_device(self, device: str) -> torch.device:
        """Determine the best device to use"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load model configuration from JSON file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    def _generate_default_boxes(self) -> torch.Tensor:
        """Generate default boxes for the model"""
        generator = DefaultBoxGenerator(self.config)
        default_boxes = generator.generate_default_boxes()
        return default_boxes.to(self.device)
    
    def load_model(self, model_path: str):
        """Load pre-trained model weights"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'state_dict' in checkpoint:
            self.network.load_state_dict(checkpoint['state_dict'])
        else:
            self.network.load_state_dict(checkpoint)
        
        print(f"Model loaded from {model_path}")
    
    def detect(self, image_path: str, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Detect objects in a single image
        
        Args:
            image_path: Path to input image
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of detection dictionaries with 'bbox', 'label', 'score' keys
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Preprocess image
        input_tensor = preprocess_image(image, self.config['input_size'])
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            start_time = time.time()
            loc_pred, conf_pred = self.network(input_tensor)
            inference_time = time.time() - start_time
        
        # Decode predictions
        decoded_boxes, decoded_labels, decoded_scores = self.box_encoder.decode(
            loc_pred, conf_pred, confidence_threshold
        )
        
        # Convert to original image coordinates
        detections = self._convert_to_image_coordinates(
            decoded_boxes[0], decoded_labels[0], decoded_scores[0], image.shape
        )
        
        # Update FPS counter
        fps = 1.0 / inference_time
        self.fps_counter.append(fps)
        if len(self.fps_counter) > 100:
            self.fps_counter.pop(0)
        
        return detections
    
    def detect_batch(self, image_paths: List[str], 
                    confidence_threshold: float = 0.5) -> List[List[Dict]]:
        """
        Detect objects in multiple images
        
        Args:
            image_paths: List of image paths
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of detection lists for each image
        """
        batch_detections = []
        
        for image_path in image_paths:
            detections = self.detect(image_path, confidence_threshold)
            batch_detections.append(detections)
        
        return batch_detections
    
    def detect_video(self, source: str = 0, 
                    confidence_threshold: float = 0.5,
                    display: bool = True,
                    save_path: Optional[str] = None):
        """
        Real-time object detection on video stream
        
        Args:
            source: Video source (0 for webcam, or video file path)
            confidence_threshold: Minimum confidence threshold
            display: Whether to display the video
            save_path: Path to save the output video
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer if save_path is provided
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        print("Starting real-time detection... Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect objects in frame
                detections = self._detect_frame(frame, confidence_threshold)
                
                # Draw detections on frame
                annotated_frame = draw_detections(frame, detections)
                
                # Add FPS information
                avg_fps = np.mean(self.fps_counter) if self.fps_counter else 0
                cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display frame
                if display:
                    cv2.imshow('SSD Object Detection', annotated_frame)
                
                # Save frame
                if writer:
                    writer.write(annotated_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
    
    def _detect_frame(self, frame: np.ndarray, 
                     confidence_threshold: float) -> List[Dict]:
        """Detect objects in a single frame"""
        # Preprocess frame
        input_tensor = preprocess_image(frame, self.config['input_size'])
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            start_time = time.time()
            loc_pred, conf_pred = self.network(input_tensor)
            inference_time = time.time() - start_time
        
        # Decode predictions
        decoded_boxes, decoded_labels, decoded_scores = self.box_encoder.decode(
            loc_pred, conf_pred, confidence_threshold
        )
        
        # Convert to frame coordinates
        detections = self._convert_to_image_coordinates(
            decoded_boxes[0], decoded_labels[0], decoded_scores[0], frame.shape
        )
        
        # Update FPS counter
        fps = 1.0 / inference_time
        self.fps_counter.append(fps)
        if len(self.fps_counter) > 100:
            self.fps_counter.pop(0)
        
        return detections
    
    def _convert_to_image_coordinates(self, boxes: torch.Tensor, 
                                     labels: torch.Tensor, 
                                     scores: torch.Tensor, 
                                     image_shape: Tuple[int, int, int]) -> List[Dict]:
        """Convert normalized coordinates to image coordinates"""
        height, width = image_shape[:2]
        
        detections = []
        for box, label, score in zip(boxes, labels, scores):
            # Convert normalized coordinates to pixel coordinates
            x1 = int(box[0].item() * width)
            y1 = int(box[1].item() * height)
            x2 = int(box[2].item() * width)
            y2 = int(box[3].item() * height)
            
            detection = {
                'bbox': [x1, y1, x2, y2],
                'label': label.item(),
                'score': score.item()
            }
            detections.append(detection)
        
        return detections
    
    def get_average_fps(self) -> float:
        """Get average FPS over the last 100 frames"""
        return np.mean(self.fps_counter) if self.fps_counter else 0
    
    def get_model_info(self) -> Dict:
        """Get information about the model"""
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        
        return {
            'input_size': self.config['input_size'],
            'num_classes': self.config['num_classes'],
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'default_boxes': len(self.default_boxes)
        }
    
    def visualize_detections(self, image_path: str, 
                           detections: List[Dict], 
                           save_path: Optional[str] = None):
        """
        Visualize detections on an image
        
        Args:
            image_path: Path to input image
            detections: List of detection dictionaries
            save_path: Path to save the annotated image
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        annotated_image = draw_detections(image, detections)
        
        if save_path:
            cv2.imwrite(save_path, annotated_image)
            print(f"Annotated image saved to {save_path}")
        
        return annotated_image
