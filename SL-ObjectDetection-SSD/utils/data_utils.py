import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple, List, Dict
import os


def preprocess_image(image: np.ndarray, input_size: int = 300) -> torch.Tensor:
    """
    Preprocess image for SSD network input
    
    Args:
        image: Input image as numpy array (BGR format)
        input_size: Target input size for the network
        
    Returns:
        Preprocessed image tensor
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image
    image_resized = cv2.resize(image_rgb, (input_size, input_size))
    
    # Normalize pixel values to [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Convert to tensor and change format from HWC to CHW
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1)
    
    # Apply ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    image_tensor = normalize(image_tensor)
    
    return image_tensor


def postprocess_image(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Postprocess image tensor back to numpy array
    
    Args:
        image_tensor: Preprocessed image tensor
        
    Returns:
        Postprocessed image as numpy array
    """
    # Denormalize
    denormalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    image_tensor = denormalize(image_tensor)
    
    # Convert to numpy and change format from CHW to HWC
    image_np = image_tensor.permute(1, 2, 0).numpy()
    
    # Clip values to [0, 1] and convert to uint8
    image_np = np.clip(image_np, 0, 1)
    image_np = (image_np * 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    return image_bgr


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from file path
    
    Args:
        image_path: Path to image file
        
    Returns:
        Loaded image as numpy array
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from: {image_path}")
    
    return image


def save_image(image: np.ndarray, output_path: str):
    """
    Save image to file
    
    Args:
        image: Image as numpy array
        output_path: Output file path
    """
    success = cv2.imwrite(output_path, image)
    if not success:
        raise ValueError(f"Could not save image to: {output_path}")


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image
        target_size: Target (width, height)
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create padded image
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Calculate padding
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    
    # Place resized image in padded image
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    
    return padded


def create_data_augmentation():
    """
    Create data augmentation transforms for training
    
    Returns:
        Data augmentation transforms
    """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(size=(300, 300), scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform


def create_validation_transform():
    """
    Create validation transforms
    
    Returns:
        Validation transforms
    """
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform


def load_video(video_path: str):
    """
    Load video file
    
    Args:
        video_path: Path to video file
        
    Returns:
        Video capture object
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    return cap


def get_video_info(video_path: str) -> Dict:
    """
    Get video information
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary containing video information
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info


def extract_frames(video_path: str, output_dir: str, frame_interval: int = 1):
    """
    Extract frames from video
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for frames
        frame_interval: Extract every nth frame
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames from video")


def create_video_from_frames(frame_dir: str, output_path: str, fps: int = 30):
    """
    Create video from frames
    
    Args:
        frame_dir: Directory containing frames
        output_path: Output video path
        fps: Frames per second
    """
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    
    if not frame_files:
        raise ValueError("No frame files found in directory")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        writer.write(frame)
    
    writer.release()
    print(f"Video created: {output_path}")


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU value
    """
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union


def non_max_suppression(boxes: List[List[float]], scores: List[float], 
                       iou_threshold: float = 0.5) -> List[int]:
    """
    Apply Non-Maximum Suppression
    
    Args:
        boxes: List of bounding boxes [x1, y1, x2, y2]
        scores: List of confidence scores
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Indices of kept boxes
    """
    if not boxes:
        return []
    
    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Sort by scores in descending order
    indices = np.argsort(scores)[::-1]
    
    keep = []
    
    while indices.size > 0:
        # Pick the box with highest score
        current = indices[0]
        keep.append(current)
        
        if indices.size == 1:
            break
        
        # Calculate IoU with remaining boxes
        remaining = indices[1:]
        ious = [calculate_iou(boxes[current], boxes[i]) for i in remaining]
        
        # Keep boxes with IoU below threshold
        indices = remaining[np.array(ious) < iou_threshold]
    
    return keep
