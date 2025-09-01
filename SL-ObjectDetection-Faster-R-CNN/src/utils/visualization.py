"""
Visualization utilities for Faster R-CNN
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import List, Dict, Tuple, Optional, Union
import os


def visualize_predictions(
    image: Union[torch.Tensor, np.ndarray, str],
    predictions: Dict[str, torch.Tensor],
    class_names: List[str],
    confidence_threshold: float = 0.5,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Visualize Faster R-CNN predictions on an image.
    
    Args:
        image: Input image (tensor, numpy array, or file path)
        predictions: Dictionary containing 'boxes', 'labels', 'scores'
        class_names: List of class names
        confidence_threshold: Minimum confidence to display
        save_path: Path to save the visualization
        show: Whether to display the plot
        figsize: Figure size for matplotlib
    """
    # Convert image to numpy array
    if isinstance(image, str):
        img = np.array(Image.open(image))
    elif isinstance(image, torch.Tensor):
        img = image.cpu().numpy()
        if img.shape[0] == 3:  # CHW format
            img = np.transpose(img, (1, 2, 0))
    else:
        img = np.array(image)
    
    # Normalize image to [0, 1] if needed
    if img.max() > 1.0:
        img = img.astype(np.uint8)
    else:
        img = (img * 255).astype(np.uint8)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img)
    
    # Get predictions
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    
    # Filter by confidence
    mask = scores >= confidence_threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]
    
    # Define colors for different classes
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    
    # Draw bounding boxes
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2,
            edgecolor=colors[label % len(colors)],
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        ax.text(
            x1, y1 - 5,
            f"{class_name}: {score:.2f}",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[label % len(colors)], alpha=0.7),
            fontsize=10,
            color='black'
        )
    
    ax.set_title(f"Faster R-CNN Predictions (Confidence ≥ {confidence_threshold})")
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Visualization saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_detections(
    image: Union[torch.Tensor, np.ndarray, str],
    detections: List[Dict[str, Union[torch.Tensor, float, int]]],
    class_names: List[str],
    confidence_threshold: float = 0.5,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot detections on an image with more detailed visualization.
    
    Args:
        image: Input image
        detections: List of detection dictionaries
        class_names: List of class names
        confidence_threshold: Minimum confidence to display
        save_path: Path to save the visualization
        show: Whether to display the plot
        figsize: Figure size for matplotlib
    """
    # Convert image to numpy array
    if isinstance(image, str):
        img = np.array(Image.open(image))
    elif isinstance(image, torch.Tensor):
        img = image.cpu().numpy()
        if img.shape[0] == 3:  # CHW format
            img = np.transpose(img, (1, 2, 0))
    else:
        img = np.array(image)
    
    # Normalize image to [0, 1] if needed
    if img.max() > 1.0:
        img = img.astype(np.uint8)
    else:
        img = (img * 255).astype(np.uint8)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img)
    
    # Define colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    # Draw detections
    for detection in detections:
        if detection['score'] < confidence_threshold:
            continue
            
        box = detection['box']
        label = detection['label']
        score = detection['score']
        
        x1, y1, x2, y2 = box
        
        # Create rectangle patch
        color = colors[label % len(colors)]
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label with background
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        text = f"{class_name}: {score:.3f}"
        
        # Create text box
        text_box = patches.FancyBboxPatch(
            (x1, y1 - 25), len(text) * 8, 20,
            boxstyle="round,pad=0.3",
            facecolor=color,
            alpha=0.8,
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(text_box)
        
        # Add text
        ax.text(
            x1 + 5, y1 - 10,
            text,
            fontsize=9,
            color='white',
            weight='bold'
        )
    
    ax.set_title(f"Object Detections (Confidence ≥ {confidence_threshold})")
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Detection plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_detection_video(
    video_path: str,
    model,
    class_names: List[str],
    output_path: str,
    confidence_threshold: float = 0.5,
    fps: int = 30,
    max_frames: Optional[int] = None,
    device: torch.device = torch.device('cpu')
) -> None:
    """
    Create a video with object detection overlays.
    
    Args:
        video_path: Path to input video
        model: Trained Faster R-CNN model
        class_names: List of class names
        output_path: Path to save output video
        confidence_threshold: Minimum confidence to display
        fps: Output video FPS
        max_frames: Maximum number of frames to process
        device: Device to run inference on
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_count = 0
    
    print(f"Processing video: {total_frames} frames")
    
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Preprocess image
        transform = get_transform(train=False)
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            predictions = model(input_tensor)
        
        # Draw detections
        frame_with_detections = draw_detections_on_frame(
            frame, predictions[0], class_names, confidence_threshold
        )
        
        # Write frame
        out.write(frame_with_detections)
        
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    # Clean up
    cap.release()
    out.release()
    print(f"Detection video saved to: {output_path}")


def draw_detections_on_frame(
    frame: np.ndarray,
    predictions: Dict[str, torch.Tensor],
    class_names: List[str],
    confidence_threshold: float = 0.5
) -> np.ndarray:
    """
    Draw detections on a single frame.
    
    Args:
        frame: Input frame (BGR format)
        predictions: Model predictions
        class_names: List of class names
        confidence_threshold: Minimum confidence to display
        
    Returns:
        Frame with detections drawn
    """
    frame_copy = frame.copy()
    
    # Get predictions
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    
    # Filter by confidence
    mask = scores >= confidence_threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]
    
    # Define colors (BGR format for OpenCV)
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (128, 128, 0),  # Olive
        (128, 0, 0),    # Maroon
    ]
    
    # Draw bounding boxes
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)
        
        # Get color
        color = colors[label % len(colors)]
        
        # Draw rectangle
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        label_text = f"{class_name}: {score:.2f}"
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Draw label background
        cv2.rectangle(
            frame_copy,
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            frame_copy,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    return frame_copy


def create_detection_grid(
    images: List[Union[torch.Tensor, np.ndarray, str]],
    predictions_list: List[Dict[str, torch.Tensor]],
    class_names: List[str],
    confidence_threshold: float = 0.5,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (20, 16)
) -> None:
    """
    Create a grid visualization of multiple images with detections.
    
    Args:
        images: List of input images
        predictions_list: List of predictions for each image
        class_names: List of class names
        confidence_threshold: Minimum confidence to display
        save_path: Path to save the visualization
        show: Whether to display the plot
        figsize: Figure size for matplotlib
    """
    n_images = len(images)
    n_cols = min(4, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_images == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Define colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    
    for i, (image, predictions) in enumerate(zip(images, predictions_list)):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Convert image to numpy array
        if isinstance(image, str):
            img = np.array(Image.open(image))
        elif isinstance(image, torch.Tensor):
            img = image.cpu().numpy()
            if img.shape[0] == 3:  # CHW format
                img = np.transpose(img, (1, 2, 0))
        else:
            img = np.array(image)
        
        # Normalize image
        if img.max() > 1.0:
            img = img.astype(np.uint8)
        else:
            img = (img * 255).astype(np.uint8)
        
        ax.imshow(img)
        
        # Get predictions
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        
        # Filter by confidence
        mask = scores >= confidence_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]
        
        # Draw bounding boxes
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2,
                edgecolor=colors[label % len(colors)],
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            class_name = class_names[label] if label < len(class_names) else f"Class {label}"
            ax.text(
                x1, y1 - 5,
                f"{class_name}: {score:.2f}",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[label % len(colors)], alpha=0.7),
                fontsize=8,
                color='black'
            )
        
        ax.set_title(f"Image {i+1}")
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"Detection Grid (Confidence ≥ {confidence_threshold})", fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Detection grid saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def save_detection_image(
    image: Union[torch.Tensor, np.ndarray, str],
    predictions: Dict[str, torch.Tensor],
    class_names: List[str],
    output_path: str,
    confidence_threshold: float = 0.5,
    image_size: Tuple[int, int] = (800, 600)
) -> None:
    """
    Save an image with detections drawn on it.
    
    Args:
        image: Input image
        predictions: Model predictions
        class_names: List of class names
        output_path: Path to save the output image
        confidence_threshold: Minimum confidence to display
        image_size: Output image size
    """
    # Convert image to PIL Image
    if isinstance(image, str):
        pil_image = Image.open(image)
    elif isinstance(image, torch.Tensor):
        img_array = image.cpu().numpy()
        if img_array.shape[0] == 3:  # CHW format
            img_array = np.transpose(img_array, (1, 2, 0))
        pil_image = Image.fromarray(img_array)
    else:
        pil_image = Image.fromarray(image)
    
    # Resize image
    pil_image = pil_image.resize(image_size, Image.Resampling.LANCZOS)
    
    # Create drawing object
    draw = ImageDraw.Draw(pil_image)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Get predictions
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    
    # Filter by confidence
    mask = scores >= confidence_threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]
    
    # Define colors
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (0, 128, 0),    # Dark Green
        (128, 0, 0),    # Maroon
    ]
    
    # Draw bounding boxes
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        
        # Scale coordinates to image size
        x1 = int(x1 * image_size[0])
        y1 = int(y1 * image_size[1])
        x2 = int(x2 * image_size[0])
        y2 = int(y2 * image_size[1])
        
        # Get color
        color = colors[label % len(colors)]
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Add label
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        label_text = f"{class_name}: {score:.2f}"
        
        # Draw label background
        bbox = draw.textbbox((x1, y1 - 20), label_text, font=font)
        draw.rectangle(bbox, fill=color)
        
        # Draw text
        draw.text((x1, y1 - 20), label_text, fill=(255, 255, 255), font=font)
    
    # Save image
    pil_image.save(output_path)
    print(f"Detection image saved to: {output_path}")


# Import the transform function for video processing
def get_transform(train: bool = True):
    """Placeholder for transform function - should be imported from data.transforms"""
    # This is a placeholder - in practice, this should be imported
    from ..data.transforms import get_transform as get_transform_actual
    return get_transform_actual(train=train)
