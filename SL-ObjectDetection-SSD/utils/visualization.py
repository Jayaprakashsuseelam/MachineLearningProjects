import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple
import seaborn as sns


# COCO class names for visualization
COCO_CLASSES = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Color palette for different classes
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64),
    (64, 64, 0), (64, 0, 64), (0, 64, 64), (192, 0, 0), (0, 192, 0),
    (0, 0, 192), (192, 192, 0), (192, 0, 192), (0, 192, 192), (96, 0, 0)
]


def draw_detections(image: np.ndarray, detections: List[Dict], 
                   class_names: List[str] = None, 
                   confidence_threshold: float = 0.5) -> np.ndarray:
    """
    Draw bounding boxes and labels on image
    
    Args:
        image: Input image
        detections: List of detection dictionaries
        class_names: List of class names
        confidence_threshold: Minimum confidence threshold
        
    Returns:
        Image with detections drawn
    """
    if class_names is None:
        class_names = COCO_CLASSES
    
    # Create a copy of the image
    annotated_image = image.copy()
    
    for detection in detections:
        bbox = detection['bbox']
        label = detection['label']
        score = detection['score']
        
        # Skip if confidence is below threshold
        if score < confidence_threshold:
            continue
        
        # Get class name and color
        class_name = class_names[label] if label < len(class_names) else f'Class {label}'
        color = COLORS[label % len(COLORS)]
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        
        # Create label text
        label_text = f'{class_name}: {score:.2f}'
        
        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        # Draw label background
        cv2.rectangle(annotated_image, (x1, y1 - text_height - 10), 
                     (x1 + text_width, y1), color, -1)
        
        # Draw label text
        cv2.putText(annotated_image, label_text, (x1, y1 - 5), 
                   font, font_scale, (255, 255, 255), thickness)
    
    return annotated_image


def draw_detections_matplotlib(image: np.ndarray, detections: List[Dict],
                             class_names: List[str] = None,
                             confidence_threshold: float = 0.5,
                             figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Draw detections using matplotlib
    
    Args:
        image: Input image
        detections: List of detection dictionaries
        class_names: List of class names
        confidence_threshold: Minimum confidence threshold
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = COCO_CLASSES
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image_rgb)
    
    for detection in detections:
        bbox = detection['bbox']
        label = detection['label']
        score = detection['score']
        
        # Skip if confidence is below threshold
        if score < confidence_threshold:
            continue
        
        # Get class name and color
        class_name = class_names[label] if label < len(class_names) else f'Class {label}'
        color = [c/255 for c in COLORS[label % len(COLORS)]]
        
        # Create rectangle patch
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        rect = patches.Rectangle((x1, y1), width, height, 
                               linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        label_text = f'{class_name}: {score:.2f}'
        ax.text(x1, y1 - 5, label_text, color='white', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))
    
    ax.set_title('Object Detection Results')
    ax.axis('off')
    
    return fig


def plot_detection_statistics(detections: List[Dict], 
                            class_names: List[str] = None,
                            figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot detection statistics
    
    Args:
        detections: List of detection dictionaries
        class_names: List of class names
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = COCO_CLASSES
    
    # Count detections by class
    class_counts = {}
    class_scores = {}
    
    for detection in detections:
        label = detection['label']
        score = detection['score']
        
        class_name = class_names[label] if label < len(class_names) else f'Class {label}'
        
        if class_name not in class_counts:
            class_counts[class_name] = 0
            class_scores[class_name] = []
        
        class_counts[class_name] += 1
        class_scores[class_name].append(score)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot detection counts
    if class_counts:
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        ax1.bar(range(len(classes)), counts, color='skyblue')
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Number of Detections')
        ax1.set_title('Detection Counts by Class')
        ax1.set_xticks(range(len(classes)))
        ax1.set_xticklabels(classes, rotation=45, ha='right')
    
    # Plot average confidence scores
    if class_scores:
        avg_scores = [np.mean(class_scores[cls]) for cls in classes]
        
        ax2.bar(range(len(classes)), avg_scores, color='lightcoral')
        ax2.set_xlabel('Classes')
        ax2.set_ylabel('Average Confidence Score')
        ax2.set_title('Average Confidence by Class')
        ax2.set_xticks(range(len(classes)))
        ax2.set_xticklabels(classes, rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def create_detection_video(input_video_path: str, output_video_path: str,
                          detector, confidence_threshold: float = 0.5,
                          class_names: List[str] = None):
    """
    Create a video with detections drawn on each frame
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path to output video
        detector: SSD detector instance
        confidence_threshold: Minimum confidence threshold
        class_names: List of class names
    """
    if class_names is None:
        class_names = COCO_CLASSES
    
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects in frame
        detections = detector._detect_frame(frame, confidence_threshold)
        
        # Draw detections
        annotated_frame = draw_detections(frame, detections, class_names, confidence_threshold)
        
        # Add frame counter
        cv2.putText(annotated_frame, f'Frame: {frame_count}', (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Write frame
        writer.write(annotated_frame)
        
        frame_count += 1
        
        # Print progress
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")
    
    # Release resources
    cap.release()
    writer.release()
    
    print(f"Detection video saved to: {output_video_path}")


def visualize_model_performance(metrics: Dict, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Visualize model performance metrics
    
    Args:
        metrics: Dictionary containing performance metrics
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Plot training loss
    if 'train_loss' in metrics:
        axes[0, 0].plot(metrics['train_loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
    
    # Plot validation loss
    if 'val_loss' in metrics:
        axes[0, 1].plot(metrics['val_loss'])
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
    
    # Plot learning rate
    if 'learning_rate' in metrics:
        axes[0, 2].plot(metrics['learning_rate'])
        axes[0, 2].set_title('Learning Rate')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
    
    # Plot mAP
    if 'map' in metrics:
        axes[1, 0].plot(metrics['map'])
        axes[1, 0].set_title('mAP')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mAP')
    
    # Plot precision
    if 'precision' in metrics:
        axes[1, 1].plot(metrics['precision'])
        axes[1, 1].set_title('Precision')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Precision')
    
    # Plot recall
    if 'recall' in metrics:
        axes[1, 2].plot(metrics['recall'])
        axes[1, 2].set_title('Recall')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Recall')
    
    plt.tight_layout()
    return fig


def create_confusion_matrix(predictions: List[int], targets: List[int],
                           class_names: List[str] = None,
                           figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Create confusion matrix visualization
    
    Args:
        predictions: List of predicted labels
        targets: List of target labels
        class_names: List of class names
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = COCO_CLASSES
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(targets, predictions)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names[:len(cm)], 
               yticklabels=class_names[:len(cm)], ax=ax)
    
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    return fig


def plot_class_distribution(dataset_labels: List[int], 
                           class_names: List[str] = None,
                           figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot class distribution in dataset
    
    Args:
        dataset_labels: List of labels from dataset
        class_names: List of class names
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = COCO_CLASSES
    
    # Count occurrences of each class
    unique, counts = np.unique(dataset_labels, return_counts=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bar chart
    bars = ax.bar(range(len(unique)), counts, color='lightblue')
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Class Distribution in Dataset')
    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels([class_names[i] for i in unique], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig
