"""
Evaluation metrics for object detection
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU) between two sets of boxes
    
    Args:
        box1: First set of boxes [N, 4] in (x1, y1, x2, y2) format
        box2: Second set of boxes [M, 4] in (x1, y1, x2, y2) format
    
    Returns:
        IoU matrix [N, M]
    """
    # Calculate areas
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    # Calculate intersection
    lt = torch.max(box1[:, None, :2], box2[:, :2])  # left-top
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # right-bottom
    
    wh = (rb - lt).clamp(min=0)  # width, height
    inter = wh[:, :, 0] * wh[:, :, 1]  # intersection
    
    # Calculate union
    union = area1[:, None] + area2 - inter
    
    # Calculate IoU
    iou = inter / union
    
    return iou


def calculate_map(predictions: List[Dict], targets: List[Dict], 
                 iou_thresholds: List[float] = [0.5], num_classes: int = 21) -> Dict[str, float]:
    """
    Calculate mean Average Precision (mAP) for object detection
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        iou_thresholds: List of IoU thresholds
        num_classes: Number of classes (including background)
    
    Returns:
        Dictionary containing mAP values for each IoU threshold
    """
    # Initialize results
    results = {}
    
    for iou_thresh in iou_thresholds:
        # Calculate AP for each class
        aps = []
        
        for class_id in range(1, num_classes):  # Skip background class
            ap = calculate_ap_per_class(predictions, targets, class_id, iou_thresh)
            aps.append(ap)
        
        # Calculate mAP
        mAP = np.mean(aps)
        
        # Store results
        if iou_thresh == 0.5:
            results[f'mAP_0.5'] = mAP
        elif iou_thresh == 0.75:
            results[f'mAP_0.75'] = mAP
        else:
            results[f'mAP_{iou_thresh}'] = mAP
    
    # Calculate mAP across all IoU thresholds
    if len(iou_thresholds) > 1:
        all_aps = []
        for iou_thresh in iou_thresholds:
            for class_id in range(1, num_classes):
                ap = calculate_ap_per_class(predictions, targets, class_id, iou_thresh)
                all_aps.append(ap)
        results['mAP_0.5_0.95'] = np.mean(all_aps)
    
    return results


def calculate_ap_per_class(predictions: List[Dict], targets: List[Dict], 
                          class_id: int, iou_threshold: float) -> float:
    """
    Calculate Average Precision (AP) for a specific class and IoU threshold
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        class_id: Class ID to evaluate
        iou_threshold: IoU threshold
    
    Returns:
        AP value for the class
    """
    # Collect all predictions and targets for this class
    all_predictions = []
    all_targets = []
    
    for pred, target in zip(predictions, targets):
        # Get predictions for this class
        if pred['labels'].numel() > 0:
            class_mask = pred['labels'] == class_id
            if class_mask.any():
                class_preds = {
                    'boxes': pred['boxes'][class_mask],
                    'scores': pred['scores'][class_mask]
                }
                all_predictions.append(class_preds)
            else:
                all_predictions.append({'boxes': torch.empty(0, 4), 'scores': torch.empty(0)})
        else:
            all_predictions.append({'boxes': torch.empty(0, 4), 'scores': torch.empty(0)})
        
        # Get targets for this class
        if target['labels'].numel() > 0:
            class_mask = target['labels'] == class_id
            if class_mask.any():
                class_targets = {
                    'boxes': target['boxes'][class_mask],
                    'image_id': target['image_id']
                }
                all_targets.append(class_targets)
            else:
                all_targets.append({'boxes': torch.empty(0, 4), 'image_id': target['image_id']})
        else:
            all_targets.append({'boxes': torch.empty(0, 4), 'image_id': target['image_id']})
    
    # Flatten predictions and targets
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_image_ids = []
    
    for i, pred in enumerate(all_predictions):
        if pred['boxes'].numel() > 0:
            all_pred_boxes.append(pred['boxes'])
            all_pred_scores.append(pred['scores'])
            all_pred_image_ids.extend([i] * pred['boxes'].shape[0])
    
    if not all_pred_boxes:
        return 0.0
    
    all_pred_boxes = torch.cat(all_pred_boxes, dim=0)
    all_pred_scores = torch.cat(all_pred_scores, dim=0)
    all_pred_image_ids = torch.tensor(all_pred_image_ids)
    
    # Sort predictions by score
    sorted_indices = torch.argsort(all_pred_scores, descending=True)
    all_pred_boxes = all_pred_boxes[sorted_indices]
    all_pred_scores = all_pred_scores[sorted_indices]
    all_pred_image_ids = all_pred_image_ids[sorted_indices]
    
    # Calculate precision and recall
    tp = torch.zeros(len(all_pred_boxes), dtype=torch.bool)
    fp = torch.zeros(len(all_pred_boxes), dtype=torch.bool)
    
    # Track which targets have been matched
    matched_targets = defaultdict(set)
    
    for pred_idx in range(len(all_pred_boxes)):
        pred_box = all_pred_boxes[pred_idx]
        pred_image_id = all_pred_image_ids[pred_idx].item()
        
        # Get targets for this image
        target_boxes = all_targets[pred_image_id]['boxes']
        
        if target_boxes.numel() == 0:
            # No targets in this image
            fp[pred_idx] = True
            continue
        
        # Calculate IoU with all targets
        ious = calculate_iou(pred_box.unsqueeze(0), target_boxes).squeeze(0)
        
        # Find best matching target
        max_iou, max_idx = torch.max(ious, dim=0)
        
        if max_iou >= iou_threshold and max_idx.item() not in matched_targets[pred_image_id]:
            # True positive
            tp[pred_idx] = True
            matched_targets[pred_image_id].add(max_idx.item())
        else:
            # False positive
            fp[pred_idx] = True
    
    # Calculate cumulative sums
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    
    # Calculate precision and recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    recall = tp_cumsum / (torch.sum(torch.tensor([len(t['boxes']) for t in all_targets])) + 1e-8)
    
    # Calculate AP using 11-point interpolation
    ap = calculate_ap_11_point(precision, recall)
    
    return ap.item()


def calculate_ap_11_point(precision: torch.Tensor, recall: torch.Tensor) -> torch.Tensor:
    """
    Calculate AP using 11-point interpolation
    
    Args:
        precision: Precision values
        recall: Recall values
    
    Returns:
        AP value
    """
    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if torch.sum(recall >= t) == 0:
            p = 0
        else:
            p = torch.max(precision[recall >= t])
        ap = ap + p / 11.0
    
    return ap


def calculate_precision_recall(predictions: List[Dict], targets: List[Dict], 
                              class_id: int, iou_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate precision-recall curve for a specific class
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        class_id: Class ID to evaluate
        iou_threshold: IoU threshold
    
    Returns:
        Tuple of (precision, recall, thresholds)
    """
    # Collect all predictions and targets for this class
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_image_ids = []
    
    for i, pred in enumerate(predictions):
        if pred['labels'].numel() > 0:
            class_mask = pred['labels'] == class_id
            if class_mask.any():
                all_pred_boxes.append(pred['boxes'][class_mask])
                all_pred_scores.append(pred['scores'][class_mask])
                all_pred_image_ids.extend([i] * class_mask.sum().item())
    
    if not all_pred_boxes:
        return np.array([]), np.array([]), np.array([])
    
    all_pred_boxes = torch.cat(all_pred_boxes, dim=0)
    all_pred_scores = torch.cat(all_pred_scores, dim=0)
    all_pred_image_ids = torch.tensor(all_pred_image_ids)
    
    # Sort predictions by score
    sorted_indices = torch.argsort(all_pred_scores, descending=True)
    all_pred_boxes = all_pred_boxes[sorted_indices]
    all_pred_scores = all_pred_scores[sorted_indices]
    all_pred_image_ids = all_pred_image_ids[sorted_indices]
    
    # Calculate precision and recall
    tp = torch.zeros(len(all_pred_boxes), dtype=torch.bool)
    fp = torch.zeros(len(all_pred_boxes), dtype=torch.bool)
    
    # Track which targets have been matched
    matched_targets = defaultdict(set)
    
    for pred_idx in range(len(all_pred_boxes)):
        pred_box = all_pred_boxes[pred_idx]
        pred_image_id = all_pred_image_ids[pred_idx].item()
        
        # Get targets for this image
        target_boxes = targets[pred_image_id]['boxes']
        target_labels = targets[pred_image_id]['labels']
        
        if target_boxes.numel() == 0:
            fp[pred_idx] = True
            continue
        
        # Find targets of the same class
        class_mask = target_labels == class_id
        if not class_mask.any():
            fp[pred_idx] = True
            continue
        
        class_target_boxes = target_boxes[class_mask]
        
        # Calculate IoU with targets of the same class
        ious = calculate_iou(pred_box.unsqueeze(0), class_target_boxes).squeeze(0)
        
        # Find best matching target
        max_iou, max_idx = torch.max(ious, dim=0)
        
        if max_iou >= iou_threshold and max_idx.item() not in matched_targets[pred_image_id]:
            tp[pred_idx] = True
            matched_targets[pred_image_id].add(max_idx.item())
        else:
            fp[pred_idx] = True
    
    # Calculate cumulative sums
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    
    # Calculate precision and recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    recall = tp_cumsum / (torch.sum(torch.tensor([torch.sum(t['labels'] == class_id) for t in targets])) + 1e-8)
    
    # Convert to numpy arrays
    precision = precision.numpy()
    recall = recall.numpy()
    scores = all_pred_scores.numpy()
    
    return precision, recall, scores


def plot_precision_recall_curve(predictions: List[Dict], targets: List[Dict], 
                               class_names: List[str], iou_threshold: float = 0.5,
                               save_path: Optional[str] = None):
    """
    Plot precision-recall curves for all classes
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        class_names: List of class names
        iou_threshold: IoU threshold
        save_path: Path to save the plot
    """
    num_classes = len(class_names)
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    
    for class_id in range(1, num_classes):  # Skip background
        ax = axes[class_id - 1]
        
        # Calculate precision-recall curve
        precision, recall, scores = calculate_precision_recall(
            predictions, targets, class_id, iou_threshold
        )
        
        if len(precision) > 0:
            # Plot curve
            ax.plot(recall, precision, linewidth=2)
            
            # Calculate AP
            ap = calculate_ap_per_class(predictions, targets, class_id, iou_threshold)
            
            # Set title and labels
            ax.set_title(f'{class_names[class_id]}\nAP = {ap:.3f}')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        else:
            ax.set_title(f'{class_names[class_id]}\nNo predictions')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
    
    # Remove extra subplots
    for i in range(num_classes - 1, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-recall curves saved to {save_path}")
    
    plt.show()


def calculate_class_wise_metrics(predictions: List[Dict], targets: List[Dict], 
                                num_classes: int = 21, iou_threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
    """
    Calculate class-wise metrics
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        num_classes: Number of classes
        iou_threshold: IoU threshold
    
    Returns:
        Dictionary containing metrics for each class
    """
    class_metrics = {}
    
    for class_id in range(1, num_classes):  # Skip background
        # Calculate AP
        ap = calculate_ap_per_class(predictions, targets, class_id, iou_threshold)
        
        # Calculate precision and recall
        precision, recall, _ = calculate_precision_recall(predictions, targets, class_id, iou_threshold)
        
        if len(precision) > 0:
            # Find precision at different recall levels
            precision_at_recall_50 = 0.0
            precision_at_recall_75 = 0.0
            
            if len(recall) > 0:
                # Find precision at recall = 0.5
                recall_50_idx = np.argmin(np.abs(recall - 0.5))
                precision_at_recall_50 = precision[recall_50_idx]
                
                # Find precision at recall = 0.75
                recall_75_idx = np.argmin(np.abs(recall - 0.75))
                precision_at_recall_75 = precision[recall_75_idx]
            
            class_metrics[f'class_{class_id}'] = {
                'AP': ap,
                'precision_at_recall_50': precision_at_recall_50,
                'precision_at_recall_75': precision_at_recall_75,
                'max_precision': np.max(precision),
                'max_recall': np.max(recall)
            }
        else:
            class_metrics[f'class_{class_id}'] = {
                'AP': 0.0,
                'precision_at_recall_50': 0.0,
                'precision_at_recall_75': 0.0,
                'max_precision': 0.0,
                'max_recall': 0.0
            }
    
    return class_metrics


def print_evaluation_summary(predictions: List[Dict], targets: List[Dict], 
                           class_names: List[str], iou_thresholds: List[float] = [0.5]):
    """
    Print comprehensive evaluation summary
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        class_names: List of class names
        iou_thresholds: List of IoU thresholds
    """
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    # Calculate overall mAP
    overall_metrics = calculate_map(predictions, targets, iou_thresholds, len(class_names))
    
    print(f"Overall Performance:")
    for metric_name, metric_value in overall_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    print("\n" + "=" * 80)
    print("Class-wise Performance (IoU = 0.5)")
    print("=" * 80)
    
    # Calculate class-wise metrics
    class_metrics = calculate_class_wise_metrics(predictions, targets, len(class_names), 0.5)
    
    # Print table header
    print(f"{'Class':<15} {'AP':<8} {'P@R50':<8} {'P@R75':<8} {'Max P':<8} {'Max R':<8}")
    print("-" * 80)
    
    # Print class-wise results
    for class_id in range(1, len(class_names)):
        metrics = class_metrics[f'class_{class_id}']
        class_name = class_names[class_id]
        
        print(f"{class_name:<15} {metrics['AP']:<8.3f} {metrics['precision_at_recall_50']:<8.3f} "
              f"{metrics['precision_at_recall_75']:<8.3f} {metrics['max_precision']:<8.3f} "
              f"{metrics['max_recall']:<8.3f}")
    
    print("=" * 80)
    
    # Calculate total predictions and targets
    total_predictions = sum(len(pred['boxes']) for pred in predictions)
    total_targets = sum(len(target['boxes']) for target in targets)
    
    print(f"Total Predictions: {total_predictions}")
    print(f"Total Targets: {total_targets}")
    print("=" * 80)
