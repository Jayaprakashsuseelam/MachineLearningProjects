"""
Evaluation Metrics for YOLO
Comprehensive evaluation metrics for object detection
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU value
    """
    # Convert to numpy arrays
    box1 = np.array(box1)
    box2 = np.array(box2)
    
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0.0
    
    return iou

def calculate_precision_recall(predictions: List[Dict[str, Any]], 
                             ground_truth: List[Dict[str, Any]], 
                             iou_threshold: float = 0.5) -> Tuple[List[float], List[float]]:
    """
    Calculate precision and recall for object detection
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of ground truth dictionaries
        iou_threshold: IoU threshold for matching
        
    Returns:
        Tuple of (precision, recall) lists
    """
    # Group predictions and ground truth by class
    pred_by_class = defaultdict(list)
    gt_by_class = defaultdict(list)
    
    for pred in predictions:
        pred_by_class[pred['class_id']].append(pred)
    
    for gt in ground_truth:
        gt_by_class[gt['class_id']].append(gt)
    
    all_precisions = []
    all_recalls = []
    
    # Calculate for each class
    for class_id in set(pred_by_class.keys()) | set(gt_by_class.keys()):
        class_preds = pred_by_class[class_id]
        class_gts = gt_by_class[class_id]
        
        if len(class_preds) == 0:
            all_precisions.append(1.0)
            all_recalls.append(0.0)
            continue
        
        if len(class_gts) == 0:
            all_precisions.append(0.0)
            all_recalls.append(1.0)
            continue
        
        # Sort predictions by confidence
        class_preds.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Initialize arrays
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        gt_matched = [False] * len(class_gts)
        
        # Match predictions to ground truth
        for i, pred in enumerate(class_preds):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt in enumerate(class_gts):
                if gt_matched[j]:
                    continue
                
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold:
                tp[i] = 1
                gt_matched[best_gt_idx] = True
            else:
                fp[i] = 1
        
        # Calculate cumulative sums
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Calculate precision and recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / len(class_gts)
        
        all_precisions.extend(precision)
        all_recalls.extend(recall)
    
    return all_precisions, all_recalls

def calculate_map(predictions: List[Dict[str, Any]], 
                 ground_truth: List[Dict[str, Any]], 
                 iou_thresholds: List[float] = None,
                 class_names: List[str] = None) -> Dict[str, float]:
    """
    Calculate mean Average Precision (mAP)
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of ground truth dictionaries
        iou_thresholds: List of IoU thresholds
        class_names: List of class names
        
    Returns:
        Dictionary with mAP results
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75, 0.5]  # mAP@0.5, mAP@0.75, mAP@0.5:0.95
    
    if class_names is None:
        class_names = [f"class_{i}" for i in range(80)]
    
    # Group by class
    pred_by_class = defaultdict(list)
    gt_by_class = defaultdict(list)
    
    for pred in predictions:
        pred_by_class[pred['class_id']].append(pred)
    
    for gt in ground_truth:
        gt_by_class[gt['class_id']].append(gt)
    
    # Calculate AP for each class and IoU threshold
    aps = defaultdict(list)
    
    for iou_thresh in iou_thresholds:
        for class_id in range(len(class_names)):
            class_preds = pred_by_class[class_id]
            class_gts = gt_by_class[class_id]
            
            if len(class_preds) == 0 or len(class_gts) == 0:
                aps[iou_thresh].append(0.0)
                continue
            
            # Sort predictions by confidence
            class_preds.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Initialize arrays
            tp = np.zeros(len(class_preds))
            fp = np.zeros(len(class_preds))
            gt_matched = [False] * len(class_gts)
            
            # Match predictions to ground truth
            for i, pred in enumerate(class_preds):
                best_iou = 0
                best_gt_idx = -1
                
                for j, gt in enumerate(class_gts):
                    if gt_matched[j]:
                        continue
                    
                    iou = calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
                
                if best_iou >= iou_thresh:
                    tp[i] = 1
                    gt_matched[best_gt_idx] = True
                else:
                    fp[i] = 1
            
            # Calculate cumulative sums
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            # Calculate precision and recall
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / len(class_gts)
            
            # Calculate AP using interpolation
            ap = calculate_average_precision(recall, precision)
            aps[iou_thresh].append(ap)
    
    # Calculate mAP for each IoU threshold
    results = {}
    for iou_thresh in iou_thresholds:
        if len(aps[iou_thresh]) > 0:
            map_value = np.mean(aps[iou_thresh])
            results[f'mAP@{iou_thresh}'] = map_value
    
    # Calculate mAP@0.5:0.95 (COCO metric)
    if 0.5 in aps and 0.75 in aps:
        map_50_95 = np.mean([np.mean(aps[0.5]), np.mean(aps[0.75])])
        results['mAP@0.5:0.95'] = map_50_95
    
    return results

def calculate_average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Calculate Average Precision using interpolation
    
    Args:
        recall: Recall values
        precision: Precision values
        
    Returns:
        Average Precision
    """
    # Add sentinel values
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    
    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Calculate AP
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap

def calculate_f1_score(precision: float, recall: float) -> float:
    """
    Calculate F1 score
    
    Args:
        precision: Precision value
        recall: Recall value
        
    Returns:
        F1 score
    """
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

def evaluate_model(model, 
                  test_loader, 
                  device: str = 'cpu',
                  conf_threshold: float = 0.25,
                  iou_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Evaluate a trained model
    
    Args:
        model: Trained YOLO model
        test_loader: Test data loader
        device: Device to run evaluation on
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
        
    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    all_predictions = []
    all_ground_truth = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = images.to(device)
            
            # Forward pass
            predictions = model(images)
            
            # Process predictions and ground truth
            for i in range(len(images)):
                # Process predictions
                pred = predictions[i]
                # Convert predictions to detection format
                # This is a simplified version - you'll need to implement
                # the actual conversion based on your model's output format
                
                # Process ground truth
                target = targets[i]
                # Convert ground truth to detection format
                
                all_predictions.extend(pred)
                all_ground_truth.extend(target)
    
    # Calculate metrics
    results = calculate_map(all_predictions, all_ground_truth, 
                          iou_thresholds=[0.5, 0.75])
    
    return results

def plot_precision_recall_curve(predictions: List[Dict[str, Any]], 
                               ground_truth: List[Dict[str, Any]], 
                               class_names: List[str] = None,
                               save_path: Optional[str] = None):
    """
    Plot precision-recall curves
    
    Args:
        predictions: List of predictions
        ground_truth: List of ground truth
        class_names: List of class names
        save_path: Path to save plot
    """
    if class_names is None:
        class_names = [f"class_{i}" for i in range(80)]
    
    # Group by class
    pred_by_class = defaultdict(list)
    gt_by_class = defaultdict(list)
    
    for pred in predictions:
        pred_by_class[pred['class_id']].append(pred)
    
    for gt in ground_truth:
        gt_by_class[gt['class_id']].append(gt)
    
    # Plot for each class
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for class_id in range(min(4, len(class_names))):
        class_preds = pred_by_class[class_id]
        class_gts = gt_by_class[class_id]
        
        if len(class_preds) == 0 or len(class_gts) == 0:
            axes[class_id].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[class_id].set_title(f'{class_names[class_id]}')
            continue
        
        # Sort predictions by confidence
        class_preds.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Calculate precision and recall
        precision, recall = calculate_precision_recall(class_preds, class_gts)
        
        # Plot
        axes[class_id].plot(recall, precision, 'b-', linewidth=2)
        axes[class_id].set_xlabel('Recall')
        axes[class_id].set_ylabel('Precision')
        axes[class_id].set_title(f'{class_names[class_id]}')
        axes[class_id].grid(True)
        axes[class_id].set_xlim([0, 1])
        axes[class_id].set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Precision-recall curves saved to {save_path}")
    else:
        plt.show()

def plot_confusion_matrix(predictions: List[Dict[str, Any]], 
                         ground_truth: List[Dict[str, Any]], 
                         class_names: List[str] = None,
                         save_path: Optional[str] = None):
    """
    Plot confusion matrix
    
    Args:
        predictions: List of predictions
        ground_truth: List of ground truth
        class_names: List of class names
        save_path: Path to save plot
    """
    if class_names is None:
        class_names = [f"class_{i}" for i in range(80)]
    
    num_classes = len(class_names)
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    # Create confusion matrix
    for pred in predictions:
        pred_class = pred['class_id']
        
        # Find best matching ground truth
        best_iou = 0
        best_gt_class = -1
        
        for gt in ground_truth:
            iou = calculate_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou and iou > 0.5:  # IoU threshold
                best_iou = iou
                best_gt_class = gt['class_id']
        
        if best_gt_class >= 0:
            confusion_matrix[best_gt_class, pred_class] += 1
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names[:10], yticklabels=class_names[:10])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

def calculate_detection_metrics(predictions: List[Dict[str, Any]], 
                               ground_truth: List[Dict[str, Any]], 
                               iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate comprehensive detection metrics
    
    Args:
        predictions: List of predictions
        ground_truth: List of ground truth
        iou_threshold: IoU threshold
        
    Returns:
        Dictionary with metrics
    """
    # Group by class
    pred_by_class = defaultdict(list)
    gt_by_class = defaultdict(list)
    
    for pred in predictions:
        pred_by_class[pred['class_id']].append(pred)
    
    for gt in ground_truth:
        gt_by_class[gt['class_id']].append(gt)
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Calculate metrics for each class
    for class_id in set(pred_by_class.keys()) | set(gt_by_class.keys()):
        class_preds = pred_by_class[class_id]
        class_gts = gt_by_class[class_id]
        
        if len(class_preds) == 0:
            total_fn += len(class_gts)
            continue
        
        if len(class_gts) == 0:
            total_fp += len(class_preds)
            continue
        
        # Sort predictions by confidence
        class_preds.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Match predictions to ground truth
        gt_matched = [False] * len(class_gts)
        
        for pred in class_preds:
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt in enumerate(class_gts):
                if gt_matched[j]:
                    continue
                
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold:
                total_tp += 1
                gt_matched[best_gt_idx] = True
            else:
                total_fp += 1
        
        # Count unmatched ground truth as false negatives
        total_fn += sum(1 for matched in gt_matched if not matched)
    
    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = calculate_f1_score(precision, recall)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn
    }

# Convenience functions
def evaluate_detections(predictions: List[Dict[str, Any]], 
                       ground_truth: List[Dict[str, Any]], 
                       **kwargs) -> Dict[str, Any]:
    """Quick evaluation function"""
    results = {}
    
    # Calculate mAP
    map_results = calculate_map(predictions, ground_truth, **kwargs)
    results.update(map_results)
    
    # Calculate other metrics
    detection_metrics = calculate_detection_metrics(predictions, ground_truth, **kwargs)
    results.update(detection_metrics)
    
    return results
