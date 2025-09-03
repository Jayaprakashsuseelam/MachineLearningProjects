import numpy as np
import torch
from typing import List, Dict, Tuple
from sklearn.metrics import precision_recall_curve, average_precision_score
import time


class DetectionMetrics:
    """
    Calculate various detection metrics for SSD evaluation
    """
    
    def __init__(self, num_classes: int, iou_threshold: float = 0.5):
        """
        Initialize metrics calculator
        
        Args:
            num_classes: Number of classes (including background)
            iou_threshold: IoU threshold for positive detection
        """
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.tp = np.zeros(self.num_classes)
        self.fp = np.zeros(self.num_classes)
        self.fn = np.zeros(self.num_classes)
        self.total_gt = np.zeros(self.num_classes)
    
    def update(self, predictions: List[Dict], ground_truth: List[Dict]):
        """
        Update metrics with new predictions and ground truth
        
        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries
        """
        # Group predictions by class
        pred_by_class = self._group_by_class(predictions)
        gt_by_class = self._group_by_class(ground_truth)
        
        # Calculate metrics for each class
        for class_id in range(self.num_classes):
            if class_id == 0:  # Skip background class
                continue
                
            pred_boxes = pred_by_class.get(class_id, [])
            gt_boxes = gt_by_class.get(class_id, [])
            
            self.total_gt[class_id] += len(gt_boxes)
            
            if len(gt_boxes) == 0:
                # No ground truth, all predictions are false positives
                self.fp[class_id] += len(pred_boxes)
            elif len(pred_boxes) == 0:
                # No predictions, all ground truth are false negatives
                self.fn[class_id] += len(gt_boxes)
            else:
                # Calculate IoU matrix
                iou_matrix = self._calculate_iou_matrix(pred_boxes, gt_boxes)
                
                # Find best matches
                matched_gt = set()
                matched_pred = set()
                
                # Sort predictions by confidence (descending)
                pred_indices = np.argsort([p['score'] for p in pred_boxes])[::-1]
                
                for pred_idx in pred_indices:
                    best_gt_idx = np.argmax(iou_matrix[pred_idx])
                    best_iou = iou_matrix[pred_idx, best_gt_idx]
                    
                    if best_iou >= self.iou_threshold and best_gt_idx not in matched_gt:
                        # True positive
                        self.tp[class_id] += 1
                        matched_gt.add(best_gt_idx)
                        matched_pred.add(pred_idx)
                    else:
                        # False positive
                        self.fp[class_id] += 1
                
                # False negatives (unmatched ground truth)
                self.fn[class_id] += len(gt_boxes) - len(matched_gt)
    
    def compute(self) -> Dict:
        """
        Compute final metrics
        
        Returns:
            Dictionary containing computed metrics
        """
        metrics = {}
        
        # Per-class metrics
        precision = np.zeros(self.num_classes)
        recall = np.zeros(self.num_classes)
        f1_score = np.zeros(self.num_classes)
        
        for class_id in range(1, self.num_classes):  # Skip background
            if self.tp[class_id] + self.fp[class_id] > 0:
                precision[class_id] = self.tp[class_id] / (self.tp[class_id] + self.fp[class_id])
            
            if self.total_gt[class_id] > 0:
                recall[class_id] = self.tp[class_id] / self.total_gt[class_id]
            
            if precision[class_id] + recall[class_id] > 0:
                f1_score[class_id] = 2 * precision[class_id] * recall[class_id] / (precision[class_id] + recall[class_id])
        
        # Overall metrics
        total_tp = np.sum(self.tp[1:])  # Exclude background
        total_fp = np.sum(self.fp[1:])
        total_fn = np.sum(self.fn[1:])
        total_gt = np.sum(self.total_gt[1:])
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / total_gt if total_gt > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        # Mean Average Precision (mAP)
        mean_precision = np.mean(precision[1:])  # Exclude background
        mean_recall = np.mean(recall[1:])
        mean_f1 = np.mean(f1_score[1:])
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tp': self.tp,
            'fp': self.fp,
            'fn': self.fn,
            'total_gt': self.total_gt,
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1': overall_f1,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'mean_f1': mean_f1,
            'mAP': mean_precision  # Simplified mAP
        }
        
        return metrics
    
    def _group_by_class(self, detections: List[Dict]) -> Dict:
        """Group detections by class ID"""
        grouped = {}
        for detection in detections:
            class_id = detection['label']
            if class_id not in grouped:
                grouped[class_id] = []
            grouped[class_id].append(detection)
        return grouped
    
    def _calculate_iou_matrix(self, pred_boxes: List[Dict], gt_boxes: List[Dict]) -> np.ndarray:
        """Calculate IoU matrix between predictions and ground truth"""
        num_pred = len(pred_boxes)
        num_gt = len(gt_boxes)
        iou_matrix = np.zeros((num_pred, num_gt))
        
        for i, pred in enumerate(pred_boxes):
            for j, gt in enumerate(gt_boxes):
                iou_matrix[i, j] = self._calculate_iou(pred['bbox'], gt['bbox'])
        
        return iou_matrix
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union


class PerformanceMetrics:
    """
    Calculate performance metrics like FPS, inference time, etc.
    """
    
    def __init__(self):
        self.inference_times = []
        self.fps_values = []
        self.memory_usage = []
    
    def update(self, inference_time: float, memory_usage: float = None):
        """
        Update performance metrics
        
        Args:
            inference_time: Time taken for inference in seconds
            memory_usage: Memory usage in MB (optional)
        """
        self.inference_times.append(inference_time)
        self.fps_values.append(1.0 / inference_time)
        
        if memory_usage is not None:
            self.memory_usage.append(memory_usage)
    
    def compute(self) -> Dict:
        """
        Compute performance statistics
        
        Returns:
            Dictionary containing performance metrics
        """
        if not self.inference_times:
            return {}
        
        metrics = {
            'avg_inference_time': np.mean(self.inference_times),
            'std_inference_time': np.std(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'avg_fps': np.mean(self.fps_values),
            'std_fps': np.std(self.fps_values),
            'min_fps': np.min(self.fps_values),
            'max_fps': np.max(self.fps_values),
            'total_frames': len(self.inference_times)
        }
        
        if self.memory_usage:
            metrics.update({
                'avg_memory_usage': np.mean(self.memory_usage),
                'max_memory_usage': np.max(self.memory_usage)
            })
        
        return metrics
    
    def reset(self):
        """Reset all performance metrics"""
        self.inference_times = []
        self.fps_values = []
        self.memory_usage = []


class ModelEvaluator:
    """
    Comprehensive model evaluator for SSD
    """
    
    def __init__(self, num_classes: int, iou_threshold: float = 0.5):
        """
        Initialize evaluator
        
        Args:
            num_classes: Number of classes
            iou_threshold: IoU threshold for evaluation
        """
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.detection_metrics = DetectionMetrics(num_classes, iou_threshold)
        self.performance_metrics = PerformanceMetrics()
    
    def evaluate_batch(self, predictions: List[List[Dict]], 
                      ground_truth: List[List[Dict]],
                      inference_times: List[float] = None):
        """
        Evaluate a batch of predictions
        
        Args:
            predictions: List of prediction lists for each image
            ground_truth: List of ground truth lists for each image
            inference_times: List of inference times (optional)
        """
        # Update detection metrics
        for pred, gt in zip(predictions, ground_truth):
            self.detection_metrics.update(pred, gt)
        
        # Update performance metrics
        if inference_times:
            for inference_time in inference_times:
                self.performance_metrics.update(inference_time)
    
    def get_results(self) -> Dict:
        """
        Get evaluation results
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        detection_results = self.detection_metrics.compute()
        performance_results = self.performance_metrics.compute()
        
        results = {**detection_results, **performance_results}
        
        return results
    
    def reset(self):
        """Reset all metrics"""
        self.detection_metrics.reset()
        self.performance_metrics.reset()
    
    def print_summary(self):
        """Print a summary of evaluation results"""
        results = self.get_results()
        
        print("=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        
        print(f"Detection Metrics:")
        print(f"  Overall Precision: {results.get('overall_precision', 0):.4f}")
        print(f"  Overall Recall: {results.get('overall_recall', 0):.4f}")
        print(f"  Overall F1-Score: {results.get('overall_f1', 0):.4f}")
        print(f"  mAP: {results.get('mAP', 0):.4f}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Average FPS: {results.get('avg_fps', 0):.2f}")
        print(f"  Average Inference Time: {results.get('avg_inference_time', 0)*1000:.2f} ms")
        print(f"  Total Frames Processed: {results.get('total_frames', 0)}")
        
        if 'avg_memory_usage' in results:
            print(f"  Average Memory Usage: {results.get('avg_memory_usage', 0):.2f} MB")
        
        print("=" * 50)


def calculate_map(predictions: List[List[Dict]], 
                 ground_truth: List[List[Dict]], 
                 num_classes: int,
                 iou_threshold: float = 0.5) -> float:
    """
    Calculate mean Average Precision (mAP)
    
    Args:
        predictions: List of prediction lists for each image
        ground_truth: List of ground truth lists for each image
        num_classes: Number of classes
        iou_threshold: IoU threshold
        
    Returns:
        mAP value
    """
    evaluator = ModelEvaluator(num_classes, iou_threshold)
    evaluator.evaluate_batch(predictions, ground_truth)
    results = evaluator.get_results()
    return results['mAP']


def benchmark_model(model, test_loader, device: str = 'cpu') -> Dict:
    """
    Benchmark model performance
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to run on
        
    Returns:
        Benchmark results
    """
    model.eval()
    performance_metrics = PerformanceMetrics()
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            
            # Measure inference time
            start_time = time.time()
            output = model(data)
            inference_time = time.time() - start_time
            
            # Update metrics
            performance_metrics.update(inference_time)
            
            # Limit benchmarking to first 100 batches
            if batch_idx >= 100:
                break
    
    return performance_metrics.compute()
