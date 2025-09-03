import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SSDLoss(nn.Module):
    """
    SSD Loss Function combining classification and localization losses
    """
    
    def __init__(self, num_classes, neg_pos_ratio=3, alpha=1.0):
        super(SSDLoss, self).__init__()
        self.num_classes = num_classes
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        
    def forward(self, predictions, targets, default_boxes):
        """
        Compute SSD loss
        
        Args:
            predictions: tuple of (loc_pred, conf_pred)
            targets: tuple of (loc_target, conf_target)
            default_boxes: default boxes for matching
            
        Returns:
            total_loss: combined classification and localization loss
        """
        loc_pred, conf_pred = predictions
        loc_target, conf_target = targets
        
        batch_size = loc_pred.size(0)
        num_default_boxes = loc_pred.size(1)
        
        # Reshape predictions
        loc_pred = loc_pred.view(batch_size, num_default_boxes, 4)
        conf_pred = conf_pred.view(batch_size, num_default_boxes, self.num_classes)
        
        # Compute classification loss
        conf_loss = self._compute_classification_loss(conf_pred, conf_target)
        
        # Compute localization loss
        loc_loss = self._compute_localization_loss(loc_pred, loc_target, conf_target)
        
        # Combine losses
        total_loss = conf_loss + self.alpha * loc_loss
        
        return total_loss, conf_loss, loc_loss
    
    def _compute_classification_loss(self, conf_pred, conf_target):
        """Compute classification loss with hard negative mining"""
        batch_size = conf_pred.size(0)
        num_default_boxes = conf_pred.size(1)
        
        # Flatten predictions and targets
        conf_pred_flat = conf_pred.view(-1, self.num_classes)
        conf_target_flat = conf_target.view(-1)
        
        # Find positive and negative samples
        pos_mask = conf_target_flat > 0
        neg_mask = conf_target_flat == 0
        
        # Compute cross entropy loss
        loss_c = F.cross_entropy(conf_pred_flat, conf_target_flat, reduction='none')
        
        # Hard negative mining
        num_pos = pos_mask.sum()
        num_neg = min(neg_mask.sum(), num_pos * self.neg_pos_ratio)
        
        if num_neg > 0:
            # Sort negative samples by loss
            neg_loss = loss_c[neg_mask]
            _, hard_neg_indices = torch.topk(neg_loss, num_neg)
            
            # Create mask for hard negatives
            hard_neg_mask = torch.zeros_like(neg_mask)
            hard_neg_mask[neg_mask][hard_neg_indices] = 1
            
            # Combine positive and hard negative masks
            final_mask = pos_mask | hard_neg_mask
        else:
            final_mask = pos_mask
        
        # Compute final classification loss
        if final_mask.sum() > 0:
            conf_loss = loss_c[final_mask].mean()
        else:
            conf_loss = torch.tensor(0.0, device=conf_pred.device)
        
        return conf_loss
    
    def _compute_localization_loss(self, loc_pred, loc_target, conf_target):
        """Compute localization loss using Smooth L1"""
        batch_size = loc_pred.size(0)
        num_default_boxes = loc_pred.size(1)
        
        # Create positive mask
        pos_mask = conf_target > 0
        
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=loc_pred.device)
        
        # Flatten predictions and targets
        loc_pred_flat = loc_pred.view(-1, 4)
        loc_target_flat = loc_target.view(-1, 4)
        
        # Apply positive mask
        loc_pred_pos = loc_pred_flat[pos_mask.view(-1)]
        loc_target_pos = loc_target_flat[pos_mask.view(-1)]
        
        # Compute Smooth L1 loss
        loc_loss = F.smooth_l1_loss(loc_pred_pos, loc_target_pos, reduction='mean')
        
        return loc_loss


class BoxMatcher:
    """
    Match ground truth boxes with default boxes using IoU
    """
    
    def __init__(self, iou_threshold=0.5, neg_threshold=0.3):
        self.iou_threshold = iou_threshold
        self.neg_threshold = neg_threshold
    
    def match_boxes(self, default_boxes, ground_truth_boxes, ground_truth_labels):
        """
        Match default boxes with ground truth boxes
        
        Args:
            default_boxes: (N, 4) default boxes in (cx, cy, w, h) format
            ground_truth_boxes: (M, 4) ground truth boxes in (cx, cy, w, h) format
            ground_truth_labels: (M,) ground truth labels
            
        Returns:
            matched_labels: (N,) matched labels (0 for background)
            matched_boxes: (N, 4) matched ground truth boxes
        """
        num_default = default_boxes.size(0)
        num_gt = ground_truth_boxes.size(0)
        
        # Convert to (x1, y1, x2, y2) format for IoU computation
        default_boxes_xyxy = self._cxcywh_to_xyxy(default_boxes)
        gt_boxes_xyxy = self._cxcywh_to_xyxy(ground_truth_boxes)
        
        # Compute IoU matrix
        iou_matrix = self._compute_iou(default_boxes_xyxy, gt_boxes_xyxy)
        
        # Find best matches for each default box
        best_gt_iou, best_gt_idx = torch.max(iou_matrix, dim=1)
        
        # Create matched labels and boxes
        matched_labels = torch.zeros(num_default, dtype=torch.long, device=default_boxes.device)
        matched_boxes = torch.zeros(num_default, 4, dtype=torch.float32, device=default_boxes.device)
        
        # Positive matches
        pos_mask = best_gt_iou >= self.iou_threshold
        matched_labels[pos_mask] = ground_truth_labels[best_gt_idx[pos_mask]]
        matched_boxes[pos_mask] = ground_truth_boxes[best_gt_idx[pos_mask]]
        
        # Negative matches
        neg_mask = best_gt_iou < self.neg_threshold
        matched_labels[neg_mask] = 0  # Background class
        
        return matched_labels, matched_boxes
    
    def _cxcywh_to_xyxy(self, boxes):
        """Convert (cx, cy, w, h) to (x1, y1, x2, y2) format"""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    def _compute_iou(self, boxes1, boxes2):
        """Compute IoU between two sets of boxes"""
        # Compute intersection
        x1 = torch.max(boxes1[:, 0:1], boxes2[:, 0])
        y1 = torch.max(boxes1[:, 1:2], boxes2[:, 1])
        x2 = torch.min(boxes1[:, 2:3], boxes2[:, 2])
        y2 = torch.min(boxes1[:, 3:4], boxes2[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Compute union
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1.unsqueeze(1) + area2 - intersection
        
        # Compute IoU
        iou = intersection / (union + 1e-6)
        
        return iou


class BoxEncoder:
    """
    Encode/decode bounding boxes for SSD training and inference
    """
    
    def __init__(self, default_boxes, variance=[0.1, 0.2]):
        self.default_boxes = default_boxes
        self.variance = variance
    
    def encode(self, ground_truth_boxes, ground_truth_labels):
        """
        Encode ground truth boxes to default box format
        
        Args:
            ground_truth_boxes: (N, 4) in (cx, cy, w, h) format
            ground_truth_labels: (N,) labels
            
        Returns:
            encoded_boxes: (num_default, 4) encoded boxes
            encoded_labels: (num_default,) encoded labels
        """
        num_default = self.default_boxes.size(0)
        num_gt = ground_truth_boxes.size(0)
        
        # Match boxes
        matcher = BoxMatcher()
        matched_labels, matched_boxes = matcher.match_boxes(
            self.default_boxes, ground_truth_boxes, ground_truth_labels
        )
        
        # Encode boxes
        encoded_boxes = self._encode_boxes(matched_boxes)
        
        return encoded_boxes, matched_labels
    
    def decode(self, loc_pred, conf_pred, confidence_threshold=0.5):
        """
        Decode network predictions to bounding boxes
        
        Args:
            loc_pred: (batch_size, num_default, 4) location predictions
            conf_pred: (batch_size, num_default, num_classes) confidence predictions
            confidence_threshold: minimum confidence threshold
            
        Returns:
            decoded_boxes: list of (N, 4) boxes for each image
            decoded_labels: list of (N,) labels for each image
            decoded_scores: list of (N,) scores for each image
        """
        batch_size = loc_pred.size(0)
        num_default = loc_pred.size(1)
        num_classes = conf_pred.size(2)
        
        decoded_boxes = []
        decoded_labels = []
        decoded_scores = []
        
        for i in range(batch_size):
            # Get predictions for current image
            loc = loc_pred[i]  # (num_default, 4)
            conf = conf_pred[i]  # (num_default, num_classes)
            
            # Decode location predictions
            boxes = self._decode_boxes(loc)
            
            # Get confidence scores and labels
            scores, labels = torch.max(conf, dim=1)
            
            # Apply confidence threshold
            mask = scores > confidence_threshold
            boxes = boxes[mask]
            labels = labels[mask]
            scores = scores[mask]
            
            decoded_boxes.append(boxes)
            decoded_labels.append(labels)
            decoded_scores.append(scores)
        
        return decoded_boxes, decoded_labels, decoded_scores
    
    def _encode_boxes(self, matched_boxes):
        """Encode matched boxes to default box format"""
        # Convert to (cx, cy, w, h) format
        default_cx = self.default_boxes[:, 0]
        default_cy = self.default_boxes[:, 1]
        default_w = self.default_boxes[:, 2]
        default_h = self.default_boxes[:, 3]
        
        matched_cx = matched_boxes[:, 0]
        matched_cy = matched_boxes[:, 1]
        matched_w = matched_boxes[:, 2]
        matched_h = matched_boxes[:, 3]
        
        # Encode
        encoded_cx = (matched_cx - default_cx) / (default_w * self.variance[0])
        encoded_cy = (matched_cy - default_cy) / (default_h * self.variance[0])
        encoded_w = torch.log(matched_w / default_w) / self.variance[1]
        encoded_h = torch.log(matched_h / default_h) / self.variance[1]
        
        return torch.stack([encoded_cx, encoded_cy, encoded_w, encoded_h], dim=1)
    
    def _decode_boxes(self, loc_pred):
        """Decode location predictions to bounding boxes"""
        # Convert to (cx, cy, w, h) format
        default_cx = self.default_boxes[:, 0]
        default_cy = self.default_boxes[:, 1]
        default_w = self.default_boxes[:, 2]
        default_h = self.default_boxes[:, 3]
        
        # Decode
        decoded_cx = default_cx + loc_pred[:, 0] * default_w * self.variance[0]
        decoded_cy = default_cy + loc_pred[:, 1] * default_h * self.variance[0]
        decoded_w = default_w * torch.exp(loc_pred[:, 2] * self.variance[1])
        decoded_h = default_h * torch.exp(loc_pred[:, 3] * self.variance[1])
        
        # Convert to (x1, y1, x2, y2) format
        x1 = decoded_cx - decoded_w / 2
        y1 = decoded_cy - decoded_h / 2
        x2 = decoded_cx + decoded_w / 2
        y2 = decoded_cy + decoded_h / 2
        
        return torch.stack([x1, y1, x2, y2], dim=1)
