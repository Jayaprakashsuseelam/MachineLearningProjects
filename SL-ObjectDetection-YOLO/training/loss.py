"""
YOLO Loss Functions
Comprehensive implementation of YOLO loss functions for training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any
import math

class YOLOLoss(nn.Module):
    """Complete YOLO loss function"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Loss weights
        self.lambda_coord = config.get('lambda_coord', 5.0)
        self.lambda_noobj = config.get('lambda_noobj', 0.5)
        self.lambda_obj = config.get('lambda_obj', 1.0)
        self.lambda_class = config.get('lambda_class', 1.0)
        
        # Anchor boxes
        self.anchors = torch.tensor(config.get('anchors', [
            [(10, 13), (16, 30), (33, 23)],      # P3/8
            [(30, 61), (62, 45), (59, 119)],     # P4/16
            [(116, 90), (156, 198), (373, 326)]  # P5/32
        ])).float()
        
        # Strides
        self.strides = torch.tensor([8, 16, 32])
        
        # Number of classes
        self.num_classes = config.get('num_classes', 80)
        
        # IoU threshold for positive samples
        self.iou_threshold = config.get('iou_threshold', 0.5)
        
        # Label smoothing
        self.label_smoothing = config.get('label_smoothing', 0.0)
        
        # Focal loss parameters
        self.focal_alpha = config.get('focal_alpha', 0.25)
        self.focal_gamma = config.get('focal_gamma', 2.0)
        
        # BCE loss for confidence
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        # MSE loss for coordinates
        self.mse_loss = nn.MSELoss(reduction='none')
        
        # Cross entropy loss for classification
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', label_smoothing=self.label_smoothing)
    
    def forward(self, predictions: List[torch.Tensor], targets: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute YOLO loss
        
        Args:
            predictions: List of prediction tensors from different scales
            targets: List of target tensors for different scales
            
        Returns:
            Dictionary containing different loss components
        """
        device = predictions[0].device
        
        # Initialize loss components
        total_loss = torch.tensor(0.0, device=device)
        coord_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)
        noobj_loss = torch.tensor(0.0, device=device)
        class_loss = torch.tensor(0.0, device=device)
        
        # Process each scale
        for scale_idx, (pred, target) in enumerate(zip(predictions, targets)):
            batch_size, _, grid_h, grid_w = pred.shape
            
            # Reshape predictions
            pred = pred.view(batch_size, 3, self.num_classes + 5, grid_h, grid_w)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            
            # Extract components
            pred_xy = pred[..., :2]
            pred_wh = pred[..., 2:4]
            pred_conf = pred[..., 4]
            pred_cls = pred[..., 5:]
            
            # Create target mask
            obj_mask = target[..., 4] > 0
            noobj_mask = ~obj_mask
            
            # Number of positive samples
            num_pos = obj_mask.sum()
            
            if num_pos > 0:
                # Coordinate loss (only for positive samples)
                target_xy = target[..., :2]
                target_wh = target[..., 2:4]
                
                # Scale coordinates to grid
                pred_xy_scaled = torch.sigmoid(pred_xy)
                pred_wh_scaled = torch.exp(pred_wh)
                
                # Coordinate loss
                xy_loss = self.mse_loss(pred_xy_scaled, target_xy)
                wh_loss = self.mse_loss(pred_wh_scaled, target_wh)
                
                coord_loss += (xy_loss + wh_loss)[obj_mask].sum()
                
                # Objectness loss (positive samples)
                target_conf = target[..., 4]
                obj_loss += self.bce_loss(pred_conf, target_conf)[obj_mask].sum()
                
                # Classification loss (positive samples)
                target_cls = target[..., 5:].argmax(dim=-1)
                class_loss += self.ce_loss(pred_cls[obj_mask], target_cls[obj_mask]).sum()
            
            # Objectness loss (negative samples)
            if noobj_mask.sum() > 0:
                target_conf_neg = target[..., 4]
                noobj_loss += self.bce_loss(pred_conf, target_conf_neg)[noobj_mask].sum()
        
        # Combine losses
        total_loss = (
            self.lambda_coord * coord_loss +
            self.lambda_obj * obj_loss +
            self.lambda_noobj * noobj_loss +
            self.lambda_class * class_loss
        )
        
        return {
            'total_loss': total_loss,
            'coord_loss': coord_loss,
            'obj_loss': obj_loss,
            'noobj_loss': noobj_loss,
            'class_loss': class_loss
        }

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss
        
        Args:
            inputs: Predicted logits
            targets: Target labels
            
        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class IoULoss(nn.Module):
    """IoU-based loss for bounding box regression"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU loss
        
        Args:
            pred_boxes: Predicted bounding boxes (x1, y1, x2, y2)
            target_boxes: Target bounding boxes (x1, y1, x2, y2)
            
        Returns:
            IoU loss
        """
        # Calculate intersection
        x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate union
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union = pred_area + target_area - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-6)
        
        # IoU loss
        iou_loss = 1 - iou
        
        if self.reduction == 'mean':
            return iou_loss.mean()
        elif self.reduction == 'sum':
            return iou_loss.sum()
        else:
            return iou_loss

class GIoULoss(nn.Module):
    """Generalized IoU Loss"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Compute GIoU loss
        
        Args:
            pred_boxes: Predicted bounding boxes (x1, y1, x2, y2)
            target_boxes: Target bounding boxes (x1, y1, x2, y2)
            
        Returns:
            GIoU loss
        """
        # Calculate intersection
        x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate union
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union = pred_area + target_area - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-6)
        
        # Calculate enclosing box
        enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
        
        # Calculate GIoU
        giou = iou - (enclose_area - union) / (enclose_area + 1e-6)
        
        # GIoU loss
        giou_loss = 1 - giou
        
        if self.reduction == 'mean':
            return giou_loss.mean()
        elif self.reduction == 'sum':
            return giou_loss.sum()
        else:
            return giou_loss

class YOLOv5Loss(YOLOLoss):
    """Enhanced YOLOv5 loss with additional components"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Additional loss components
        self.use_focal = config.get('use_focal', False)
        self.use_giou = config.get('use_giou', False)
        
        if self.use_focal:
            self.focal_loss = FocalLoss(self.focal_alpha, self.focal_gamma)
        
        if self.use_giou:
            self.giou_loss = GIoULoss()
    
    def forward(self, predictions: List[torch.Tensor], targets: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute enhanced YOLOv5 loss
        
        Args:
            predictions: List of prediction tensors
            targets: List of target tensors
            
        Returns:
            Dictionary containing loss components
        """
        # Get base loss
        base_losses = super().forward(predictions, targets)
        
        # Add focal loss if enabled
        if self.use_focal:
            focal_loss = torch.tensor(0.0, device=predictions[0].device)
            for pred, target in zip(predictions, targets):
                if target[..., 4].sum() > 0:  # If there are positive samples
                    obj_mask = target[..., 4] > 0
                    pred_cls = pred.view(pred.shape[0], 3, self.num_classes + 5, *pred.shape[2:])
                    pred_cls = pred_cls.permute(0, 1, 3, 4, 2)[..., 5:][obj_mask]
                    target_cls = target[..., 5:].argmax(dim=-1)[obj_mask]
                    focal_loss += self.focal_loss(pred_cls, target_cls)
            
            base_losses['focal_loss'] = focal_loss
            base_losses['total_loss'] += focal_loss
        
        return base_losses

def create_loss_function(config: Dict[str, Any]) -> nn.Module:
    """Factory function to create loss function"""
    loss_type = config.get('loss_type', 'yolo')
    
    if loss_type == 'yolo':
        return YOLOLoss(config)
    elif loss_type == 'yolov5':
        return YOLOv5Loss(config)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

# Loss weight scheduling
class LossWeightScheduler:
    """Scheduler for loss weights during training"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.current_epoch = 0
        
        # Initial weights
        self.lambda_coord = config.get('lambda_coord', 5.0)
        self.lambda_noobj = config.get('lambda_noobj', 0.5)
        self.lambda_obj = config.get('lambda_obj', 1.0)
        self.lambda_class = config.get('lambda_class', 1.0)
        
        # Scheduling parameters
        self.warmup_epochs = config.get('warmup_epochs', 3)
        self.coord_warmup = config.get('coord_warmup', True)
    
    def step(self):
        """Update weights for current epoch"""
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs and self.coord_warmup:
            # Gradually increase coordinate loss weight
            progress = self.current_epoch / self.warmup_epochs
            self.lambda_coord = 5.0 * progress
    
    def get_weights(self) -> Dict[str, float]:
        """Get current loss weights"""
        return {
            'lambda_coord': self.lambda_coord,
            'lambda_noobj': self.lambda_noobj,
            'lambda_obj': self.lambda_obj,
            'lambda_class': self.lambda_class
        }
