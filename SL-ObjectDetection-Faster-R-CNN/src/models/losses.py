"""
Loss functions for Faster R-CNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


class RPNLoss(nn.Module):
    """RPN loss function"""
    
    def __init__(self, fg_iou_thresh: float = 0.7, bg_iou_thresh: float = 0.3,
                 batch_size_per_image: int = 256, positive_fraction: float = 0.5):
        super().__init__()
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
    
    def forward(self, cls_logits: torch.Tensor, bbox_reg: torch.Tensor,
                anchors: List[torch.Tensor], targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Calculate RPN loss
        
        Args:
            cls_logits: Classification logits [B, N, 1]
            bbox_reg: Bounding box regression [B, N, 4]
            anchors: List of anchor boxes for each image
            targets: Ground truth targets
        
        Returns:
            Dictionary containing classification and regression losses
        """
        batch_size = cls_logits.shape[0]
        
        # Sample anchors for each image
        sampled_anchors, sampled_targets = self._sample_anchors(anchors, targets)
        
        # Calculate classification loss
        cls_loss = self._classification_loss(cls_logits, sampled_targets)
        
        # Calculate regression loss
        reg_loss = self._regression_loss(bbox_reg, sampled_anchors, sampled_targets)
        
        return {
            'rpn_cls_loss': cls_loss,
            'rpn_reg_loss': reg_loss
        }
    
    def _sample_anchors(self, anchors: List[torch.Tensor], 
                        targets: List[Dict]) -> Tuple[List[torch.Tensor], List[Dict]]:
        """Sample positive and negative anchors"""
        sampled_anchors = []
        sampled_targets = []
        
        for i, (anchor, target) in enumerate(zip(anchors, targets)):
            # Calculate IoU between anchors and ground truth boxes
            gt_boxes = target['boxes']
            iou = self._box_iou(anchor, gt_boxes)
            
            # Assign labels based on IoU thresholds
            max_iou, max_idx = iou.max(dim=1)
            
            # Positive anchors: IoU > fg_iou_thresh
            pos_mask = max_iou >= self.fg_iou_thresh
            
            # Negative anchors: IoU < bg_iou_thresh
            neg_mask = max_iou < self.bg_iou_thresh
            
            # Sample anchors
            pos_indices = torch.where(pos_mask)[0]
            neg_indices = torch.where(neg_mask)[0]
            
            # Ensure balanced sampling
            num_pos = min(pos_indices.numel(), 
                         int(self.batch_size_per_image * self.positive_fraction))
            num_neg = min(neg_indices.numel(), 
                         self.batch_size_per_image - num_pos)
            
            if num_pos > 0:
                pos_indices = pos_indices[torch.randperm(pos_indices.numel())[:num_pos]]
            if num_neg > 0:
                neg_indices = neg_indices[torch.randperm(neg_indices.numel())[:num_neg]]
            
            # Combine indices
            sampled_indices = torch.cat([pos_indices, neg_indices])
            
            sampled_anchors.append(anchor[sampled_indices])
            sampled_targets.append({
                'boxes': target['boxes'][max_idx[sampled_indices]],
                'labels': (max_iou[sampled_indices] >= self.fg_iou_thresh).long()
            })
        
        return sampled_anchors, sampled_targets
    
    def _classification_loss(self, cls_logits: torch.Tensor, 
                           sampled_targets: List[Dict]) -> torch.Tensor:
        """Calculate classification loss"""
        # Flatten logits and targets
        cls_logits = cls_logits.view(-1)
        labels = torch.cat([t['labels'] for t in sampled_targets])
        
        # Binary cross-entropy loss
        cls_loss = F.binary_cross_entropy_with_logits(cls_logits, labels.float(), 
                                                     reduction='mean')
        
        return cls_loss
    
    def _regression_loss(self, bbox_reg: torch.Tensor, sampled_anchors: List[torch.Tensor],
                        sampled_targets: List[Dict]) -> torch.Tensor:
        """Calculate regression loss"""
        # Only calculate loss for positive anchors
        pos_mask = torch.cat([t['labels'] for t in sampled_targets]).bool()
        
        if not pos_mask.any():
            return torch.tensor(0.0, device=bbox_reg.device)
        
        # Get positive samples
        pos_bbox_reg = bbox_reg.view(-1, 4)[pos_mask]
        pos_anchors = torch.cat(sampled_anchors)[pos_mask]
        pos_targets = torch.cat([t['boxes'] for t in sampled_targets])[pos_mask]
        
        # Calculate target deltas
        target_deltas = self._get_target_deltas(pos_anchors, pos_targets)
        
        # Smooth L1 loss
        reg_loss = F.smooth_l1_loss(pos_bbox_reg, target_deltas, reduction='mean')
        
        return reg_loss
    
    def _get_target_deltas(self, anchors: torch.Tensor, 
                          targets: torch.Tensor) -> torch.Tensor:
        """Calculate target deltas for regression"""
        # Convert to center format
        anchors_cxcywh = self._xyxy_to_cxcywh(anchors)
        targets_cxcywh = self._xyxy_to_cxcywh(targets)
        
        # Calculate deltas
        deltas = torch.zeros_like(anchors_cxcywh)
        deltas[:, 0] = (targets_cxcywh[:, 0] - anchors_cxcywh[:, 0]) / anchors_cxcywh[:, 2]
        deltas[:, 1] = (targets_cxcywh[:, 1] - anchors_cxcywh[:, 1]) / anchors_cxcywh[:, 3]
        deltas[:, 2] = torch.log(targets_cxcywh[:, 2] / anchors_cxcywh[:, 2])
        deltas[:, 3] = torch.log(targets_cxcywh[:, 3] / anchors_cxcywh[:, 3])
        
        return deltas
    
    def _xyxy_to_cxcywh(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h)"""
        x1, y1, x2, y2 = boxes.unbind(-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return torch.stack([cx, cy, w, h], dim=-1)
    
    def _box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between two sets of boxes"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2 - inter
        iou = inter / union
        
        return iou


class DetectionLoss(nn.Module):
    """Fast R-CNN detection loss function"""
    
    def __init__(self, fg_iou_thresh: float = 0.5, bg_iou_thresh: float = 0.5,
                 batch_size_per_image: int = 512, positive_fraction: float = 0.25,
                 bbox_reg_weights: Optional[List[float]] = None):
        super().__init__()
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        
        if bbox_reg_weights is None:
            bbox_reg_weights = [1.0, 1.0, 1.0, 1.0]
        self.bbox_reg_weights = torch.tensor(bbox_reg_weights)
    
    def forward(self, cls_scores: torch.Tensor, bbox_preds: torch.Tensor,
                proposals: List[torch.Tensor], targets: List[Dict],
                image_sizes: List[Tuple[int, int]]) -> Dict[str, torch.Tensor]:
        """
        Calculate detection loss
        
        Args:
            cls_scores: Classification scores [N, num_classes]
            bbox_preds: Bounding box predictions [N, num_classes * 4]
            proposals: Region proposals
            targets: Ground truth targets
            image_sizes: Original image sizes
        
        Returns:
            Dictionary containing classification and regression losses
        """
        # Sample proposals for training
        sampled_proposals, sampled_targets = self._sample_proposals(
            proposals, targets, image_sizes
        )
        
        # Calculate classification loss
        cls_loss = self._classification_loss(cls_scores, sampled_targets)
        
        # Calculate regression loss
        reg_loss = self._regression_loss(bbox_preds, sampled_proposals, sampled_targets)
        
        return {
            'det_cls_loss': cls_loss,
            'det_reg_loss': reg_loss
        }
    
    def _sample_proposals(self, proposals: List[torch.Tensor], 
                         targets: List[Dict],
                         image_sizes: List[Tuple[int, int]]) -> Tuple[List[torch.Tensor], List[Dict]]:
        """Sample positive and negative proposals"""
        sampled_proposals = []
        sampled_targets = []
        
        for i, (proposal, target, image_size) in enumerate(zip(proposals, targets, image_sizes)):
            if proposal.numel() == 0:
                sampled_proposals.append(torch.empty(0, 4, device=proposal.device))
                sampled_targets.append({
                    'boxes': torch.empty(0, 4, device=proposal.device),
                    'labels': torch.empty(0, dtype=torch.long, device=proposal.device)
                })
                continue
            
            # Remove batch index from proposals
            proposal_boxes = proposal[:, 1:5]
            
            # Calculate IoU between proposals and ground truth
            gt_boxes = target['boxes']
            if gt_boxes.numel() == 0:
                # No ground truth boxes
                sampled_proposals.append(torch.empty(0, 4, device=proposal.device))
                sampled_targets.append({
                    'boxes': torch.empty(0, 4, device=proposal.device),
                    'labels': torch.empty(0, dtype=torch.long, device=proposal.device)
                })
                continue
            
            iou = self._box_iou(proposal_boxes, gt_boxes)
            max_iou, max_idx = iou.max(dim=1)
            
            # Assign labels based on IoU thresholds
            pos_mask = max_iou >= self.fg_iou_thresh
            neg_mask = max_iou < self.bg_iou_thresh
            
            # Sample proposals
            pos_indices = torch.where(pos_mask)[0]
            neg_indices = torch.where(neg_mask)[0]
            
            # Ensure balanced sampling
            num_pos = min(pos_indices.numel(), 
                         int(self.batch_size_per_image * self.positive_fraction))
            num_neg = min(neg_indices.numel(), 
                         self.batch_size_per_image - num_pos)
            
            if num_pos > 0:
                pos_indices = pos_indices[torch.randperm(pos_indices.numel())[:num_pos]]
            if num_neg > 0:
                neg_indices = neg_indices[torch.randperm(neg_indices.numel())[:num_neg]]
            
            # Combine indices
            sampled_indices = torch.cat([pos_indices, neg_indices])
            
            sampled_proposals.append(proposal_boxes[sampled_indices])
            sampled_targets.append({
                'boxes': gt_boxes[max_idx[sampled_indices]],
                'labels': target['labels'][max_idx[sampled_indices]]
            })
        
        return sampled_proposals, sampled_targets
    
    def _classification_loss(self, cls_scores: torch.Tensor, 
                           sampled_targets: List[Dict]) -> torch.Tensor:
        """Calculate classification loss"""
        # Get labels for sampled proposals
        labels = []
        for target in sampled_targets:
            if target['labels'].numel() > 0:
                labels.append(target['labels'])
        
        if not labels:
            return torch.tensor(0.0, device=cls_scores.device)
        
        labels = torch.cat(labels)
        
        # Cross-entropy loss
        cls_loss = F.cross_entropy(cls_scores, labels, reduction='mean')
        
        return cls_loss
    
    def _regression_loss(self, bbox_preds: torch.Tensor, 
                        sampled_proposals: List[torch.Tensor],
                        sampled_targets: List[Dict]) -> torch.Tensor:
        """Calculate regression loss"""
        # Only calculate loss for positive proposals
        pos_proposals = []
        pos_targets = []
        pos_labels = []
        
        for proposal, target in zip(sampled_proposals, sampled_targets):
            if proposal.numel() == 0:
                continue
            
            # Get positive proposals
            pos_mask = target['labels'] > 0
            if pos_mask.any():
                pos_proposals.append(proposal[pos_mask])
                pos_targets.append(target['boxes'][pos_mask])
                pos_labels.append(target['labels'][pos_mask])
        
        if not pos_proposals:
            return torch.tensor(0.0, device=bbox_preds.device)
        
        # Concatenate positive samples
        pos_proposals = torch.cat(pos_proposals, dim=0)
        pos_targets = torch.cat(pos_targets, dim=0)
        pos_labels = torch.cat(pos_labels, dim=0)
        
        # Get bbox predictions for the target class
        bbox_preds_per_class = bbox_preds.view(-1, bbox_preds.size(-1) // 4, 4)
        bbox_preds_per_class = bbox_preds_per_class[torch.arange(pos_labels.size(0)), pos_labels]
        
        # Calculate target deltas
        target_deltas = self._get_target_deltas(pos_proposals, pos_targets)
        
        # Apply bbox regression weights
        target_deltas = target_deltas / self.bbox_reg_weights.to(target_deltas.device)
        
        # Smooth L1 loss
        reg_loss = F.smooth_l1_loss(bbox_preds_per_class, target_deltas, reduction='mean')
        
        return reg_loss
    
    def _get_target_deltas(self, proposals: torch.Tensor, 
                          targets: torch.Tensor) -> torch.Tensor:
        """Calculate target deltas for regression"""
        # Convert to center format
        proposals_cxcywh = self._xyxy_to_cxcywh(proposals)
        targets_cxcywh = self._xyxy_to_cxcywh(targets)
        
        # Calculate deltas
        deltas = torch.zeros_like(proposals_cxcywh)
        deltas[:, 0] = (targets_cxcywh[:, 0] - proposals_cxcywh[:, 0]) / proposals_cxcywh[:, 2]
        deltas[:, 1] = (targets_cxcywh[:, 1] - proposals_cxcywh[:, 1]) / proposals_cxcywh[:, 3]
        deltas[:, 2] = torch.log(targets_cxcywh[:, 2] / proposals_cxcywh[:, 2])
        deltas[:, 3] = torch.log(targets_cxcywh[:, 3] / proposals_cxcywh[:, 3])
        
        return deltas
    
    def _xyxy_to_cxcywh(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h)"""
        x1, y1, x2, y2 = boxes.unbind(-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return torch.stack([cx, cy, w, h], dim=-1)
    
    def _box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between two sets of boxes"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2 - inter
        iou = inter / union
        
        return iou


class CombinedLoss(nn.Module):
    """Combined loss for Faster R-CNN training"""
    
    def __init__(self, rpn_loss: RPNLoss, detection_loss: DetectionLoss,
                 rpn_weight: float = 1.0, detection_weight: float = 1.0):
        super().__init__()
        self.rpn_loss = rpn_loss
        self.detection_loss = detection_loss
        self.rpn_weight = rpn_weight
        self.detection_weight = detection_weight
    
    def forward(self, rpn_outputs: Dict, detection_outputs: Dict,
                targets: List[Dict], image_sizes: List[Tuple[int, int]]) -> Dict[str, torch.Tensor]:
        """
        Calculate combined loss
        
        Args:
            rpn_outputs: RPN outputs containing proposals and losses
            detection_outputs: Detection outputs containing scores and predictions
            targets: Ground truth targets
            image_sizes: Original image sizes
        
        Returns:
            Dictionary containing all losses
        """
        # RPN losses
        rpn_losses = rpn_outputs.get('losses', {})
        
        # Detection losses
        detection_losses = self.detection_loss(
            detection_outputs['cls_scores'],
            detection_outputs['bbox_preds'],
            rpn_outputs['proposals'],
            targets,
            image_sizes
        )
        
        # Combine losses
        total_loss = 0.0
        combined_losses = {}
        
        # Add RPN losses
        for name, loss in rpn_losses.items():
            combined_losses[name] = loss
            total_loss += self.rpn_weight * loss
        
        # Add detection losses
        for name, loss in detection_losses.items():
            combined_losses[name] = loss
            total_loss += self.detection_weight * loss
        
        combined_losses['total_loss'] = total_loss
        
        return combined_losses


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss
        
        Args:
            inputs: Input logits [N, C]
            targets: Target labels [N]
        
        Returns:
            Focal loss value
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
        Calculate IoU loss
        
        Args:
            pred_boxes: Predicted boxes [N, 4] in (x1, y1, x2, y2) format
            target_boxes: Target boxes [N, 4] in (x1, y1, x2, y2) format
        
        Returns:
            IoU loss value
        """
        # Calculate IoU
        iou = self._box_iou(pred_boxes, target_boxes)
        
        # IoU loss = 1 - IoU
        iou_loss = 1 - iou
        
        if self.reduction == 'mean':
            return iou_loss.mean()
        elif self.reduction == 'sum':
            return iou_loss.sum()
        else:
            return iou_loss
    
    def _box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between two sets of boxes"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2 - inter
        iou = inter / union
        
        return iou
