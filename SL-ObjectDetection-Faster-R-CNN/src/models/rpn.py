"""
Region Proposal Network (RPN) implementation for Faster R-CNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np


class AnchorGenerator(nn.Module):
    """Generate anchors for RPN"""
    
    def __init__(self, sizes: List[int], aspect_ratios: List[float], 
                 stride: int = 16):
        super().__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.stride = stride
        
        # Generate base anchors
        self.base_anchors = self._generate_base_anchors()
    
    def _generate_base_anchors(self) -> torch.Tensor:
        """Generate base anchors at origin (0, 0)"""
        base_anchors = []
        
        for size in self.sizes:
            for aspect_ratio in self.aspect_ratios:
                # Calculate width and height
                w = size * np.sqrt(aspect_ratio)
                h = size / np.sqrt(aspect_ratio)
                
                # Create anchor coordinates (x1, y1, x2, y2)
                x1 = -w / 2
                y1 = -h / 2
                x2 = w / 2
                y2 = h / 2
                
                base_anchors.append([x1, y1, x2, y2])
        
        return torch.tensor(base_anchors, dtype=torch.float32)
    
    def forward(self, feature_map: torch.Tensor, 
                image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Generate anchors for the entire feature map
        
        Args:
            feature_map: Feature map tensor [B, C, H, W]
            image_size: Original image size (height, width)
        
        Returns:
            anchors: Generated anchors [H*W*num_anchors, 4]
        """
        batch_size, channels, height, width = feature_map.shape
        
        # Generate grid coordinates
        shifts_x = torch.arange(0, width * self.stride, self.stride, 
                              dtype=torch.float32, device=feature_map.device)
        shifts_y = torch.arange(0, height * self.stride, self.stride, 
                              dtype=torch.float32, device=feature_map.device)
        
        # Create meshgrid
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        shifts = torch.stack([shift_x.reshape(-1), shift_y.reshape(-1),
                            shift_x.reshape(-1), shift_y.reshape(-1)], dim=1)
        
        # Add base anchors to shifts
        anchors = (self.base_anchors.view(1, -1, 4) + 
                  shifts.view(-1, 1, 4)).reshape(-1, 4)
        
        # Clip anchors to image boundaries
        anchors[:, 0::2] = torch.clamp(anchors[:, 0::2], 0, image_size[1])
        anchors[:, 1::2] = torch.clamp(anchors[:, 1::2], 0, image_size[0])
        
        return anchors


class RPNHead(nn.Module):
    """RPN classification and regression heads"""
    
    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__()
        self.num_anchors = num_anchors
        
        # 3x3 conv layer
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                              padding=1)
        
        # Classification head (objectness score)
        self.cls_head = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        
        # Regression head (bbox deltas)
        self.reg_head = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through RPN head
        
        Args:
            x: Input feature map [B, C, H, W]
        
        Returns:
            cls_logits: Classification logits [B, num_anchors, H, W]
            bbox_reg: Bounding box regression [B, num_anchors*4, H, W]
        """
        x = F.relu(self.conv(x))
        
        cls_logits = self.cls_head(x)
        bbox_reg = self.reg_head(x)
        
        return cls_logits, bbox_reg


class RPN(nn.Module):
    """Region Proposal Network"""
    
    def __init__(self, in_channels: int, anchor_sizes: List[int] = None,
                 anchor_ratios: List[float] = None, stride: int = 16,
                 pre_nms_top_n_train: int = 2000,
                 post_nms_top_n_train: int = 2000,
                 pre_nms_top_n_test: int = 1000,
                 post_nms_top_n_test: int = 1000,
                 nms_thresh: float = 0.7,
                 fg_iou_thresh: float = 0.7,
                 bg_iou_thresh: float = 0.3,
                 batch_size_per_image: int = 256,
                 positive_fraction: float = 0.5):
        super().__init__()
        
        # Anchor settings
        if anchor_sizes is None:
            anchor_sizes = [8, 16, 32]
        if anchor_ratios is None:
            anchor_ratios = [0.5, 1.0, 2.0]
        
        self.anchor_sizes = anchor_sizes
        self.anchor_ratios = anchor_ratios
        self.stride = stride
        self.num_anchors = len(anchor_sizes) * len(anchor_ratios)
        
        # RPN parameters
        self.pre_nms_top_n_train = pre_nms_top_n_train
        self.post_nms_top_n_train = post_nms_top_n_train
        self.pre_nms_top_n_test = pre_nms_top_n_test
        self.post_nms_top_n_test = post_nms_top_n_test
        self.nms_thresh = nms_thresh
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        
        # Components
        self.anchor_generator = AnchorGenerator(anchor_sizes, anchor_ratios, stride)
        self.rpn_head = RPNHead(in_channels, self.num_anchors)
        
        # Loss function
        self.rpn_loss = RPNLoss()
    
    def forward(self, features: torch.Tensor, targets: Optional[List[Dict]] = None,
                image_sizes: Optional[List[Tuple[int, int]]] = None) -> Dict:
        """
        Forward pass through RPN
        
        Args:
            features: Feature maps from backbone [B, C, H, W]
            targets: Ground truth targets (for training)
            image_sizes: Original image sizes
        
        Returns:
            Dictionary containing proposals and losses
        """
        batch_size = features.shape[0]
        
        # Generate anchors for each image
        anchors_list = []
        for i in range(batch_size):
            if image_sizes is not None:
                image_size = image_sizes[i]
            else:
                # Default image size if not provided
                image_size = (800, 800)
            
            anchors = self.anchor_generator(features[i:i+1], image_size)
            anchors_list.append(anchors)
        
        # RPN head forward pass
        cls_logits, bbox_reg = self.rpn_head(features)
        
        # Reshape outputs
        cls_logits = cls_logits.permute(0, 2, 3, 1).contiguous()
        bbox_reg = bbox_reg.permute(0, 2, 3, 1).contiguous()
        
        # Flatten spatial dimensions
        cls_logits = cls_logits.view(batch_size, -1, 1)
        bbox_reg = bbox_reg.view(batch_size, -1, 4)
        
        # Generate proposals
        proposals = self._generate_proposals(anchors_list, cls_logits, bbox_reg, 
                                          image_sizes)
        
        # Calculate losses if training
        losses = {}
        if self.training and targets is not None:
            losses = self.rpn_loss(cls_logits, bbox_reg, anchors_list, targets)
        
        return {
            'proposals': proposals,
            'losses': losses,
            'anchors': anchors_list
        }
    
    def _generate_proposals(self, anchors_list: List[torch.Tensor],
                           cls_logits: torch.Tensor, bbox_reg: torch.Tensor,
                           image_sizes: Optional[List[Tuple[int, int]]]) -> List[torch.Tensor]:
        """Generate region proposals from RPN outputs"""
        proposals = []
        
        for i, anchors in enumerate(anchors_list):
            # Get scores and bbox deltas for this image
            scores = cls_logits[i].squeeze(-1)
            deltas = bbox_reg[i]
            
            # Apply bbox deltas to anchors
            proposals_i = self._apply_deltas(anchors, deltas)
            
            # Clip proposals to image boundaries
            if image_sizes is not None:
                h, w = image_sizes[i]
                proposals_i[:, 0::2] = torch.clamp(proposals_i[:, 0::2], 0, w)
                proposals_i[:, 1::2] = torch.clamp(proposals_i[:, 1::2], 0, h)
            
            # Filter proposals by score
            top_n = self.pre_nms_top_n_train if self.training else self.pre_nms_top_n_test
            if scores.shape[0] > top_n:
                top_scores, top_indices = torch.topk(scores, top_n)
                proposals_i = proposals_i[top_indices]
                scores = scores[top_indices]
            
            # Apply NMS
            keep = self._nms(proposals_i, scores)
            proposals_i = proposals_i[keep]
            
            # Limit number of proposals
            post_nms_top_n = self.post_nms_top_n_train if self.training else self.post_nms_top_n_test
            if proposals_i.shape[0] > post_nms_top_n:
                proposals_i = proposals_i[:post_nms_top_n]
            
            proposals.append(proposals_i)
        
        return proposals
    
    def _apply_deltas(self, anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """Apply bounding box deltas to anchors"""
        # Convert anchors from (x1, y1, x2, y2) to (cx, cy, w, h)
        anchors_cxcywh = self._xyxy_to_cxcywh(anchors)
        
        # Apply deltas
        proposals_cxcywh = anchors_cxcywh.clone()
        proposals_cxcywh[:, 0] += deltas[:, 0] * anchors_cxcywh[:, 2]  # cx
        proposals_cxcywh[:, 1] += deltas[:, 1] * anchors_cxcywh[:, 3]  # cy
        proposals_cxcywh[:, 2] *= torch.exp(deltas[:, 2])  # w
        proposals_cxcywh[:, 3] *= torch.exp(deltas[:, 3])  # h
        
        # Convert back to (x1, y1, x2, y2)
        proposals = self._cxcywh_to_xyxy(proposals_cxcywh)
        
        return proposals
    
    def _xyxy_to_cxcywh(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h)"""
        x1, y1, x2, y2 = boxes.unbind(-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return torch.stack([cx, cy, w, h], dim=-1)
    
    def _cxcywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2)"""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    def _nms(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Apply Non-Maximum Suppression"""
        if boxes.shape[0] == 0:
            return torch.empty(0, dtype=torch.long, device=boxes.device)
        
        # Sort by scores in descending order
        _, order = scores.sort(descending=True)
        
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
            
            i = order[0]
            keep.append(i)
            
            # Calculate IoU with remaining boxes
            iou = self._box_iou(boxes[i:i+1], boxes[order[1:]])
            
            # Keep boxes with IoU < threshold
            idx = (iou <= self.nms_thresh).nonzero().squeeze()
            if idx.numel() == 0:
                break
            order = order[idx + 1]
        
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)
    
    def _box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between two sets of boxes"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom
        
        wh = (rb - lt).clamp(min=0)  # width, height
        inter = wh[:, :, 0] * wh[:, :, 1]  # intersection
        
        union = area1[:, None] + area2 - inter
        iou = inter / union
        
        return iou


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
