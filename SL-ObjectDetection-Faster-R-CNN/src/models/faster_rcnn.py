"""
Main Faster R-CNN model implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import torchvision

from .backbone import Backbone, get_backbone
from .rpn import RPN
from .roi_pooling import RoIPooling, get_roi_pooling
from .losses import DetectionLoss


class FastRCNNPredictor(nn.Module):
    """Fast R-CNN predictor head"""
    
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better training"""
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Fast R-CNN predictor
        
        Args:
            x: Input features [N, in_channels]
        
        Returns:
            cls_score: Classification scores [N, num_classes]
            bbox_pred: Bounding box predictions [N, num_classes * 4]
        """
        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        
        return cls_score, bbox_pred


class FasterRCNN(nn.Module):
    """Faster R-CNN model"""
    
    def __init__(self, backbone_name: str = "resnet50", num_classes: int = 21,
                 pretrained: bool = True, freeze_backbone: bool = False,
                 rpn_anchor_sizes: List[int] = None, rpn_anchor_ratios: List[float] = None,
                 rpn_stride: int = 16, rpn_pre_nms_top_n_train: int = 2000,
                 rpn_post_nms_top_n_train: int = 2000, rpn_pre_nms_top_n_test: int = 1000,
                 rpn_post_nms_top_n_test: int = 1000, rpn_nms_thresh: float = 0.7,
                 rpn_fg_iou_thresh: float = 0.7, rpn_bg_iou_thresh: float = 0.3,
                 rpn_batch_size_per_image: int = 256, rpn_positive_fraction: float = 0.5,
                 box_fg_iou_thresh: float = 0.5, box_bg_iou_thresh: float = 0.5,
                 box_batch_size_per_image: int = 512, box_positive_fraction: float = 0.25,
                 bbox_reg_weights: Optional[List[float]] = None, score_thresh: float = 0.05,
                 nms_thresh: float = 0.5, detections_per_img: int = 100,
                 roi_pooling_type: str = "pooling", roi_pool_size: Tuple[int, int] = (7, 7)):
        super().__init__()
        
        # Model parameters
        self.num_classes = num_classes
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        
        # Backbone network
        self.backbone = get_backbone(backbone_name, pretrained, 
                                   freeze_layers=2 if freeze_backbone else 0)
        
        # RPN
        self.rpn = RPN(
            in_channels=self.backbone.out_channels,
            anchor_sizes=rpn_anchor_sizes,
            anchor_ratios=rpn_anchor_ratios,
            stride=rpn_stride,
            pre_nms_top_n_train=rpn_pre_nms_top_n_train,
            post_nms_top_n_train=rpn_post_nms_top_n_train,
            pre_nms_top_n_test=rpn_pre_nms_top_n_test,
            post_nms_top_n_test=rpn_post_nms_top_n_test,
            nms_thresh=rpn_nms_thresh,
            fg_iou_thresh=rpn_fg_iou_thresh,
            bg_iou_thresh=rpn_bg_iou_thresh,
            batch_size_per_image=rpn_batch_size_per_image,
            positive_fraction=rpn_positive_fraction
        )
        
        # RoI pooling
        self.roi_pooling = get_roi_pooling(
            pooling_type=roi_pooling_type,
            output_size=roi_pool_size,
            spatial_scale=1.0 / rpn_stride
        )
        
        # Fast R-CNN head
        self.fast_rcnn_head = FastRCNNPredictor(
            in_channels=self.backbone.out_channels * roi_pool_size[0] * roi_pool_size[1],
            num_classes=num_classes
        )
        
        # Detection loss
        self.detection_loss = DetectionLoss(
            fg_iou_thresh=box_fg_iou_thresh,
            bg_iou_thresh=box_bg_iou_thresh,
            batch_size_per_image=box_batch_size_per_image,
            positive_fraction=box_positive_fraction,
            bbox_reg_weights=bbox_reg_weights
        )
        
        # Set training mode
        self.train()
    
    def forward(self, images: torch.Tensor, targets: Optional[List[Dict]] = None) -> Dict:
        """
        Forward pass through Faster R-CNN
        
        Args:
            images: Input images [B, C, H, W]
            targets: Ground truth targets (for training)
        
        Returns:
            Dictionary containing predictions and losses
        """
        # Get image sizes for RPN
        image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        
        # Backbone feature extraction
        features = self.backbone(images)
        
        # RPN forward pass
        rpn_outputs = self.rpn(features, targets, image_sizes)
        proposals = rpn_outputs['proposals']
        rpn_losses = rpn_outputs.get('losses', {})
        
        # Prepare proposals for RoI pooling
        if self.training and targets is not None:
            # During training, use both RPN proposals and ground truth boxes
            proposals = self._prepare_training_proposals(proposals, targets, image_sizes)
        
        # RoI pooling
        roi_features = self._roi_pooling(features, proposals)
        
        # Fast R-CNN head
        cls_scores, bbox_preds = self.fast_rcnn_head(roi_features)
        
        # Post-processing
        if self.training:
            # During training, calculate detection loss
            detection_losses = self.detection_loss(
                cls_scores, bbox_preds, proposals, targets, image_sizes
            )
            
            # Combine all losses
            losses = {**rpn_losses, **detection_losses}
            
            return {
                'losses': losses,
                'proposals': proposals,
                'cls_scores': cls_scores,
                'bbox_preds': bbox_preds
            }
        else:
            # During inference, post-process predictions
            detections = self._post_process_detections(
                cls_scores, bbox_preds, proposals, image_sizes
            )
            
            return {
                'detections': detections,
                'proposals': proposals
            }
    
    def _prepare_training_proposals(self, proposals: List[torch.Tensor], 
                                  targets: List[Dict], 
                                  image_sizes: List[Tuple[int, int]]) -> List[torch.Tensor]:
        """Prepare proposals for training by adding ground truth boxes"""
        training_proposals = []
        
        for i, (proposal, target, image_size) in enumerate(zip(proposals, targets, image_sizes)):
            # Add batch index to proposals
            if proposal.numel() > 0:
                batch_indices = torch.full((proposal.shape[0], 1), i, 
                                         dtype=proposal.dtype, device=proposal.device)
                proposal_with_batch = torch.cat([batch_indices, proposal], dim=1)
            else:
                proposal_with_batch = torch.empty(0, 5, dtype=torch.float32, 
                                               device=proposal.device)
            
            # Add ground truth boxes with batch index
            gt_boxes = target['boxes']
            if gt_boxes.numel() > 0:
                batch_indices = torch.full((gt_boxes.shape[0], 1), i, 
                                         dtype=gt_boxes.dtype, device=gt_boxes.device)
                gt_with_batch = torch.cat([batch_indices, gt_boxes], dim=1)
                
                # Combine proposals and ground truth
                combined = torch.cat([proposal_with_batch, gt_with_batch], dim=0)
            else:
                combined = proposal_with_batch
            
            training_proposals.append(combined)
        
        return training_proposals
    
    def _roi_pooling(self, features: torch.Tensor, 
                     proposals: List[torch.Tensor]) -> torch.Tensor:
        """Apply RoI pooling to features using proposals"""
        # Concatenate all proposals with batch indices
        all_proposals = []
        for i, proposal in enumerate(proposals):
            if proposal.numel() > 0:
                all_proposals.append(proposal)
        
        if not all_proposals:
            return torch.empty(0, self.backbone.out_channels * 7 * 7, 
                             device=features.device)
        
        all_proposals = torch.cat(all_proposals, dim=0)
        
        # Apply RoI pooling
        roi_features = self.roi_pooling(features, all_proposals)
        
        # Flatten features
        roi_features = roi_features.view(roi_features.size(0), -1)
        
        return roi_features
    
    def _post_process_detections(self, cls_scores: torch.Tensor, 
                                bbox_preds: torch.Tensor, 
                                proposals: List[torch.Tensor],
                                image_sizes: List[Tuple[int, int]]) -> List[Dict]:
        """Post-process predictions to get final detections"""
        # Apply softmax to get probabilities
        cls_probs = F.softmax(cls_scores, dim=-1)
        
        # Reshape bbox predictions
        bbox_preds = bbox_preds.view(bbox_preds.size(0), -1, 4)
        
        # Process each image
        detections = []
        proposal_idx = 0
        
        for i, (proposal, image_size) in enumerate(zip(proposals, image_sizes)):
            if proposal.numel() == 0:
                detections.append({
                    'boxes': torch.empty(0, 4, device=cls_scores.device),
                    'labels': torch.empty(0, dtype=torch.long, device=cls_scores.device),
                    'scores': torch.empty(0, device=cls_scores.device)
                })
                continue
            
            # Get proposals for this image
            num_proposals = proposal.shape[0]
            img_cls_probs = cls_probs[proposal_idx:proposal_idx + num_proposals]
            img_bbox_preds = bbox_preds[proposal_idx:proposal_idx + num_proposals]
            
            # Remove background class (index 0)
            img_cls_probs = img_cls_probs[:, 1:]
            img_bbox_preds = img_bbox_preds[:, 1:]
            
            # Get best class for each proposal
            best_scores, best_labels = img_cls_probs.max(dim=1)
            
            # Filter by score threshold
            keep_mask = best_scores >= self.score_thresh
            if not keep_mask.any():
                detections.append({
                    'boxes': torch.empty(0, 4, device=cls_scores.device),
                    'labels': torch.empty(0, dtype=torch.long, device=cls_scores.device),
                    'scores': torch.empty(0, device=cls_scores.device)
                })
                proposal_idx += num_proposals
                continue
            
            # Apply score threshold
            filtered_scores = best_scores[keep_mask]
            filtered_labels = best_labels[keep_mask]
            filtered_bbox_preds = img_bbox_preds[keep_mask]
            filtered_proposals = proposal[keep_mask, 1:5]  # Remove batch index
            
            # Apply bounding box regression
            filtered_boxes = self._apply_bbox_regression(
                filtered_proposals, filtered_bbox_preds, filtered_labels
            )
            
            # Clip boxes to image boundaries
            filtered_boxes[:, 0::2] = torch.clamp(filtered_boxes[:, 0::2], 0, image_size[1])
            filtered_boxes[:, 1::2] = torch.clamp(filtered_boxes[:, 1::2], 0, image_size[0])
            
            # Apply NMS
            keep_indices = self._nms(filtered_boxes, filtered_scores)
            
            # Limit number of detections
            if keep_indices.numel() > self.detections_per_img:
                keep_indices = keep_indices[:self.detections_per_img]
            
            # Final detections
            final_boxes = filtered_boxes[keep_indices]
            final_labels = filtered_labels[keep_indices]
            final_scores = filtered_scores[keep_indices]
            
            detections.append({
                'boxes': final_boxes,
                'labels': final_labels,
                'scores': final_scores
            })
            
            proposal_idx += num_proposals
        
        return detections
    
    def _apply_bbox_regression(self, proposals: torch.Tensor, 
                              bbox_preds: torch.Tensor, 
                              labels: torch.Tensor) -> torch.Tensor:
        """Apply bounding box regression to proposals"""
        # Convert proposals to center format
        proposals_cxcywh = self._xyxy_to_cxcywh(proposals)
        
        # Get bbox predictions for the predicted class
        bbox_preds_per_class = bbox_preds[torch.arange(bbox_preds.size(0)), labels]
        
        # Apply deltas
        refined_boxes_cxcywh = proposals_cxcywh.clone()
        refined_boxes_cxcywh[:, 0] += bbox_preds_per_class[:, 0] * proposals_cxcywh[:, 2]
        refined_boxes_cxcywh[:, 1] += bbox_preds_per_class[:, 1] * proposals_cxcywh[:, 3]
        refined_boxes_cxcywh[:, 2] *= torch.exp(bbox_preds_per_class[:, 2])
        refined_boxes_cxcywh[:, 3] *= torch.exp(bbox_preds_per_class[:, 3])
        
        # Convert back to xyxy format
        refined_boxes = self._cxcywh_to_xyxy(refined_boxes_cxcywh)
        
        return refined_boxes
    
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
        if boxes.numel() == 0:
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
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2 - inter
        iou = inter / union
        
        return iou


def faster_rcnn_resnet50(num_classes: int = 21, pretrained: bool = True, 
                         **kwargs) -> FasterRCNN:
    """Create Faster R-CNN with ResNet-50 backbone"""
    return FasterRCNN(backbone_name="resnet50", num_classes=num_classes, 
                      pretrained=pretrained, **kwargs)


def faster_rcnn_resnet101(num_classes: int = 21, pretrained: bool = True, 
                          **kwargs) -> FasterRCNN:
    """Create Faster R-CNN with ResNet-101 backbone"""
    return FasterRCNN(backbone_name="resnet101", num_classes=num_classes, 
                      pretrained=pretrained, **kwargs)


def faster_rcnn_vgg16(num_classes: int = 21, pretrained: bool = True, 
                      **kwargs) -> FasterRCNN:
    """Create Faster R-CNN with VGG-16 backbone"""
    return FasterRCNN(backbone_name="vgg16", num_classes=num_classes, 
                      pretrained=pretrained, **kwargs)
