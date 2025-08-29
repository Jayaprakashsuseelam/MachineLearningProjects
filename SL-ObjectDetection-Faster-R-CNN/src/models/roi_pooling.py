"""
RoI Pooling layer implementation for Faster R-CNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np


class RoIPooling(nn.Module):
    """Region of Interest Pooling layer"""
    
    def __init__(self, output_size: Tuple[int, int], spatial_scale: float = 1.0):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
    
    def forward(self, features: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RoI pooling
        
        Args:
            features: Feature maps [B, C, H, W]
            rois: Region proposals [N, 5] where first column is batch index
        
        Returns:
            Pooled features [N, C, output_h, output_w]
        """
        if rois.numel() == 0:
            return torch.empty(0, features.size(1), self.output_size[0], 
                             self.output_size[1], device=features.device)
        
        # Extract batch indices and coordinates
        batch_indices = rois[:, 0].long()
        roi_coords = rois[:, 1:5]
        
        # Scale ROI coordinates to feature map coordinates
        roi_coords = roi_coords * self.spatial_scale
        
        # Convert to integer coordinates
        roi_coords = roi_coords.long()
        
        # Ensure coordinates are within feature map bounds
        roi_coords[:, 0::2] = torch.clamp(roi_coords[:, 0::2], 0, features.size(3) - 1)
        roi_coords[:, 1::2] = torch.clamp(roi_coords[:, 1::2], 0, features.size(2) - 1)
        
        # Perform RoI pooling for each ROI
        pooled_features = []
        for i, (batch_idx, coords) in enumerate(zip(batch_indices, roi_coords)):
            x1, y1, x2, y2 = coords
            
            # Extract ROI from feature map
            roi_features = features[batch_idx, :, y1:y2+1, x1:x2+1]
            
            # Apply adaptive pooling to get fixed size
            pooled_roi = F.adaptive_max_pool2d(roi_features, self.output_size)
            pooled_features.append(pooled_roi)
        
        # Stack all pooled features
        if pooled_features:
            return torch.stack(pooled_features, dim=0)
        else:
            return torch.empty(0, features.size(1), self.output_size[0], 
                             self.output_size[1], device=features.device)


class RoIAlign(nn.Module):
    """Region of Interest Align layer (more accurate than RoI Pooling)"""
    
    def __init__(self, output_size: Tuple[int, int], spatial_scale: float = 1.0,
                 sampling_ratio: int = -1, aligned: bool = True):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned
    
    def forward(self, features: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RoI Align
        
        Args:
            features: Feature maps [B, C, H, W]
            rois: Region proposals [N, 5] where first column is batch index
        
        Returns:
            Aligned features [N, C, output_h, output_w]
        """
        if rois.numel() == 0:
            return torch.empty(0, features.size(1), self.output_size[0], 
                             self.output_size[1], device=features.device)
        
        # Extract batch indices and coordinates
        batch_indices = rois[:, 0].long()
        roi_coords = rois[:, 1:5]
        
        # Scale ROI coordinates to feature map coordinates
        roi_coords = roi_coords * self.spatial_scale
        
        # Perform RoI Align for each ROI
        aligned_features = []
        for i, (batch_idx, coords) in enumerate(zip(batch_indices, roi_coords)):
            x1, y1, x2, y2 = coords
            
            # Extract ROI from feature map
            roi_features = features[batch_idx, :, y1:y2+1, x1:x2+1]
            
            # Apply bilinear interpolation to get fixed size
            aligned_roi = F.interpolate(roi_features.unsqueeze(0), 
                                      size=self.output_size, 
                                      mode='bilinear', 
                                      align_corners=self.aligned)
            aligned_features.append(aligned_roi.squeeze(0))
        
        # Stack all aligned features
        if aligned_features:
            return torch.stack(aligned_features, dim=0)
        else:
            return torch.empty(0, features.size(1), self.output_size[0], 
                             self.output_size[1], device=features.device)


class RoIPoolingV2(nn.Module):
    """Improved RoI Pooling with better gradient handling"""
    
    def __init__(self, output_size: Tuple[int, int], spatial_scale: float = 1.0):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
    
    def forward(self, features: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through improved RoI pooling
        
        Args:
            features: Feature maps [B, C, H, W]
            rois: Region proposals [N, 5] where first column is batch index
        
        Returns:
            Pooled features [N, C, output_h, output_w]
        """
        if rois.numel() == 0:
            return torch.empty(0, features.size(1), self.output_size[0], 
                             self.output_size[1], device=features.device)
        
        # Extract batch indices and coordinates
        batch_indices = rois[:, 0].long()
        roi_coords = rois[:, 1:5]
        
        # Scale ROI coordinates to feature map coordinates
        roi_coords = roi_coords * self.spatial_scale
        
        # Perform RoI pooling for each ROI
        pooled_features = []
        for i, (batch_idx, coords) in enumerate(zip(batch_indices, roi_coords)):
            x1, y1, x2, y2 = coords
            
            # Extract ROI from feature map
            roi_features = features[batch_idx, :, y1:y2+1, x1:x2+1]
            
            # Calculate pooling regions
            roi_h, roi_w = roi_features.shape[-2:]
            bin_h = roi_h / self.output_size[0]
            bin_w = roi_w / self.output_size[1]
            
            # Apply max pooling with proper binning
            pooled_roi = self._roi_pool_single(roi_features, bin_h, bin_w)
            pooled_features.append(pooled_roi)
        
        # Stack all pooled features
        if pooled_features:
            return torch.stack(pooled_features, dim=0)
        else:
            return torch.empty(0, features.size(1), self.output_size[0], 
                             self.output_size[1], device=features.device)
    
    def _roi_pool_single(self, roi_features: torch.Tensor, bin_h: float, 
                         bin_w: float) -> torch.Tensor:
        """Pool a single ROI with proper binning"""
        channels, roi_h, roi_w = roi_features.shape
        output_h, output_w = self.output_size
        
        # Initialize output tensor
        pooled = torch.zeros(channels, output_h, output_w, 
                           device=roi_features.device, dtype=roi_features.dtype)
        
        # Apply pooling for each output bin
        for h in range(output_h):
            h_start = int(h * bin_h)
            h_end = int((h + 1) * bin_h)
            h_start = min(h_start, roi_h - 1)
            h_end = min(h_end, roi_h)
            
            for w in range(output_w):
                w_start = int(w * bin_w)
                w_end = int((w + 1) * bin_w)
                w_start = min(w_start, roi_w - 1)
                w_end = min(w_end, roi_w)
                
                if h_end > h_start and w_end > w_start:
                    # Extract region and apply max pooling
                    region = roi_features[:, h_start:h_end, w_start:w_end]
                    pooled[:, h, w] = region.max(dim=-1)[0].max(dim=-1)[0]
        
        return pooled


class RoIPooling3D(nn.Module):
    """3D RoI Pooling for video or volumetric data"""
    
    def __init__(self, output_size: Tuple[int, int, int], 
                 spatial_scale: float = 1.0, temporal_scale: float = 1.0):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.temporal_scale = temporal_scale
    
    def forward(self, features: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through 3D RoI pooling
        
        Args:
            features: Feature maps [B, C, T, H, W]
            rois: Region proposals [N, 6] where first column is batch index
        
        Returns:
            Pooled features [N, C, output_t, output_h, output_w]
        """
        if rois.numel() == 0:
            return torch.empty(0, features.size(1), self.output_size[0], 
                             self.output_size[1], self.output_size[2], 
                             device=features.device)
        
        # Extract batch indices and coordinates
        batch_indices = rois[:, 0].long()
        roi_coords = rois[:, 1:6]  # t1, x1, y1, t2, x2, y2
        
        # Scale ROI coordinates
        roi_coords[:, [1, 3]] *= self.spatial_scale  # x coordinates
        roi_coords[:, [2, 4]] *= self.spatial_scale  # y coordinates
        roi_coords[:, [0, 5]] *= self.temporal_scale  # t coordinates
        
        # Perform 3D RoI pooling for each ROI
        pooled_features = []
        for i, (batch_idx, coords) in enumerate(zip(batch_indices, roi_coords)):
            t1, x1, y1, t2, x2, y2 = coords
            
            # Extract 3D ROI from feature map
            roi_features = features[batch_idx, :, t1:t2+1, y1:y2+1, x1:x2+1]
            
            # Apply adaptive pooling to get fixed size
            pooled_roi = F.adaptive_max_pool3d(roi_features, self.output_size)
            pooled_features.append(pooled_roi)
        
        # Stack all pooled features
        if pooled_features:
            return torch.stack(pooled_features, dim=0)
        else:
            return torch.empty(0, features.size(1), self.output_size[0], 
                             self.output_size[1], self.output_size[2], 
                             device=features.device)


def get_roi_pooling(pooling_type: str = "pooling", **kwargs) -> nn.Module:
    """Factory function to get RoI pooling layer"""
    if pooling_type == "pooling":
        return RoIPooling(**kwargs)
    elif pooling_type == "align":
        return RoIAlign(**kwargs)
    elif pooling_type == "pooling_v2":
        return RoIPoolingV2(**kwargs)
    elif pooling_type == "3d":
        return RoIPooling3D(**kwargs)
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")


# Convenience functions for common use cases
def roi_pool(features: torch.Tensor, rois: torch.Tensor, 
             output_size: Tuple[int, int], spatial_scale: float = 1.0) -> torch.Tensor:
    """Convenience function for RoI pooling"""
    pool_layer = RoIPooling(output_size, spatial_scale)
    return pool_layer(features, rois)


def roi_align(features: torch.Tensor, rois: torch.Tensor, 
              output_size: Tuple[int, int], spatial_scale: float = 1.0) -> torch.Tensor:
    """Convenience function for RoI align"""
    align_layer = RoIAlign(output_size, spatial_scale)
    return align_layer(features, rois)
