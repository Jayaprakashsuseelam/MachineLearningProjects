"""
YOLOv5 Model Implementation
Comprehensive implementation of YOLOv5 architecture with all components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any
import math

class ConvBNSiLU(nn.Module):
    """Convolution + BatchNorm + SiLU activation"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, 
                 stride: int = 1, padding: int = 0, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    """Standard bottleneck"""
    
    def __init__(self, in_channels: int, out_channels: int, shortcut: bool = True, 
                 expansion: float = 0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = ConvBNSiLU(in_channels, hidden_channels, 1, 1)
        self.cv2 = ConvBNSiLU(hidden_channels, out_channels, 3, 1, 1)
        self.use_add = shortcut and in_channels == out_channels
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.use_add else self.cv2(self.cv1(x))

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions"""
    
    def __init__(self, in_channels: int, out_channels: int, n: int = 1, shortcut: bool = True, 
                 expansion: float = 0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = ConvBNSiLU(in_channels, hidden_channels, 1, 1)
        self.cv2 = ConvBNSiLU(in_channels, hidden_channels, 1, 1)
        self.cv3 = ConvBNSiLU(2 * hidden_channels, out_channels, 1)
        self.m = nn.Sequential(*[Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0) for _ in range(n)])
    
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5"""
    
    def __init__(self, in_channels: int, out_channels: int, k: int = 5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = ConvBNSiLU(in_channels, hidden_channels, 1, 1)
        self.cv2 = ConvBNSiLU(hidden_channels * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

class CSPDarknet(nn.Module):
    """CSPDarknet backbone for YOLOv5"""
    
    def __init__(self, base_channels: int = 64, base_depth: int = 3):
        super().__init__()
        
        # Initial convolution
        self.conv1 = ConvBNSiLU(3, base_channels, 3, 2, 1)
        self.conv2 = ConvBNSiLU(base_channels, base_channels * 2, 3, 2, 1)
        
        # C3 blocks
        self.c3_1 = C3(base_channels * 2, base_channels * 2, base_depth)
        self.conv3 = ConvBNSiLU(base_channels * 2, base_channels * 4, 3, 2, 1)
        self.c3_2 = C3(base_channels * 4, base_channels * 4, base_depth * 2)
        self.conv4 = ConvBNSiLU(base_channels * 4, base_channels * 8, 3, 2, 1)
        self.c3_3 = C3(base_channels * 8, base_channels * 8, base_depth * 3)
        self.conv5 = ConvBNSiLU(base_channels * 8, base_channels * 16, 3, 2, 1)
        self.c3_4 = C3(base_channels * 16, base_channels * 16, base_depth)
        self.conv6 = ConvBNSiLU(base_channels * 16, base_channels * 32, 3, 2, 1)
        self.c3_5 = C3(base_channels * 32, base_channels * 32, base_depth)
        
        # SPPF
        self.sppf = SPPF(base_channels * 32, base_channels * 32)
    
    def forward(self, x):
        # Backbone features
        x1 = self.conv1(x)  # 1/2
        x2 = self.conv2(x1)  # 1/4
        x3 = self.c3_1(x2)
        x4 = self.conv3(x3)  # 1/8
        x5 = self.c3_2(x4)
        x6 = self.conv4(x5)  # 1/16
        x7 = self.c3_3(x6)
        x8 = self.conv5(x7)  # 1/32
        x9 = self.c3_4(x8)
        x10 = self.conv6(x9)  # 1/64
        x11 = self.c3_5(x10)
        x12 = self.sppf(x11)
        
        return [x7, x9, x12]  # P3, P4, P5

class PANet(nn.Module):
    """Path Aggregation Network (PANet) for YOLOv5"""
    
    def __init__(self, channels: List[int]):
        super().__init__()
        self.channels = channels
        
        # Upsample path (P5 -> P4 -> P3)
        self.up_conv1 = ConvBNSiLU(channels[2], channels[1], 1, 1)
        self.up_conv2 = ConvBNSiLU(channels[1], channels[0], 1, 1)
        
        # Downsample path (P3 -> P4 -> P5)
        self.down_conv1 = ConvBNSiLU(channels[0], channels[1], 3, 2, 1)
        self.down_conv2 = ConvBNSiLU(channels[1], channels[2], 3, 2, 1)
        
        # C3 blocks for feature fusion
        self.c3_up1 = C3(channels[1] * 2, channels[1], 1, False)
        self.c3_up2 = C3(channels[0] * 2, channels[0], 1, False)
        self.c3_down1 = C3(channels[1] * 2, channels[1], 1, False)
        self.c3_down2 = C3(channels[2] * 2, channels[2], 1, False)
    
    def forward(self, features):
        p3, p4, p5 = features
        
        # Upsample path
        up_p5 = self.up_conv1(p5)
        up_p5 = F.interpolate(up_p5, scale_factor=2, mode='nearest')
        p4_fused = self.c3_up1(torch.cat([up_p5, p4], 1))
        
        up_p4 = self.up_conv2(p4_fused)
        up_p4 = F.interpolate(up_p4, scale_factor=2, mode='nearest')
        p3_fused = self.c3_up2(torch.cat([up_p4, p3], 1))
        
        # Downsample path
        down_p3 = self.down_conv1(p3_fused)
        p4_fused_down = self.c3_down1(torch.cat([down_p3, p4_fused], 1))
        
        down_p4 = self.down_conv2(p4_fused_down)
        p5_fused = self.c3_down2(torch.cat([down_p4, p5], 1))
        
        return [p3_fused, p4_fused_down, p5_fused]

class DetectionHead(nn.Module):
    """Detection head for YOLOv5"""
    
    def __init__(self, in_channels: int, num_classes: int, num_anchors: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Detection layers
        self.conv1 = ConvBNSiLU(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, num_anchors * (5 + num_classes), 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize detection head weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class YOLOv5(nn.Module):
    """Complete YOLOv5 model"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Model parameters
        self.num_classes = config.get('num_classes', 80)
        self.depth_multiple = config.get('depth_multiple', 1.0)
        self.width_multiple = config.get('width_multiple', 1.0)
        
        # Calculate channel sizes
        base_channels = int(64 * self.width_multiple)
        channels = [
            int(base_channels * 2),   # P3
            int(base_channels * 4),   # P4
            int(base_channels * 8)    # P5
        ]
        
        # Backbone
        self.backbone = CSPDarknet(base_channels, int(3 * self.depth_multiple))
        
        # Neck (PANet)
        self.neck = PANet(channels)
        
        # Detection heads
        self.heads = nn.ModuleList([
            DetectionHead(channels[0], self.num_classes),  # P3
            DetectionHead(channels[1], self.num_classes),  # P4
            DetectionHead(channels[2], self.num_classes)   # P5
        ])
        
        # Anchor boxes (COCO dataset anchors)
        self.anchors = torch.tensor([
            [10, 13], [16, 30], [33, 23],      # P3/8
            [30, 61], [62, 45], [59, 119],     # P4/16
            [116, 90], [156, 198], [373, 326]  # P5/32
        ]).float()
        
        self.stride = torch.tensor([8, 16, 32])
        self.grid = [torch.zeros(1)] * 3
        self.anchor_grid = [torch.zeros(1)] * 3
    
    def forward(self, x):
        # Backbone
        features = self.backbone(x)
        
        # Neck
        features = self.neck(features)
        
        # Detection heads
        outputs = []
        for i, (feature, head) in enumerate(zip(features, self.heads)):
            output = head(feature)
            outputs.append(output)
        
        if self.training:
            return outputs
        else:
            return self.inference(outputs, x.shape[2:])
    
    def inference(self, outputs, img_shape):
        """Inference with post-processing"""
        z = []
        for i, output in enumerate(outputs):
            bs, _, ny, nx = output.shape
            output = output.view(bs, self.num_anchors, self.num_classes + 5, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            if not self.training:
                if self.grid[i].shape[2:4] != output.shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                
                y = output.sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.num_classes + 5))
        
        return torch.cat(z, 1) if len(z) > 1 else z[0]
    
    def _make_grid(self, nx, ny, i):
        """Create grid for anchor boxes"""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        
        shape = 1, self.num_anchors, ny, nx, 2
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.num_anchors, 1, 1, 2)).expand(shape)
        
        return grid, anchor_grid

def create_yolov5_model(config: Dict[str, Any]) -> YOLOv5:
    """Factory function to create YOLOv5 model"""
    return YOLOv5(config)

# Model size configurations
def yolov5n(config: Dict[str, Any]) -> YOLOv5:
    """YOLOv5 nano model"""
    config.update({'depth_multiple': 0.33, 'width_multiple': 0.25})
    return YOLOv5(config)

def yolov5s(config: Dict[str, Any]) -> YOLOv5:
    """YOLOv5 small model"""
    config.update({'depth_multiple': 0.33, 'width_multiple': 0.50})
    return YOLOv5(config)

def yolov5m(config: Dict[str, Any]) -> YOLOv5:
    """YOLOv5 medium model"""
    config.update({'depth_multiple': 0.67, 'width_multiple': 0.75})
    return YOLOv5(config)

def yolov5l(config: Dict[str, Any]) -> YOLOv5:
    """YOLOv5 large model"""
    config.update({'depth_multiple': 1.0, 'width_multiple': 1.0})
    return YOLOv5(config)

def yolov5x(config: Dict[str, Any]) -> YOLOv5:
    """YOLOv5 extra large model"""
    config.update({'depth_multiple': 1.33, 'width_multiple': 1.25})
    return YOLOv5(config)

# Model registry
MODEL_REGISTRY = {
    'yolov5n': yolov5n,
    'yolov5s': yolov5s,
    'yolov5m': yolov5m,
    'yolov5l': yolov5l,
    'yolov5x': yolov5x,
}

def get_yolov5_model(model_name: str, config: Dict[str, Any]) -> YOLOv5:
    """Get YOLOv5 model by name"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_name](config)
