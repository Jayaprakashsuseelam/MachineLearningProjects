import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np


class L2Norm(nn.Module):
    """
    L2 normalization layer for feature maps
    """
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.scale = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.scale)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = x / norm * self.weight.view(1, -1, 1, 1)
        return x


class SSDNetwork(nn.Module):
    """
    Single Shot Detector Network Architecture
    Based on VGG16 backbone with additional feature layers
    """
    
    def __init__(self, config):
        super(SSDNetwork, self).__init__()
        self.config = config
        self.num_classes = config['num_classes']
        self.input_size = config['input_size']
        
        # VGG16 base network
        self.vgg = self._make_vgg_layers()
        
        # Additional feature layers
        self.extras = self._make_extras_layers()
        
        # L2 normalization layer
        self.l2norm = L2Norm(512, 20)
        
        # Localization and classification layers
        self.loc_layers, self.conf_layers = self._make_detection_layers()
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_vgg_layers(self):
        """Create VGG16 base network layers"""
        layers = []
        in_channels = 3
        
        # VGG16 configuration
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 
               512, 512, 512, 'M', 512, 512, 512]
        
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
                
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
        
        return nn.ModuleList(layers)
    
    def _make_extras_layers(self):
        """Create additional feature extraction layers"""
        layers = []
        in_channels = 1024
        
        # Extra layers configuration
        cfg = [256, 512, 128, 256, 128, 256, 128, 256]
        
        flag = False
        for k, v in enumerate(cfg):
            if in_channels != 'S':
                if v == 'S':
                    layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                       kernel_size=(1, 3)[flag], stride=2, padding=1)]
                else:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            layers += [nn.ReLU(inplace=True)]
            in_channels = v
            flag = not flag
            
        return nn.ModuleList(layers)
    
    def _make_detection_layers(self):
        """Create detection heads for localization and classification"""
        loc_layers = []
        conf_layers = []
        
        # Number of default boxes per feature map location
        num_default_boxes = len(self.config['aspect_ratios']) + 1
        
        # Feature map sizes
        feature_maps = self.config['feature_maps']
        
        for i, feature_map_size in enumerate(feature_maps):
            # Localization layers
            loc_layers += [nn.Conv2d(512 if i == 0 else 1024 if i == 1 else 512,
                                   num_default_boxes * 4,
                                   kernel_size=3, padding=1)]
            
            # Classification layers
            conf_layers += [nn.Conv2d(512 if i == 0 else 1024 if i == 1 else 512,
                                    num_default_boxes * self.num_classes,
                                    kernel_size=3, padding=1)]
        
        return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, x):
        """Forward pass through the network"""
        features = []
        loc = []
        conf = []
        
        # VGG layers
        for i in range(23):
            x = self.vgg[i](x)
        s = self.l2norm(x)
        features.append(s)
        
        # Apply vgg up to fc7
        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)
        features.append(x)
        
        # Apply extra layers and build feature list
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                features.append(x)
        
        # Apply detection layers
        for (x, l, c) in zip(features, self.loc_layers, self.conf_layers):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        
        # Reshape outputs
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        
        return loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes)


class DefaultBoxGenerator:
    """
    Generate default boxes (anchor boxes) for SSD
    """
    
    def __init__(self, config):
        self.config = config
        self.feature_maps = config['feature_maps']
        self.min_sizes = config['min_sizes']
        self.max_sizes = config['max_sizes']
        self.steps = config['steps']
        self.aspect_ratios = config['aspect_ratios']
        self.clip = config['clip']
        
    def generate_default_boxes(self):
        """Generate default boxes for all feature maps"""
        default_boxes = []
        
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            max_sizes = self.max_sizes[k]
            
            f_k = self.input_size / self.steps[k]
            
            # Generate center coordinates
            for i, j in itertools.product(range(f), repeat=2):
                # Unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                
                # Aspect ratio 1:1
                s_k = min_sizes / self.input_size
                default_boxes.append([cx, cy, s_k, s_k])
                
                # Aspect ratio 1:1 with max size
                s_k = math.sqrt(min_sizes * max_sizes) / self.input_size
                default_boxes.append([cx, cy, s_k, s_k])
                
                # Other aspect ratios
                for ar in self.aspect_ratios:
                    default_boxes.append([cx, cy, s_k * math.sqrt(ar), s_k / math.sqrt(ar)])
                    default_boxes.append([cx, cy, s_k / math.sqrt(ar), s_k * math.sqrt(ar)])
        
        default_boxes = torch.tensor(default_boxes, dtype=torch.float32)
        if self.clip:
            default_boxes.clamp_(min=0, max=1)
            
        return default_boxes


# Import itertools for the default box generator
import itertools
