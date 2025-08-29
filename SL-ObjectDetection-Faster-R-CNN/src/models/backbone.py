"""
Backbone network implementations for Faster R-CNN
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Tuple, Optional


class Backbone(nn.Module):
    """Base backbone class for feature extraction"""
    
    def __init__(self, name: str = "resnet50", pretrained: bool = True, 
                 freeze_layers: int = 0):
        super().__init__()
        self.name = name
        self.pretrained = pretrained
        self.freeze_layers = freeze_layers
        
        self._build_backbone()
        self._freeze_layers()
    
    def _build_backbone(self):
        """Build the backbone network"""
        if self.name.startswith("resnet"):
            self._build_resnet()
        elif self.name.startswith("vgg"):
            self._build_vgg()
        else:
            raise ValueError(f"Unsupported backbone: {self.name}")
    
    def _build_resnet(self):
        """Build ResNet backbone"""
        if self.name == "resnet18":
            backbone = models.resnet18(pretrained=self.pretrained)
            self.out_channels = 512
        elif self.name == "resnet34":
            backbone = models.resnet34(pretrained=self.pretrained)
            self.out_channels = 512
        elif self.name == "resnet50":
            backbone = models.resnet50(pretrained=self.pretrained)
            self.out_channels = 2048
        elif self.name == "resnet101":
            backbone = models.resnet101(pretrained=self.pretrained)
            self.out_channels = 2048
        elif self.name == "resnet152":
            backbone = models.resnet152(pretrained=self.pretrained)
            self.out_channels = 2048
        else:
            raise ValueError(f"Unsupported ResNet version: {self.name}")
        
        # Extract layers for feature extraction
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # ResNet layer 1
        self.layer2 = backbone.layer2  # ResNet layer 2
        self.layer3 = backbone.layer3  # ResNet layer 3
        self.layer4 = backbone.layer4  # ResNet layer 4
        
        # Remove classification layers
        self.avgpool = None
        self.fc = None
    
    def _build_vgg(self):
        """Build VGG backbone"""
        if self.name == "vgg11":
            backbone = models.vgg11(pretrained=self.pretrained)
            self.out_channels = 512
        elif self.name == "vgg13":
            backbone = models.vgg13(pretrained=self.pretrained)
            self.out_channels = 512
        elif self.name == "vgg16":
            backbone = models.vgg16(pretrained=self.pretrained)
            self.out_channels = 512
        elif self.name == "vgg19":
            backbone = models.vgg19(pretrained=self.pretrained)
            self.out_channels = 512
        else:
            raise ValueError(f"Unsupported VGG version: {self.name}")
        
        # Extract features from VGG
        self.features = backbone.features
        
        # Remove classification layers
        self.avgpool = None
        self.classifier = None
    
    def _freeze_layers(self):
        """Freeze specified number of layers"""
        if self.freeze_layers <= 0:
            return
        
        if self.name.startswith("resnet"):
            layers_to_freeze = [
                self.conv1, self.bn1, self.maxpool, self.layer1
            ]
            if self.freeze_layers >= 2:
                layers_to_freeze.append(self.layer2)
            if self.freeze_layers >= 3:
                layers_to_freeze.append(self.layer3)
            
            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
                    
        elif self.name.startswith("vgg"):
            # Freeze early layers in VGG
            for i, layer in enumerate(self.features):
                if i < self.freeze_layers * 5:  # Approximate layer count
                    for param in layer.parameters():
                        param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone"""
        if self.name.startswith("resnet"):
            return self._forward_resnet(x)
        elif self.name.startswith("vgg"):
            return self._forward_vgg(x)
    
    def _forward_resnet(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)  # 1/4 scale
        x = self.layer2(x)  # 1/8 scale
        x = self.layer3(x)  # 1/16 scale
        x = self.layer4(x)  # 1/32 scale
        
        return x
    
    def _forward_vgg(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through VGG"""
        x = self.features(x)
        return x
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get feature maps at different scales (for FPN support)"""
        if self.name.startswith("resnet"):
            return self._get_resnet_feature_maps(x)
        elif self.name.startswith("vgg"):
            return self._get_vgg_feature_maps(x)
    
    def _get_resnet_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get ResNet feature maps at different scales"""
        feature_maps = {}
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        feature_maps['layer1'] = x  # 1/4 scale
        
        x = self.layer2(x)
        feature_maps['layer2'] = x  # 1/8 scale
        
        x = self.layer3(x)
        feature_maps['layer3'] = x  # 1/16 scale
        
        x = self.layer4(x)
        feature_maps['layer4'] = x  # 1/32 scale
        
        return feature_maps
    
    def _get_vgg_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get VGG feature maps at different scales"""
        feature_maps = {}
        current_scale = 1
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            # Store feature maps at specific layers
            if i in [10, 17, 24, 31]:  # After maxpool layers
                feature_maps[f'layer{i}'] = x
                current_scale *= 2
        
        return feature_maps


class ResNetBackbone(Backbone):
    """ResNet-specific backbone implementation"""
    
    def __init__(self, name: str = "resnet50", pretrained: bool = True, 
                 freeze_layers: int = 0):
        if not name.startswith("resnet"):
            raise ValueError(f"ResNetBackbone only supports ResNet architectures, got {name}")
        super().__init__(name, pretrained, freeze_layers)


class VGGBackbone(Backbone):
    """VGG-specific backbone implementation"""
    
    def __init__(self, name: str = "vgg16", pretrained: bool = True, 
                 freeze_layers: int = 0):
        if not name.startswith("vgg"):
            raise ValueError(f"VGGBackbone only supports VGG architectures, got {name}")
        super().__init__(name, pretrained, freeze_layers)


def get_backbone(name: str, pretrained: bool = True, 
                freeze_layers: int = 0) -> Backbone:
    """Factory function to get backbone network"""
    return Backbone(name, pretrained, freeze_layers)
