"""
Advanced Model Architectures for CIFAR-10 Classification

This module contains various state-of-the-art neural network architectures
specifically adapted for CIFAR-10 image classification.

Author: AI Assistant
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel attention.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x

class ChannelAttention(nn.Module):
    """Channel attention module."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Spatial attention module."""
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class EfficientNetBlock(nn.Module):
    """
    EfficientNet building block with depthwise separable convolution.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, expand_ratio: int = 6, se_ratio: float = 0.25):
        super(EfficientNetBlock, self).__init__()
        
        self.expand_ratio = expand_ratio
        self.stride = stride
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, expanded_channels, 1, bias=False)
            self.expand_bn = nn.BatchNorm2d(expanded_channels)
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(
            expanded_channels, expanded_channels, kernel_size, 
            stride=stride, padding=kernel_size//2, groups=expanded_channels, bias=False
        )
        self.depthwise_bn = nn.BatchNorm2d(expanded_channels)
        
        # Squeeze-and-Excitation
        se_channels = max(1, int(expanded_channels * se_ratio))
        self.se = SEBlock(expanded_channels, se_channels)
        
        # Projection phase
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.use_skip = stride == 1 and in_channels == out_channels
    
    def forward(self, x):
        residual = x
        
        # Expansion
        if self.expand_ratio != 1:
            x = F.relu(self.expand_bn(self.expand_conv(x)))
        
        # Depthwise convolution
        x = F.relu(self.depthwise_bn(self.depthwise_conv(x)))
        
        # Squeeze-and-Excitation
        x = self.se(x)
        
        # Projection
        x = self.project_bn(self.project_conv(x))
        
        # Skip connection
        if self.use_skip:
            x = x + residual
        
        return x

class DenseBlock(nn.Module):
    """
    DenseNet dense block.
    """
    
    def __init__(self, num_layers: int, in_channels: int, growth_rate: int, 
                 bn_size: int = 4, drop_rate: float = 0.0):
        super(DenseBlock, self).__init__()
        
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, drop_rate))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)

class DenseLayer(nn.Module):
    """
    DenseNet dense layer.
    """
    
    def __init__(self, in_channels: int, growth_rate: int, bn_size: int, drop_rate: float):
        super(DenseLayer, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, 3, padding=1, bias=False)
        self.drop_rate = drop_rate
    
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out

class TransitionBlock(nn.Module):
    """
    DenseNet transition block.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(TransitionBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)
    
    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out

class AdvancedCNN(nn.Module):
    """
    Advanced CNN with attention mechanisms and modern techniques.
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5, 
                 use_attention: bool = True, attention_type: str = 'se'):
        super(AdvancedCNN, self).__init__()
        
        self.use_attention = use_attention
        self.attention_type = attention_type
        
        # Feature extraction with attention
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Second block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        # Attention modules
        if self.use_attention:
            if attention_type == 'se':
                self.attention1 = SEBlock(64)
                self.attention2 = SEBlock(128)
                self.attention3 = SEBlock(256)
            elif attention_type == 'cbam':
                self.attention1 = CBAMBlock(64)
                self.attention2 = CBAMBlock(128)
                self.attention3 = CBAMBlock(256)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # First block
        x = self.features[0](x)  # Conv2d
        x = self.features[1](x)  # BatchNorm
        x = self.features[2](x)  # ReLU
        x = self.features[3](x)  # Conv2d
        x = self.features[4](x)  # BatchNorm
        x = self.features[5](x)  # ReLU
        if self.use_attention:
            x = self.attention1(x)
        x = self.features[6](x)  # MaxPool2d
        x = self.features[7](x)  # Dropout2d
        
        # Second block
        x = self.features[8](x)  # Conv2d
        x = self.features[9](x)  # BatchNorm
        x = self.features[10](x) # ReLU
        x = self.features[11](x) # Conv2d
        x = self.features[12](x) # BatchNorm
        x = self.features[13](x) # ReLU
        if self.use_attention:
            x = self.attention2(x)
        x = self.features[14](x) # MaxPool2d
        x = self.features[15](x) # Dropout2d
        
        # Third block
        x = self.features[16](x) # Conv2d
        x = self.features[17](x) # BatchNorm
        x = self.features[18](x) # ReLU
        x = self.features[19](x) # Conv2d
        x = self.features[20](x) # BatchNorm
        x = self.features[21](x) # ReLU
        if self.use_attention:
            x = self.attention3(x)
        x = self.features[22](x) # MaxPool2d
        x = self.features[23](x) # Dropout2d
        
        x = self.classifier(x)
        return x

class EfficientNetCIFAR10(nn.Module):
    """
    EfficientNet adapted for CIFAR-10.
    """
    
    def __init__(self, num_classes: int = 10, width_coefficient: float = 1.0, 
                 depth_coefficient: float = 1.0, dropout_rate: float = 0.2):
        super(EfficientNetCIFAR10, self).__init__()
        
        # Calculate scaled dimensions
        def round_filters(filters):
            multiplier = width_coefficient
            divisor = 8
            min_depth = None
            filters *= multiplier
            min_depth = min_depth or divisor
            new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
            if new_filters < 0.9 * filters:
                new_filters += divisor
            return int(new_filters)
        
        def round_repeats(repeats):
            return int(math.ceil(depth_coefficient * repeats))
        
        # Stem
        in_channels = 3
        out_channels = round_filters(32)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # EfficientNet blocks
        self.blocks = nn.ModuleList()
        
        # Block 1
        self.blocks.append(EfficientNetBlock(out_channels, round_filters(16), 3, 1, 1))
        out_channels = round_filters(16)
        
        # Block 2
        self.blocks.append(EfficientNetBlock(out_channels, round_filters(24), 3, 1, 6))
        out_channels = round_filters(24)
        
        # Block 3
        self.blocks.append(EfficientNetBlock(out_channels, round_filters(40), 5, 2, 6))
        out_channels = round_filters(40)
        
        # Block 4
        self.blocks.append(EfficientNetBlock(out_channels, round_filters(80), 3, 2, 6))
        out_channels = round_filters(80)
        
        # Block 5
        self.blocks.append(EfficientNetBlock(out_channels, round_filters(112), 5, 1, 6))
        out_channels = round_filters(112)
        
        # Block 6
        self.blocks.append(EfficientNetBlock(out_channels, round_filters(192), 5, 2, 6))
        out_channels = round_filters(192)
        
        # Block 7
        self.blocks.append(EfficientNetBlock(out_channels, round_filters(320), 3, 1, 6))
        out_channels = round_filters(320)
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(out_channels, round_filters(1280), 1, bias=False),
            nn.BatchNorm2d(round_filters(1280)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(round_filters(1280), num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.head(x)
        return x

class DenseNetCIFAR10(nn.Module):
    """
    DenseNet adapted for CIFAR-10.
    """
    
    def __init__(self, num_classes: int = 10, growth_rate: int = 12, 
                 num_layers: List[int] = [6, 12, 24, 16], 
                 num_init_features: int = 24, drop_rate: float = 0.0):
        super(DenseNetCIFAR10, self).__init__()
        
        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True)
        )
        
        num_features = num_init_features
        
        # Dense blocks and transition blocks
        for i, num_layer in enumerate(num_layers):
            # Dense block
            block = DenseBlock(num_layer, num_features, growth_rate, drop_rate=drop_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layer * growth_rate
            
            # Transition block (except for the last block)
            if i != len(num_layers) - 1:
                trans = TransitionBlock(num_features, num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        out = self.classifier(features)
        return out

class MobileNetCIFAR10(nn.Module):
    """
    MobileNet adapted for CIFAR-10.
    """
    
    def __init__(self, num_classes: int = 10, width_multiplier: float = 1.0):
        super(MobileNetCIFAR10, self).__init__()
        
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
        
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
        
        # Calculate output channels based on width multiplier
        def make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        
        input_channel = make_divisible(32 * width_multiplier, 8)
        self.features = [conv_bn(3, input_channel, 1)]
        
        # MobileNet blocks
        mobile_net_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        for t, c, n, s in mobile_net_setting:
            output_channel = make_divisible(c * width_multiplier, 8)
            for i in range(n):
                if i == 0:
                    self.features.append(conv_dw(input_channel, output_channel, s))
                else:
                    self.features.append(conv_dw(input_channel, output_channel, 1))
                input_channel = output_channel
        
        self.features = nn.Sequential(*self.features)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(input_channel, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_model(model_name: str, num_classes: int = 10, **kwargs):
    """
    Factory function to create different model architectures.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        **kwargs: Additional arguments for model creation
        
    Returns:
        Model instance
    """
    models = {
        'cnn': lambda: CIFAR10CNN(num_classes, **kwargs),
        'resnet': lambda: ResNetCIFAR10(num_classes),
        'advanced_cnn': lambda: AdvancedCNN(num_classes, **kwargs),
        'efficientnet': lambda: EfficientNetCIFAR10(num_classes, **kwargs),
        'densenet': lambda: DenseNetCIFAR10(num_classes, **kwargs),
        'mobilenet': lambda: MobileNetCIFAR10(num_classes, **kwargs)
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")
    
    return models[model_name]()

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test different model architectures
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models_to_test = ['cnn', 'resnet', 'advanced_cnn', 'efficientnet', 'densenet', 'mobilenet']
    
    print("Model Architecture Comparison:")
    print("=" * 60)
    
    for model_name in models_to_test:
        try:
            model = get_model(model_name)
            param_count = count_parameters(model)
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, 32, 32)
            output = model(dummy_input)
            
            print(f"{model_name.upper():<15} | Parameters: {param_count:>8,} | Output shape: {output.shape}")
            
        except Exception as e:
            print(f"{model_name.upper():<15} | Error: {str(e)}")
    
    print("=" * 60)
