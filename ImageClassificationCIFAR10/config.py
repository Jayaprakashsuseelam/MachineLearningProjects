"""
Configuration file for CIFAR-10 Image Classification

This file contains default configurations and hyperparameters for training
different model architectures on the CIFAR-10 dataset.

Author: AI Assistant
Date: 2024
"""

import torch
from typing import Dict, Any, List

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset configuration
DATASET_CONFIG = {
    'data_dir': './data',
    'download': True,
    'train_val_split': 0.8,
    'num_classes': 10,
    'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck'],
    'image_size': (32, 32),
    'channels': 3
}

# Data augmentation configuration
AUGMENTATION_CONFIG = {
    'random_crop': {
        'size': 32,
        'padding': 4
    },
    'random_horizontal_flip': {
        'probability': 0.5
    },
    'random_rotation': {
        'degrees': 10
    },
    'color_jitter': {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1
    },
    'cutout': {
        'n_holes': 1,
        'length': 8
    }
}

# Normalization values for CIFAR-10
NORMALIZATION = {
    'mean': (0.4914, 0.4822, 0.4465),
    'std': (0.2023, 0.1994, 0.2010)
}

# Training configuration
TRAINING_CONFIG = {
    'epochs': 50,
    'batch_size': 128,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'num_workers': 2,
    'pin_memory': True,
    'gradient_clipping': None,  # Set to a value to enable gradient clipping
    'mixed_precision': False,  # Enable automatic mixed precision
    'save_best_model': True,
    'early_stopping': {
        'enabled': False,
        'patience': 10,
        'min_delta': 0.001
    }
}

# Model-specific configurations
MODEL_CONFIGS = {
    'cnn': {
        'dropout_rate': 0.5,
        'use_batch_norm': True,
        'activation': 'relu',
        'pooling': 'max'
    },
    'resnet': {
        'num_blocks': [2, 2, 2],
        'num_channels': [64, 128, 256],
        'strides': [1, 2, 2],
        'use_bottleneck': False
    },
    'efficientnet': {
        'width_coefficient': 1.0,
        'depth_coefficient': 1.0,
        'dropout_rate': 0.2,
        'se_ratio': 0.25
    },
    'densenet': {
        'growth_rate': 12,
        'num_layers': [6, 12, 24, 16],
        'num_init_features': 24,
        'drop_rate': 0.0,
        'bn_size': 4
    },
    'mobilenet': {
        'width_multiplier': 1.0,
        'resolution_multiplier': 1.0
    },
    'advanced_cnn': {
        'dropout_rate': 0.5,
        'use_attention': True,
        'attention_type': 'se',  # 'se' or 'cbam'
        'attention_reduction': 16
    }
}

# Optimizer configurations
OPTIMIZER_CONFIGS = {
    'adam': {
        'lr': 0.001,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 1e-4
    },
    'sgd': {
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'nesterov': True
    },
    'rmsprop': {
        'lr': 0.01,
        'alpha': 0.99,
        'eps': 1e-8,
        'weight_decay': 1e-4,
        'momentum': 0.9
    }
}

# Learning rate scheduler configurations
SCHEDULER_CONFIGS = {
    'reduce_lr_on_plateau': {
        'mode': 'min',
        'factor': 0.5,
        'patience': 5,
        'threshold': 0.0001,
        'min_lr': 1e-6
    },
    'cosine_annealing': {
        'T_max': 50,
        'eta_min': 1e-6
    },
    'step_lr': {
        'step_size': 20,
        'gamma': 0.1
    },
    'exponential_lr': {
        'gamma': 0.95
    }
}

# Loss function configurations
LOSS_CONFIGS = {
    'cross_entropy': {
        'weight': None,  # Set to class weights if needed
        'reduction': 'mean',
        'label_smoothing': 0.0
    },
    'focal_loss': {
        'alpha': 1.0,
        'gamma': 2.0,
        'reduction': 'mean'
    }
}

# Evaluation configuration
EVALUATION_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
    'save_predictions': True,
    'save_probabilities': True,
    'top_k_accuracy': [1, 5],
    'confusion_matrix': True,
    'classification_report': True
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    'save_dir': './plots',
    'dpi': 300,
    'figsize': (12, 8),
    'style': 'seaborn-v0_8',
    'color_palette': 'husl',
    'save_format': 'png',
    'show_plots': True
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': './logs/training.log',
    'console': True,
    'tensorboard': False,
    'wandb': False
}

# Experiment tracking
EXPERIMENT_CONFIG = {
    'project_name': 'cifar10_classification',
    'experiment_name': None,  # Will be auto-generated if None
    'tags': ['cifar10', 'image_classification', 'deep_learning'],
    'notes': None,
    'save_model': True,
    'save_checkpoints': True,
    'checkpoint_frequency': 10  # Save checkpoint every N epochs
}

# Hyperparameter search configuration
HYPERPARAMETER_SEARCH_CONFIG = {
    'method': 'grid',  # 'grid', 'random', 'bayesian'
    'n_trials': 20,
    'search_space': {
        'learning_rate': [0.001, 0.01, 0.1],
        'batch_size': [64, 128, 256],
        'dropout_rate': [0.3, 0.5, 0.7],
        'weight_decay': [1e-5, 1e-4, 1e-3]
    },
    'optimization_metric': 'val_accuracy',
    'direction': 'maximize'
}

# Model ensemble configuration
ENSEMBLE_CONFIG = {
    'enabled': False,
    'models': ['cnn', 'resnet', 'efficientnet'],
    'voting_method': 'soft',  # 'soft' or 'hard'
    'weights': None,  # None for equal weights
    'save_individual_predictions': True
}

# Data loading configuration
DATA_LOADING_CONFIG = {
    'batch_size': 128,
    'shuffle': True,
    'num_workers': 2,
    'pin_memory': True,
    'persistent_workers': True,
    'prefetch_factor': 2
}

# Memory optimization configuration
MEMORY_CONFIG = {
    'gradient_accumulation_steps': 1,
    'gradient_checkpointing': False,
    'empty_cache_frequency': 10,  # Clear cache every N epochs
    'max_memory_usage': 0.9  # Maximum GPU memory usage (0.0-1.0)
}

# Reproducibility configuration
REPRODUCIBILITY_CONFIG = {
    'seed': 42,
    'deterministic': True,
    'benchmark': True,  # Set to False for reproducibility
    'cudnn_deterministic': False
}

# Default experiment configurations
EXPERIMENTS = {
    'quick_test': {
        'epochs': 5,
        'batch_size': 64,
        'learning_rate': 0.01,
        'model': 'cnn',
        'augmentation': False
    },
    'standard_training': {
        'epochs': 50,
        'batch_size': 128,
        'learning_rate': 0.001,
        'model': 'resnet',
        'augmentation': True
    },
    'extensive_training': {
        'epochs': 100,
        'batch_size': 128,
        'learning_rate': 0.001,
        'model': 'efficientnet',
        'augmentation': True,
        'scheduler': 'cosine_annealing'
    },
    'model_comparison': {
        'epochs': 30,
        'batch_size': 128,
        'learning_rate': 0.001,
        'models': ['cnn', 'resnet', 'efficientnet', 'densenet', 'mobilenet'],
        'augmentation': True
    }
}

def get_config(experiment_name: str = 'standard_training') -> Dict[str, Any]:
    """
    Get configuration for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment configuration
        
    Returns:
        Dictionary containing the configuration
    """
    if experiment_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}. "
                        f"Available experiments: {list(EXPERIMENTS.keys())}")
    
    config = EXPERIMENTS[experiment_name].copy()
    
    # Add default configurations
    config.update({
        'device': DEVICE,
        'dataset': DATASET_CONFIG,
        'augmentation': AUGMENTATION_CONFIG,
        'normalization': NORMALIZATION,
        'training': TRAINING_CONFIG,
        'evaluation': EVALUATION_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'logging': LOGGING_CONFIG,
        'experiment': EXPERIMENT_CONFIG
    })
    
    return config

def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values.
    
    Args:
        config: Original configuration
        updates: Updates to apply
        
    Returns:
        Updated configuration
    """
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    return deep_update(config.copy(), updates)

def print_config(config: Dict[str, Any], indent: int = 0):
    """
    Print configuration in a readable format.
    
    Args:
        config: Configuration to print
        indent: Indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print('  ' * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print('  ' * indent + f"{key}: {value}")

if __name__ == "__main__":
    # Example usage
    print("Available Experiments:")
    for name in EXPERIMENTS.keys():
        print(f"  - {name}")
    
    print("\nStandard Training Configuration:")
    config = get_config('standard_training')
    print_config(config)
    
    print("\nCustom Configuration Example:")
    custom_config = update_config(config, {
        'training': {'epochs': 100, 'batch_size': 64},
        'model': 'efficientnet'
    })
    print_config(custom_config)
