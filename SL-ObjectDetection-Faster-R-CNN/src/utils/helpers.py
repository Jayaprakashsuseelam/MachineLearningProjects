"""
Helper utility functions for Faster R-CNN
"""
import os
import json
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
from pathlib import Path
import pickle
import yaml


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available devices for training/inference.
    
    Returns:
        Dictionary containing device information
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': 'cpu'
    }
    
    if torch.cuda.is_available():
        device_info['current_device'] = 'cuda'
        device_info['cuda_version'] = torch.version.cuda
        device_info['cudnn_version'] = torch.backends.cudnn.version()
        device_info['cudnn_enabled'] = torch.backends.cudnn.enabled
        
        # Get GPU information
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                'id': i,
                'name': gpu_props.name,
                'memory_total': gpu_props.total_memory / 1024**3,  # GB
                'memory_allocated': torch.cuda.memory_allocated(i) / 1024**3,  # GB
                'memory_cached': torch.cuda.memory_reserved(i) / 1024**3,  # GB
                'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
            })
        device_info['gpus'] = gpu_info
    
    return device_info


def get_device(device_name: Optional[str] = None) -> torch.device:
    """
    Get the best available device for training/inference.
    
    Args:
        device_name: Specific device name ('cpu', 'cuda', 'cuda:0', etc.)
        
    Returns:
        torch.device object
    """
    if device_name:
        return torch.device(device_name)
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def print_device_info() -> None:
    """
    Print detailed device information to console.
    """
    device_info = get_device_info()
    
    print("=== Device Information ===")
    print(f"CUDA Available: {device_info['cuda_available']}")
    print(f"Current Device: {device_info['current_device']}")
    
    if device_info['cuda_available']:
        print(f"CUDA Version: {device_info['cuda_version']}")
        print(f"cuDNN Version: {device_info['cudnn_version']}")
        print(f"cuDNN Enabled: {device_info['cudnn_enabled']}")
        print(f"GPU Count: {device_info['cuda_device_count']}")
        
        for gpu in device_info['gpus']:
            print(f"\nGPU {gpu['id']}: {gpu['name']}")
            print(f"  Memory: {gpu['memory_total']:.1f} GB total")
            print(f"  Allocated: {gpu['memory_allocated']:.1f} GB")
            print(f"  Cached: {gpu['memory_cached']:.1f} GB")
            print(f"  Compute Capability: {gpu['compute_capability']}")
    
    print("=" * 30)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    filepath: str,
    save_optimizer: bool = True,
    save_scheduler: bool = True
) -> None:
    """
    Save a training checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer state
        scheduler: The scheduler state (optional)
        epoch: Current epoch number
        loss: Current loss value
        metrics: Dictionary of evaluation metrics
        filepath: Path to save the checkpoint
        save_optimizer: Whether to save optimizer state
        save_scheduler: Whether to save scheduler state
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    
    if save_optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if save_scheduler and scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save checkpoint
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = torch.device('cpu'),
    load_optimizer: bool = True,
    load_scheduler: bool = True
) -> Dict[str, Any]:
    """
    Load a training checkpoint.
    
    Args:
        filepath: Path to the checkpoint file
        model: The model to load weights into
        optimizer: The optimizer to load state into (optional)
        scheduler: The scheduler to load state into (optional)
        device: Device to load the checkpoint on
        load_optimizer: Whether to load optimizer state
        load_scheduler: Whether to load scheduler state
        
    Returns:
        Dictionary containing checkpoint information
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if requested
    if load_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if requested
    if load_scheduler and scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    
    return checkpoint


def save_config(config: Dict[str, Any], filepath: str) -> None:
    """
    Save configuration to a file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save the configuration
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Determine file format based on extension
    if filepath.endswith('.json'):
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    elif filepath.endswith('.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(config, f)
    else:
        # Default to JSON
        with open(filepath + '.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {filepath}")


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Args:
        filepath: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    # Determine file format based on extension
    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            config = json.load(f)
    elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
    elif filepath.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            config = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    print(f"Configuration loaded from {filepath}")
    return config


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count the number of parameters in a model.
    
    Args:
        model: The model to count parameters for
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def print_model_summary(model: nn.Module, input_size: Tuple[int, ...] = (3, 800, 800)) -> None:
    """
    Print a summary of the model architecture and parameters.
    
    Args:
        model: The model to summarize
        input_size: Input tensor size for parameter calculation
    """
    param_counts = count_parameters(model)
    
    print("=== Model Summary ===")
    print(f"Total Parameters: {param_counts['total']:,}")
    print(f"Trainable Parameters: {param_counts['trainable']:,}")
    print(f"Non-trainable Parameters: {param_counts['non_trainable']:,}")
    
    # Calculate model size in MB
    model_size = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_mb = model_size / (1024 * 1024)
    print(f"Model Size: {model_size_mb:.2f} MB")
    
    print("\n=== Model Architecture ===")
    print(model)
    print("=" * 30)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """
    Create a directory for an experiment.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
        
    Returns:
        Path to the experiment directory
    """
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    
    # Create subdirectories
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "configs"), exist_ok=True)
    
    print(f"Experiment directory created: {experiment_dir}")
    return experiment_dir


def save_experiment_results(
    results: Dict[str, Any],
    experiment_dir: str,
    filename: str = "results.json"
) -> None:
    """
    Save experiment results to a file.
    
    Args:
        results: Dictionary containing experiment results
        experiment_dir: Directory to save results in
        filename: Name of the results file
    """
    results_file = os.path.join(experiment_dir, filename)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_numpy(results)
    
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Experiment results saved to {results_file}")


def get_latest_checkpoint(checkpoint_dir: str, pattern: str = "*.pth") -> Optional[str]:
    """
    Get the path to the latest checkpoint file.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        pattern: File pattern to match
        
    Returns:
        Path to the latest checkpoint, or None if no checkpoints found
    """
    import glob
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    return checkpoint_files[0]


def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def calculate_eta(
    current_step: int,
    total_steps: int,
    elapsed_time: float
) -> str:
    """
    Calculate estimated time of arrival (ETA) for completion.
    
    Args:
        current_step: Current step number
        total_steps: Total number of steps
        elapsed_time: Time elapsed so far
        
    Returns:
        ETA string
    """
    if current_step == 0:
        return "Unknown"
    
    steps_remaining = total_steps - current_step
    time_per_step = elapsed_time / current_step
    eta_seconds = steps_remaining * time_per_step
    
    return format_time(eta_seconds)


def print_progress_bar(
    current: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    length: int = 50,
    fill: str = "█"
) -> None:
    """
    Print a progress bar to the console.
    
    Args:
        current: Current progress value
        total: Total value
        prefix: Prefix string
        suffix: Suffix string
        length: Length of the progress bar
        fill: Character to fill the bar with
    """
    percent = f"{100 * (current / float(total)):.1f}"
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    
    if current == total:
        print()
