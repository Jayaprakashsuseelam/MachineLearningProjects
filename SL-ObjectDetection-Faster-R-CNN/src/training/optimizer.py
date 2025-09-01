"""
Optimizer and scheduler factory functions for Faster R-CNN training
"""
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,
    ReduceLROnPlateau, OneCycleLR, CosineAnnealingWarmRestarts
)
from typing import Dict, List, Optional, Union, Any


def get_optimizer(
    model: torch.nn.Module,
    optimizer_name: str = "SGD",
    lr: float = 0.001,
    weight_decay: float = 0.0005,
    momentum: float = 0.9,
    nesterov: bool = False,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Factory function to create an optimizer for Faster R-CNN training.
    
    Args:
        model: The model to optimize
        optimizer_name: Name of the optimizer ('SGD', 'Adam', 'AdamW', 'RMSprop')
        lr: Learning rate
        weight_decay: Weight decay (L2 regularization)
        momentum: Momentum for SGD
        nesterov: Whether to use Nesterov momentum
        **kwargs: Additional optimizer-specific arguments
        
    Returns:
        Configured optimizer
    """
    # Filter out parameters that don't require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_name.lower() == "sgd":
        return optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            **kwargs
        )
    
    elif optimizer_name.lower() == "adam":
        return optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    
    elif optimizer_name.lower() == "adamw":
        return optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    
    elif optimizer_name.lower() == "rmsprop":
        return optim.RMSprop(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            **kwargs
        )
    
    elif optimizer_name.lower() == "adagrad":
        return optim.Adagrad(
            params,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "StepLR",
    step_size: int = 7,
    gamma: float = 0.1,
    milestones: Optional[List[int]] = None,
    patience: int = 10,
    factor: float = 0.1,
    min_lr: float = 1e-6,
    T_max: int = 100,
    T_0: int = 10,
    **kwargs
) -> Any:
    """
    Factory function to create a learning rate scheduler.
    
    Args:
        optimizer: The optimizer to schedule
        scheduler_name: Name of the scheduler
        step_size: Step size for StepLR
        gamma: Multiplicative factor for StepLR and MultiStepLR
        milestones: List of epoch indices for MultiStepLR
        patience: Number of epochs with no improvement for ReduceLROnPlateau
        factor: Factor by which to reduce learning rate
        min_lr: Minimum learning rate
        T_max: Maximum number of iterations for CosineAnnealingLR
        T_0: Number of iterations for the first restart
        **kwargs: Additional scheduler-specific arguments
        
    Returns:
        Configured scheduler
    """
    if scheduler_name.lower() == "steplr":
        return StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
            **kwargs
        )
    
    elif scheduler_name.lower() == "multisteplr":
        if milestones is None:
            milestones = [30, 80]
        return MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
            **kwargs
        )
    
    elif scheduler_name.lower() == "exponentiallr":
        return ExponentialLR(
            optimizer,
            gamma=gamma,
            **kwargs
        )
    
    elif scheduler_name.lower() == "cosineannealinglr":
        return CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=min_lr,
            **kwargs
        )
    
    elif scheduler_name.lower() == "cosineannealingwarmrestarts":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=2,
            eta_min=min_lr,
            **kwargs
        )
    
    elif scheduler_name.lower() == "reducelronplateau":
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            **kwargs
        )
    
    elif scheduler_name.lower() == "onecyclelr":
        # Default to 100 epochs if not specified
        total_steps = kwargs.get('total_steps', 100)
        return OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', 0.01),
            total_steps=total_steps,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def get_optimizer_with_scheduler(
    model: torch.nn.Module,
    optimizer_config: Dict[str, Any],
    scheduler_config: Dict[str, Any]
) -> tuple[torch.optim.Optimizer, Any]:
    """
    Create both optimizer and scheduler from configurations.
    
    Args:
        model: The model to optimize
        optimizer_config: Dictionary containing optimizer configuration
        scheduler_config: Dictionary containing scheduler configuration
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    optimizer = get_optimizer(model, **optimizer_config)
    scheduler = get_scheduler(optimizer, **scheduler_config)
    
    return optimizer, scheduler


def create_optimizer_groups(
    model: torch.nn.Module,
    base_lr: float = 0.001,
    backbone_lr_multiplier: float = 0.1,
    rpn_lr_multiplier: float = 1.0,
    roi_head_lr_multiplier: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Create parameter groups with different learning rates for different parts of the model.
    
    This is useful for Faster R-CNN where different components may benefit from
    different learning rates (e.g., lower LR for pre-trained backbone).
    
    Args:
        model: The Faster R-CNN model
        base_lr: Base learning rate
        backbone_lr_multiplier: Learning rate multiplier for backbone
        rpn_lr_multiplier: Learning rate multiplier for RPN
        roi_head_lr_multiplier: Learning rate multiplier for ROI head
        
    Returns:
        List of parameter groups for the optimizer
    """
    param_groups = []
    
    # Backbone parameters (usually pre-trained, lower LR)
    backbone_params = []
    # RPN parameters
    rpn_params = []
    # ROI head parameters
    roi_head_params = []
    # Other parameters
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'backbone' in name:
            backbone_params.append(param)
        elif 'rpn' in name:
            rpn_params.append(param)
        elif 'roi_heads' in name or 'box_head' in name or 'box_predictor' in name:
            roi_head_params.append(param)
        else:
            other_params.append(param)
    
    # Add parameter groups
    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': base_lr * backbone_lr_multiplier,
            'name': 'backbone'
        })
    
    if rpn_params:
        param_groups.append({
            'params': rpn_params,
            'lr': base_lr * rpn_lr_multiplier,
            'name': 'rpn'
        })
    
    if roi_head_params:
        param_groups.append({
            'params': roi_head_params,
            'lr': base_lr * roi_head_lr_multiplier,
            'name': 'roi_head'
        })
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'name': 'other'
        })
    
    return param_groups


def get_optimizer_with_groups(
    model: torch.nn.Module,
    optimizer_name: str = "SGD",
    base_lr: float = 0.001,
    backbone_lr_multiplier: float = 0.1,
    rpn_lr_multiplier: float = 1.0,
    roi_head_lr_multiplier: float = 1.0,
    weight_decay: float = 0.0005,
    momentum: float = 0.9,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create optimizer with parameter groups for different model components.
    
    Args:
        model: The Faster R-CNN model
        optimizer_name: Name of the optimizer
        base_lr: Base learning rate
        backbone_lr_multiplier: Learning rate multiplier for backbone
        rpn_lr_multiplier: Learning rate multiplier for RPN
        roi_head_lr_multiplier: Learning rate multiplier for ROI head
        weight_decay: Weight decay
        momentum: Momentum for SGD
        **kwargs: Additional optimizer arguments
        
    Returns:
        Configured optimizer with parameter groups
    """
    param_groups = create_optimizer_groups(
        model,
        base_lr=base_lr,
        backbone_lr_multiplier=backbone_lr_multiplier,
        rpn_lr_multiplier=rpn_lr_multiplier,
        roi_head_lr_multiplier=roi_head_lr_multiplier
    )
    
    if optimizer_name.lower() == "sgd":
        return optim.SGD(
            param_groups,
            momentum=momentum,
            weight_decay=weight_decay,
            **kwargs
        )
    
    elif optimizer_name.lower() == "adam":
        return optim.Adam(
            param_groups,
            weight_decay=weight_decay,
            **kwargs
        )
    
    elif optimizer_name.lower() == "adamw":
        return optim.AdamW(
            param_groups,
            weight_decay=weight_decay,
            **kwargs
        )
    
    else:
        # For other optimizers, use the simple approach
        return get_optimizer(model, optimizer_name, base_lr, weight_decay, momentum, **kwargs)


def print_optimizer_info(optimizer: torch.optim.Optimizer) -> None:
    """
    Print information about the optimizer configuration.
    
    Args:
        optimizer: The optimizer to inspect
    """
    print(f"Optimizer: {type(optimizer).__name__}")
    
    if hasattr(optimizer, 'param_groups'):
        for i, group in enumerate(optimizer.param_groups):
            print(f"  Group {i}:")
            if 'name' in group:
                print(f"    Name: {group['name']}")
            print(f"    Learning Rate: {group['lr']}")
            print(f"    Weight Decay: {group.get('weight_decay', 0)}")
            if 'momentum' in group:
                print(f"    Momentum: {group['momentum']}")
            print(f"    Parameters: {len(group['params'])}")
    else:
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']}")
        print(f"  Weight Decay: {optimizer.param_groups[0].get('weight_decay', 0)}")
        if 'momentum' in optimizer.param_groups[0]:
            print(f"  Momentum: {optimizer.param_groups[0]['momentum']}")


def print_scheduler_info(scheduler: Any) -> None:
    """
    Print information about the scheduler configuration.
    
    Args:
        scheduler: The scheduler to inspect
    """
    print(f"Scheduler: {type(scheduler).__name__}")
    
    if hasattr(scheduler, 'step_size'):
        print(f"  Step Size: {scheduler.step_size}")
    if hasattr(scheduler, 'gamma'):
        print(f"  Gamma: {scheduler.gamma}")
    if hasattr(scheduler, 'milestones'):
        print(f"  Milestones: {scheduler.milestones}")
    if hasattr(scheduler, 'patience'):
        print(f"  Patience: {scheduler.patience}")
    if hasattr(scheduler, 'factor'):
        print(f"  Factor: {scheduler.factor}")
    if hasattr(scheduler, 'min_lr'):
        print(f"  Min LR: {scheduler.min_lr}")
    if hasattr(scheduler, 'T_max'):
        print(f"  T_max: {scheduler.T_max}")
    if hasattr(scheduler, 'T_0'):
        print(f"  T_0: {scheduler.T_0}")
