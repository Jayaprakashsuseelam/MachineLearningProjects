"""
YOLO Trainer
Comprehensive training module with training loops, validation, and monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.yolo_v5 import get_yolov5_model
from training.loss import create_loss_function, LossWeightScheduler
from utils.metrics import calculate_map, calculate_precision_recall
from config.yolo_config import TRAINING_CONFIGS

class YOLOTrainer:
    """YOLO Training Manager"""
    
    def __init__(self, model: nn.Module, 
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 config: Optional[Dict[str, Any]] = None,
                 device: str = 'auto'):
        """
        Initialize YOLO trainer
        
        Args:
            model: YOLO model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TRAINING_CONFIGS['default']
        self.device = self._set_device(device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize components
        self.criterion = create_loss_function(self.config)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.loss_scheduler = LossWeightScheduler(self.config)
        
        # Training state
        self.current_epoch = 0
        self.best_map = 0.0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_map': [],
            'val_map': [],
            'learning_rate': []
        }
        
        # Logging
        self.log_dir = Path(self.config.get('log_dir', 'logs'))
        self.log_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Checkpointing
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Mixed precision training
        self.use_amp = self.config.get('amp', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
    
    def _set_device(self, device: str) -> torch.device:
        """Set the device for training"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = 'cpu'
        
        return torch.device(device)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        optimizer_type = self.config.get('optimizer', 'SGD')
        lr = self.config.get('learning_rate', 0.01)
        momentum = self.config.get('momentum', 0.937)
        weight_decay = self.config.get('weight_decay', 0.0005)
        
        if optimizer_type.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'cosine')
        epochs = self.config.get('epochs', 300)
        
        if scheduler_type.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        elif scheduler_type.lower() == 'step':
            step_size = self.config.get('step_size', 100)
            gamma = self.config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type.lower() == 'multistep':
            milestones = self.config.get('milestones', [100, 200])
            gamma = self.config.get('gamma', 0.1)
            return optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # Move data to device
            images = images.to(self.device)
            targets = [target.to(self.device) for target in targets]
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    predictions = self.model(images)
                    loss_dict = self.criterion(predictions, targets)
                    loss = loss_dict['total_loss']
            else:
                predictions = self.model(images)
                loss_dict = self.criterion(predictions, targets)
                loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
            
            # Log to tensorboard
            if batch_idx % 10 == 0:
                step = self.current_epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), step)
                self.writer.add_scalar('Train/Coord_Loss', loss_dict['coord_loss'].item(), step)
                self.writer.add_scalar('Train/Obj_Loss', loss_dict['obj_loss'].item(), step)
                self.writer.add_scalar('Train/NoObj_Loss', loss_dict['noobj_loss'].item(), step)
                self.writer.add_scalar('Train/Class_Loss', loss_dict['class_loss'].item(), step)
        
        return {'train_loss': total_loss / num_batches}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validation'):
                # Move data to device
                images = images.to(self.device)
                targets = [target.to(self.device) for target in targets]
                
                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(images)
                        loss_dict = self.criterion(predictions, targets)
                        loss = loss_dict['total_loss']
                else:
                    predictions = self.model(images)
                    loss_dict = self.criterion(predictions, targets)
                    loss = loss_dict['total_loss']
                
                total_loss += loss.item()
                
                # Store predictions and targets for mAP calculation
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Calculate mAP
        val_loss = total_loss / len(self.val_loader)
        val_map = self._calculate_map(all_predictions, all_targets)
        
        return {
            'val_loss': val_loss,
            'val_map': val_map
        }
    
    def _calculate_map(self, predictions: List[torch.Tensor], 
                      targets: List[torch.Tensor]) -> float:
        """Calculate mAP for validation set"""
        # This is a simplified mAP calculation
        # In practice, you would need to convert predictions to detection format
        # and use a proper evaluation library like pycocotools
        
        # For now, return a placeholder
        return 0.0
    
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Train the model
        
        Args:
            num_epochs: Number of epochs to train (overrides config)
            
        Returns:
            Training history
        """
        epochs = num_epochs or self.config.get('epochs', 300)
        
        print(f"Starting training for {epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Update loss weights
            self.loss_scheduler.step()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            self.training_history['train_loss'].append(train_metrics['train_loss'])
            self.training_history['learning_rate'].append(current_lr)
            
            if val_metrics:
                self.training_history['val_loss'].append(val_metrics['val_loss'])
                self.training_history['val_map'].append(val_metrics['val_map'])
            
            # Log to tensorboard
            self.writer.add_scalar('Epoch/Train_Loss', train_metrics['train_loss'], epoch)
            self.writer.add_scalar('Epoch/Learning_Rate', current_lr, epoch)
            
            if val_metrics:
                self.writer.add_scalar('Epoch/Val_Loss', val_metrics['val_loss'], epoch)
                self.writer.add_scalar('Epoch/Val_mAP', val_metrics['val_map'], epoch)
            
            # Print progress
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            if val_metrics:
                print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                print(f"  Val mAP: {val_metrics['val_map']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_period', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
            
            # Save best model
            if val_metrics and val_metrics['val_map'] > self.best_map:
                self.best_map = val_metrics['val_map']
                self.save_checkpoint('best_model.pth')
                print(f"  New best model saved! mAP: {self.best_map:.4f}")
        
        # Save final model
        self.save_checkpoint('final_model.pth')
        
        # Close tensorboard writer
        self.writer.close()
        
        return self.training_history
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_map': self.best_map,
            'training_history': self.training_history,
            'config': self.config
        }
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_map = checkpoint['best_map']
        self.training_history = checkpoint['training_history']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch + 1}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        axes[0, 0].plot(self.training_history['train_loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        
        # Validation loss
        if self.training_history['val_loss']:
            axes[0, 1].plot(self.training_history['val_loss'])
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
        
        # Learning rate
        axes[1, 0].plot(self.training_history['learning_rate'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        
        # Validation mAP
        if self.training_history['val_map']:
            axes[1, 1].plot(self.training_history['val_map'])
            axes[1, 1].set_title('Validation mAP')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('mAP')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training history plot saved to {save_path}")
        else:
            plt.show()
    
    def export_model(self, export_path: str, format: str = 'torchscript'):
        """Export trained model"""
        self.model.eval()
        
        if format == 'torchscript':
            # Create dummy input
            dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
            
            # Export to TorchScript
            traced_model = torch.jit.trace(self.model, dummy_input)
            torch.jit.save(traced_model, export_path)
            
        elif format == 'onnx':
            # Create dummy input
            dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
            
            # Export to ONNX
            torch.onnx.export(
                self.model,
                dummy_input,
                export_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        print(f"Model exported to {export_path}")

# Convenience functions
def create_trainer(model: nn.Module, 
                  train_loader: DataLoader,
                  val_loader: Optional[DataLoader] = None,
                  config: Optional[Dict[str, Any]] = None) -> YOLOTrainer:
    """Create a YOLO trainer"""
    return YOLOTrainer(model, train_loader, val_loader, config)

def train_yolo_model(model: nn.Module,
                    train_loader: DataLoader,
                    val_loader: Optional[DataLoader] = None,
                    config: Optional[Dict[str, Any]] = None,
                    num_epochs: Optional[int] = None) -> Dict[str, List[float]]:
    """Quick function to train a YOLO model"""
    trainer = create_trainer(model, train_loader, val_loader, config)
    return trainer.train(num_epochs)
