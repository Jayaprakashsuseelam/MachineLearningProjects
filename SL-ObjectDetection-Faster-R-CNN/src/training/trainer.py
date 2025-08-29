"""
Trainer for Faster R-CNN
"""
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import json
import copy

from ..models import FasterRCNN
from ..data import VOCDataset, get_transform
from ..utils import calculate_map
from .optimizer import get_optimizer, get_scheduler


class Trainer:
    """Trainer class for Faster R-CNN"""
    
    def __init__(self, model: FasterRCNN, train_dataset: VOCDataset, 
                 val_dataset: Optional[VOCDataset] = None, config: Dict = None):
        """
        Initialize trainer
        
        Args:
            model: Faster R-CNN model
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            config: Training configuration
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or {}
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Data loaders
        self.train_loader = self._create_data_loader(train_dataset, train=True)
        self.val_loader = self._create_data_loader(val_dataset, train=False) if val_dataset else None
        
        # Optimizer and scheduler
        self.optimizer = get_optimizer(model, config.get('optimizer', {}))
        self.scheduler = get_scheduler(self.optimizer, config.get('scheduler', {}))
        
        # Training state
        self.current_epoch = 0
        self.best_map = 0.0
        self.train_losses = []
        self.val_metrics = []
        
        # Logging
        self.log_dir = config.get('log_dir', './logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Checkpointing
        self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Load checkpoint if exists
        if config.get('resume_from'):
            self.load_checkpoint(config['resume_from'])
    
    def _create_data_loader(self, dataset: VOCDataset, train: bool = True) -> DataLoader:
        """Create data loader for dataset"""
        if dataset is None:
            return None
        
        # Get transforms
        transform = get_transform(
            train=train,
            image_size=self.config.get('image_size', (800, 800)),
            max_size=self.config.get('max_size', 1000)
        )
        
        # Apply transforms to dataset
        dataset.transform = transform
        
        # Create data loader
        loader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 2),
            shuffle=train,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=self._collate_fn
        )
        
        return loader
    
    def _collate_fn(self, batch: List[Tuple]) -> Tuple[List[torch.Tensor], List[Dict]]:
        """Custom collate function for variable-sized images"""
        images = []
        targets = []
        
        for image, target in batch:
            images.append(image)
            targets.append(target)
        
        return images, targets
    
    def train(self, epochs: int, save_freq: int = 1, eval_freq: int = 1):
        """
        Train the model
        
        Args:
            epochs: Number of epochs to train
            save_freq: Frequency of saving checkpoints
            eval_freq: Frequency of evaluation
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Training on device: {self.device}")
        print(f"Training samples: {len(self.train_dataset)}")
        if self.val_dataset:
            print(f"Validation samples: {len(self.val_dataset)}")
        
        # Training loop
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_loss = self._train_epoch()
            
            # Log training loss
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.train_losses.append(train_loss)
            
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}")
            
            # Evaluation
            if self.val_loader and (epoch + 1) % eval_freq == 0:
                val_metrics = self._validate_epoch()
                self.val_metrics.append(val_metrics)
                
                # Log validation metrics
                for metric_name, metric_value in val_metrics.items():
                    self.writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)
                
                print(f"Validation - mAP@0.5: {val_metrics['mAP_0.5']:.4f}, "
                      f"mAP@0.5:0.95: {val_metrics['mAP_0.5_0.95']:.4f}")
                
                # Save best model
                if val_metrics['mAP_0.5'] > self.best_map:
                    self.best_map = val_metrics['mAP_0.5']
                    self.save_checkpoint('best_model.pth')
            
            # Save checkpoint
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
                print(f"Learning Rate: {current_lr:.6f}")
        
        # Save final model
        self.save_checkpoint('final_model.pth')
        print("Training completed!")
        
        # Close tensorboard writer
        self.writer.close()
    
    def _train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {self.current_epoch + 1}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # Move to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in target.items()} for target in targets]
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, targets)
            
            # Calculate loss
            losses = outputs['losses']
            total_loss_value = sum(losses.values())
            
            # Backward pass
            total_loss_value.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip'):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                             self.config['gradient_clip'])
            
            # Update weights
            self.optimizer.step()
            
            # Update progress
            total_loss += total_loss_value.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{total_loss_value.item():.4f}",
                'Avg Loss': f"{total_loss / num_batches:.4f}"
            })
            
            # Log batch losses
            if batch_idx % self.config.get('log_freq', 100) == 0:
                for loss_name, loss_value in losses.items():
                    self.writer.add_scalar(f'Batch_Loss/{loss_name}', 
                                         loss_value.item(), 
                                         self.current_epoch * len(self.train_loader) + batch_idx)
        
        return total_loss / num_batches
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in target.items()} for target in targets]
                
                # Forward pass
                outputs = self.model(images)
                predictions = outputs['detections']
                
                # Collect predictions and targets
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Calculate metrics
        metrics = calculate_map(all_predictions, all_targets, 
                              iou_thresholds=[0.5, 0.75])
        
        return metrics
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_map': self.best_map,
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return
        
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_map = checkpoint.get('best_map', 0.0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        
        print(f"Checkpoint loaded. Resuming from epoch {self.current_epoch + 1}")
    
    def evaluate(self, test_dataset: Optional[VOCDataset] = None) -> Dict[str, float]:
        """
        Evaluate the model
        
        Args:
            test_dataset: Test dataset (uses validation dataset if not provided)
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if test_dataset is None:
            test_dataset = self.val_dataset
        
        if test_dataset is None:
            raise ValueError("No test dataset provided")
        
        # Create test data loader
        test_loader = self._create_data_loader(test_dataset, train=False)
        
        # Set model to evaluation mode
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        print("Evaluating model...")
        with torch.no_grad():
            for images, targets in tqdm(test_loader, desc="Evaluation"):
                # Move to device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in target.items()} for target in targets]
                
                # Forward pass
                outputs = self.model(images)
                predictions = outputs['detections']
                
                # Collect predictions and targets
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Calculate metrics
        metrics = calculate_map(all_predictions, all_targets, 
                              iou_thresholds=[0.5, 0.75])
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        print("="*50)
        
        return metrics
    
    def predict(self, image_path: str, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Make prediction on a single image
        
        Args:
            image_path: Path to input image
            confidence_threshold: Confidence threshold for detections
        
        Returns:
            Dictionary containing predictions
        """
        from PIL import Image
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = get_transform(train=False, 
                               image_size=self.config.get('image_size', (800, 800)),
                               max_size=self.config.get('max_size', 1000))
        
        # Apply transforms
        image_tensor, _ = transform(image, {})
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(image_tensor)
            predictions = outputs['detections'][0]
            
            # Filter by confidence
            if predictions['scores'].numel() > 0:
                keep_mask = predictions['scores'] >= confidence_threshold
                predictions = {
                    'boxes': predictions['boxes'][keep_mask],
                    'labels': predictions['labels'][keep_mask],
                    'scores': predictions['scores'][keep_mask]
                }
        
        return predictions
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        return {
            'current_epoch': self.current_epoch,
            'best_map': self.best_map,
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
            'config': self.config
        }
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves"""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training loss
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot validation metrics
        if self.val_metrics:
            epochs = range(1, len(self.val_metrics) + 1)
            mAP_50 = [metrics['mAP_0.5'] for metrics in self.val_metrics]
            mAP_75 = [metrics['mAP_0.5_0.95'] for metrics in self.val_metrics]
            
            ax2.plot(epochs, mAP_50, label='mAP@0.5', marker='o')
            ax2.plot(epochs, mAP_75, label='mAP@0.5:0.95', marker='s')
            ax2.set_title('Validation mAP')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('mAP')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def export_model(self, export_path: str, format: str = 'torchscript'):
        """
        Export model for deployment
        
        Args:
            export_path: Path to save exported model
            format: Export format ('torchscript', 'onnx')
        """
        self.model.eval()
        
        if format.lower() == 'torchscript':
            # Export as TorchScript
            dummy_input = torch.randn(1, 3, 800, 800).to(self.device)
            traced_model = torch.jit.trace(self.model, dummy_input)
            torch.jit.save(traced_model, export_path)
            print(f"Model exported as TorchScript to {export_path}")
        
        elif format.lower() == 'onnx':
            # Export as ONNX
            dummy_input = torch.randn(1, 3, 800, 800).to(self.device)
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
            print(f"Model exported as ONNX to {export_path}")
        
        else:
            raise ValueError(f"Unsupported export format: {format}")


def create_trainer(model: FasterRCNN, config: Dict) -> Trainer:
    """Factory function to create trainer"""
    # Create datasets
    train_dataset = VOCDataset(
        root=config['data']['voc_root'],
        year=config['data']['voc_year'],
        image_set='train'
    )
    
    val_dataset = VOCDataset(
        root=config['data']['voc_root'],
        year=config['data']['voc_year'],
        image_set='val'
    )
    
    # Create trainer
    trainer = Trainer(model, train_dataset, val_dataset, config)
    
    return trainer
