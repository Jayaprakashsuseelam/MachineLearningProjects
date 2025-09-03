import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from typing import Dict, List, Optional
from tqdm import tqdm
import time

from models.ssd_network import SSDNetwork, DefaultBoxGenerator
from models.ssd_loss import SSDLoss, BoxEncoder
from utils.metrics import ModelEvaluator
from utils.visualization import visualize_model_performance


class SSDTrainer:
    """
    Trainer class for SSD model
    """
    
    def __init__(self, config_path: str, dataset_path: str, 
                 output_dir: str = 'outputs'):
        """
        Initialize SSD trainer
        
        Args:
            config_path: Path to model configuration
            dataset_path: Path to dataset
            output_dir: Output directory for checkpoints and logs
        """
        self.config_path = config_path
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model
        self.model = SSDNetwork(self.config)
        
        # Generate default boxes
        self.default_boxes = self._generate_default_boxes()
        
        # Initialize loss function
        self.criterion = SSDLoss(
            num_classes=self.config['num_classes'],
            neg_pos_ratio=3,
            alpha=1.0
        )
        
        # Initialize box encoder
        self.box_encoder = BoxEncoder(self.default_boxes, self.config['variance'])
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(
            num_classes=self.config['num_classes'],
            iou_threshold=0.5
        )
        
        # Training state
        self.current_epoch = 0
        self.best_map = 0.0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'map': [],
            'precision': [],
            'recall': []
        }
    
    def _generate_default_boxes(self):
        """Generate default boxes for the model"""
        generator = DefaultBoxGenerator(self.config)
        return generator.generate_default_boxes()
    
    def setup_training(self, learning_rate: float = 0.001, 
                      weight_decay: float = 0.0005,
                      device: str = 'auto'):
        """
        Setup training components
        
        Args:
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            device: Device to train on
        """
        # Determine device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=60,
            gamma=0.1
        )
        
        print(f"Training setup complete. Device: {self.device}")
    
    def load_dataset(self, batch_size: int = 16, num_workers: int = 4):
        """
        Load training and validation datasets
        
        Args:
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
        """
        # This is a placeholder - you would implement your own dataset loading
        # For now, we'll create dummy data loaders
        print("Loading datasets...")
        
        # Create dummy datasets (replace with your actual dataset implementation)
        self.train_loader = self._create_dummy_loader(batch_size, num_workers, is_train=True)
        self.val_loader = self._create_dummy_loader(batch_size, num_workers, is_train=False)
        
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
    
    def _create_dummy_loader(self, batch_size: int, num_workers: int, is_train: bool):
        """Create dummy data loader for demonstration"""
        # This is just for demonstration - replace with your actual dataset
        from torch.utils.data import Dataset, DataLoader
        import numpy as np
        
        class DummyDataset(Dataset):
            def __init__(self, num_samples=1000, input_size=300):
                self.num_samples = num_samples
                self.input_size = input_size
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                # Create dummy image and annotations
                image = torch.randn(3, self.input_size, self.input_size)
                boxes = torch.randn(5, 4)  # 5 boxes per image
                labels = torch.randint(1, self.config['num_classes'], (5,))
                
                return image, {'boxes': boxes, 'labels': labels}
        
        dataset = DummyDataset()
        return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, 
                        num_workers=num_workers)
    
    def train_epoch(self) -> float:
        """
        Train for one epoch
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            # Move data to device
            images = images.to(self.device)
            
            # Prepare targets
            batch_targets = self._prepare_targets(targets)
            
            # Forward pass
            self.optimizer.zero_grad()
            loc_pred, conf_pred = self.model(images)
            
            # Calculate loss
            loss = self.criterion((loc_pred, conf_pred), batch_targets, self.default_boxes)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
        
        return total_loss / num_batches
    
    def validate_epoch(self) -> Dict:
        """
        Validate for one epoch
        
        Returns:
            Validation metrics
        """
        self.model.eval()
        self.evaluator.reset()
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validation'):
                # Move data to device
                images = images.to(self.device)
                
                # Forward pass
                loc_pred, conf_pred = self.model(images)
                
                # Decode predictions
                predictions = self._decode_predictions(loc_pred, conf_pred)
                
                # Prepare ground truth
                ground_truth = self._prepare_ground_truth(targets)
                
                # Update evaluator
                self.evaluator.evaluate_batch(predictions, ground_truth)
        
        return self.evaluator.get_results()
    
    def _prepare_targets(self, targets: List[Dict]) -> tuple:
        """Prepare targets for training"""
        batch_size = len(targets)
        num_default_boxes = len(self.default_boxes)
        
        # Initialize target tensors
        loc_targets = torch.zeros(batch_size, num_default_boxes, 4, device=self.device)
        conf_targets = torch.zeros(batch_size, num_default_boxes, dtype=torch.long, device=self.device)
        
        for i, target in enumerate(targets):
            boxes = target['boxes']
            labels = target['labels']
            
            # Encode targets
            encoded_boxes, encoded_labels = self.box_encoder.encode(boxes, labels)
            
            loc_targets[i] = encoded_boxes
            conf_targets[i] = encoded_labels
        
        return (loc_targets, conf_targets)
    
    def _prepare_ground_truth(self, targets: List[Dict]) -> List[List[Dict]]:
        """Prepare ground truth for evaluation"""
        ground_truth = []
        
        for target in targets:
            boxes = target['boxes']
            labels = target['labels']
            
            gt_boxes = []
            for box, label in zip(boxes, labels):
                gt_boxes.append({
                    'bbox': box.tolist(),
                    'label': label.item()
                })
            
            ground_truth.append(gt_boxes)
        
        return ground_truth
    
    def _decode_predictions(self, loc_pred: torch.Tensor, conf_pred: torch.Tensor) -> List[List[Dict]]:
        """Decode model predictions"""
        batch_size = loc_pred.size(0)
        predictions = []
        
        for i in range(batch_size):
            # Decode predictions for current image
            decoded_boxes, decoded_labels, decoded_scores = self.box_encoder.decode(
                loc_pred[i:i+1], conf_pred[i:i+1], 
                confidence_threshold=self.config['confidence_threshold']
            )
            
            # Convert to list of dictionaries
            image_predictions = []
            for box, label, score in zip(decoded_boxes[0], decoded_labels[0], decoded_scores[0]):
                image_predictions.append({
                    'bbox': box.tolist(),
                    'label': label.item(),
                    'score': score.item()
                })
            
            predictions.append(image_predictions)
        
        return predictions
    
    def train(self, epochs: int, save_interval: int = 10):
        """
        Train the model
        
        Args:
            epochs: Number of epochs to train
            save_interval: Save model every N epochs
        """
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step()
            
            # Update training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_metrics.get('overall_f1', 0))  # Using F1 as val loss
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.training_history['map'].append(val_metrics.get('mAP', 0))
            self.training_history['precision'].append(val_metrics.get('overall_precision', 0))
            self.training_history['recall'].append(val_metrics.get('overall_recall', 0))
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val mAP: {val_metrics.get('mAP', 0):.4f}")
            print(f"Val Precision: {val_metrics.get('overall_precision', 0):.4f}")
            print(f"Val Recall: {val_metrics.get('overall_recall', 0):.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            current_map = val_metrics.get('mAP', 0)
            if current_map > self.best_map:
                self.best_map = current_map
                self.save_checkpoint('best_model.pth')
                print(f"New best model saved! mAP: {current_map:.4f}")
            
            # Save checkpoint periodically
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
        
        # Save final model
        self.save_checkpoint('final_model.pth')
        
        # Create training visualization
        self._create_training_visualization()
        
        print(f"Training completed! Best mAP: {self.best_map:.4f}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_map': self.best_map,
            'training_history': self.training_history,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(self.output_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_map = checkpoint['best_map']
        self.training_history = checkpoint['training_history']
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch + 1}")
    
    def _create_training_visualization(self):
        """Create training visualization"""
        fig = visualize_model_performance(self.training_history)
        fig.savefig(os.path.join(self.output_dir, 'training_curves.png'))
        plt.close(fig)
        print("Training visualization saved")
    
    def evaluate_model(self, test_loader) -> Dict:
        """
        Evaluate model on test set
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation results
        """
        self.model.eval()
        self.evaluator.reset()
        
        print("Evaluating model...")
        
        with torch.no_grad():
            for images, targets in tqdm(test_loader, desc='Evaluation'):
                # Move data to device
                images = images.to(self.device)
                
                # Forward pass
                loc_pred, conf_pred = self.model(images)
                
                # Decode predictions
                predictions = self._decode_predictions(loc_pred, conf_pred)
                
                # Prepare ground truth
                ground_truth = self._prepare_ground_truth(targets)
                
                # Update evaluator
                self.evaluator.evaluate_batch(predictions, ground_truth)
        
        results = self.evaluator.get_results()
        self.evaluator.print_summary()
        
        return results
