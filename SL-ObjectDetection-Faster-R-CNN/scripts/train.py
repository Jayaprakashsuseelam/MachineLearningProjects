#!/usr/bin/env python3
"""
Training script for Faster R-CNN
"""
import os
import sys
import argparse
import torch
import yaml
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import faster_rcnn_resnet50
from training import Trainer
from data import VOCDataset, get_transform
from config import get_config


def main():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN')
    parser.add_argument('--config', type=str, default='config/config.py',
                       help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=12,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.005,
                       help='Learning rate')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Output directory for checkpoints and logs')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        config = get_config(args.config)
    else:
        config = get_config()
    
    # Override config with command line arguments
    config.training.epochs = args.epochs
    config.training.learning_rate = args.lr
    config.data.batch_size = args.batch_size
    config.training.resume_from = args.resume
    config.training.save_dir = args.output_dir
    config.training.log_dir = os.path.join(args.output_dir, 'logs')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(config.training.save_dir, exist_ok=True)
    os.makedirs(config.training.log_dir, exist_ok=True)
    
    # Save configuration
    config.save(os.path.join(args.output_dir, 'config.yaml'))
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("Creating Faster R-CNN model...")
    model = faster_rcnn_resnet50(
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
        freeze_backbone=config.model.freeze_backbone
    )
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = VOCDataset(
        root=config.data.voc_root,
        year=config.data.voc_year,
        image_set='train'
    )
    
    val_dataset = VOCDataset(
        root=config.data.voc_root,
        year=config.data.voc_year,
        image_set='val'
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config={
            'batch_size': config.data.batch_size,
            'num_workers': config.data.num_workers,
            'image_size': config.data.image_size,
            'max_size': config.data.max_size,
            'learning_rate': config.training.learning_rate,
            'momentum': config.training.momentum,
            'weight_decay': config.training.weight_decay,
            'gradient_clip': config.training.gradient_clip,
            'log_dir': config.training.log_dir,
            'checkpoint_dir': config.training.save_dir,
            'resume_from': config.training.resume_from
        }
    )
    
    # Start training
    print("Starting training...")
    trainer.train(
        epochs=config.training.epochs,
        save_freq=config.training.save_freq,
        eval_freq=1
    )
    
    # Final evaluation
    print("Performing final evaluation...")
    final_metrics = trainer.evaluate()
    
    # Save final results
    results_path = os.path.join(args.output_dir, 'final_results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(final_metrics, f, default_flow_style=False)
    
    print(f"Training completed! Final results saved to {results_path}")
    print("Final metrics:")
    for metric_name, metric_value in final_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")


if __name__ == '__main__':
    main()
