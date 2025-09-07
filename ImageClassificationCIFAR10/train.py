"""
Simple Training Script for CIFAR-10 Classification

This script provides a simplified interface for training models using
the configuration system.

Author: AI Assistant
Date: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time
import os

# Import our modules
from models import get_model, count_parameters
from data_utils import CIFAR10DataProcessor
from visualization import CIFAR10Visualizer
from config import get_config, update_config

def train_model(config):
    """
    Train a model using the provided configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, history, test_loader)
    """
    device = config['device']
    print(f"Using device: {device}")
    
    # Create model
    model_name = config.get('model', 'cnn')
    model = get_model(model_name, num_classes=config['dataset']['num_classes'])
    model = model.to(device)
    
    param_count = count_parameters(model)
    print(f"Model: {model_name.upper()}")
    print(f"Parameters: {param_count:,}")
    
    # Setup data
    processor = CIFAR10DataProcessor(
        data_dir=config['dataset']['data_dir'],
        download=config['dataset']['download']
    )
    
    use_augmentation = config.get('augmentation', True)
    train_loader, val_loader, test_loader = processor.create_data_loaders(
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        use_augmentation=use_augmentation
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    epochs = config['training']['epochs']
    best_val_acc = 0.0
    
    print(f"\nStarting training for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if config['training']['save_best_model']:
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"New best model saved! Val Acc: {val_acc:.2f}%")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    return model, history, test_loader, training_time

def evaluate_model(model, test_loader, device, class_names):
    """
    Evaluate the model on the test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to use
        class_names: List of class names
        
    Returns:
        Tuple of (accuracy, predictions, targets)
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = 100. * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    return accuracy, all_preds, all_targets

def main():
    """
    Main function for training.
    """
    parser = argparse.ArgumentParser(description='CIFAR-10 Training Script')
    parser.add_argument('--experiment', type=str, default='standard_training',
                       choices=['quick_test', 'standard_training', 'extensive_training', 'model_comparison'],
                       help='Experiment configuration to use')
    parser.add_argument('--model', type=str, default=None,
                       choices=['cnn', 'resnet', 'efficientnet', 'densenet', 'mobilenet', 'advanced_cnn'],
                       help='Override model architecture')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--no_augmentation', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Get base configuration
    config = get_config(args.experiment)
    
    # Apply overrides
    overrides = {}
    if args.model:
        overrides['model'] = args.model
    if args.epochs:
        overrides['training'] = {'epochs': args.epochs}
    if args.batch_size:
        overrides['training'] = {'batch_size': args.batch_size}
    if args.lr:
        overrides['training'] = {'learning_rate': args.lr}
    if args.no_augmentation:
        overrides['augmentation'] = False
    
    if overrides:
        config = update_config(config, overrides)
    
    # Update save directory
    config['visualization']['save_dir'] = args.save_dir
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("CIFAR-10 Image Classification Training")
    print("=" * 50)
    print(f"Experiment: {args.experiment}")
    print(f"Model: {config.get('model', 'cnn')}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Learning Rate: {config['training']['learning_rate']}")
    print(f"Augmentation: {config.get('augmentation', True)}")
    print("=" * 50)
    
    # Train model
    model, history, test_loader, training_time = train_model(config)
    
    # Load best model for evaluation
    if config['training']['save_best_model'] and os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth'))
        print("Loaded best model for evaluation")
    
    # Evaluate model
    print("\nEvaluating on test set...")
    accuracy, preds, targets = evaluate_model(
        model, test_loader, config['device'], config['dataset']['class_names']
    )
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualizer = CIFAR10Visualizer(
        class_names=config['dataset']['class_names'],
        save_dir=config['visualization']['save_dir']
    )
    
    # Plot training history
    visualizer.plot_training_history(history)
    
    # Plot confusion matrix
    visualizer.plot_confusion_matrix(targets, preds)
    
    # Plot per-class accuracy
    visualizer.plot_class_accuracy(targets, preds)
    
    # Create summary report
    results = {
        'test_accuracy': accuracy,
        'training_time': training_time,
        'parameters': count_parameters(model),
        'history': history,
        'y_true': targets,
        'y_pred': preds
    }
    
    visualizer.create_summary_report(results)
    
    print(f"\nTraining completed successfully!")
    print(f"Final Test Accuracy: {accuracy:.2f}%")
    print(f"Results saved to: {args.save_dir}")

if __name__ == "__main__":
    main()
