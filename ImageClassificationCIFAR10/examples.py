"""
Example Usage Scripts for CIFAR-10 Classification

This module provides example scripts demonstrating different ways to use
the CIFAR-10 classification system.

Author: AI Assistant
Date: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
from typing import Dict, List

# Import our modules
from models import get_model, count_parameters
from data_utils import CIFAR10DataProcessor
from visualization import CIFAR10Visualizer

class ExampleTrainer:
    """
    Example trainer class demonstrating different training approaches.
    """
    
    def __init__(self, model_name: str, device: torch.device):
        self.model_name = model_name
        self.device = device
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        
    def train_basic_model(self, epochs: int = 20, batch_size: int = 128, 
                         learning_rate: float = 0.001):
        """
        Train a basic model with standard settings.
        """
        print(f"Training {self.model_name} model...")
        
        # Create model
        model = get_model(self.model_name)
        model = model.to(self.device)
        
        # Setup data
        processor = CIFAR10DataProcessor()
        train_loader, val_loader, test_loader = processor.create_data_loaders(
            batch_size=batch_size, use_augmentation=True
        )
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
            
            # Update history
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            scheduler.step(val_loss)
            
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        return model, history, test_loader
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader):
        """
        Evaluate model on test set.
        """
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                
                test_loss += nn.CrossEntropyLoss()(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = 100. * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        
        return accuracy, all_preds, all_targets

def example_1_basic_training():
    """
    Example 1: Basic model training
    """
    print("=" * 60)
    print("EXAMPLE 1: Basic Model Training")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = ExampleTrainer('cnn', device)
    
    # Train model
    model, history, test_loader = trainer.train_basic_model(epochs=10)
    
    # Evaluate
    accuracy, preds, targets = trainer.evaluate_model(model, test_loader)
    
    # Visualize results
    visualizer = CIFAR10Visualizer()
    visualizer.plot_training_history(history)
    
    print(f"Final Test Accuracy: {accuracy:.2f}%")

def example_2_model_comparison():
    """
    Example 2: Compare different model architectures
    """
    print("=" * 60)
    print("EXAMPLE 2: Model Architecture Comparison")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models_to_compare = ['cnn', 'resnet', 'mobilenet']
    
    results = {}
    
    for model_name in models_to_compare:
        print(f"\nTraining {model_name.upper()}...")
        trainer = ExampleTrainer(model_name, device)
        
        start_time = time.time()
        model, history, test_loader = trainer.train_basic_model(epochs=5)
        training_time = time.time() - start_time
        
        accuracy, _, _ = trainer.evaluate_model(model, test_loader)
        param_count = count_parameters(model)
        
        results[model_name] = {
            'accuracy': accuracy,
            'parameters': param_count,
            'training_time': training_time,
            'history': history
        }
        
        print(f"{model_name.upper()} Results:")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Parameters: {param_count:,}")
        print(f"  Training Time: {training_time:.2f}s")
    
    # Visualize comparison
    visualizer = CIFAR10Visualizer()
    
    # Plot learning curves comparison
    histories = {name: results[name]['history'] for name in models_to_compare}
    visualizer.plot_learning_curves_comparison(histories, metric='val_acc')
    
    # Plot model complexity analysis
    models_info = [
        {
            'name': name,
            'accuracy': results[name]['accuracy'],
            'parameters': results[name]['parameters'],
            'training_time': results[name]['training_time']
        }
        for name in models_to_compare
    ]
    visualizer.plot_model_complexity_analysis(models_info)

def example_3_data_augmentation_analysis():
    """
    Example 3: Analyze effect of data augmentation
    """
    print("=" * 60)
    print("EXAMPLE 3: Data Augmentation Analysis")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train with augmentation
    print("Training with data augmentation...")
    processor_aug = CIFAR10DataProcessor()
    train_loader_aug, val_loader_aug, test_loader = processor_aug.create_data_loaders(
        batch_size=128, use_augmentation=True
    )
    
    trainer_aug = ExampleTrainer('cnn', device)
    model_aug, history_aug, _ = trainer_aug.train_basic_model(epochs=10)
    accuracy_aug, _, _ = trainer_aug.evaluate_model(model_aug, test_loader)
    
    # Train without augmentation
    print("\nTraining without data augmentation...")
    processor_no_aug = CIFAR10DataProcessor()
    train_loader_no_aug, val_loader_no_aug, _ = processor_no_aug.create_data_loaders(
        batch_size=128, use_augmentation=False
    )
    
    trainer_no_aug = ExampleTrainer('cnn', device)
    model_no_aug, history_no_aug, _ = trainer_no_aug.train_basic_model(epochs=10)
    accuracy_no_aug, _, _ = trainer_no_aug.evaluate_model(model_no_aug, test_loader)
    
    print(f"\nResults:")
    print(f"With Augmentation: {accuracy_aug:.2f}%")
    print(f"Without Augmentation: {accuracy_no_aug:.2f}%")
    print(f"Improvement: {accuracy_aug - accuracy_no_aug:.2f}%")
    
    # Visualize comparison
    visualizer = CIFAR10Visualizer()
    
    # Compare training histories
    histories = {
        'With Augmentation': history_aug,
        'Without Augmentation': history_no_aug
    }
    visualizer.plot_learning_curves_comparison(histories, metric='val_acc')

def example_4_hyperparameter_tuning():
    """
    Example 4: Hyperparameter tuning example
    """
    print("=" * 60)
    print("EXAMPLE 4: Hyperparameter Tuning")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define hyperparameter grid
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [64, 128, 256]
    
    best_accuracy = 0
    best_params = {}
    results = []
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"\nTesting LR={lr}, Batch Size={batch_size}")
            
            trainer = ExampleTrainer('cnn', device)
            model, history, test_loader = trainer.train_basic_model(
                epochs=5, batch_size=batch_size, learning_rate=lr
            )
            
            accuracy, _, _ = trainer.evaluate_model(model, test_loader)
            
            results.append({
                'lr': lr,
                'batch_size': batch_size,
                'accuracy': accuracy
            })
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'lr': lr, 'batch_size': batch_size}
            
            print(f"Accuracy: {accuracy:.2f}%")
    
    print(f"\nBest Parameters: {best_params}")
    print(f"Best Accuracy: {best_accuracy:.2f}%")
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    
    # Plot accuracy vs learning rate
    plt.subplot(1, 2, 1)
    for bs in batch_sizes:
        lrs = [r['lr'] for r in results if r['batch_size'] == bs]
        accs = [r['accuracy'] for r in results if r['batch_size'] == bs]
        plt.plot(lrs, accs, marker='o', label=f'Batch Size {bs}')
    
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Learning Rate')
    plt.legend()
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy vs batch size
    plt.subplot(1, 2, 2)
    for lr in learning_rates:
        bss = [r['batch_size'] for r in results if r['lr'] == lr]
        accs = [r['accuracy'] for r in results if r['lr'] == lr]
        plt.plot(bss, accs, marker='s', label=f'LR {lr}')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Batch Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def example_5_error_analysis():
    """
    Example 5: Error analysis and debugging
    """
    print("=" * 60)
    print("EXAMPLE 5: Error Analysis")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train a model
    trainer = ExampleTrainer('cnn', device)
    model, history, test_loader = trainer.train_basic_model(epochs=15)
    
    # Get predictions
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    all_images = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            _, predicted = output.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_images.extend(data.cpu())
    
    # Convert to tensors
    all_images = torch.stack(all_images)
    all_probs = torch.tensor(all_probs)
    
    # Visualize errors
    visualizer = CIFAR10Visualizer()
    
    # Confusion matrix
    visualizer.plot_confusion_matrix(all_targets, all_preds)
    
    # Per-class accuracy
    visualizer.plot_class_accuracy(all_targets, all_preds)
    
    # Error analysis
    visualizer.plot_error_analysis(
        all_images, all_targets, all_preds, all_probs
    )
    
    # Prediction samples
    visualizer.plot_prediction_samples(
        all_images[:16], all_targets[:16], all_preds[:16], all_probs[:16]
    )

def main():
    """
    Main function to run examples.
    """
    parser = argparse.ArgumentParser(description='CIFAR-10 Example Scripts')
    parser.add_argument('--example', type=int, default=1, 
                       choices=[1, 2, 3, 4, 5],
                       help='Example to run (1-5)')
    parser.add_argument('--all', action='store_true',
                       help='Run all examples')
    
    args = parser.parse_args()
    
    if args.all:
        examples = [example_1_basic_training, example_2_model_comparison,
                   example_3_data_augmentation_analysis, example_4_hyperparameter_tuning,
                   example_5_error_analysis]
        
        for i, example_func in enumerate(examples, 1):
            print(f"\n{'='*80}")
            print(f"RUNNING EXAMPLE {i}")
            print(f"{'='*80}")
            try:
                example_func()
            except Exception as e:
                print(f"Error in Example {i}: {e}")
    else:
        examples = {
            1: example_1_basic_training,
            2: example_2_model_comparison,
            3: example_3_data_augmentation_analysis,
            4: example_4_hyperparameter_tuning,
            5: example_5_error_analysis
        }
        
        if args.example in examples:
            examples[args.example]()
        else:
            print(f"Example {args.example} not found!")

if __name__ == "__main__":
    main()
