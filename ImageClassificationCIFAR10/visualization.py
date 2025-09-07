"""
Visualization Utilities for CIFAR-10 Classification

This module provides comprehensive visualization tools for analyzing model performance,
training progress, and data characteristics in CIFAR-10 classification tasks.

Author: AI Assistant
Date: 2024
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from typing import List, Dict, Tuple, Optional
import os

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CIFAR10Visualizer:
    """
    Comprehensive visualization class for CIFAR-10 classification results.
    """
    
    def __init__(self, class_names: List[str] = None, save_dir: str = './plots'):
        """
        Initialize the visualizer.
        
        Args:
            class_names: List of class names for CIFAR-10
            save_dir: Directory to save plots
        """
        self.class_names = class_names or ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                                          'dog', 'frog', 'horse', 'ship', 'truck']
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set up color palette
        self.colors = sns.color_palette("husl", len(self.class_names))
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                            title: str = "Training History", 
                            save_name: str = "training_history.png"):
        """
        Plot training and validation loss/accuracy curves.
        
        Args:
            history: Dictionary containing training history
            title: Plot title
            save_name: Filename to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], 
                            title: str = "Confusion Matrix",
                            save_name: str = "confusion_matrix.png"):
        """
        Plot confusion matrix with detailed annotations.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            save_name: Filename to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Number of Samples'})
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_class_accuracy(self, y_true: List[int], y_pred: List[int],
                          title: str = "Per-Class Accuracy",
                          save_name: str = "class_accuracy.png"):
        """
        Plot accuracy for each class.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            save_name: Filename to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(self.class_names, class_accuracies, color=self.colors)
        
        # Add value labels on bars
        for bar, acc in zip(bars, class_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Classes')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
        
        return class_accuracies
    
    def plot_prediction_samples(self, images: torch.Tensor, true_labels: List[int], 
                              pred_labels: List[int], probabilities: torch.Tensor,
                              num_samples: int = 16, title: str = "Prediction Samples",
                              save_name: str = "prediction_samples.png"):
        """
        Visualize model predictions on sample images.
        
        Args:
            images: Batch of images
            true_labels: True labels
            pred_labels: Predicted labels
            probabilities: Prediction probabilities
            num_samples: Number of samples to show
            title: Plot title
            save_name: Filename to save the plot
        """
        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(images))):
            # Denormalize image for visualization
            img = self.denormalize_image(images[i])
            img = img.permute(1, 2, 0)
            
            axes[i].imshow(img)
            
            # Color the title based on correctness
            is_correct = true_labels[i] == pred_labels[i]
            color = 'green' if is_correct else 'red'
            
            confidence = probabilities[i][pred_labels[i]].item()
            axes[i].set_title(f'True: {self.class_names[true_labels[i]]}\n'
                             f'Pred: {self.class_names[pred_labels[i]]}\n'
                             f'Conf: {confidence:.3f}', 
                             color=color, fontweight='bold')
            axes[i].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_visualization(self, model: torch.nn.Module, images: torch.Tensor,
                                 layer_name: str = None, title: str = "Feature Maps",
                                 save_name: str = "feature_maps.png"):
        """
        Visualize feature maps from a specific layer.
        
        Args:
            model: Trained model
            layer_name: Name of the layer to visualize
            images: Input images
            title: Plot title
            save_name: Filename to save the plot
        """
        model.eval()
        
        # Hook to capture feature maps
        feature_maps = {}
        def hook_fn(module, input, output):
            feature_maps['features'] = output
        
        # Register hook
        if layer_name:
            layer = dict(model.named_modules())[layer_name]
            hook = layer.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            _ = model(images)
        
        if layer_name:
            hook.remove()
        
        if 'features' in feature_maps:
            features = feature_maps['features']
            
            # Select first image and some feature maps
            img_features = features[0]  # First image
            num_maps = min(16, img_features.shape[0])
            
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            axes = axes.ravel()
            
            for i in range(num_maps):
                feature_map = img_features[i].cpu().numpy()
                axes[i].imshow(feature_map, cmap='viridis')
                axes[i].set_title(f'Feature Map {i+1}')
                axes[i].axis('off')
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_learning_curves_comparison(self, histories: Dict[str, Dict[str, List[float]]],
                                      metric: str = 'val_acc',
                                      title: str = "Model Comparison",
                                      save_name: str = "model_comparison.png"):
        """
        Compare learning curves of different models.
        
        Args:
            histories: Dictionary of model histories
            metric: Metric to compare ('val_acc', 'val_loss', etc.)
            title: Plot title
            save_name: Filename to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        for model_name, history in histories.items():
            epochs = range(1, len(history[metric]) + 1)
            plt.plot(epochs, history[metric], label=model_name, linewidth=2, marker='o')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_complexity_analysis(self, models_info: List[Dict],
                                     title: str = "Model Complexity Analysis",
                                     save_name: str = "model_complexity.png"):
        """
        Analyze model complexity vs performance.
        
        Args:
            models_info: List of dictionaries with model information
            title: Plot title
            save_name: Filename to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        model_names = [info['name'] for info in models_info]
        parameters = [info['parameters'] for info in models_info]
        accuracies = [info['accuracy'] for info in models_info]
        training_times = [info.get('training_time', 0) for info in models_info]
        
        # Parameters vs Accuracy
        scatter = ax1.scatter(parameters, accuracies, s=100, alpha=0.7, c=range(len(model_names)), cmap='viridis')
        for i, name in enumerate(model_names):
            ax1.annotate(name, (parameters[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax1.set_xlabel('Number of Parameters')
        ax1.set_ylabel('Test Accuracy (%)')
        ax1.set_title('Parameters vs Accuracy')
        ax1.grid(True, alpha=0.3)
        
        # Training Time vs Accuracy
        scatter2 = ax2.scatter(training_times, accuracies, s=100, alpha=0.7, c=range(len(model_names)), cmap='viridis')
        for i, name in enumerate(model_names):
            ax2.annotate(name, (training_times[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax2.set_xlabel('Training Time (seconds)')
        ax2.set_ylabel('Test Accuracy (%)')
        ax2.set_title('Training Time vs Accuracy')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_error_analysis(self, images: torch.Tensor, true_labels: List[int], 
                          pred_labels: List[int], probabilities: torch.Tensor,
                          title: str = "Error Analysis",
                          save_name: str = "error_analysis.png"):
        """
        Analyze prediction errors and show most confident wrong predictions.
        
        Args:
            images: Batch of images
            true_labels: True labels
            pred_labels: Predicted labels
            probabilities: Prediction probabilities
            title: Plot title
            save_name: Filename to save the plot
        """
        # Find wrong predictions
        wrong_indices = [i for i, (true, pred) in enumerate(zip(true_labels, pred_labels)) 
                        if true != pred]
        
        if not wrong_indices:
            print("No wrong predictions found!")
            return
        
        # Sort by confidence (highest confidence wrong predictions first)
        wrong_confidences = [probabilities[i][pred_labels[i]].item() for i in wrong_indices]
        sorted_indices = sorted(zip(wrong_indices, wrong_confidences), 
                              key=lambda x: x[1], reverse=True)
        
        # Show top 16 wrong predictions
        num_samples = min(16, len(sorted_indices))
        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        axes = axes.ravel()
        
        for i, (idx, conf) in enumerate(sorted_indices[:num_samples]):
            img = self.denormalize_image(images[idx])
            img = img.permute(1, 2, 0)
            
            axes[i].imshow(img)
            axes[i].set_title(f'True: {self.class_names[true_labels[idx]]}\n'
                             f'Pred: {self.class_names[pred_labels[idx]]}\n'
                             f'Conf: {conf:.3f}', 
                             color='red', fontweight='bold')
            axes[i].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_data_distribution(self, dataset, title: str = "Dataset Distribution",
                              save_name: str = "data_distribution.png"):
        """
        Plot class distribution in the dataset.
        
        Args:
            dataset: Dataset to analyze
            title: Plot title
            save_name: Filename to save the plot
        """
        class_counts = [0] * len(self.class_names)
        
        for _, label in dataset:
            if isinstance(label, torch.Tensor):
                label = label.item()
            class_counts[label] += 1
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(self.class_names, class_counts, color=self.colors)
        
        # Add value labels on bars
        for bar, count in zip(bars, class_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Classes')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
        
        return class_counts
    
    def denormalize_image(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize a tensor image for visualization.
        
        Args:
            tensor: Normalized tensor image
            
        Returns:
            Denormalized tensor image
        """
        # CIFAR-10 normalization values
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        
        return tensor
    
    def create_summary_report(self, results: Dict, save_name: str = "summary_report.png"):
        """
        Create a comprehensive summary report.
        
        Args:
            results: Dictionary containing all results
            save_name: Filename to save the plot
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Create a grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Training history
        ax1 = fig.add_subplot(gs[0, :2])
        epochs = range(1, len(results['history']['train_loss']) + 1)
        ax1.plot(epochs, results['history']['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, results['history']['val_loss'], 'r-', label='Val Loss')
        ax1.set_title('Training History - Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy history
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.plot(epochs, results['history']['train_acc'], 'b-', label='Train Acc')
        ax2.plot(epochs, results['history']['val_acc'], 'r-', label='Val Acc')
        ax2.set_title('Training History - Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Confusion matrix
        ax3 = fig.add_subplot(gs[1, :2])
        cm = confusion_matrix(results['y_true'], results['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names, ax=ax3)
        ax3.set_title('Confusion Matrix')
        
        # Per-class accuracy
        ax4 = fig.add_subplot(gs[1, 2:])
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        bars = ax4.bar(self.class_names, class_accuracies, color=self.colors)
        ax4.set_title('Per-Class Accuracy')
        ax4.set_ylabel('Accuracy')
        ax4.tick_params(axis='x', rotation=45)
        
        # Model info
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        info_text = f"""
        Model Summary:
        • Test Accuracy: {results['test_accuracy']:.2f}%
        • Training Time: {results.get('training_time', 'N/A')} seconds
        • Model Parameters: {results.get('parameters', 'N/A'):,}
        • Best Validation Accuracy: {max(results['history']['val_acc']):.2f}%
        • Final Training Loss: {results['history']['train_loss'][-1]:.4f}
        • Final Validation Loss: {results['history']['val_loss'][-1]:.4f}
        """
        
        ax5.text(0.1, 0.5, info_text, fontsize=14, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        plt.suptitle('CIFAR-10 Classification Summary Report', fontsize=20, fontweight='bold')
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()

def plot_data_augmentation_comparison(dataset_with_aug, dataset_without_aug, 
                                    num_samples: int = 8, save_dir: str = './plots'):
    """
    Compare original and augmented data samples.
    
    Args:
        dataset_with_aug: Dataset with augmentation
        dataset_without_aug: Dataset without augmentation
        num_samples: Number of samples to show
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(num_samples):
        # Original image
        img_orig, label_orig = dataset_without_aug[i]
        if isinstance(img_orig, torch.Tensor):
            img_orig = img_orig.permute(1, 2, 0)
        
        axes[0, i].imshow(img_orig)
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Augmented image
        img_aug, label_aug = dataset_with_aug[i]
        if isinstance(img_aug, torch.Tensor):
            img_aug = img_aug.permute(1, 2, 0)
        
        axes[1, i].imshow(img_aug)
        axes[1, i].set_title('Augmented')
        axes[1, i].axis('off')
    
    plt.suptitle('Data Augmentation Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'data_augmentation_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Example usage
    visualizer = CIFAR10Visualizer()
    
    # Example training history
    example_history = {
        'train_loss': [2.3, 1.8, 1.5, 1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        'val_loss': [2.4, 1.9, 1.6, 1.3, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6],
        'train_acc': [20, 35, 45, 55, 65, 70, 75, 80, 85, 90],
        'val_acc': [18, 32, 42, 52, 62, 68, 73, 78, 83, 88]
    }
    
    visualizer.plot_training_history(example_history)
