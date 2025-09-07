"""
Data Preprocessing and Augmentation Utilities for CIFAR-10

This module provides comprehensive data preprocessing, augmentation, and visualization
utilities specifically designed for the CIFAR-10 dataset.

Author: AI Assistant
Date: 2024
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
from typing import Tuple, List, Optional
import os

class CIFAR10DataProcessor:
    """
    Comprehensive data processor for CIFAR-10 dataset with various augmentation strategies.
    """
    
    def __init__(self, data_dir: str = './data', download: bool = True):
        """
        Initialize the data processor.
        
        Args:
            data_dir: Directory to store the CIFAR-10 dataset
            download: Whether to download the dataset if not present
        """
        self.data_dir = data_dir
        self.download = download
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        
        # CIFAR-10 normalization values
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2023, 0.1994, 0.2010)
    
    def get_basic_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """
        Get basic transforms for training and testing.
        
        Returns:
            Tuple of (train_transform, test_transform)
        """
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        return train_transform, test_transform
    
    def get_augmented_transforms(self, 
                                crop_padding: int = 4,
                                horizontal_flip_prob: float = 0.5,
                                rotation_degrees: int = 10,
                                color_jitter_strength: float = 0.2) -> Tuple[transforms.Compose, transforms.Compose]:
        """
        Get transforms with data augmentation for training.
        
        Args:
            crop_padding: Padding for random crop
            horizontal_flip_prob: Probability of horizontal flip
            rotation_degrees: Maximum rotation degrees
            color_jitter_strength: Strength of color jittering
            
        Returns:
            Tuple of (train_transform, test_transform)
        """
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=crop_padding),
            transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
            transforms.RandomRotation(rotation_degrees),
            transforms.ColorJitter(
                brightness=color_jitter_strength,
                contrast=color_jitter_strength,
                saturation=color_jitter_strength,
                hue=color_jitter_strength/2
            ),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        return train_transform, test_transform
    
    def get_advanced_augmentation_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """
        Get advanced augmentation transforms including Cutout and Mixup.
        
        Returns:
            Tuple of (train_transform, test_transform)
        """
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            Cutout(n_holes=1, length=8),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        return train_transform, test_transform
    
    def load_datasets(self, 
                     use_augmentation: bool = True,
                     train_val_split: float = 0.8) -> Tuple[torch.utils.data.Dataset, 
                                                           torch.utils.data.Dataset, 
                                                           torch.utils.data.Dataset]:
        """
        Load CIFAR-10 datasets with optional augmentation.
        
        Args:
            use_augmentation: Whether to use data augmentation
            train_val_split: Fraction of training data to use for training
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if use_augmentation:
            train_transform, test_transform = self.get_augmented_transforms()
        else:
            train_transform, test_transform = self.get_basic_transforms()
        
        # Load full training dataset
        full_train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, 
            train=True, 
            download=self.download, 
            transform=train_transform
        )
        
        # Load test dataset
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, 
            train=False, 
            download=self.download, 
            transform=test_transform
        )
        
        # Split training data into train and validation
        train_size = int(train_val_split * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        
        # Create indices for splitting
        indices = list(range(len(full_train_dataset)))
        random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create subsets
        train_dataset = Subset(full_train_dataset, train_indices)
        
        # Create validation dataset with test transforms
        val_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, 
            train=True, 
            download=False, 
            transform=test_transform
        )
        val_dataset = Subset(val_dataset, val_indices)
        
        return train_dataset, val_dataset, test_dataset
    
    def create_data_loaders(self, 
                           batch_size: int = 128,
                           num_workers: int = 2,
                           use_augmentation: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create data loaders for training, validation, and testing.
        
        Args:
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            use_augmentation: Whether to use data augmentation
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_dataset, val_dataset, test_dataset = self.load_datasets(use_augmentation)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def visualize_dataset_samples(self, dataset, num_samples: int = 16, title: str = "Dataset Samples"):
        """
        Visualize random samples from the dataset.
        
        Args:
            dataset: Dataset to visualize
            num_samples: Number of samples to show
            title: Title for the plot
        """
        # Get random indices
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            if i >= num_samples:
                break
                
            image, label = dataset[idx]
            
            # Denormalize image for visualization
            if isinstance(image, torch.Tensor):
                image = self.denormalize_image(image)
                image = image.permute(1, 2, 0)
            
            axes[i].imshow(image)
            axes[i].set_title(f'{self.class_names[label]}')
            axes[i].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def denormalize_image(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize a tensor image for visualization.
        
        Args:
            tensor: Normalized tensor image
            
        Returns:
            Denormalized tensor image
        """
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        
        return tensor
    
    def analyze_dataset_distribution(self, dataset):
        """
        Analyze the class distribution in the dataset.
        
        Args:
            dataset: Dataset to analyze
        """
        class_counts = [0] * 10
        
        for _, label in dataset:
            if isinstance(label, torch.Tensor):
                label = label.item()
            class_counts[label] += 1
        
        # Plot distribution
        plt.figure(figsize=(12, 6))
        bars = plt.bar(self.class_names, class_counts)
        plt.title('Class Distribution in Dataset')
        plt.xlabel('Classes')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, class_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return class_counts

class Cutout:
    """
    Cutout data augmentation technique.
    
    Randomly masks out square regions of input image.
    """
    
    def __init__(self, n_holes: int = 1, length: int = 16):
        """
        Initialize Cutout augmentation.
        
        Args:
            n_holes: Number of holes to cut out
            length: Length of the square hole
        """
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img):
        """
        Apply cutout augmentation to image.
        
        Args:
            img: PIL Image or Tensor
            
        Returns:
            Augmented image
        """
        h = img.size(1) if isinstance(img, torch.Tensor) else img.size[1]
        w = img.size(2) if isinstance(img, torch.Tensor) else img.size[0]
        
        mask = np.ones((h, w), np.float32)
        
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            mask[y1: y2, x1: x2] = 0.
        
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        
        return img

class MixupDataset:
    """
    Dataset wrapper for Mixup data augmentation.
    """
    
    def __init__(self, dataset, alpha: float = 1.0):
        """
        Initialize Mixup dataset.
        
        Args:
            dataset: Base dataset
            alpha: Mixup parameter
        """
        self.dataset = dataset
        self.alpha = alpha
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        """
        Get item with mixup augmentation.
        
        Args:
            index: Index of the item
            
        Returns:
            Mixed image and labels
        """
        # Get original item
        img1, label1 = self.dataset[index]
        
        # Get random second item
        index2 = random.randint(0, len(self.dataset) - 1)
        img2, label2 = self.dataset[index2]
        
        # Generate mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix images
        mixed_img = lam * img1 + (1 - lam) * img2
        
        # Mix labels (one-hot encoding)
        label1_onehot = torch.zeros(10)
        label2_onehot = torch.zeros(10)
        label1_onehot[label1] = 1
        label2_onehot[label2] = 1
        
        mixed_label = lam * label1_onehot + (1 - lam) * label2_onehot
        
        return mixed_img, mixed_label

def create_balanced_subset(dataset, samples_per_class: int = 500):
    """
    Create a balanced subset of the dataset with equal samples per class.
    
    Args:
        dataset: Original dataset
        samples_per_class: Number of samples per class
        
    Returns:
        Balanced subset dataset
    """
    class_indices = [[] for _ in range(10)]
    
    # Collect indices for each class
    for idx, (_, label) in enumerate(dataset):
        if isinstance(label, torch.Tensor):
            label = label.item()
        class_indices[label].append(idx)
    
    # Select balanced samples
    selected_indices = []
    for class_idx in range(10):
        selected_indices.extend(
            random.sample(class_indices[class_idx], 
                         min(samples_per_class, len(class_indices[class_idx])))
        )
    
    return Subset(dataset, selected_indices)

def visualize_data_augmentation(dataset, num_samples: int = 8):
    """
    Visualize the effect of data augmentation on sample images.
    
    Args:
        dataset: Dataset with augmentation
        num_samples: Number of samples to show
    """
    processor = CIFAR10DataProcessor()
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(num_samples):
        # Original image (without augmentation)
        original_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(processor.mean, processor.std)
        ])
        
        # Get original dataset
        original_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=False, transform=original_transform
        )
        
        img, label = original_dataset[i]
        img = processor.denormalize_image(img).permute(1, 2, 0)
        
        axes[0, i].imshow(img)
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Augmented image
        img_aug, _ = dataset[i]
        img_aug = processor.denormalize_image(img_aug).permute(1, 2, 0)
        
        axes[1, i].imshow(img_aug)
        axes[1, i].set_title('Augmented')
        axes[1, i].axis('off')
    
    plt.suptitle('Data Augmentation Comparison', fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    processor = CIFAR10DataProcessor()
    
    # Create data loaders with augmentation
    train_loader, val_loader, test_loader = processor.create_data_loaders(
        batch_size=128, 
        use_augmentation=True
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Visualize dataset samples
    train_dataset, _, _ = processor.load_datasets()
    processor.visualize_dataset_samples(train_dataset, title="Training Dataset Samples")
    
    # Analyze class distribution
    processor.analyze_dataset_distribution(train_dataset)
