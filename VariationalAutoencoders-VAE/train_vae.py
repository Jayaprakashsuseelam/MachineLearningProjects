"""
Training script for Variational Autoencoder (VAE)

This script demonstrates how to train a VAE on MNIST dataset with
comprehensive visualization and evaluation.

Usage:
    python train_vae.py --epochs 50 --batch_size 128 --latent_dim 2
"""

import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from vae_model import VAE, VAETrainer, visualize_reconstruction, visualize_latent_space, generate_samples
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_data_loaders(batch_size: int = 128, 
                    data_dir: str = './data') -> tuple:
    """
    Get MNIST data loaders.
    
    Args:
        batch_size: Batch size for data loaders
        data_dir: Directory to store data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Flatten to 1D vector
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def plot_training_history(trainer: VAETrainer, save_path: str = None) -> None:
    """
    Plot training history.
    
    Args:
        trainer: Trained VAE trainer
        save_path: Path to save the plot
    """
    epochs = range(1, len(trainer.train_losses) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Total loss
    axes[0].plot(epochs, [loss['total_loss'] for loss in trainer.train_losses], 
                 label='Train', color='blue')
    axes[0].plot(epochs, [loss['total_loss'] for loss in trainer.val_losses], 
                 label='Validation', color='red')
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[1].plot(epochs, [loss['reconstruction_loss'] for loss in trainer.train_losses], 
                 label='Train', color='blue')
    axes[1].plot(epochs, [loss['reconstruction_loss'] for loss in trainer.val_losses], 
                 label='Validation', color='red')
    axes[1].set_title('Reconstruction Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # KL divergence loss
    axes[2].plot(epochs, [loss['kl_loss'] for loss in trainer.train_losses], 
                 label='Train', color='blue')
    axes[2].plot(epochs, [loss['kl_loss'] for loss in trainer.val_losses], 
                 label='Validation', color='red')
    axes[2].set_title('KL Divergence Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()


def interpolate_in_latent_space(model: VAE, 
                               device: torch.device,
                               num_steps: int = 10,
                               save_path: str = None) -> None:
    """
    Demonstrate latent space interpolation.
    
    Args:
        model: Trained VAE model
        device: Device
        num_steps: Number of interpolation steps
        save_path: Path to save the visualization
    """
    model.eval()
    
    # Generate two random points in latent space
    z1 = torch.randn(1, model.latent_dim).to(device)
    z2 = torch.randn(1, model.latent_dim).to(device)
    
    # Create interpolation
    interpolations = []
    for i in range(num_steps):
        alpha = i / (num_steps - 1)
        z_interp = (1 - alpha) * z1 + alpha * z2
        interpolations.append(z_interp)
    
    # Generate images
    with torch.no_grad():
        generated_images = []
        for z in interpolations:
            img = model.decoder(z)
            generated_images.append(img.cpu().numpy().reshape(28, 28))
    
    # Visualize interpolation
    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 2, 2))
    
    for i, img in enumerate(generated_images):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Step {i+1}')
        axes[i].axis('off')
    
    plt.suptitle('Latent Space Interpolation')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Interpolation visualization saved to {save_path}")
    
    plt.show()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train VAE on MNIST')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--latent_dim', type=int, default=2, help='Latent dimension')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta parameter for β-VAE')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--save_dir', type=str, default='./results', help='Save directory')
    parser.add_argument('--device', type=str, default='auto', help='Device (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Training parameters: epochs={args.epochs}, batch_size={args.batch_size}, "
               f"latent_dim={args.latent_dim}, lr={args.learning_rate}, beta={args.beta}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(args.batch_size, args.data_dir)
    
    # Model parameters
    input_dim = 784  # 28x28 for MNIST
    hidden_dims = [512, 256]
    
    # Create model
    model = VAE(input_dim, hidden_dims, args.latent_dim)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = VAETrainer(model, device, args.learning_rate, args.beta)
    
    # Train model
    logger.info("Starting training...")
    trainer.train(train_loader, val_loader, args.epochs, 
                  save_path=os.path.join(args.save_dir, 'vae_model'))
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    
    # Training history
    plot_training_history(trainer, 
                         save_path=os.path.join(args.save_dir, 'training_history.png'))
    
    # Reconstructions
    visualize_reconstruction(model, val_loader, device, 
                           save_path=os.path.join(args.save_dir, 'reconstructions.png'))
    
    # Latent space visualization (only for 2D latent space)
    if args.latent_dim == 2:
        visualize_latent_space(model, val_loader, device,
                              save_path=os.path.join(args.save_dir, 'latent_space.png'))
    
    # Generated samples
    generate_samples(model, device, 
                    save_path=os.path.join(args.save_dir, 'generated_samples.png'))
    
    # Latent space interpolation
    if args.latent_dim == 2:
        interpolate_in_latent_space(model, device,
                                   save_path=os.path.join(args.save_dir, 'interpolation.png'))
    
    logger.info("Training completed successfully!")
    logger.info(f"Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
