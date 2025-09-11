"""
Example Usage and Demonstrations for Variational Autoencoder (VAE)

This module provides comprehensive examples demonstrating VAE capabilities
including training, evaluation, and advanced techniques.

Author: AI Assistant
Date: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from vae_model import VAE, VAETrainer, visualize_reconstruction, visualize_latent_space, generate_samples
from theoretical_foundations import TheoreticalVAE
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VAEExamples:
    """
    Comprehensive examples demonstrating VAE capabilities.
    """
    
    def __init__(self, device: torch.device):
        """
        Initialize VAE examples.
        
        Args:
            device: Device to run examples on
        """
        self.device = device
        self.theoretical_vae = TheoreticalVAE()
    
    def example_1_basic_training(self):
        """
        Example 1: Basic VAE training on MNIST.
        """
        logger.info("=== Example 1: Basic VAE Training ===")
        
        # Model parameters
        input_dim = 784  # 28x28 for MNIST
        hidden_dims = [512, 256]
        latent_dim = 2
        
        # Create model
        model = VAE(input_dim, hidden_dims, latent_dim)
        trainer = VAETrainer(model, self.device, learning_rate=1e-3, beta=1.0)
        
        # Get data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        val_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        
        # Train for a few epochs
        logger.info("Training VAE for 5 epochs...")
        trainer.train(train_loader, val_loader, epochs=5)
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        visualize_reconstruction(model, val_loader, self.device, num_samples=8)
        visualize_latent_space(model, val_loader, self.device)
        generate_samples(model, self.device, num_samples=16)
        
        logger.info("Example 1 completed!")
    
    def example_2_beta_vae_comparison(self):
        """
        Example 2: Compare different β values in β-VAE.
        """
        logger.info("=== Example 2: β-VAE Comparison ===")
        
        # Model parameters
        input_dim = 784
        hidden_dims = [512, 256]
        latent_dim = 2
        
        # Different β values to test
        beta_values = [0.1, 1.0, 5.0]
        
        # Get data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        
        models = {}
        trainers = {}
        
        # Train models with different β values
        for beta in beta_values:
            logger.info(f"Training VAE with β = {beta}")
            
            model = VAE(input_dim, hidden_dims, latent_dim)
            trainer = VAETrainer(model, self.device, learning_rate=1e-3, beta=beta)
            
            # Train for a few epochs
            trainer.train(train_loader, train_loader, epochs=3)  # Using train as val for simplicity
            
            models[beta] = model
            trainers[beta] = trainer
        
        # Compare reconstructions
        fig, axes = plt.subplots(len(beta_values), 8, figsize=(16, 6))
        
        # Get sample data
        data_iter = iter(train_loader)
        original, _ = next(data_iter)
        original = original[:8].view(8, -1).to(self.device)
        
        for i, beta in enumerate(beta_values):
            model = models[beta]
            model.eval()
            
            with torch.no_grad():
                reconstructed, _, _ = model(original)
            
            # Convert to numpy
            original_np = original.cpu().numpy()
            reconstructed_np = reconstructed.cpu().numpy()
            
            for j in range(8):
                if i == 0:  # Show original only once
                    axes[i, j].imshow(original_np[j].reshape(28, 28), cmap='gray')
                    axes[i, j].set_title(f'Original' if j == 0 else '')
                else:
                    axes[i, j].imshow(reconstructed_np[j].reshape(28, 28), cmap='gray')
                    axes[i, j].set_title(f'β={beta}' if j == 0 else '')
                axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Compare training losses
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        for beta in beta_values:
            trainer = trainers[beta]
            epochs = range(1, len(trainer.train_losses) + 1)
            plt.plot(epochs, [loss['reconstruction_loss'] for loss in trainer.train_losses], 
                    label=f'β={beta}', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Reconstruction Loss')
        plt.title('Reconstruction Loss Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        for beta in beta_values:
            trainer = trainers[beta]
            epochs = range(1, len(trainer.train_losses) + 1)
            plt.plot(epochs, [loss['kl_loss'] for loss in trainer.train_losses], 
                    label=f'β={beta}', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('KL Loss')
        plt.title('KL Loss Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        logger.info("Example 2 completed!")
    
    def example_3_latent_space_interpolation(self):
        """
        Example 3: Demonstrate latent space interpolation.
        """
        logger.info("=== Example 3: Latent Space Interpolation ===")
        
        # Train a VAE first
        input_dim = 784
        hidden_dims = [512, 256]
        latent_dim = 2
        
        model = VAE(input_dim, hidden_dims, latent_dim)
        trainer = VAETrainer(model, self.device, learning_rate=1e-3, beta=1.0)
        
        # Get data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        
        # Train for a few epochs
        trainer.train(train_loader, train_loader, epochs=3)
        
        # Get two random samples
        data_iter = iter(train_loader)
        sample1, _ = next(data_iter)
        sample2, _ = next(data_iter)
        
        sample1 = sample1[0:1].view(1, -1).to(self.device)
        sample2 = sample2[0:1].view(1, -1).to(self.device)
        
        model.eval()
        
        # Encode samples
        with torch.no_grad():
            mu1, logvar1 = model.encoder(sample1)
            mu2, logvar2 = model.encoder(sample2)
            
            z1 = model.reparameterize(mu1, logvar1)
            z2 = model.reparameterize(mu2, logvar2)
        
        # Create interpolation
        num_steps = 10
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
        plt.show()
        
        # Show original samples
        fig, axes = plt.subplots(1, 2, figsize=(4, 2))
        
        axes[0].imshow(sample1.cpu().numpy().reshape(28, 28), cmap='gray')
        axes[0].set_title('Sample 1')
        axes[0].axis('off')
        
        axes[1].imshow(sample2.cpu().numpy().reshape(28, 28), cmap='gray')
        axes[1].set_title('Sample 2')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        logger.info("Example 3 completed!")
    
    def example_4_different_latent_dimensions(self):
        """
        Example 4: Compare different latent dimensions.
        """
        logger.info("=== Example 4: Different Latent Dimensions ===")
        
        # Different latent dimensions to test
        latent_dims = [2, 10, 50]
        
        # Get data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        
        models = {}
        trainers = {}
        
        # Train models with different latent dimensions
        for latent_dim in latent_dims:
            logger.info(f"Training VAE with latent_dim = {latent_dim}")
            
            model = VAE(784, [512, 256], latent_dim)
            trainer = VAETrainer(model, self.device, learning_rate=1e-3, beta=1.0)
            
            # Train for a few epochs
            trainer.train(train_loader, train_loader, epochs=3)
            
            models[latent_dim] = model
            trainers[latent_dim] = trainer
        
        # Compare reconstructions
        fig, axes = plt.subplots(len(latent_dims), 8, figsize=(16, 6))
        
        # Get sample data
        data_iter = iter(train_loader)
        original, _ = next(data_iter)
        original = original[:8].view(8, -1).to(self.device)
        
        for i, latent_dim in enumerate(latent_dims):
            model = models[latent_dim]
            model.eval()
            
            with torch.no_grad():
                reconstructed, _, _ = model(original)
            
            # Convert to numpy
            reconstructed_np = reconstructed.cpu().numpy()
            
            for j in range(8):
                axes[i, j].imshow(reconstructed_np[j].reshape(28, 28), cmap='gray')
                axes[i, j].set_title(f'Latent Dim={latent_dim}' if j == 0 else '')
                axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Compare training losses
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        for latent_dim in latent_dims:
            trainer = trainers[latent_dim]
            epochs = range(1, len(trainer.train_losses) + 1)
            plt.plot(epochs, [loss['reconstruction_loss'] for loss in trainer.train_losses], 
                    label=f'Latent Dim={latent_dim}', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Reconstruction Loss')
        plt.title('Reconstruction Loss Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        for latent_dim in latent_dims:
            trainer = trainers[latent_dim]
            epochs = range(1, len(trainer.train_losses) + 1)
            plt.plot(epochs, [loss['kl_loss'] for loss in trainer.train_losses], 
                    label=f'Latent Dim={latent_dim}', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('KL Loss')
        plt.title('KL Loss Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        logger.info("Example 4 completed!")
    
    def example_5_advanced_techniques(self):
        """
        Example 5: Advanced VAE techniques and tips.
        """
        logger.info("=== Example 5: Advanced Techniques ===")
        
        # Demonstrate different architectures
        architectures = {
            "shallow": [256],
            "medium": [512, 256],
            "deep": [1024, 512, 256, 128]
        }
        
        # Get data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        
        models = {}
        trainers = {}
        
        # Train models with different architectures
        for arch_name, hidden_dims in architectures.items():
            logger.info(f"Training VAE with {arch_name} architecture")
            
            model = VAE(784, hidden_dims, 2)
            trainer = VAETrainer(model, self.device, learning_rate=1e-3, beta=1.0)
            
            # Train for a few epochs
            trainer.train(train_loader, train_loader, epochs=3)
            
            models[arch_name] = model
            trainers[arch_name] = trainer
        
        # Compare model sizes
        model_sizes = {}
        for arch_name, model in models.items():
            model_sizes[arch_name] = sum(p.numel() for p in model.parameters())
        
        # Plot model sizes
        plt.figure(figsize=(8, 6))
        arch_names = list(model_sizes.keys())
        sizes = list(model_sizes.values())
        
        plt.bar(arch_names, sizes)
        plt.xlabel('Architecture')
        plt.ylabel('Number of Parameters')
        plt.title('Model Size Comparison')
        plt.grid(True, alpha=0.3)
        
        for i, size in enumerate(sizes):
            plt.text(i, size + max(sizes) * 0.01, f'{size:,}', ha='center')
        
        plt.tight_layout()
        plt.show()
        
        # Demonstrate learning rate scheduling
        logger.info("Demonstrating learning rate scheduling...")
        
        model = VAE(784, [512, 256], 2)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        
        trainer = VAETrainer(model, self.device, learning_rate=1e-3, beta=1.0)
        trainer.optimizer = optimizer
        
        # Train with learning rate scheduling
        for epoch in range(5):
            trainer.train_epoch(train_loader)
            scheduler.step()
            logger.info(f"Epoch {epoch+1}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        logger.info("Example 5 completed!")
    
    def run_all_examples(self):
        """
        Run all examples.
        """
        logger.info("Running all VAE examples...")
        
        try:
            self.example_1_basic_training()
        except Exception as e:
            logger.error(f"Example 1 failed: {e}")
        
        try:
            self.example_2_beta_vae_comparison()
        except Exception as e:
            logger.error(f"Example 2 failed: {e}")
        
        try:
            self.example_3_latent_space_interpolation()
        except Exception as e:
            logger.error(f"Example 3 failed: {e}")
        
        try:
            self.example_4_different_latent_dimensions()
        except Exception as e:
            logger.error(f"Example 4 failed: {e}")
        
        try:
            self.example_5_advanced_techniques()
        except Exception as e:
            logger.error(f"Example 5 failed: {e}")
        
        logger.info("All examples completed!")


def main():
    """Main function to run examples."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create examples
    examples = VAEExamples(device)
    
    # Run all examples
    examples.run_all_examples()


if __name__ == "__main__":
    main()
