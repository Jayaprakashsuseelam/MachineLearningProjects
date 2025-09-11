"""
Variational Autoencoder (VAE) Implementation

This module contains the complete implementation of a Variational Autoencoder
with theoretical foundations and practical applications.

Author: AI Assistant
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from typing import Tuple, Optional, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    """
    Encoder network for VAE.
    
    Maps input data to latent space parameters (mean and log variance).
    """
    
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        """
        Initialize the encoder.
        
        Args:
            input_dim: Dimension of input data
            hidden_dims: List of hidden layer dimensions
            latent_dim: Dimension of latent space
        """
        super(Encoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Output layers for mean and log variance
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor
            
        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder network for VAE.
    
    Maps latent space samples back to data space.
    """
    
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int):
        """
        Initialize the decoder.
        
        Args:
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions (in reverse order)
            output_dim: Dimension of output data
        """
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Build decoder layers
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # Ensure output is in [0,1] for image data
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            z: Latent tensor
            
        Returns:
            Reconstructed data
        """
        return self.decoder(z)


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) implementation.
    
    This class implements the complete VAE architecture with:
    - Encoder: Maps data to latent space parameters
    - Decoder: Maps latent samples to data space
    - Reparameterization trick for differentiable sampling
    - ELBO loss computation
    """
    
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        """
        Initialize VAE.
        
        Args:
            input_dim: Dimension of input data
            hidden_dims: List of hidden layer dimensions
            latent_dim: Dimension of latent space
        """
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Initialize encoder and decoder
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims[::-1], input_dim)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for differentiable sampling.
        
        Instead of sampling directly from N(μ, σ²), we sample ε ~ N(0,1)
        and compute z = μ + σ * ε, where σ = exp(logvar/2).
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input tensor
            
        Returns:
            reconstructed: Reconstructed data
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate samples from the VAE.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            Generated samples
        """
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decoder(z)
        return samples
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space.
        
        Args:
            x: Input tensor
            
        Returns:
            Latent representation
        """
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)


def vae_loss(reconstructed: torch.Tensor, 
             x: torch.Tensor, 
             mu: torch.Tensor, 
             logvar: torch.Tensor,
             beta: float = 1.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute VAE loss (ELBO).
    
    The Evidence Lower BOund (ELBO) consists of:
    1. Reconstruction loss: -E[log p(x|z)]
    2. KL divergence: D_KL(q(z|x) || p(z))
    
    Args:
        reconstructed: Reconstructed data
        x: Original data
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence (β-VAE)
        
    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary containing individual loss components
    """
    # Reconstruction loss (Binary Cross Entropy for image data)
    recon_loss = F.binary_cross_entropy(reconstructed, x, reduction='sum')
    
    # KL divergence: D_KL(q(z|x) || p(z))
    # KL(q(z|x) || N(0,I)) = 0.5 * sum(μ² + σ² - log(σ²) - 1)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss (negative ELBO)
    total_loss = recon_loss + beta * kl_loss
    
    loss_dict = {
        'reconstruction_loss': recon_loss,
        'kl_loss': kl_loss,
        'total_loss': total_loss
    }
    
    return total_loss, loss_dict


class VAETrainer:
    """
    Trainer class for VAE with comprehensive training utilities.
    """
    
    def __init__(self, 
                 model: VAE, 
                 device: torch.device,
                 learning_rate: float = 1e-3,
                 beta: float = 1.0):
        """
        Initialize VAE trainer.
        
        Args:
            model: VAE model
            device: Device to train on
            learning_rate: Learning rate for optimizer
            beta: Weight for KL divergence
        """
        self.model = model.to(device)
        self.device = device
        self.beta = beta
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Dictionary of average losses
        """
        self.model.train()
        total_losses = {'reconstruction_loss': 0, 'kl_loss': 0, 'total_loss': 0}
        num_batches = 0
        
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.view(data.size(0), -1).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstructed, mu, logvar = self.model(data)
            
            # Compute loss
            loss, loss_dict = vae_loss(reconstructed, data, mu, logvar, self.beta)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            for key in total_losses:
                total_losses[key] += loss_dict[key].item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                logger.info(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
        
        # Average losses
        avg_losses = {key: val / num_batches for key, val in total_losses.items()}
        self.train_losses.append(avg_losses)
        
        return avg_losses
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary of average losses
        """
        self.model.eval()
        total_losses = {'reconstruction_loss': 0, 'kl_loss': 0, 'total_loss': 0}
        num_batches = 0
        
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.view(data.size(0), -1).to(self.device)
                
                # Forward pass
                reconstructed, mu, logvar = self.model(data)
                
                # Compute loss
                loss, loss_dict = vae_loss(reconstructed, data, mu, logvar, self.beta)
                
                # Accumulate losses
                for key in total_losses:
                    total_losses[key] += loss_dict[key].item()
                num_batches += 1
        
        # Average losses
        avg_losses = {key: val / num_batches for key, val in total_losses.items()}
        self.val_losses.append(avg_losses)
        
        return avg_losses
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader, 
              epochs: int,
              save_path: Optional[str] = None) -> None:
        """
        Train the VAE model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            save_path: Path to save the trained model
        """
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
            train_losses = self.train_epoch(train_loader)
            
            # Validation
            val_losses = self.validate(val_loader)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"Train Loss: {train_losses['total_loss']:.4f} "
                       f"(Recon: {train_losses['reconstruction_loss']:.4f}, "
                       f"KL: {train_losses['kl_loss']:.4f})")
            logger.info(f"Val Loss: {val_losses['total_loss']:.4f} "
                       f"(Recon: {val_losses['reconstruction_loss']:.4f}, "
                       f"KL: {val_losses['kl_loss']:.4f})")
            
            # Save model
            if save_path and (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                }, f"{save_path}_epoch_{epoch+1}.pth")
        
        # Save final model
        if save_path:
            torch.save({
                'epoch': epochs,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
            }, f"{save_path}_final.pth")
            logger.info(f"Model saved to {save_path}_final.pth")


def visualize_reconstruction(model: VAE, 
                           dataloader: DataLoader, 
                           device: torch.device,
                           num_samples: int = 8,
                           save_path: Optional[str] = None) -> None:
    """
    Visualize VAE reconstructions.
    
    Args:
        model: Trained VAE model
        dataloader: Data loader
        device: Device
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization
    """
    model.eval()
    
    # Get sample data
    data_iter = iter(dataloader)
    original, _ = next(data_iter)
    original = original[:num_samples].view(num_samples, -1).to(device)
    
    with torch.no_grad():
        reconstructed, _, _ = model(original)
    
    # Convert to numpy for visualization
    original = original.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    
    for i in range(num_samples):
        # Original images
        axes[0, i].imshow(original[i].reshape(28, 28), cmap='gray')
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Reconstructed images
        axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Reconstruction visualization saved to {save_path}")
    
    plt.show()


def visualize_latent_space(model: VAE, 
                          dataloader: DataLoader, 
                          device: torch.device,
                          save_path: Optional[str] = None) -> None:
    """
    Visualize latent space representation.
    
    Args:
        model: Trained VAE model
        dataloader: Data loader
        device: Device
        save_path: Path to save the visualization
    """
    model.eval()
    
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data = data.view(data.size(0), -1).to(device)
            mu, logvar = model.encoder(data)
            z = model.reparameterize(mu, logvar)
            
            latent_vectors.append(z.cpu().numpy())
            labels.append(target.numpy())
    
    # Concatenate all batches
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # Plot latent space
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('VAE Latent Space Visualization')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Latent space visualization saved to {save_path}")
    
    plt.show()


def generate_samples(model: VAE, 
                    device: torch.device,
                    num_samples: int = 16,
                    save_path: Optional[str] = None) -> None:
    """
    Generate new samples from the VAE.
    
    Args:
        model: Trained VAE model
        device: Device
        num_samples: Number of samples to generate
        save_path: Path to save the visualization
    """
    model.eval()
    
    with torch.no_grad():
        samples = model.sample(num_samples, device)
    
    # Convert to numpy for visualization
    samples = samples.cpu().numpy()
    
    # Create grid visualization
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < num_samples:
                axes[i, j].imshow(samples[idx].reshape(28, 28), cmap='gray')
                axes[i, j].axis('off')
    
    plt.suptitle('Generated Samples from VAE')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Generated samples saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Model parameters
    input_dim = 784  # 28x28 for MNIST
    hidden_dims = [512, 256]
    latent_dim = 2
    
    # Create model
    model = VAE(input_dim, hidden_dims, latent_dim)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = VAETrainer(model, device, learning_rate=1e-3, beta=1.0)
    
    logger.info("VAE implementation ready for training!")
