"""
Test script for VAE implementation

This script verifies that all components work correctly.
"""

import torch
import torch.nn as nn
from vae_model import VAE, VAETrainer, vae_loss
from theoretical_foundations import TheoreticalVAE
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_vae_components():
    """Test individual VAE components."""
    logger.info("Testing VAE components...")
    
    # Test parameters
    input_dim = 784
    hidden_dims = [512, 256]
    latent_dim = 2
    batch_size = 32
    
    # Create model
    model = VAE(input_dim, hidden_dims, latent_dim)
    
    # Test forward pass (use normalized data for binary cross entropy)
    x = torch.rand(batch_size, input_dim)  # Random values between 0 and 1
    reconstructed, mu, logvar = model(x)
    
    # Check shapes
    assert reconstructed.shape == x.shape, f"Reconstruction shape mismatch: {reconstructed.shape} vs {x.shape}"
    assert mu.shape == (batch_size, latent_dim), f"Mu shape mismatch: {mu.shape}"
    assert logvar.shape == (batch_size, latent_dim), f"Logvar shape mismatch: {logvar.shape}"
    
    # Test loss computation
    loss, loss_dict = vae_loss(reconstructed, x, mu, logvar)
    assert loss.item() > 0, "Loss should be positive"
    assert 'reconstruction_loss' in loss_dict, "Missing reconstruction loss"
    assert 'kl_loss' in loss_dict, "Missing KL loss"
    assert 'total_loss' in loss_dict, "Missing total loss"
    
    # Test sampling
    samples = model.sample(16, torch.device('cpu'))
    assert samples.shape == (16, input_dim), f"Sample shape mismatch: {samples.shape}"
    
    # Test encoding
    z = model.encode(x)
    assert z.shape == (batch_size, latent_dim), f"Encoded shape mismatch: {z.shape}"
    
    logger.info("✓ All VAE components working correctly!")


def test_trainer():
    """Test VAE trainer."""
    logger.info("Testing VAE trainer...")
    
    # Create model and trainer
    model = VAE(784, [512, 256], 2)
    device = torch.device('cpu')
    trainer = VAETrainer(model, device, learning_rate=1e-3, beta=1.0)
    
    # Create dummy data (normalized for binary cross entropy)
    x = torch.rand(32, 784)  # Random values between 0 and 1
    dataloader = [(x, torch.zeros(32))]  # Dummy labels
    
    # Test training step
    trainer.model.train()
    loss_dict = trainer.train_epoch(dataloader)
    
    assert 'reconstruction_loss' in loss_dict, "Missing reconstruction loss in trainer"
    assert 'kl_loss' in loss_dict, "Missing KL loss in trainer"
    assert 'total_loss' in loss_dict, "Missing total loss in trainer"
    
    logger.info("✓ VAE trainer working correctly!")


def test_theoretical_foundations():
    """Test theoretical foundations."""
    logger.info("Testing theoretical foundations...")
    
    theoretical_vae = TheoreticalVAE()
    
    # Test explanations
    intuition = theoretical_vae.explain_vae_intuition()
    assert len(intuition) > 0, "Intuition explanation should not be empty"
    
    math_foundations = theoretical_vae.mathematical_foundations()
    assert len(math_foundations) > 0, "Mathematical foundations should not be empty"
    
    analysis = theoretical_vae.analyze_latent_space_properties()
    assert len(analysis) > 0, "Latent space analysis should not be empty"
    
    comparisons = theoretical_vae.compare_with_other_models()
    assert len(comparisons) > 0, "Model comparisons should not be empty"
    
    logger.info("✓ Theoretical foundations working correctly!")


def test_different_configurations():
    """Test different VAE configurations."""
    logger.info("Testing different VAE configurations...")
    
    configurations = [
        {"input_dim": 784, "hidden_dims": [256], "latent_dim": 2},
        {"input_dim": 784, "hidden_dims": [512, 256], "latent_dim": 10},
        {"input_dim": 784, "hidden_dims": [1024, 512, 256], "latent_dim": 50},
    ]
    
    for config in configurations:
        model = VAE(**config)
        x = torch.rand(16, config["input_dim"])  # Normalized data
        
        # Test forward pass
        reconstructed, mu, logvar = model(x)
        
        # Check shapes
        assert reconstructed.shape == x.shape
        assert mu.shape == (16, config["latent_dim"])
        assert logvar.shape == (16, config["latent_dim"])
        
        # Test sampling
        samples = model.sample(8, torch.device('cpu'))
        assert samples.shape == (8, config["input_dim"])
    
    logger.info("✓ Different configurations working correctly!")


def main():
    """Run all tests."""
    logger.info("Starting VAE implementation tests...")
    
    try:
        test_vae_components()
        test_trainer()
        test_theoretical_foundations()
        test_different_configurations()
        
        logger.info("🎉 All tests passed successfully!")
        logger.info("VAE implementation is ready to use!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
