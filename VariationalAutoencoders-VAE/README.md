# Variational Autoencoders (VAE) - Comprehensive Implementation

A complete implementation of Variational Autoencoders with theoretical foundations, practical examples, and comprehensive documentation.

## Table of Contents

- [Overview](#overview)
- [Theoretical Foundations](#theoretical-foundations)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Advanced Features](#advanced-features)
- [Mathematical Background](#mathematical-background)
- [Visualizations](#visualizations)
- [Performance Tips](#performance-tips)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project provides a comprehensive implementation of Variational Autoencoders (VAEs), including:

- **Complete VAE Architecture**: Encoder, decoder, and reparameterization trick
- **Theoretical Foundations**: Mathematical explanations and intuitions
- **Practical Examples**: Multiple use cases and demonstrations
- **Advanced Techniques**: β-VAE, different architectures, and optimization strategies
- **Comprehensive Visualizations**: Training progress, latent space, and generated samples

### Key Features

- 🧠 **Theoretical Understanding**: Deep mathematical foundations with intuitive explanations
- 🔧 **Practical Implementation**: Production-ready code with comprehensive error handling
- 📊 **Rich Visualizations**: Training progress, latent space analysis, and sample generation
- 🚀 **Advanced Techniques**: β-VAE, different architectures, and optimization strategies
- 📚 **Educational Examples**: Step-by-step demonstrations and comparisons

## Theoretical Foundations

### What is a VAE?

A Variational Autoencoder is a generative model that learns to:
1. **Encode**: Map high-dimensional data to a lower-dimensional latent space
2. **Decode**: Reconstruct original data from latent representations
3. **Generate**: Create new data by sampling from the latent space

### Key Concepts

- **Probabilistic Encoding**: Instead of deterministic mapping, VAEs learn probabilistic distributions
- **Reparameterization Trick**: Enables differentiable sampling from latent distributions
- **Evidence Lower BOund (ELBO)**: Objective function balancing reconstruction and regularization
- **Continuous Latent Space**: Enables smooth interpolation and meaningful arithmetic

### Mathematical Formulation

The VAE optimizes the Evidence Lower BOund:

```
ELBO = E[log p(x|z)] - D_KL(q(z|x) || p(z))
```

Where:
- `E[log p(x|z)]`: Reconstruction term (likelihood)
- `D_KL(q(z|x) || p(z))`: Regularization term (KL divergence)
- `q(z|x)`: Encoder distribution (approximate posterior)
- `p(z)`: Prior distribution (typically N(0, I))

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/VariationalAutoencoders-VAE.git
cd VariationalAutoencoders-VAE

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```python
import torch
from vae_model import VAE
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Quick Start

### Basic Usage

```python
import torch
from vae_model import VAE, VAETrainer
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model
model = VAE(input_dim=784, hidden_dims=[512, 256], latent_dim=2)

# Create trainer
trainer = VAETrainer(model, device, learning_rate=1e-3, beta=1.0)

# Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Train model
trainer.train(train_loader, train_loader, epochs=50)
```

### Command Line Training

```bash
# Basic training
python train_vae.py --epochs 50 --batch_size 128 --latent_dim 2

# β-VAE training
python train_vae.py --epochs 50 --beta 5.0 --latent_dim 2

# High-dimensional latent space
python train_vae.py --epochs 50 --latent_dim 10 --learning_rate 1e-4
```

## Project Structure

```
VariationalAutoencoders-VAE/
├── vae_model.py              # Core VAE implementation
├── train_vae.py              # Training script with CLI
├── theoretical_foundations.py # Mathematical foundations and theory
├── examples.py               # Comprehensive examples and demonstrations
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── results/                  # Training outputs and visualizations
    ├── vae_model_final.pth   # Trained model weights
    ├── training_history.png  # Training progress plots
    ├── reconstructions.png   # Reconstruction examples
    ├── latent_space.png      # Latent space visualization
    └── generated_samples.png # Generated samples
```

## Usage Examples

### Example 1: Basic VAE Training

```python
from examples import VAEExamples

# Initialize examples
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
examples = VAEExamples(device)

# Run basic training example
examples.example_1_basic_training()
```

### Example 2: β-VAE Comparison

```python
# Compare different β values
examples.example_2_beta_vae_comparison()
```

### Example 3: Latent Space Interpolation

```python
# Demonstrate smooth interpolation in latent space
examples.example_3_latent_space_interpolation()
```

### Example 4: Different Latent Dimensions

```python
# Compare models with different latent dimensions
examples.example_4_different_latent_dimensions()
```

### Example 5: Advanced Techniques

```python
# Explore different architectures and optimization strategies
examples.example_5_advanced_techniques()
```

## Advanced Features

### β-VAE Implementation

β-VAE introduces a hyperparameter β to control the trade-off between reconstruction quality and disentanglement:

```python
# Standard VAE (β = 1.0)
trainer = VAETrainer(model, device, beta=1.0)

# β-VAE with emphasis on disentanglement (β > 1.0)
trainer = VAETrainer(model, device, beta=5.0)

# β-VAE with emphasis on reconstruction (β < 1.0)
trainer = VAETrainer(model, device, beta=0.1)
```

### Custom Architectures

```python
# Shallow architecture
model = VAE(input_dim=784, hidden_dims=[256], latent_dim=2)

# Deep architecture
model = VAE(input_dim=784, hidden_dims=[1024, 512, 256, 128], latent_dim=2)

# Wide architecture
model = VAE(input_dim=784, hidden_dims=[1024, 1024], latent_dim=2)
```

### Advanced Training Options

```python
# Learning rate scheduling
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Early stopping
# (Implementation details in train_vae.py)
```

## Mathematical Background

### Reparameterization Trick

Instead of sampling directly from N(μ, σ²), we sample ε ~ N(0,1) and compute:

```
z = μ + σ ⊙ ε
```

This enables gradients to flow through the sampling process.

### KL Divergence

For Gaussian distributions, the KL divergence has a closed form:

```
D_KL(N(μ, σ²) || N(0, I)) = ½ Σᵢ(μᵢ² + σᵢ² - log(σᵢ²) - 1)
```

### Loss Function

The VAE loss is the negative ELBO:

```
L = -ELBO = -E[log p(x|z)] + D_KL(q(z|x) || p(z))
```

For binary data (e.g., MNIST), the reconstruction loss is:

```
-E[log p(x|z)] = -Σᵢ(xᵢ log(x̂ᵢ) + (1-xᵢ)log(1-x̂ᵢ))
```

## Visualizations

### Training Progress

The training script generates comprehensive visualizations:

- **Training History**: Loss curves for reconstruction and KL divergence
- **Reconstructions**: Original vs. reconstructed samples
- **Latent Space**: 2D visualization of learned representations
- **Generated Samples**: New samples created from the model
- **Interpolation**: Smooth transitions between latent codes

### Custom Visualizations

```python
from vae_model import visualize_reconstruction, visualize_latent_space, generate_samples

# Reconstruction visualization
visualize_reconstruction(model, dataloader, device, num_samples=8)

# Latent space visualization (2D only)
visualize_latent_space(model, dataloader, device)

# Generated samples
generate_samples(model, device, num_samples=16)
```

## Performance Tips

### Training Optimization

1. **Batch Size**: Use larger batches (128-512) for stable training
2. **Learning Rate**: Start with 1e-3, adjust based on convergence
3. **Architecture**: Deeper networks can learn more complex mappings
4. **Latent Dimension**: Balance between expressiveness and regularization

### Common Issues and Solutions

1. **Blurry Reconstructions**: 
   - Increase β to encourage better latent representations
   - Use deeper decoder architecture
   - Increase latent dimension

2. **Posterior Collapse**: 
   - Decrease β to prioritize reconstruction
   - Use skip connections in decoder
   - Implement KL annealing

3. **Mode Collapse**: 
   - Increase latent dimension
   - Use different initialization strategies
   - Implement diversity regularization

### Hardware Recommendations

- **CPU**: Multi-core processor for data loading
- **GPU**: NVIDIA GPU with CUDA support for faster training
- **Memory**: 8GB+ RAM for larger datasets
- **Storage**: SSD for faster data loading

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black *.py

# Lint code
flake8 *.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original VAE paper: "Auto-Encoding Variational Bayes" by Kingma & Welling (2014)
- β-VAE paper: "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" by Higgins et al. (2017)
- PyTorch team for the excellent deep learning framework
- MNIST dataset creators for the standard benchmark

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{vae_implementation,
  title={Variational Autoencoders - Comprehensive Implementation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/VariationalAutoencoders-VAE}
}
```

---

**Happy Learning!** 🚀

For questions, issues, or contributions, please open an issue on GitHub or contact the maintainers.
