# Deep Q-Networks (DQN) - Reinforcement Learning Implementation

A comprehensive implementation of Deep Q-Networks (DQN) with theoretical understanding and practical implementation for solving reinforcement learning problems.

## Table of Contents

1. [Theoretical Background](#theoretical-background)
2. [DQN Algorithm](#dqn-algorithm)
3. [Key Innovations](#key-innovizations)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Implementation Details](#implementation-details)
8. [Results and Visualization](#results-and-visualization)
9. [References](#references)

## Theoretical Background

### Q-Learning Fundamentals

Q-Learning is a model-free reinforcement learning algorithm that learns the quality of actions, telling an agent what action to take under what circumstances. The Q-value represents the expected cumulative reward for taking action `a` in state `s` and following the optimal policy thereafter.

**Q-Learning Update Rule:**
```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

Where:
- `α` (alpha) is the learning rate
- `γ` (gamma) is the discount factor
- `r` is the immediate reward
- `s'` is the next state

### Deep Q-Networks (DQN)

DQN extends traditional Q-Learning to handle high-dimensional state spaces by using deep neural networks to approximate the Q-function. This allows the algorithm to work with complex environments like video games or robotics.

**Key Challenges Addressed by DQN:**
1. **Function Approximation**: Neural networks can approximate complex Q-functions
2. **High-Dimensional State Spaces**: Can handle raw pixel inputs
3. **Non-linear Relationships**: Deep networks can learn complex state-action relationships

## DQN Algorithm

### Core Algorithm Steps

1. **Initialize** replay memory `D` with capacity `N`
2. **Initialize** action-value function `Q` with random weights `θ`
3. **Initialize** target action-value function `Q̂` with weights `θ⁻ = θ`
4. **For** each episode:
   - Initialize sequence `s₁ = {x₁}` and preprocessed sequence `φ₁ = φ(s₁)`
   - **For** `t = 1` to `T`:
     - With probability `ε` select random action `aₜ`, otherwise `aₜ = argmax_a Q(φ(sₜ), a; θ)`
     - Execute action `aₜ` and observe reward `rₜ` and image `xₜ₊₁`
     - Set `sₜ₊₁ = sₜ, aₜ, xₜ₊₁` and preprocess `φₜ₊₁ = φ(sₜ₊₁)`
     - Store transition `(φₜ, aₜ, rₜ, φₜ₊₁)` in `D`
     - Sample random minibatch of transitions `(φⱼ, aⱼ, rⱼ, φⱼ₊₁)` from `D`
     - Set `yⱼ = rⱼ` if episode terminates at step `j+1`, otherwise `yⱼ = rⱼ + γ max_a' Q̂(φⱼ₊₁, a'; θ⁻)`
     - Perform gradient descent step on `(yⱼ - Q(φⱼ, aⱼ; θ))²` with respect to network parameters `θ`
     - Every `C` steps reset `Q̂ = Q`

## Key Innovations

### 1. Experience Replay
- **Problem**: Sequential data is highly correlated, leading to unstable learning
- **Solution**: Store experiences in a replay buffer and sample random batches
- **Benefits**: 
  - Breaks correlation between consecutive samples
  - Enables learning from past experiences multiple times
  - Improves sample efficiency

### 2. Target Network
- **Problem**: Moving target problem - network updates affect the target it's trying to learn
- **Solution**: Use a separate target network that's updated less frequently
- **Benefits**:
  - Stabilizes training
  - Reduces correlation between current and target Q-values

### 3. ε-Greedy Exploration
- **Problem**: Need to balance exploration vs exploitation
- **Solution**: Start with high exploration (ε=1) and decay over time
- **Benefits**:
  - Ensures sufficient exploration early in training
  - Gradually shifts to exploitation as learning progresses

## Project Structure

```
RL-DeepQNetworks-DQN/
├── README.md
├── requirements.txt
├── dqn/
│   ├── __init__.py
│   ├── agent.py          # DQN Agent implementation
│   ├── network.py        # Neural network architecture
│   ├── replay_buffer.py  # Experience replay buffer
│   └── utils.py          # Utility functions
├── environments/
│   ├── __init__.py
│   └── wrappers.py       # Environment wrappers
├── training/
│   ├── __init__.py
│   ├── trainer.py        # Training loop
│   └── config.py         # Hyperparameter configuration
├── visualization/
│   ├── __init__.py
│   ├── plotter.py        # Plotting utilities
│   └── monitor.py        # Training monitoring
├── examples/
│   ├── cartpole_demo.py  # CartPole environment demo
│   ├── atari_demo.py     # Atari environment demo
│   └── custom_env_demo.py # Custom environment demo
└── notebooks/
    ├── dqn_theory.ipynb  # Theoretical explanation notebook
    └── experiments.ipynb # Experimentation notebook
```

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd RL-DeepQNetworks-DQN
```

2. **Quick setup (recommended):**
```bash
python setup.py
```

3. **Manual installation:**
```bash
pip install -r requirements.txt
python setup.py --skip-install
```

4. **Verify installation:**
```bash
python examples/cartpole_demo.py --quick
```

## Usage

### Command Line Training

```bash
# Basic training on CartPole
python train_dqn.py --env CartPole-v1 --episodes 1000

# Training with custom hyperparameters
python train_dqn.py --env CartPole-v1 --learning-rate 0.0005 --gamma 0.95 --epsilon 0.9

# Training with different agent types
python train_dqn.py --env CartPole-v1 --agent-type double
python train_dqn.py --env CartPole-v1 --agent-type dueling
python train_dqn.py --env CartPole-v1 --agent-type rainbow

# Training on different environments
python train_dqn.py --env MountainCar-v0 --episodes 2000
python train_dqn.py --env LunarLander-v2 --episodes 2000
```

### Python API Usage

```python
from dqn.agent import DQNAgent
from environments.wrappers import make_env
from training.config import DQNConfig
from training.trainer import DQNTrainer

# Create configuration
config = DQNConfig(
    env_name='CartPole-v1',
    learning_rate=0.001,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01,
    max_episodes=1000
)

# Create trainer
trainer = DQNTrainer(config)

# Train the agent
results = trainer.train()

# Test the trained agent
test_scores = trainer.test(episodes=100)
```

### Examples and Demos

```bash
# CartPole environment demo
python examples/cartpole_demo.py

# Atari environment demo (requires gym[atari])
python examples/atari_demo.py

# Custom environment demo
python examples/custom_env_demo.py

# Quick demos (faster training)
python examples/cartpole_demo.py --quick
python examples/atari_demo.py --quick
python examples/custom_env_demo.py --quick
```

### Jupyter Notebooks

```bash
# Start Jupyter notebook server
jupyter notebook notebooks/

# Or use JupyterLab
jupyter lab notebooks/
```

The notebooks provide:
- **dqn_theory.ipynb**: Comprehensive theoretical understanding
- **experiments.ipynb**: Hands-on experiments and analysis

### Advanced Usage

```python
# Custom hyperparameters
config = {
    'learning_rate': 0.0005,
    'gamma': 0.95,
    'epsilon': 1.0,
    'epsilon_decay': 0.99,
    'epsilon_min': 0.05,
    'batch_size': 64,
    'memory_size': 10000,
    'target_update_frequency': 10,
    'hidden_layers': [128, 64]
}

agent = DQNAgent(**config)
```

## Implementation Details

### Neural Network Architecture

The DQN uses a feedforward neural network with the following architecture:
- **Input Layer**: State size (e.g., 4 for CartPole)
- **Hidden Layers**: Configurable (default: [128, 64])
- **Output Layer**: Action size (e.g., 2 for CartPole)
- **Activation**: ReLU for hidden layers, linear for output
- **Optimizer**: Adam optimizer

### Experience Replay Buffer

- **Capacity**: Configurable buffer size (default: 10,000)
- **Sampling**: Random uniform sampling for training batches
- **Storage**: Efficient deque-based implementation

### Training Process

1. **Episode Loop**: Run episodes until convergence
2. **Action Selection**: ε-greedy policy
3. **Experience Storage**: Store (state, action, reward, next_state, done)
4. **Batch Training**: Sample random batches from replay buffer
5. **Target Updates**: Periodic target network updates

## Results and Visualization

### Training Metrics

The implementation provides comprehensive monitoring of:
- **Episode Rewards**: Track learning progress
- **Loss Values**: Monitor training stability
- **Epsilon Decay**: Track exploration vs exploitation
- **Q-Values**: Monitor value function learning

### Visualization Tools

- **Training Curves**: Plot rewards and loss over time
- **Performance Metrics**: Success rate and convergence analysis
- **Real-time Monitoring**: Live training progress visualization

## Hyperparameter Tuning

### Key Hyperparameters

| Parameter | Description | Typical Range | Impact |
|-----------|-------------|---------------|---------|
| `learning_rate` | Neural network learning rate | 0.0001 - 0.01 | Higher = faster learning, less stable |
| `gamma` | Discount factor | 0.9 - 0.99 | Higher = more long-term thinking |
| `epsilon` | Initial exploration rate | 0.9 - 1.0 | Higher = more exploration |
| `epsilon_decay` | Exploration decay rate | 0.99 - 0.999 | Slower decay = more exploration |
| `batch_size` | Training batch size | 32 - 128 | Larger = more stable, slower |
| `memory_size` | Replay buffer size | 1000 - 100000 | Larger = more diverse experiences |

### Tuning Guidelines

1. **Start Conservative**: Begin with proven hyperparameters
2. **One at a Time**: Change one parameter at a time
3. **Monitor Stability**: Watch for training instability
4. **Environment Specific**: Adjust based on environment characteristics

## Common Issues and Solutions

### Training Instability
- **Problem**: Loss spikes or diverging rewards
- **Solutions**: 
  - Reduce learning rate
  - Increase target update frequency
  - Adjust epsilon decay rate

### Slow Learning
- **Problem**: Agent takes too long to learn
- **Solutions**:
  - Increase learning rate (carefully)
  - Improve reward shaping
  - Increase replay buffer size

### Poor Performance
- **Problem**: Agent doesn't reach optimal performance
- **Solutions**:
  - Extend training time
  - Adjust network architecture
  - Fine-tune hyperparameters

## References

1. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature, 518(7540), 529-533.

2. Mnih, V., et al. (2013). "Playing Atari with Deep Reinforcement Learning." arXiv preprint arXiv:1312.5602.

3. Sutton, R. S., & Barto, A. G. (2018). "Reinforcement learning: An introduction." MIT press.

4. Van Hasselt, H., Guez, A., & Silver, D. (2016). "Deep reinforcement learning with double Q-learning." AAAI.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- OpenAI Gym for providing excellent RL environments
- PyTorch team for the deep learning framework
- The reinforcement learning community for continuous research and development
