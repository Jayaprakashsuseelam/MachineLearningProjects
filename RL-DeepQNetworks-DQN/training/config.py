"""
Training Configuration and Hyperparameters for DQN

This module provides configuration classes and hyperparameter management
for DQN training.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json
import os


@dataclass
class DQNConfig:
    """
    Configuration class for DQN training hyperparameters.
    
    This class contains all the hyperparameters needed for DQN training,
    with sensible defaults and validation.
    """
    
    # Environment settings
    env_name: str = "CartPole-v1"
    render_mode: Optional[str] = None
    
    # Agent hyperparameters
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    
    # Network architecture
    network_type: str = "dqn"  # dqn, dueling, cnn
    hidden_layers: List[int] = field(default_factory=lambda: [128, 64])
    dropout_rate: float = 0.1
    use_batch_norm: bool = False
    
    # Training hyperparameters
    batch_size: int = 64
    memory_size: int = 10000
    target_update_frequency: int = 10
    
    # Training settings
    max_episodes: int = 1000
    max_steps_per_episode: int = 1000
    learning_starts: int = 100  # Start learning after this many steps
    
    # Evaluation settings
    eval_frequency: int = 100  # Evaluate every N episodes
    eval_episodes: int = 10
    eval_epsilon: float = 0.0  # No exploration during evaluation
    
    # Device settings
    device: str = "cpu"  # cpu or cuda
    
    # Logging and saving
    save_frequency: int = 200  # Save model every N episodes
    log_frequency: int = 10  # Log metrics every N episodes
    save_path: str = "models"
    log_path: str = "logs"
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 100  # Episodes to wait for improvement
    min_improvement: float = 0.01
    
    # Random seed
    seed: Optional[int] = None
    
    # Advanced settings
    gradient_clip: float = 1.0
    double_dqn: bool = False
    dueling_dqn: bool = False
    prioritized_replay: bool = False
    n_step: bool = False
    n_steps: int = 3
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert 0 < self.gamma <= 1, "Gamma must be in (0, 1]"
        assert 0 <= self.epsilon <= 1, "Epsilon must be in [0, 1]"
        assert 0 < self.epsilon_decay < 1, "Epsilon decay must be in (0, 1)"
        assert 0 <= self.epsilon_min <= self.epsilon, "Epsilon min must be <= epsilon"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.memory_size > 0, "Memory size must be positive"
        assert self.target_update_frequency > 0, "Target update frequency must be positive"
        assert self.max_episodes > 0, "Max episodes must be positive"
        assert self.max_steps_per_episode > 0, "Max steps per episode must be positive"
        assert self.eval_frequency > 0, "Eval frequency must be positive"
        assert self.eval_episodes > 0, "Eval episodes must be positive"
        assert self.gradient_clip > 0, "Gradient clip must be positive"
        
        # Validate network type
        valid_network_types = ["dqn", "dueling", "cnn"]
        assert self.network_type in valid_network_types, f"Network type must be one of {valid_network_types}"
        
        # Validate device
        valid_devices = ["cpu", "cuda"]
        assert self.device in valid_devices, f"Device must be one of {valid_devices}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'env_name': self.env_name,
            'render_mode': self.render_mode,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'network_type': self.network_type,
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'batch_size': self.batch_size,
            'memory_size': self.memory_size,
            'target_update_frequency': self.target_update_frequency,
            'max_episodes': self.max_episodes,
            'max_steps_per_episode': self.max_steps_per_episode,
            'learning_starts': self.learning_starts,
            'eval_frequency': self.eval_frequency,
            'eval_episodes': self.eval_episodes,
            'eval_epsilon': self.eval_epsilon,
            'device': self.device,
            'save_frequency': self.save_frequency,
            'log_frequency': self.log_frequency,
            'save_path': self.save_path,
            'log_path': self.log_path,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'min_improvement': self.min_improvement,
            'seed': self.seed,
            'gradient_clip': self.gradient_clip,
            'double_dqn': self.double_dqn,
            'dueling_dqn': self.dueling_dqn,
            'prioritized_replay': self.prioritized_replay,
            'n_step': self.n_step,
            'n_steps': self.n_steps
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DQNConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'DQNConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Predefined configurations for common environments
CART_POLE_CONFIG = DQNConfig(
    env_name="CartPole-v1",
    learning_rate=0.001,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01,
    hidden_layers=[128, 64],
    batch_size=64,
    memory_size=10000,
    target_update_frequency=10,
    max_episodes=1000,
    max_steps_per_episode=500,
    eval_frequency=50,
    eval_episodes=10
)

MOUNTAIN_CAR_CONFIG = DQNConfig(
    env_name="MountainCar-v0",
    learning_rate=0.001,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.999,
    epsilon_min=0.01,
    hidden_layers=[128, 64],
    batch_size=64,
    memory_size=10000,
    target_update_frequency=10,
    max_episodes=2000,
    max_steps_per_episode=200,
    eval_frequency=100,
    eval_episodes=10
)

ATARI_CONFIG = DQNConfig(
    env_name="Breakout-v4",
    learning_rate=0.00025,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.999,
    epsilon_min=0.1,
    network_type="cnn",
    batch_size=32,
    memory_size=100000,
    target_update_frequency=10000,
    max_episodes=10000,
    max_steps_per_episode=10000,
    eval_frequency=500,
    eval_episodes=5,
    learning_starts=10000
)

LUNAR_LANDER_CONFIG = DQNConfig(
    env_name="LunarLander-v2",
    learning_rate=0.001,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01,
    hidden_layers=[128, 64],
    batch_size=64,
    memory_size=10000,
    target_update_frequency=10,
    max_episodes=2000,
    max_steps_per_episode=1000,
    eval_frequency=100,
    eval_episodes=10
)

# Configuration registry
CONFIG_REGISTRY = {
    "cartpole": CART_POLE_CONFIG,
    "mountain_car": MOUNTAIN_CAR_CONFIG,
    "atari": ATARI_CONFIG,
    "lunar_lander": LUNAR_LANDER_CONFIG
}


def get_config(config_name: str) -> DQNConfig:
    """
    Get predefined configuration by name.
    
    Args:
        config_name: Name of the configuration
        
    Returns:
        DQNConfig: Configuration object
    """
    if config_name not in CONFIG_REGISTRY:
        raise ValueError(f"Unknown config name: {config_name}. Available: {list(CONFIG_REGISTRY.keys())}")
    
    return CONFIG_REGISTRY[config_name]


def create_config(
    env_name: str,
    **overrides
) -> DQNConfig:
    """
    Create configuration for a specific environment with overrides.
    
    Args:
        env_name: Name of the environment
        **overrides: Configuration overrides
        
    Returns:
        DQNConfig: Configuration object
    """
    # Start with default configuration
    config = DQNConfig(env_name=env_name)
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
    
    return config


if __name__ == "__main__":
    # Test configuration
    print("Testing DQN Configuration...")
    
    # Test default configuration
    config = DQNConfig()
    print(f"Default config: {config.env_name}")
    
    # Test predefined configurations
    cartpole_config = get_config("cartpole")
    print(f"CartPole config: {cartpole_config.env_name}")
    
    # Test configuration with overrides
    custom_config = create_config("CartPole-v1", learning_rate=0.0005, epsilon=0.9)
    print(f"Custom config learning rate: {custom_config.learning_rate}")
    
    # Test save/load
    config.save("test_config.json")
    loaded_config = DQNConfig.load("test_config.json")
    print(f"Loaded config: {loaded_config.env_name}")
    
    # Clean up
    import os
    if os.path.exists("test_config.json"):
        os.remove("test_config.json")
    
    print("Configuration tested successfully!")
