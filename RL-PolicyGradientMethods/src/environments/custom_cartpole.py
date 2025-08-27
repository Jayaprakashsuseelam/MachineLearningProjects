"""
Custom CartPole Environment with Enhanced Features
"""
import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt


class CustomCartPoleEnv(Env):
    """
    Custom CartPole environment with additional features for policy gradient methods
    
    Features:
    - Continuous reward shaping
    - Additional state information
    - Configurable difficulty
    - Performance tracking
    """
    
    def __init__(self, max_steps: int = 500, reward_shaping: bool = True, 
                 difficulty: float = 1.0, render_mode: Optional[str] = None):
        """
        Initialize custom CartPole environment
        
        Args:
            max_steps: Maximum steps per episode
            reward_shaping: Whether to use continuous reward shaping
            difficulty: Difficulty multiplier (affects gravity)
            render_mode: Rendering mode
        """
        super().__init__()
        
        self.max_steps = max_steps
        self.reward_shaping = reward_shaping
        self.difficulty = difficulty
        self.render_mode = render_mode
        
        # CartPole parameters
        self.gravity = 9.8 * difficulty
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.total_mass = self.cart_mass + self.pole_mass
        self.length = 0.5  # Actually half the pole's length
        self.pole_mass_length = self.pole_mass * self.length
        self.force_magnitude = 10.0
        self.tau = 0.02  # Seconds between state updates
        
        # Thresholds
        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.state = None
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset episode tracking
        self.current_step = 0
        
        # Initialize state with small random values
        if seed is not None:
            np.random.seed(seed)
        
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            'episode_length': 0,
            'total_reward': 0.0
        }
        
        return observation, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation with additional features"""
        x, x_dot, theta, theta_dot = self.state
        
        # Basic state
        observation = np.array([
            x,
            x_dot,
            theta,
            theta_dot
        ], dtype=np.float32)
        
        # Additional features for better learning
        additional_features = np.array([
            np.sin(theta),  # Sine of angle (helps with angle wrapping)
            np.cos(theta)   # Cosine of angle
        ], dtype=np.float32)
        
        return np.concatenate([observation, additional_features])
    
    def _is_done(self) -> bool:
        """Check if episode is done"""
        x, x_dot, theta, theta_dot = self.state
        
        # Check if pole fell or cart went out of bounds
        done = (
            x < -self.x_threshold or
            x > self.x_threshold or
            theta < -self.theta_threshold_radians or
            theta > self.theta_threshold_radians
        )
        
        return done
    
    def _calculate_reward(self, action: int) -> float:
        """Calculate reward for current state and action"""
        x, x_dot, theta, theta_dot = self.state
        
        if not self.reward_shaping:
            # Simple binary reward
            return 1.0 if not self._is_done() else 0.0
        
        # Continuous reward shaping
        reward = 1.0  # Base reward for staying alive
        
        # Penalty for being far from center
        x_penalty = -abs(x) / self.x_threshold
        
        # Penalty for large angle
        theta_penalty = -abs(theta) / self.theta_threshold_radians
        
        # Penalty for high velocity
        velocity_penalty = -abs(x_dot) / 10.0 - abs(theta_dot) / 10.0
        
        # Bonus for staying balanced
        balance_bonus = 0.1 if abs(theta) < 0.1 else 0.0
        
        # Combine rewards
        total_reward = reward + x_penalty + theta_penalty + velocity_penalty + balance_bonus
        
        return total_reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        x, x_dot, theta, theta_dot = self.state
        
        # Apply force
        force = self.force_magnitude if action == 1 else -self.force_magnitude
        
        # Calculate derivatives
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        temp = (force + self.pole_mass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0/3.0 - self.pole_mass * costheta**2 / self.total_mass)
        )
        xacc = temp - self.pole_mass_length * thetaacc * costheta / self.total_mass
        
        # Update state
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check if done
        terminated = self._is_done()
        truncated = self.current_step >= self.max_steps - 1
        
        # Update step counter
        self.current_step += 1
        
        # Get next observation
        observation = self._get_observation()
        
        # Info
        info = {
            'episode_length': self.current_step,
            'total_reward': reward,
            'x': x,
            'theta': theta,
            'x_dot': x_dot,
            'theta_dot': theta_dot
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode: str = 'human'):
        """Render the environment"""
        if mode == 'human':
            x, x_dot, theta, theta_dot = self.state
            print(f"Step: {self.current_step}, x: {x:.3f}, theta: {theta:.3f}, "
                  f"x_dot: {x_dot:.3f}, theta_dot: {theta_dot:.3f}")
    
    def get_state(self) -> np.ndarray:
        """Get current state"""
        return self.state.copy()
    
    def set_state(self, state: np.ndarray):
        """Set current state (for testing)"""
        self.state = state.copy()
    
    def plot_performance(self):
        """Plot performance metrics"""
        if not self.episode_rewards:
            print("No episode data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot episode rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True)
        
        # Plot episode lengths
        ax2.plot(self.episode_lengths)
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        if not self.episode_rewards:
            return {}
        
        return {
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'mean_length': np.mean(self.episode_lengths),
            'std_length': np.std(self.episode_lengths),
            'max_reward': np.max(self.episode_rewards),
            'max_length': np.max(self.episode_lengths)
        }
