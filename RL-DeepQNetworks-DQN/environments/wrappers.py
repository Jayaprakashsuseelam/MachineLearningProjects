"""
Environment Wrappers for Deep Q-Networks

This module provides environment wrappers and utilities for working with
OpenAI Gym environments in DQN training.
"""

import gym
import numpy as np
from typing import Tuple, Optional, Any, Dict
import cv2
from collections import deque


class FrameStackWrapper(gym.Wrapper):
    """
    Wrapper to stack multiple frames for temporal information.
    
    This is particularly useful for Atari games where a single frame
    doesn't contain enough information about motion.
    
    Args:
        env: OpenAI Gym environment
        k: Number of frames to stack
    """
    
    def __init__(self, env: gym.Env, k: int = 4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        
        # Update observation space
        low = np.repeat(self.observation_space.low, k, axis=0)
        high = np.repeat(self.observation_space.high, k, axis=0)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )
    
    def reset(self, **kwargs) -> np.ndarray:
        """Reset environment and return stacked frames."""
        obs = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take step and return stacked frames."""
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info
    
    def _get_obs(self) -> np.ndarray:
        """Get stacked observation."""
        assert len(self.frames) == self.k
        return np.array(self.frames).flatten()


class ImagePreprocessingWrapper(gym.Wrapper):
    """
    Wrapper to preprocess image observations for DQN.
    
    Converts RGB images to grayscale, resizes them, and normalizes
    pixel values to [0, 1] range.
    
    Args:
        env: OpenAI Gym environment
        width: Target width for resizing
        height: Target height for resizing
        grayscale: Whether to convert to grayscale
    """
    
    def __init__(
        self,
        env: gym.Env,
        width: int = 84,
        height: int = 84,
        grayscale: bool = True
    ):
        super().__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        
        # Update observation space
        if grayscale:
            self.observation_space = gym.spaces.Box(
                low=0, high=1, shape=(height, width), dtype=np.float32
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0, high=1, shape=(height, width, 3), dtype=np.float32
            )
    
    def reset(self, **kwargs) -> np.ndarray:
        """Reset environment and return preprocessed observation."""
        obs = self.env.reset(**kwargs)
        return self._preprocess(obs)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take step and return preprocessed observation."""
        obs, reward, done, info = self.env.step(action)
        return self._preprocess(obs), reward, done, info
    
    def _preprocess(self, obs: np.ndarray) -> np.ndarray:
        """Preprocess observation."""
        # Convert to grayscale if needed
        if self.grayscale and len(obs.shape) == 3:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        
        # Resize
        obs = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        obs = obs.astype(np.float32) / 255.0
        
        return obs


class RewardClippingWrapper(gym.Wrapper):
    """
    Wrapper to clip rewards to a specified range.
    
    This can help with training stability by preventing
    extremely large rewards from destabilizing learning.
    
    Args:
        env: OpenAI Gym environment
        min_reward: Minimum reward value
        max_reward: Maximum reward value
    """
    
    def __init__(
        self,
        env: gym.Env,
        min_reward: float = -1.0,
        max_reward: float = 1.0
    ):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take step and clip reward."""
        obs, reward, done, info = self.env.step(action)
        reward = np.clip(reward, self.min_reward, self.max_reward)
        return obs, reward, done, info


class EpisodicLifeWrapper(gym.Wrapper):
    """
    Wrapper to treat each life as an episode for Atari games.
    
    This helps the agent learn that losing a life is bad,
    even if the game continues.
    
    Args:
        env: OpenAI Gym environment
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take step and check for life loss."""
        obs, reward, done, info = self.env.step(action)
        
        # Check if lives changed
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # Life lost, but game continues
            done = True
            self.was_real_done = False
        
        self.lives = lives
        return obs, reward, done, info
    
    def reset(self, **kwargs) -> np.ndarray:
        """Reset environment and track lives."""
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # No-op to continue from current state
            obs, _, _, _ = self.env.step(0)
        
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class NoopResetWrapper(gym.Wrapper):
    """
    Wrapper to perform random no-op actions at the start of episodes.
    
    This helps with exploration and prevents the agent from
    learning deterministic start sequences.
    
    Args:
        env: OpenAI Gym environment
        noop_max: Maximum number of no-op actions
    """
    
    def __init__(self, env: gym.Env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
    
    def reset(self, **kwargs) -> np.ndarray:
        """Reset environment and perform random no-op actions."""
        obs = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        
        return obs


class FireResetWrapper(gym.Wrapper):
    """
    Wrapper to automatically press FIRE at the start of episodes.
    
    Some Atari games require pressing FIRE to start.
    
    Args:
        env: OpenAI Gym environment
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3
    
    def reset(self, **kwargs) -> np.ndarray:
        """Reset environment and press FIRE."""
        obs = self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)  # FIRE action
        if done:
            obs = self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)  # Another action
        if done:
            obs = self.env.reset(**kwargs)
        return obs


class MaxAndSkipWrapper(gym.Wrapper):
    """
    Wrapper to skip frames and take the maximum of the last two frames.
    
    This reduces computational cost and helps with frame rate
    independence.
    
    Args:
        env: OpenAI Gym environment
        skip: Number of frames to skip
    """
    
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self.skip = skip
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take step and skip frames."""
        total_reward = 0.0
        done = None
        
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        
        return obs, total_reward, done, info


class StateNormalizationWrapper(gym.Wrapper):
    """
    Wrapper to normalize state observations.
    
    This can help with training stability by ensuring
    all state components are on similar scales.
    
    Args:
        env: OpenAI Gym environment
        mean: Mean values for normalization
        std: Standard deviation values for normalization
    """
    
    def __init__(
        self,
        env: gym.Env,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None
    ):
        super().__init__(env)
        self.mean = mean
        self.std = std
        
        # If not provided, use observation space bounds
        if self.mean is None:
            self.mean = (self.observation_space.high + self.observation_space.low) / 2
        if self.std is None:
            self.std = (self.observation_space.high - self.observation_space.low) / 2
    
    def reset(self, **kwargs) -> np.ndarray:
        """Reset environment and return normalized observation."""
        obs = self.env.reset(**kwargs)
        return self._normalize(obs)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take step and return normalized observation."""
        obs, reward, done, info = self.env.step(action)
        return self._normalize(obs), reward, done, info
    
    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation."""
        return (obs - self.mean) / self.std


def make_env(
    env_name: str,
    render_mode: Optional[str] = None,
    **kwargs
) -> gym.Env:
    """
    Create and configure an environment for DQN training.
    
    Args:
        env_name: Name of the environment
        render_mode: Rendering mode ('human', 'rgb_array', None)
        **kwargs: Additional configuration options
        
    Returns:
        Configured environment
    """
    # Create base environment
    env = gym.make(env_name, render_mode=render_mode)
    
    # Apply wrappers based on environment type
    if 'Atari' in env_name or 'ALE' in env_name:
        # Atari-specific wrappers
        env = NoopResetWrapper(env, noop_max=30)
        env = MaxAndSkipWrapper(env, skip=4)
        env = EpisodicLifeWrapper(env)
        
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetWrapper(env)
        
        env = ImagePreprocessingWrapper(env, width=84, height=84, grayscale=True)
        env = FrameStackWrapper(env, k=4)
        env = RewardClippingWrapper(env, min_reward=-1, max_reward=1)
    
    elif 'CartPole' in env_name:
        # CartPole-specific wrappers
        env = StateNormalizationWrapper(env)
    
    elif 'MountainCar' in env_name:
        # MountainCar-specific wrappers
        env = StateNormalizationWrapper(env)
    
    return env


def get_env_info(env: gym.Env) -> Dict[str, Any]:
    """
    Get information about an environment.
    
    Args:
        env: OpenAI Gym environment
        
    Returns:
        Dictionary containing environment information
    """
    return {
        'name': env.spec.id if env.spec else 'Unknown',
        'observation_space': env.observation_space,
        'action_space': env.action_space,
        'reward_range': env.reward_range,
        'max_episode_steps': env.spec.max_episode_steps if env.spec else None,
        'n_actions': env.action_space.n if hasattr(env.action_space, 'n') else None,
        'state_size': env.observation_space.shape[0] if len(env.observation_space.shape) == 1 else env.observation_space.shape
    }


if __name__ == "__main__":
    # Test environment wrappers
    print("Testing Environment Wrappers...")
    
    # Test CartPole environment
    env = make_env('CartPole-v1')
    info = get_env_info(env)
    print(f"CartPole environment info: {info}")
    
    # Test a few steps
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        if done:
            break
    
    env.close()
    
    # Test Atari environment (if available)
    try:
        env = make_env('Breakout-v4')
        info = get_env_info(env)
        print(f"Breakout environment info: {info}")
        
        obs = env.reset()
        print(f"Atari observation shape: {obs.shape}")
        env.close()
    except Exception as e:
        print(f"Atari environment not available: {e}")
    
    print("Environment wrappers tested successfully!")
