"""
Experience Replay Buffer for Deep Q-Networks

This module implements the experience replay buffer, which is a key component
of DQN that stores and samples past experiences to break correlation
between consecutive samples and improve training stability.
"""

import numpy as np
import torch
from collections import deque
from typing import Tuple, Optional, List
import random


class ExperienceReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling experiences.
    
    The replay buffer stores transitions (state, action, reward, next_state, done)
    and provides methods to sample random batches for training. This helps to:
    1. Break correlation between consecutive samples
    2. Enable learning from past experiences multiple times
    3. Improve sample efficiency and training stability
    
    Args:
        capacity (int): Maximum number of experiences to store
        seed (Optional[int]): Random seed for reproducibility
    """
    
    def __init__(self, capacity: int = 10000, seed: Optional[int] = None):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add a new experience to the buffer.
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Whether episode is done
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size (int): Number of experiences to sample
            
        Returns:
            Tuple[torch.Tensor, ...]: Batch of states, actions, rewards, next_states, dones
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack the batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """Check if the buffer is full."""
        return len(self.buffer) == self.capacity
    
    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self.buffer.clear()


class PrioritizedExperienceReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    
    This buffer samples experiences based on their TD-error priority,
    giving higher priority to experiences with larger prediction errors.
    This can lead to more efficient learning by focusing on experiences
    that are harder to predict.
    
    Args:
        capacity (int): Maximum number of experiences to store
        alpha (float): Prioritization exponent (0 = uniform, 1 = full prioritization)
        beta (float): Importance sampling exponent
        beta_increment (float): Beta increment per sampling
        epsilon (float): Small constant to avoid zero priority
        seed (Optional[int]): Random seed for reproducibility
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
        seed: Optional[int] = None
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # Use a list for experiences and priorities
        self.experiences = []
        self.priorities = []
        self.max_priority = 1.0
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: Optional[float] = None
    ) -> None:
        """
        Add a new experience to the buffer.
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Whether episode is done
            priority (Optional[float]): Priority of the experience
        """
        experience = (state, action, reward, next_state, done)
        
        if priority is None:
            priority = self.max_priority
        
        if len(self.experiences) >= self.capacity:
            # Remove oldest experience
            self.experiences.pop(0)
            self.priorities.pop(0)
        
        self.experiences.append(experience)
        self.priorities.append(priority)
        
        # Update max priority
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of experiences based on priority.
        
        Args:
            batch_size (int): Number of experiences to sample
            
        Returns:
            Tuple[torch.Tensor, ...]: Batch of experiences and importance weights
        """
        if len(self.experiences) < batch_size:
            batch_size = len(self.experiences)
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(
            len(self.experiences),
            size=batch_size,
            replace=False,
            p=probabilities
        )
        
        # Get experiences and calculate importance weights
        experiences = [self.experiences[i] for i in indices]
        weights = []
        
        for i in indices:
            # Importance sampling weight
            weight = (len(self.experiences) * probabilities[i]) ** (-self.beta)
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights /= weights.max()
        
        # Unpack experiences
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)
        weights = torch.FloatTensor(weights)
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """
        Update priorities for specific experiences.
        
        Args:
            indices (List[int]): Indices of experiences to update
            priorities (List[float]): New priorities
        """
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority + self.epsilon
                self.max_priority = max(self.max_priority, self.priorities[idx])
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.experiences)
    
    def is_full(self) -> bool:
        """Check if the buffer is full."""
        return len(self.experiences) == self.capacity
    
    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self.experiences.clear()
        self.priorities.clear()
        self.max_priority = 1.0


class NStepReplayBuffer:
    """
    N-Step Experience Replay Buffer.
    
    This buffer stores n-step returns instead of single-step returns,
    which can help with credit assignment and reduce variance in
    the value estimates.
    
    Args:
        capacity (int): Maximum number of experiences to store
        n_steps (int): Number of steps for n-step returns
        gamma (float): Discount factor
        seed (Optional[int]): Random seed for reproducibility
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        n_steps: int = 3,
        gamma: float = 0.99,
        seed: Optional[int] = None
    ):
        self.capacity = capacity
        self.n_steps = n_steps
        self.gamma = gamma
        self.buffer = deque(maxlen=capacity)
        
        # Temporary storage for n-step calculation
        self.temp_buffer = deque(maxlen=n_steps)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add a new experience and calculate n-step returns.
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Whether episode is done
        """
        # Add to temporary buffer
        self.temp_buffer.append((state, action, reward, next_state, done))
        
        # If we have enough steps, calculate n-step return
        if len(self.temp_buffer) == self.n_steps:
            self._calculate_n_step_return()
    
    def _calculate_n_step_return(self) -> None:
        """Calculate n-step return and add to main buffer."""
        if len(self.temp_buffer) < self.n_steps:
            return
        
        # Get the first experience
        first_state, first_action, first_reward, _, _ = self.temp_buffer[0]
        
        # Calculate n-step return
        n_step_reward = 0
        gamma_power = 1
        
        for i in range(self.n_steps):
            _, _, reward, _, done = self.temp_buffer[i]
            n_step_reward += gamma_power * reward
            gamma_power *= self.gamma
            
            if done:
                break
        
        # Get the last next_state
        _, _, _, last_next_state, last_done = self.temp_buffer[-1]
        
        # Add to main buffer
        experience = (first_state, first_action, n_step_reward, last_next_state, last_done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of n-step experiences.
        
        Args:
            batch_size (int): Number of experiences to sample
            
        Returns:
            Tuple[torch.Tensor, ...]: Batch of experiences
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack the batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """Check if the buffer is full."""
        return len(self.buffer) == self.capacity
    
    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self.buffer.clear()
        self.temp_buffer.clear()


def create_replay_buffer(
    buffer_type: str = "standard",
    capacity: int = 10000,
    **kwargs
):
    """
    Factory function to create different types of replay buffers.
    
    Args:
        buffer_type (str): Type of buffer ('standard', 'prioritized', 'nstep')
        capacity (int): Buffer capacity
        **kwargs: Additional arguments for buffer creation
        
    Returns:
        Replay buffer instance
    """
    if buffer_type.lower() == "standard":
        return ExperienceReplayBuffer(capacity, **kwargs)
    elif buffer_type.lower() == "prioritized":
        return PrioritizedExperienceReplayBuffer(capacity, **kwargs)
    elif buffer_type.lower() == "nstep":
        return NStepReplayBuffer(capacity, **kwargs)
    else:
        raise ValueError(f"Unknown buffer type: {buffer_type}")


if __name__ == "__main__":
    # Test the replay buffers
    print("Testing Replay Buffers...")
    
    # Test standard buffer
    buffer = ExperienceReplayBuffer(capacity=1000)
    
    # Add some experiences
    for i in range(100):
        state = np.random.randn(4)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = i % 10 == 0
        
        buffer.push(state, action, reward, next_state, done)
    
    # Sample a batch
    batch = buffer.sample(32)
    print(f"Standard buffer batch shapes: {[t.shape for t in batch]}")
    
    # Test prioritized buffer
    prioritized_buffer = PrioritizedExperienceReplayBuffer(capacity=1000)
    
    # Add experiences with priorities
    for i in range(100):
        state = np.random.randn(4)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = i % 10 == 0
        priority = np.random.rand()
        
        prioritized_buffer.push(state, action, reward, next_state, done, priority)
    
    # Sample a batch
    batch = prioritized_buffer.sample(32)
    print(f"Prioritized buffer batch shapes: {[t.shape for t in batch]}")
    
    print("All replay buffers tested successfully!")
