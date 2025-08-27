"""
Replay Buffer for Policy Gradient Methods
"""
import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict, Any
import torch


class ReplayBuffer:
    """
    Replay buffer for storing and sampling experiences
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool, **kwargs):
        """
        Add experience to buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            **kwargs: Additional information
        """
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            **kwargs
        }
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Sample batch of experiences
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of sampled experiences
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def sample_episode(self) -> List[Dict[str, Any]]:
        """
        Sample a complete episode
        
        Returns:
            List of experiences from one episode
        """
        if not self.buffer:
            return []
        
        # Find a random episode
        episode_start = np.random.randint(0, len(self.buffer))
        
        # Find the start of the episode
        while episode_start > 0 and not self.buffer[episode_start - 1]['done']:
            episode_start -= 1
        
        # Collect the episode
        episode = []
        for i in range(episode_start, len(self.buffer)):
            episode.append(self.buffer[i])
            if self.buffer[i]['done']:
                break
        
        return episode
    
    def sample_episodes(self, num_episodes: int) -> List[List[Dict[str, Any]]]:
        """
        Sample multiple complete episodes
        
        Args:
            num_episodes: Number of episodes to sample
            
        Returns:
            List of episodes
        """
        episodes = []
        for _ in range(num_episodes):
            episode = self.sample_episode()
            if episode:
                episodes.append(episode)
        
        return episodes
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
    
    def __len__(self) -> int:
        """Return current size of buffer"""
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return len(self.buffer) == self.capacity


class EpisodeBuffer:
    """
    Buffer for storing complete episodes
    """
    
    def __init__(self, capacity: int = 1000):
        """
        Initialize episode buffer
        
        Args:
            capacity: Maximum number of episodes to store
        """
        self.capacity = capacity
        self.episodes = deque(maxlen=capacity)
        self.current_episode = []
        
    def add_step(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool, **kwargs):
        """
        Add step to current episode
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            **kwargs: Additional information
        """
        step = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            **kwargs
        }
        
        self.current_episode.append(step)
        
        if done:
            self.episodes.append(self.current_episode.copy())
            self.current_episode.clear()
    
    def get_episode(self, episode_idx: int) -> List[Dict[str, Any]]:
        """
        Get specific episode
        
        Args:
            episode_idx: Index of episode to retrieve
            
        Returns:
            List of steps in the episode
        """
        if 0 <= episode_idx < len(self.episodes):
            return self.episodes[episode_idx]
        return []
    
    def get_latest_episode(self) -> List[Dict[str, Any]]:
        """
        Get the most recent episode
        
        Returns:
            List of steps in the latest episode
        """
        if self.episodes:
            return self.episodes[-1]
        return []
    
    def get_episodes(self, num_episodes: int) -> List[List[Dict[str, Any]]]:
        """
        Get multiple recent episodes
        
        Args:
            num_episodes: Number of episodes to retrieve
            
        Returns:
            List of episodes
        """
        if not self.episodes:
            return []
        
        start_idx = max(0, len(self.episodes) - num_episodes)
        return list(self.episodes)[start_idx:]
    
    def clear(self):
        """Clear all episodes"""
        self.episodes.clear()
        self.current_episode.clear()
    
    def __len__(self) -> int:
        """Return number of complete episodes"""
        return len(self.episodes)
    
    def get_total_steps(self) -> int:
        """Return total number of steps across all episodes"""
        return sum(len(episode) for episode in self.episodes)


class PrioritizedReplayBuffer:
    """
    Prioritized replay buffer that samples experiences based on their importance
    """
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        """
        Initialize prioritized replay buffer
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: Prioritization exponent
            beta: Importance sampling exponent
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = []
        self.position = 0
        
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool, priority: float = None, **kwargs):
        """
        Add experience to buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            priority: Priority of the experience
            **kwargs: Additional information
        """
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
        
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            **kwargs
        }
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Dict[str, Any]], List[int], List[float]]:
        """
        Sample batch of experiences with priorities
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (experiences, indices, weights)
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer), list(range(len(self.buffer))), [1.0] * len(self.buffer)
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities[:len(self.buffer)])
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        return experiences, indices.tolist(), weights.tolist()
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """
        Update priorities for specific experiences
        
        Args:
            indices: Indices of experiences to update
            priorities: New priorities
        """
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        self.priorities.clear()
        self.position = 0
    
    def __len__(self) -> int:
        """Return current size of buffer"""
        return len(self.buffer)
