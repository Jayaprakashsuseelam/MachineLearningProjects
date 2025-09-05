"""
Deep Q-Network (DQN) Agent Implementation

This module implements the DQN agent with all the key components:
- Neural network for Q-value approximation
- Experience replay buffer for training stability
- Target network for stable learning
- Epsilon-greedy exploration strategy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional, Dict, Any
import random
from collections import deque

from .network import DQNNetwork, create_network
from .replay_buffer import ExperienceReplayBuffer, create_replay_buffer


class DQNAgent:
    """
    Deep Q-Network Agent implementation.
    
    This agent implements the DQN algorithm with experience replay,
    target network, and epsilon-greedy exploration.
    
    Args:
        state_size (int): Size of the state space
        action_size (int): Size of the action space
        learning_rate (float): Learning rate for the optimizer
        gamma (float): Discount factor for future rewards
        epsilon (float): Initial exploration rate
        epsilon_decay (float): Rate of epsilon decay
        epsilon_min (float): Minimum exploration rate
        batch_size (int): Batch size for training
        memory_size (int): Size of the replay buffer
        target_update_frequency (int): Frequency of target network updates
        network_type (str): Type of network ('dqn', 'dueling', 'cnn')
        hidden_layers (List[int]): Hidden layer sizes for the network
        device (str): Device to run the network on ('cpu' or 'cuda')
        seed (Optional[int]): Random seed for reproducibility
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        batch_size: int = 64,
        memory_size: int = 10000,
        target_update_frequency: int = 10,
        network_type: str = "dqn",
        hidden_layers: list = [128, 64],
        device: str = "cpu",
        seed: Optional[int] = None,
        **kwargs
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.device = torch.device(device)
        
        # Set random seeds
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        # Create networks
        network_kwargs = {
            'state_size': state_size,
            'action_size': action_size,
            'hidden_layers': hidden_layers,
            **kwargs
        }
        
        self.q_network = create_network(network_type, **network_kwargs).to(self.device)
        self.target_network = create_network(network_type, **network_kwargs).to(self.device)
        
        # Initialize target network with same weights as main network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Create optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Create replay buffer
        self.memory = create_replay_buffer("standard", memory_size, seed=seed)
        
        # Training statistics
        self.t_step = 0
        self.losses = deque(maxlen=100)
        self.q_values_history = deque(maxlen=100)
        
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state (np.ndarray): Current state
            training (bool): Whether in training mode (affects exploration)
            
        Returns:
            int: Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.choice(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Take a step in the environment and store the experience.
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Whether episode is done
        """
        # Store experience in replay buffer
        self.memory.push(state, action, reward, next_state, done)
        
        # Learn every few steps
        self.t_step += 1
        if self.t_step % 4 == 0:  # Learn every 4 steps
            self.learn()
        
        # Update target network
        if self.t_step % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def learn(self) -> Optional[float]:
        """
        Learn from a batch of experiences.
        
        Returns:
            Optional[float]: Loss value if learning occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Get current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Store statistics
        self.losses.append(loss.item())
        self.q_values_history.append(current_q_values.mean().item())
        
        return loss.item()
    
    def save(self, filepath: str) -> None:
        """
        Save the agent's state.
        
        Args:
            filepath (str): Path to save the model
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            't_step': self.t_step,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'batch_size': self.batch_size,
            'target_update_frequency': self.target_update_frequency
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """
        Load the agent's state.
        
        Args:
            filepath (str): Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.epsilon = checkpoint['epsilon']
        self.t_step = checkpoint['t_step']
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get training statistics.
        
        Returns:
            Dict[str, Any]: Dictionary containing training statistics
        """
        return {
            'epsilon': self.epsilon,
            't_step': self.t_step,
            'memory_size': len(self.memory),
            'avg_loss': np.mean(self.losses) if self.losses else 0.0,
            'avg_q_value': np.mean(self.q_values_history) if self.q_values_history else 0.0,
            'loss_std': np.std(self.losses) if self.losses else 0.0,
            'q_value_std': np.std(self.q_values_history) if self.q_values_history else 0.0
        }
    
    def reset_stats(self) -> None:
        """Reset training statistics."""
        self.losses.clear()
        self.q_values_history.clear()


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN Agent implementation.
    
    Double DQN addresses the overestimation bias in DQN by using the main
    network to select actions and the target network to evaluate them.
    
    Args:
        Same as DQNAgent
    """
    
    def learn(self) -> Optional[float]:
        """
        Learn from a batch of experiences using Double DQN.
        
        Returns:
            Optional[float]: Loss value if learning occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Get current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: Use main network to select actions, target network to evaluate
        with torch.no_grad():
            # Select actions using main network
            next_actions = self.q_network(next_states).argmax(1)
            # Evaluate using target network
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Store statistics
        self.losses.append(loss.item())
        self.q_values_history.append(current_q_values.mean().item())
        
        return loss.item()


class DuelingDQNAgent(DQNAgent):
    """
    Dueling DQN Agent implementation.
    
    Uses the Dueling DQN architecture which separates the Q-value into
    value and advantage components.
    """
    
    def __init__(self, *args, **kwargs):
        # Force network type to dueling
        kwargs['network_type'] = 'dueling'
        super().__init__(*args, **kwargs)


class RainbowDQNAgent(DQNAgent):
    """
    Rainbow DQN Agent implementation.
    
    Combines multiple improvements: Double DQN, Dueling DQN, Prioritized
    Experience Replay, and N-step returns.
    """
    
    def __init__(
        self,
        *args,
        use_prioritized_replay: bool = True,
        use_n_step: bool = True,
        n_steps: int = 3,
        **kwargs
    ):
        # Force network type to dueling
        kwargs['network_type'] = 'dueling'
        super().__init__(*args, **kwargs)
        
        self.use_prioritized_replay = use_prioritized_replay
        self.use_n_step = use_n_step
        self.n_steps = n_steps
        
        # Replace replay buffer with prioritized version if requested
        if use_prioritized_replay:
            memory_size = kwargs.get('memory_size', 10000)
            self.memory = create_replay_buffer("prioritized", memory_size)
    
    def learn(self) -> Optional[float]:
        """
        Learn from a batch of experiences using Rainbow DQN.
        
        Returns:
            Optional[float]: Loss value if learning occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        if self.use_prioritized_replay:
            states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(self.batch_size)
            weights = weights.to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
            indices = None
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Get current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: Use main network to select actions, target network to evaluate
        with torch.no_grad():
            # Select actions using main network
            next_actions = self.q_network(next_states).argmax(1)
            # Evaluate using target network
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        # Compute TD errors for prioritized replay
        td_errors = torch.abs(current_q_values - target_q_values).squeeze()
        
        # Compute loss with importance sampling weights
        loss = (weights * nn.MSELoss(reduction='none')(current_q_values.squeeze(), target_q_values.squeeze())).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update priorities if using prioritized replay
        if self.use_prioritized_replay and indices is not None:
            self.memory.update_priorities(indices, td_errors.cpu().numpy())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Store statistics
        self.losses.append(loss.item())
        self.q_values_history.append(current_q_values.mean().item())
        
        return loss.item()


def create_agent(agent_type: str, *args, **kwargs) -> DQNAgent:
    """
    Factory function to create different types of DQN agents.
    
    Args:
        agent_type (str): Type of agent ('dqn', 'double', 'dueling', 'rainbow')
        *args: Arguments for agent creation
        **kwargs: Keyword arguments for agent creation
        
    Returns:
        DQNAgent: The created agent
    """
    if agent_type.lower() == 'dqn':
        return DQNAgent(*args, **kwargs)
    elif agent_type.lower() == 'double':
        return DoubleDQNAgent(*args, **kwargs)
    elif agent_type.lower() == 'dueling':
        return DuelingDQNAgent(*args, **kwargs)
    elif agent_type.lower() == 'rainbow':
        return RainbowDQNAgent(*args, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


if __name__ == "__main__":
    # Test the agents
    print("Testing DQN Agents...")
    
    # Test basic DQN agent
    agent = DQNAgent(
        state_size=4,
        action_size=2,
        learning_rate=0.001,
        epsilon=0.1
    )
    
    # Test action selection
    state = np.random.randn(4)
    action = agent.act(state)
    print(f"Selected action: {action}")
    
    # Test learning step
    next_state = np.random.randn(4)
    reward = np.random.randn()
    done = False
    
    agent.step(state, action, reward, next_state, done)
    
    # Test statistics
    stats = agent.get_stats()
    print(f"Agent stats: {stats}")
    
    print("DQN Agent tested successfully!")
