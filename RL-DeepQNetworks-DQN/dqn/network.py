"""
Deep Q-Network (DQN) Neural Network Architecture

This module implements the neural network architecture for Deep Q-Networks,
including both the main Q-network and target network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


class DQNNetwork(nn.Module):
    """
    Deep Q-Network neural network architecture.
    
    This network takes state observations as input and outputs Q-values
    for each possible action.
    
    Args:
        state_size (int): Size of the state space
        action_size (int): Size of the action space
        hidden_layers (List[int]): List of hidden layer sizes
        dropout_rate (float): Dropout rate for regularization
        use_batch_norm (bool): Whether to use batch normalization
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_layers: List[int] = [128, 64],
        dropout_rate: float = 0.1,
        use_batch_norm: bool = False
    ):
        super(DQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Build the network layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList()
        
        # Input layer
        prev_size = state_size
        
        # Hidden layers
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, action_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state (torch.Tensor): Input state tensor
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        x = state
        
        # Pass through hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Apply batch normalization if enabled
            if self.use_batch_norm and self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            # Apply activation function
            x = F.relu(x)
            
            # Apply dropout
            x = self.dropouts[i](x)
        
        # Output layer (no activation for Q-values)
        q_values = self.output_layer(x)
        
        return q_values
    
    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state (torch.Tensor): Input state tensor
            epsilon (float): Exploration rate
            
        Returns:
            int: Selected action
        """
        if np.random.random() < epsilon:
            return np.random.choice(self.action_size)
        
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax().item()
    
    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values for a given state.
        
        Args:
            state (torch.Tensor): Input state tensor
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        return self.forward(state)


class DuelingDQNNetwork(nn.Module):
    """
    Dueling Deep Q-Network architecture.
    
    This network separates the Q-value into value and advantage components,
    which can lead to better learning in environments where the value
    of a state is independent of the action taken.
    
    Args:
        state_size (int): Size of the state space
        action_size (int): Size of the action space
        hidden_layers (List[int]): List of hidden layer sizes
        dropout_rate (float): Dropout rate for regularization
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_layers: List[int] = [128, 64],
        dropout_rate: float = 0.1
    ):
        super(DuelingDQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layers = hidden_layers
        
        # Shared feature layers
        self.feature_layers = nn.ModuleList()
        prev_size = state_size
        
        for hidden_size in hidden_layers[:-1]:  # All but last layer
            self.feature_layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_size, hidden_layers[-1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layers[-1], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_size, hidden_layers[-1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layers[-1], action_size)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for layer in self.feature_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        
        for stream in [self.value_stream, self.advantage_stream]:
            for layer in stream:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dueling network.
        
        Args:
            state (torch.Tensor): Input state tensor
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        x = state
        
        # Pass through shared feature layers
        for layer in self.feature_layers:
            x = F.relu(layer(x))
        
        # Calculate value and advantage
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage to get Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class CNN_DQNNetwork(nn.Module):
    """
    Convolutional Neural Network for DQN (useful for image-based environments).
    
    This network uses convolutional layers to process image observations
    and outputs Q-values for each action.
    
    Args:
        input_channels (int): Number of input channels
        action_size (int): Size of the action space
        hidden_size (int): Size of the final hidden layer
    """
    
    def __init__(
        self,
        input_channels: int = 4,  # Stack of 4 frames
        action_size: int = 4,
        hidden_size: int = 512
    ):
        super(CNN_DQNNetwork, self).__init__()
        
        self.input_channels = input_channels
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size after convolutions
        # Assuming input size of 84x84
        conv_output_size = self._get_conv_output_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_conv_output_size(self) -> int:
        """Calculate the output size after convolutional layers."""
        # Simulate forward pass with dummy input
        dummy_input = torch.zeros(1, self.input_channels, 84, 84)
        x = F.relu(self.conv1(dummy_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return int(np.prod(x.size()[1:]))
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for layer in [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2]:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN network.
        
        Args:
            state (torch.Tensor): Input state tensor (image)
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        
        return q_values


def create_network(
    network_type: str,
    state_size: int,
    action_size: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create different types of DQN networks.
    
    Args:
        network_type (str): Type of network ('dqn', 'dueling', 'cnn')
        state_size (int): Size of the state space
        action_size (int): Size of the action space
        **kwargs: Additional arguments for network creation
        
    Returns:
        nn.Module: The created network
    """
    if network_type.lower() == 'dqn':
        return DQNNetwork(state_size, action_size, **kwargs)
    elif network_type.lower() == 'dueling':
        return DuelingDQNNetwork(state_size, action_size, **kwargs)
    elif network_type.lower() == 'cnn':
        return CNN_DQNNetwork(**kwargs)
    else:
        raise ValueError(f"Unknown network type: {network_type}")


if __name__ == "__main__":
    # Test the networks
    print("Testing DQN Networks...")
    
    # Test standard DQN
    dqn = DQNNetwork(state_size=4, action_size=2)
    test_state = torch.randn(1, 4)
    q_values = dqn(test_state)
    print(f"DQN output shape: {q_values.shape}")
    
    # Test Dueling DQN
    dueling_dqn = DuelingDQNNetwork(state_size=4, action_size=2)
    q_values_dueling = dueling_dqn(test_state)
    print(f"Dueling DQN output shape: {q_values_dueling.shape}")
    
    # Test CNN DQN
    cnn_dqn = CNN_DQNNetwork(input_channels=4, action_size=4)
    test_image = torch.randn(1, 4, 84, 84)
    q_values_cnn = cnn_dqn(test_image)
    print(f"CNN DQN output shape: {q_values_cnn.shape}")
    
    print("All networks tested successfully!")
