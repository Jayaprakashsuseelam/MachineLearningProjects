"""
Value Networks for Policy Gradient Methods
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNetwork(nn.Module):
    """
    Value network for estimating state values V(s)
    """
    def __init__(self, state_dim, hidden_dims=[128, 128], activation=nn.ReLU):
        super(ValueNetwork, self).__init__()
        
        self.state_dim = state_dim
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation()
            ])
            prev_dim = hidden_dim
        
        # Output layer for state value
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state):
        """
        Forward pass through the value network
        Returns estimated state value
        """
        return self.network(state)
    
    def get_value(self, state):
        """
        Get state value estimate
        """
        with torch.no_grad():
            return self.forward(state)


class QNetwork(nn.Module):
    """
    Q-network for estimating action values Q(s,a)
    """
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128], activation=nn.ReLU):
        super(QNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        prev_dim = state_dim + action_dim  # Concatenate state and action
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation()
            ])
            prev_dim = hidden_dim
        
        # Output layer for Q-value
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state, action):
        """
        Forward pass through the Q-network
        Returns estimated Q-value
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        return self.network(x)
    
    def get_q_value(self, state, action):
        """
        Get Q-value estimate
        """
        with torch.no_grad():
            return self.forward(state, action)


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-network that separates value and advantage estimation
    """
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128], activation=nn.ReLU):
        super(DuelingQNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature layers
        shared_layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims[:-1]:  # All but last hidden layer
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation()
            ])
            prev_dim = hidden_dim
        
        self.shared_network = nn.Sequential(*shared_layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            activation(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            activation(),
            nn.Linear(hidden_dims[-1], action_dim)
        )
        
    def forward(self, state):
        """
        Forward pass through the dueling network
        Returns Q-values for all actions
        """
        features = self.shared_network(state)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q_values
    
    def get_q_value(self, state, action):
        """
        Get Q-value for specific state-action pair
        """
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.gather(1, action)
