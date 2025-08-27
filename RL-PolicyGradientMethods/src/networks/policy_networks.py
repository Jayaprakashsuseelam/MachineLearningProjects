"""
Policy Networks for Policy Gradient Methods
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PolicyNetwork(nn.Module):
    """
    Policy network for discrete action spaces
    """
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128], activation=nn.ReLU):
        super(PolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation()
            ])
            prev_dim = hidden_dim
        
        # Output layer for action probabilities
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state):
        """
        Forward pass through the policy network
        Returns log probabilities of actions
        """
        logits = self.network(state)
        return F.log_softmax(logits, dim=-1)
    
    def get_action(self, state, deterministic=False):
        """
        Sample an action from the policy
        """
        with torch.no_grad():
            log_probs = self.forward(state)
            
            if deterministic:
                # Take the action with highest probability
                action = torch.argmax(log_probs, dim=-1)
                return action.item() if action.dim() == 0 else action
            else:
                # Sample from the policy
                probs = torch.exp(log_probs)
                action = torch.multinomial(probs, 1)
                return action.item() if action.dim() == 0 else action
    
    def get_action_and_log_prob(self, state):
        """
        Get action and its log probability for training
        """
        log_probs = self.forward(state)
        probs = torch.exp(log_probs)
        action = torch.multinomial(probs, 1)
        
        # Get log probability of the selected action
        log_prob = log_probs.gather(1, action)
        
        return action, log_prob


class ContinuousPolicyNetwork(nn.Module):
    """
    Policy network for continuous action spaces
    """
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128], activation=nn.ReLU):
        super(ContinuousPolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation()
            ])
            prev_dim = hidden_dim
        
        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)
        
        # Initialize log_std to be small
        self.log_std_layer.weight.data.fill_(-0.5)
        self.log_std_layer.bias.data.fill_(-0.5)
        
        self.base_network = nn.Sequential(*layers)
        
    def forward(self, state):
        """
        Forward pass through the policy network
        Returns mean and log_std of action distribution
        """
        features = self.base_network(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        
        # Clamp log_std to prevent numerical instability
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std
    
    def get_action(self, state, deterministic=False):
        """
        Sample an action from the policy
        """
        with torch.no_grad():
            mean, log_std = self.forward(state)
            
            if deterministic:
                return mean
            else:
                std = torch.exp(log_std)
                action = torch.normal(mean, std)
                return action
    
    def get_action_and_log_prob(self, state):
        """
        Get action and its log probability for training
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # Sample action
        normal = torch.distributions.Normal(mean, std)
        action = normal.sample()
        
        # Calculate log probability
        log_prob = normal.log_prob(action)
        
        # Sum over action dimensions
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob


class GaussianPolicyNetwork(ContinuousPolicyNetwork):
    """
    Alias for ContinuousPolicyNetwork for clarity
    """
    pass
