"""
REINFORCE Algorithm Implementation
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


class REINFORCE:
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm
    
    This is the simplest policy gradient method that uses the complete return
    from each episode to estimate the policy gradient.
    """
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 hidden_dims=[128, 128], device='cpu'):
        """
        Initialize REINFORCE algorithm
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            lr: Learning rate
            gamma: Discount factor
            hidden_dims: Hidden layer dimensions
            device: Device to run on ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.device = device
        
        # Initialize policy network
        from ..networks.policy_networks import PolicyNetwork
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Training history
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        
    def select_action(self, state, deterministic=False):
        """
        Select action using current policy
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.policy.get_action(state, deterministic)
    
    def update_policy(self, states, actions, rewards):
        """
        Update policy using REINFORCE algorithm
        
        Args:
            states: List of states from episode
            actions: List of actions from episode
            rewards: List of rewards from episode
        """
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        
        # Calculate returns (discounted cumulative rewards)
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns (optional but often helps)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        log_probs = self.policy.forward(states)
        log_probs = log_probs.gather(1, actions).squeeze()
        
        # REINFORCE loss: -log_prob * return
        loss = -(log_probs * returns).mean()
        
        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_episode(self, env, max_steps=1000):
        """
        Train for one episode
        """
        state, _ = env.reset()
        states, actions, rewards = [], [], []
        
        for step in range(max_steps):
            # Select action
            action = self.select_action(state)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
            
            if done:
                break
        
        # Update policy
        loss = self.update_policy(states, actions, rewards)
        
        # Store episode statistics
        total_reward = sum(rewards)
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(len(rewards))
        self.losses.append(loss)
        
        return total_reward, len(rewards), loss
    
    def train(self, env, num_episodes=1000, max_steps=1000, 
              eval_interval=100, num_eval_episodes=10):
        """
        Train the REINFORCE agent
        
        Args:
            env: Environment to train on
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            eval_interval: Episodes between evaluations
            num_eval_episodes: Number of episodes for evaluation
        """
        print("Starting REINFORCE training...")
        
        for episode in range(num_episodes):
            # Train one episode
            total_reward, episode_length, loss = self.train_episode(env, max_steps)
            
            # Print progress
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                      f"Episode Reward: {total_reward:.2f}, Loss: {loss:.4f}")
            
            # Evaluate policy
            if episode % eval_interval == 0 and episode > 0:
                eval_reward = self.evaluate(env, num_eval_episodes, max_steps)
                print(f"Evaluation at episode {episode}: {eval_reward:.2f}")
        
        print("Training completed!")
    
    def evaluate(self, env, num_episodes=10, max_steps=1000):
        """
        Evaluate the current policy
        """
        total_rewards = []
        
        for _ in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = self.select_action(state, deterministic=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            total_rewards.append(total_reward)
        
        return np.mean(total_rewards)
    
    def plot_training_progress(self):
        """
        Plot training progress
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot episode rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        
        # Plot episode lengths
        ax2.plot(self.episode_lengths)
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.grid(True)
        
        # Plot losses
        ax3.plot(self.losses)
        ax3.set_title('Policy Loss')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses
        }, filepath)
    
    def load_model(self, filepath):
        """
        Load a trained model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.losses = checkpoint['losses']
