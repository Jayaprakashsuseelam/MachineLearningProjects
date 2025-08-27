"""
Actor-Critic Algorithm Implementation
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


class ActorCritic:
    """
    Actor-Critic Algorithm
    
    This algorithm uses two networks:
    - Actor: Policy network that selects actions
    - Critic: Value network that estimates state values
    
    The critic provides lower variance estimates of the policy gradient
    compared to REINFORCE.
    """
    
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=3e-4, 
                 gamma=0.99, hidden_dims=[128, 128], device='cpu'):
        """
        Initialize Actor-Critic algorithm
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            gamma: Discount factor
            hidden_dims: Hidden layer dimensions
            device: Device to run on ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.device = device
        
        # Initialize networks
        from ..networks.policy_networks import PolicyNetwork
        from ..networks.value_networks import ValueNetwork
        
        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.critic = ValueNetwork(state_dim, hidden_dims).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Training history
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        
    def select_action(self, state, deterministic=False):
        """
        Select action using current policy
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.actor.get_action(state, deterministic)
    
    def get_value(self, state):
        """
        Get state value estimate
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.critic.forward(state)
    
    def update_networks(self, states, actions, rewards, next_states, dones):
        """
        Update both actor and critic networks
        
        Args:
            states: List of states
            actions: List of actions
            rewards: List of rewards
            next_states: List of next states
            dones: List of done flags
        """
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Calculate TD targets
        with torch.no_grad():
            next_values = self.critic.forward(next_states).squeeze()
            targets = rewards + self.gamma * next_values * (~dones)
        
        # Current values
        current_values = self.critic.forward(states).squeeze()
        
        # Critic loss (TD error)
        critic_loss = nn.MSELoss()(current_values, targets)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor loss (policy gradient with value function baseline)
        log_probs = self.actor.forward(states)
        log_probs = log_probs.gather(1, actions).squeeze()
        
        # Advantage = TD error
        advantages = targets - current_values.detach()
        
        # Actor loss
        actor_loss = -(log_probs * advantages).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()
    
    def train_episode(self, env, max_steps=1000):
        """
        Train for one episode
        """
        state, _ = env.reset()
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
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
            next_states.append(next_state)
            dones.append(done)
            
            state = next_state
            
            if done:
                break
        
        # Update networks
        actor_loss, critic_loss = self.update_networks(states, actions, rewards, next_states, dones)
        
        # Store episode statistics
        total_reward = sum(rewards)
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(len(rewards))
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        
        return total_reward, len(rewards), actor_loss, critic_loss
    
    def train(self, env, num_episodes=1000, max_steps=1000, 
              eval_interval=100, num_eval_episodes=10):
        """
        Train the Actor-Critic agent
        """
        print("Starting Actor-Critic training...")
        
        for episode in range(num_episodes):
            # Train one episode
            total_reward, episode_length, actor_loss, critic_loss = self.train_episode(env, max_steps)
            
            # Print progress
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                      f"Episode Reward: {total_reward:.2f}, "
                      f"Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
            
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
        import matplotlib.pyplot as plt
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
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
        
        # Plot actor losses
        ax3.plot(self.actor_losses)
        ax3.set_title('Actor Loss')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss')
        ax3.grid(True)
        
        # Plot critic losses
        ax4.plot(self.critic_losses)
        ax4.set_title('Critic Loss')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Loss')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses
        }, filepath)
    
    def load_model(self, filepath):
        """
        Load a trained model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.actor_losses = checkpoint['actor_losses']
        self.critic_losses = checkpoint['critic_losses']
