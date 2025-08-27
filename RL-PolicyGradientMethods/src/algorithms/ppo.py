"""
Proximal Policy Optimization (PPO) Algorithm Implementation
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


class PPO:
    """
    Proximal Policy Optimization (PPO) Algorithm
    
    PPO is a policy gradient method that uses a clipped objective function
    to prevent large policy updates, making training more stable.
    """
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 eps_clip=0.2, k_epochs=4, hidden_dims=[128, 128], 
                 device='cpu', value_coef=0.5, entropy_coef=0.01):
        """
        Initialize PPO algorithm
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            lr: Learning rate
            gamma: Discount factor
            eps_clip: Clipping parameter for PPO
            k_epochs: Number of epochs to update policy
            hidden_dims: Hidden layer dimensions
            device: Device to run on ('cpu' or 'cuda')
            value_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy loss
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = device
        
        # Initialize networks
        from ..networks.policy_networks import PolicyNetwork
        from ..networks.value_networks import ValueNetwork
        
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.value_net = ValueNetwork(state_dim, hidden_dims).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam([
            {'params': self.policy.parameters()},
            {'params': self.value_net.parameters()}
        ], lr=lr)
        
        # Training history
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        
    def select_action(self, state, deterministic=False):
        """
        Select action using current policy
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.policy.get_action(state, deterministic)
    
    def get_value(self, state):
        """
        Get state value estimate
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.value_net.forward(state)
    
    def compute_gae(self, rewards, values, next_values, dones, gamma=0.99, lam=0.95):
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            next_values: List of next value estimates
            dones: List of done flags
            gamma: Discount factor
            lam: GAE parameter
        """
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t] if not dones[t] else 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update_policy(self, states, actions, old_log_probs, rewards, 
                     values, next_values, dones):
        """
        Update policy using PPO algorithm
        """
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        next_values = torch.FloatTensor(next_values).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Compute advantages using GAE
        advantages = self.compute_gae(rewards, values, next_values, dones)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute returns
        returns = advantages + values
        
        # Update policy for k_epochs
        for _ in range(self.k_epochs):
            # Get current policy outputs
            current_log_probs = self.policy.forward(states)
            current_log_probs = current_log_probs.gather(1, actions).squeeze()
            
            # Compute entropy
            entropy = -(torch.exp(current_log_probs) * current_log_probs).sum(dim=-1).mean()
            
            # Compute ratio
            ratio = torch.exp(current_log_probs - old_log_probs)
            
            # Compute surrogates
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Policy loss
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            current_values = self.value_net.forward(states).squeeze()
            value_loss = nn.MSELoss()(current_values, returns)
            
            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Update networks
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        return policy_loss.item(), value_loss.item(), entropy.item()
    
    def train_episode(self, env, max_steps=1000):
        """
        Train for one episode
        """
        state, _ = env.reset()
        states, actions, old_log_probs, rewards, values, next_values, dones = [], [], [], [], [], [], []
        
        for step in range(max_steps):
            # Get current value
            current_value = self.get_value(state).item()
            
            # Select action
            action = self.select_action(state)
            
            # Get log probability of selected action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            log_probs = self.policy.forward(state_tensor)
            log_prob = log_probs[0, action].item()
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Get next value
            next_value = self.get_value(next_state).item() if not done else 0
            
            # Store experience
            states.append(state)
            actions.append(action)
            old_log_probs.append(log_prob)
            rewards.append(reward)
            values.append(current_value)
            next_values.append(next_value)
            dones.append(done)
            
            state = next_state
            
            if done:
                break
        
        # Update policy
        policy_loss, value_loss, entropy_loss = self.update_policy(
            states, actions, old_log_probs, rewards, values, next_values, dones
        )
        
        # Store episode statistics
        total_reward = sum(rewards)
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(len(rewards))
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropy_losses.append(entropy_loss)
        
        return total_reward, len(rewards), policy_loss, value_loss, entropy_loss
    
    def train(self, env, num_episodes=1000, max_steps=1000, 
              eval_interval=100, num_eval_episodes=10):
        """
        Train the PPO agent
        """
        print("Starting PPO training...")
        
        for episode in range(num_episodes):
            # Train one episode
            total_reward, episode_length, policy_loss, value_loss, entropy_loss = self.train_episode(env, max_steps)
            
            # Print progress
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                      f"Episode Reward: {total_reward:.2f}, "
                      f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, "
                      f"Entropy: {entropy_loss:.4f}")
            
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
        
        # Plot policy losses
        ax3.plot(self.policy_losses)
        ax3.set_title('Policy Loss')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss')
        ax3.grid(True)
        
        # Plot value losses
        ax4.plot(self.value_losses)
        ax4.set_title('Value Loss')
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
            'policy_state_dict': self.policy.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropy_losses': self.entropy_losses
        }, filepath)
    
    def load_model(self, filepath):
        """
        Load a trained model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.policy_losses = checkpoint['policy_losses']
        self.value_losses = checkpoint['value_losses']
        self.entropy_losses = checkpoint['entropy_losses']
