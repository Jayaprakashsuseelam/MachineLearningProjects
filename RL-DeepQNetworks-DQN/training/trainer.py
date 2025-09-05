"""
DQN Training Module

This module provides the main training loop and utilities for training
Deep Q-Network agents.
"""

import os
import time
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import json
from datetime import datetime

from ..dqn.agent import DQNAgent, create_agent
from ..environments.wrappers import make_env, get_env_info
from .config import DQNConfig


class DQNTrainer:
    """
    DQN Trainer class for training Deep Q-Network agents.
    
    This class handles the complete training pipeline including:
    - Environment setup
    - Agent initialization
    - Training loop
    - Evaluation
    - Logging and saving
    """
    
    def __init__(self, config: DQNConfig):
        """
        Initialize the DQN trainer.
        
        Args:
            config: DQN configuration object
        """
        self.config = config
        
        # Create directories
        os.makedirs(config.save_path, exist_ok=True)
        os.makedirs(config.log_path, exist_ok=True)
        
        # Initialize environment
        self.env = make_env(config.env_name, config.render_mode)
        self.env_info = get_env_info(self.env)
        
        # Determine state and action sizes
        if len(self.env_info['state_size']) == 1:
            self.state_size = self.env_info['state_size'][0]
        else:
            self.state_size = self.env_info['state_size']
        
        self.action_size = self.env_info['n_actions']
        
        # Initialize agent
        agent_kwargs = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'learning_rate': config.learning_rate,
            'gamma': config.gamma,
            'epsilon': config.epsilon,
            'epsilon_decay': config.epsilon_decay,
            'epsilon_min': config.epsilon_min,
            'batch_size': config.batch_size,
            'memory_size': config.memory_size,
            'target_update_frequency': config.target_update_frequency,
            'network_type': config.network_type,
            'hidden_layers': config.hidden_layers,
            'device': config.device,
            'seed': config.seed
        }
        
        # Add advanced settings
        if config.double_dqn:
            agent_kwargs['agent_type'] = 'double'
        elif config.dueling_dqn:
            agent_kwargs['agent_type'] = 'dueling'
        elif config.prioritized_replay or config.n_step:
            agent_kwargs['agent_type'] = 'rainbow'
            agent_kwargs['use_prioritized_replay'] = config.prioritized_replay
            agent_kwargs['use_n_step'] = config.n_step
            agent_kwargs['n_steps'] = config.n_steps
        
        self.agent = create_agent('dqn', **agent_kwargs)
        
        # Training statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_losses = deque(maxlen=100)
        self.eval_scores = deque(maxlen=100)
        
        # Training state
        self.episode = 0
        self.total_steps = 0
        self.best_eval_score = float('-inf')
        self.episodes_without_improvement = 0
        self.training_start_time = None
        
        # Logging
        self.training_log = []
        self.eval_log = []
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the DQN agent.
        
        Returns:
            Dict containing training statistics
        """
        print(f"Starting DQN training on {self.config.env_name}")
        print(f"State size: {self.state_size}, Action size: {self.action_size}")
        print(f"Device: {self.config.device}")
        print(f"Max episodes: {self.config.max_episodes}")
        print("-" * 50)
        
        self.training_start_time = time.time()
        
        for episode in range(self.config.max_episodes):
            self.episode = episode
            
            # Train one episode
            episode_reward, episode_length, episode_loss = self._train_episode()
            
            # Store statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            if episode_loss is not None:
                self.episode_losses.append(episode_loss)
            
            # Logging
            if episode % self.config.log_frequency == 0:
                self._log_training_progress(episode)
            
            # Evaluation
            if episode % self.config.eval_frequency == 0 and episode > 0:
                eval_score = self.evaluate()
                self.eval_scores.append(eval_score)
                self._log_evaluation(episode, eval_score)
                
                # Check for improvement
                if eval_score > self.best_eval_score:
                    self.best_eval_score = eval_score
                    self.episodes_without_improvement = 0
                    self.save_model(f"best_model_episode_{episode}.pth")
                else:
                    self.episodes_without_improvement += 1
            
            # Save model
            if episode % self.config.save_frequency == 0 and episode > 0:
                self.save_model(f"model_episode_{episode}.pth")
            
            # Early stopping
            if (self.config.early_stopping and 
                self.episodes_without_improvement >= self.config.patience):
                print(f"Early stopping at episode {episode}")
                break
        
        # Final evaluation
        final_eval_score = self.evaluate()
        print(f"Final evaluation score: {final_eval_score:.2f}")
        
        # Save final model and logs
        self.save_model("final_model.pth")
        self.save_training_log()
        
        training_time = time.time() - self.training_start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        return {
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths),
            'episode_losses': list(self.episode_losses),
            'eval_scores': list(self.eval_scores),
            'training_log': self.training_log,
            'eval_log': self.eval_log
        }
    
    def _train_episode(self) -> Tuple[float, int, Optional[float]]:
        """
        Train the agent for one episode.
        
        Returns:
            Tuple of (episode_reward, episode_length, average_loss)
        """
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        
        for step in range(self.config.max_steps_per_episode):
            # Select action
            action = self.agent.act(state, training=True)
            
            # Take step
            next_state, reward, done, _ = self.env.step(action)
            
            # Store experience and learn
            self.agent.step(state, action, reward, next_state, done)
            
            # Learn if we have enough experiences
            if (self.total_steps >= self.config.learning_starts and 
                len(self.agent.memory) >= self.config.batch_size):
                loss = self.agent.learn()
                if loss is not None:
                    episode_losses.append(loss)
            
            # Update state and statistics
            state = next_state
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
            
            if done:
                break
        
        avg_loss = np.mean(episode_losses) if episode_losses else None
        return episode_reward, episode_length, avg_loss
    
    def evaluate(self, render: bool = False) -> float:
        """
        Evaluate the agent's performance.
        
        Args:
            render: Whether to render the environment
            
        Returns:
            Average evaluation score
        """
        eval_scores = []
        
        for _ in range(self.config.eval_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for _ in range(self.config.max_steps_per_episode):
                if render:
                    self.env.render()
                
                action = self.agent.act(state, training=False)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            eval_scores.append(episode_reward)
        
        return np.mean(eval_scores)
    
    def test(self, episodes: int = 10, render: bool = False) -> List[float]:
        """
        Test the trained agent.
        
        Args:
            episodes: Number of test episodes
            render: Whether to render the environment
            
        Returns:
            List of test scores
        """
        print(f"Testing agent for {episodes} episodes...")
        
        test_scores = []
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for _ in range(self.config.max_steps_per_episode):
                if render:
                    self.env.render()
                
                action = self.agent.act(state, training=False)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            test_scores.append(episode_reward)
            print(f"Episode {episode + 1}: Score = {episode_reward:.2f}")
        
        avg_score = np.mean(test_scores)
        std_score = np.std(test_scores)
        
        print(f"Test Results:")
        print(f"Average Score: {avg_score:.2f} ± {std_score:.2f}")
        print(f"Best Score: {max(test_scores):.2f}")
        print(f"Worst Score: {min(test_scores):.2f}")
        
        return test_scores
    
    def _log_training_progress(self, episode: int) -> None:
        """Log training progress."""
        avg_reward = np.mean(self.episode_rewards)
        avg_length = np.mean(self.episode_lengths)
        avg_loss = np.mean(self.episode_losses) if self.episode_losses else 0.0
        
        stats = self.agent.get_stats()
        
        log_entry = {
            'episode': episode,
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'avg_loss': avg_loss,
            'epsilon': stats['epsilon'],
            'memory_size': stats['memory_size'],
            'total_steps': self.total_steps,
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_log.append(log_entry)
        
        print(f"Episode {episode:4d} | "
              f"Avg Reward: {avg_reward:8.2f} | "
              f"Avg Length: {avg_length:6.1f} | "
              f"Avg Loss: {avg_loss:8.4f} | "
              f"Epsilon: {stats['epsilon']:6.3f} | "
              f"Memory: {stats['memory_size']:5d}")
    
    def _log_evaluation(self, episode: int, eval_score: float) -> None:
        """Log evaluation results."""
        log_entry = {
            'episode': episode,
            'eval_score': eval_score,
            'best_eval_score': self.best_eval_score,
            'episodes_without_improvement': self.episodes_without_improvement,
            'timestamp': datetime.now().isoformat()
        }
        
        self.eval_log.append(log_entry)
        
        print(f"Evaluation at episode {episode}: Score = {eval_score:.2f} "
              f"(Best: {self.best_eval_score:.2f})")
    
    def save_model(self, filename: str) -> None:
        """Save the trained model."""
        filepath = os.path.join(self.config.save_path, filename)
        self.agent.save(filepath)
    
    def load_model(self, filename: str) -> None:
        """Load a trained model."""
        filepath = os.path.join(self.config.save_path, filename)
        self.agent.load(filepath)
    
    def save_training_log(self) -> None:
        """Save training logs to JSON files."""
        # Save training log
        training_log_path = os.path.join(self.config.log_path, "training_log.json")
        with open(training_log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        
        # Save evaluation log
        eval_log_path = os.path.join(self.config.log_path, "eval_log.json")
        with open(eval_log_path, 'w') as f:
            json.dump(self.eval_log, f, indent=2)
        
        # Save configuration
        config_path = os.path.join(self.config.log_path, "config.json")
        self.config.save(config_path)
    
    def close(self) -> None:
        """Close the environment."""
        self.env.close()


def train_dqn(
    config: DQNConfig,
    resume_from: Optional[str] = None
) -> Dict[str, List[float]]:
    """
    Convenience function to train a DQN agent.
    
    Args:
        config: DQN configuration
        resume_from: Path to model to resume from
        
    Returns:
        Training statistics
    """
    trainer = DQNTrainer(config)
    
    if resume_from:
        trainer.load_model(resume_from)
        print(f"Resumed training from {resume_from}")
    
    try:
        results = trainer.train()
        return results
    finally:
        trainer.close()


if __name__ == "__main__":
    # Test the trainer
    print("Testing DQN Trainer...")
    
    from .config import CART_POLE_CONFIG
    
    # Create a small test configuration
    test_config = CART_POLE_CONFIG
    test_config.max_episodes = 10
    test_config.eval_frequency = 5
    test_config.log_frequency = 2
    
    # Train the agent
    trainer = DQNTrainer(test_config)
    results = trainer.train()
    
    print(f"Training completed. Final results: {len(results['episode_rewards'])} episodes")
    
    trainer.close()
    print("DQN Trainer tested successfully!")
