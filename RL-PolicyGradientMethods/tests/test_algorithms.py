"""
Test suite for policy gradient algorithms
"""

import unittest
import numpy as np
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithms import REINFORCE, ActorCritic, PPO
from environments import CustomCartPoleEnv, TradingEnvironment


class TestAlgorithms(unittest.TestCase):
    """Test cases for policy gradient algorithms"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')
        self.state_dim = 6
        self.action_dim = 2
        
    def test_reinforce_initialization(self):
        """Test REINFORCE initialization"""
        agent = REINFORCE(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )
        
        self.assertEqual(agent.state_dim, self.state_dim)
        self.assertEqual(agent.action_dim, self.action_dim)
        self.assertIsNotNone(agent.policy)
        self.assertIsNotNone(agent.optimizer)
        
    def test_actor_critic_initialization(self):
        """Test Actor-Critic initialization"""
        agent = ActorCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )
        
        self.assertEqual(agent.state_dim, self.state_dim)
        self.assertEqual(agent.action_dim, self.action_dim)
        self.assertIsNotNone(agent.actor)
        self.assertIsNotNone(agent.critic)
        self.assertIsNotNone(agent.actor_optimizer)
        self.assertIsNotNone(agent.critic_optimizer)
        
    def test_ppo_initialization(self):
        """Test PPO initialization"""
        agent = PPO(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )
        
        self.assertEqual(agent.state_dim, self.state_dim)
        self.assertEqual(agent.action_dim, self.action_dim)
        self.assertIsNotNone(agent.policy)
        self.assertIsNotNone(agent.value_net)
        self.assertIsNotNone(agent.optimizer)
        
    def test_action_selection(self):
        """Test action selection for all algorithms"""
        algorithms = [
            REINFORCE(self.state_dim, self.action_dim, device=self.device),
            ActorCritic(self.state_dim, self.action_dim, device=self.device),
            PPO(self.state_dim, self.action_dim, device=self.device)
        ]
        
        state = np.random.randn(self.state_dim)
        
        for agent in algorithms:
            action = agent.select_action(state)
            self.assertIsInstance(action, int)
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, self.action_dim)
            
    def test_training_episode(self):
        """Test training episode for all algorithms"""
        env = CustomCartPoleEnv(max_steps=100, reward_shaping=True)
        
        algorithms = [
            REINFORCE(self.state_dim, self.action_dim, device=self.device),
            ActorCritic(self.state_dim, self.action_dim, device=self.device),
            PPO(self.state_dim, self.action_dim, device=self.device)
        ]
        
        for agent in algorithms:
            # Train one episode
            total_reward, episode_length, *losses = agent.train_episode(env, max_steps=100)
            
            self.assertIsInstance(total_reward, (int, float))
            self.assertIsInstance(episode_length, int)
            self.assertGreater(episode_length, 0)
            
    def test_evaluation(self):
        """Test evaluation for all algorithms"""
        env = CustomCartPoleEnv(max_steps=100, reward_shaping=True)
        
        algorithms = [
            REINFORCE(self.state_dim, self.action_dim, device=self.device),
            ActorCritic(self.state_dim, self.action_dim, device=self.device),
            PPO(self.state_dim, self.action_dim, device=self.device)
        ]
        
        for agent in algorithms:
            # Evaluate policy
            performance = agent.evaluate(env, num_episodes=2, max_steps=100)
            
            self.assertIsInstance(performance, (int, float))
            
    def test_trading_environment(self):
        """Test trading environment"""
        env = TradingEnvironment(
            symbol="AAPL",
            start_date="2020-01-01",
            end_date="2020-02-01",  # Short period for testing
            initial_balance=1000.0,
            transaction_cost=0.001,
            lookback_window=5
        )
        
        # Test reset
        state, info = env.reset()
        self.assertEqual(len(state), env.observation_space.shape[0])
        self.assertIn('portfolio_value', info)
        
        # Test step
        action = 0  # Hold
        next_state, reward, terminated, truncated, info = env.step(action)
        
        self.assertEqual(len(next_state), env.observation_space.shape[0])
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        
    def test_custom_cartpole_environment(self):
        """Test custom CartPole environment"""
        env = CustomCartPoleEnv(max_steps=100, reward_shaping=True)
        
        # Test reset
        state, info = env.reset()
        self.assertEqual(len(state), env.observation_space.shape[0])
        self.assertIn('episode_length', info)
        
        # Test step
        action = 0
        next_state, reward, terminated, truncated, info = env.step(action)
        
        self.assertEqual(len(next_state), env.observation_space.shape[0])
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)


if __name__ == '__main__':
    unittest.main()
