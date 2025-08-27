#!/usr/bin/env python3
"""
Trading Demo Script

This script demonstrates how to use policy gradient methods
for algorithmic trading.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from algorithms import PPO
from environments import TradingEnvironment
from utils import plot_training_progress, calculate_metrics

def main():
    """Main demo function"""
    print("Policy Gradient Methods Demo - Algorithmic Trading")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create trading environment
    print("Creating trading environment...")
    trading_env = TradingEnvironment(
        symbol="AAPL",
        start_date="2020-01-01",
        end_date="2023-01-01",
        initial_balance=10000.0,
        transaction_cost=0.001,
        lookback_window=20
    )
    
    print(f"Environment created with state dimension: {trading_env.observation_space.shape[0]}")
    print(f"Action space: {trading_env.action_space.n}")
    
    # Test environment
    state, info = trading_env.reset()
    print(f"Initial portfolio value: ${info['portfolio_value']:.2f}")
    
    # Initialize PPO agent for trading
    print("\nInitializing PPO agent...")
    trading_agent = PPO(
        state_dim=trading_env.observation_space.shape[0],
        action_dim=trading_env.action_space.n,
        lr=1e-4,  # Lower learning rate for trading
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4,
        device=device
    )
    
    # Train the agent
    print("\nTraining trading agent...")
    trading_agent.train(trading_env, num_episodes=300, eval_interval=50)
    
    # Plot training progress
    trading_agent.plot_training_progress()
    
    # Evaluate final performance
    print("\nEvaluating final performance...")
    final_performance = trading_agent.evaluate(trading_env, num_episodes=5)
    print(f"Final trading performance: {final_performance:.2f}")
    
    # Run a test episode for analysis
    print("\nRunning test episode for analysis...")
    state, info = trading_env.reset()
    total_reward = 0
    steps = 0
    
    while steps < 1000:  # Limit steps for analysis
        action = trading_agent.select_action(state, deterministic=True)
        next_state, reward, terminated, truncated, info = trading_env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        steps += 1
        state = next_state
        
        if done:
            break
    
    print(f"Test episode reward: {total_reward:.2f}")
    print(f"Final portfolio value: ${info['portfolio_value']:.2f}")
    print(f"Number of trades: {len(trading_env.trades)}")
    
    # Get performance metrics
    metrics = trading_env.get_performance_metrics()
    print("\nPerformance Metrics:")
    print("=" * 30)
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'return' in key.lower() or 'rate' in key.lower():
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Plot trading performance
    trading_env.plot_performance()
    
    # Calculate training metrics
    training_metrics = calculate_metrics(
        trading_agent.episode_rewards, 
        trading_agent.episode_lengths
    )
    
    print("\nTraining Metrics:")
    print("=" * 30)
    print(f"Mean Episode Reward: {training_metrics['mean_reward']:.2f}")
    print(f"Recent Mean Reward: {training_metrics.get('recent_mean_reward', 0):.2f}")
    print(f"Success Rate: {training_metrics['success_rate']:.2f}")
    print(f"Consistency: {training_metrics.get('consistency', 0):.2f}")
    
    print("\nTrading demo completed successfully!")
    print("\nKey Insights:")
    print("- The agent learns to make trading decisions based on market data")
    print("- Performance is measured by portfolio value changes")
    print("- Transaction costs are factored into the reward function")
    print("- The agent can adapt to different market conditions")

if __name__ == "__main__":
    main()
