#!/usr/bin/env python3
"""
CartPole Demo Script

This script demonstrates how to use the policy gradient methods
on the CartPole environment.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from algorithms import REINFORCE, ActorCritic, PPO
from environments import CustomCartPoleEnv
from utils import plot_training_progress, calculate_metrics

def main():
    """Main demo function"""
    print("Policy Gradient Methods Demo - CartPole Environment")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create environment
    env = CustomCartPoleEnv(max_steps=500, reward_shaping=True)
    print(f"Environment created with state dimension: {env.observation_space.shape[0]}")
    
    # Test different algorithms
    algorithms = {
        'REINFORCE': REINFORCE(
            state_dim=6,
            action_dim=2,
            lr=3e-4,
            gamma=0.99,
            device=device
        ),
        'Actor-Critic': ActorCritic(
            state_dim=6,
            action_dim=2,
            lr_actor=3e-4,
            lr_critic=3e-4,
            gamma=0.99,
            device=device
        ),
        'PPO': PPO(
            state_dim=6,
            action_dim=2,
            lr=3e-4,
            gamma=0.99,
            eps_clip=0.2,
            k_epochs=4,
            device=device
        )
    }
    
    results = {}
    
    for name, agent in algorithms.items():
        print(f"\nTraining {name}...")
        print("-" * 30)
        
        # Train the agent
        agent.train(env, num_episodes=500, eval_interval=100)
        
        # Evaluate performance
        final_performance = agent.evaluate(env, num_episodes=10)
        print(f"Final {name} performance: {final_performance:.2f}")
        
        # Store results
        results[name] = {
            'episode_rewards': agent.episode_rewards,
            'episode_lengths': agent.episode_lengths,
            'final_performance': final_performance
        }
        
        # Plot training progress
        agent.plot_training_progress()
    
    # Compare algorithms
    print("\nAlgorithm Comparison:")
    print("=" * 30)
    for name, result in results.items():
        metrics = calculate_metrics(result['episode_rewards'], result['episode_lengths'])
        print(f"{name}:")
        print(f"  Mean Reward: {metrics['mean_reward']:.2f}")
        print(f"  Recent Mean Reward: {metrics.get('recent_mean_reward', 0):.2f}")
        print(f"  Success Rate: {metrics['success_rate']:.2f}")
        print(f"  Final Performance: {result['final_performance']:.2f}")
        print()
    
    # Plot comparison
    from utils.visualization import plot_algorithm_comparison
    plot_algorithm_comparison(results, "CartPole Algorithm Comparison")
    
    print("Demo completed successfully!")

if __name__ == "__main__":
    main()
