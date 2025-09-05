"""
CartPole DQN Demo

This script demonstrates how to train a DQN agent on the CartPole environment.
It includes training, evaluation, and visualization of results.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqn.agent import DQNAgent
from environments.wrappers import make_env, get_env_info
from training.config import CART_POLE_CONFIG
from training.trainer import DQNTrainer
from visualization.monitor import TrainingMonitor
from visualization.plotter import DQNPlotter


def main():
    """Main function to run the CartPole DQN demo."""
    print("=" * 60)
    print("CARTPOLE DQN DEMO")
    print("=" * 60)
    
    # Configuration
    config = CART_POLE_CONFIG
    config.max_episodes = 500
    config.eval_frequency = 50
    config.log_frequency = 10
    config.save_frequency = 100
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Environment: {config.env_name}")
    print(f"Device: {config.device}")
    print(f"Max episodes: {config.max_episodes}")
    print("-" * 60)
    
    # Create trainer
    trainer = DQNTrainer(config)
    
    try:
        # Train the agent
        print("Starting training...")
        results = trainer.train()
        
        # Test the trained agent
        print("\nTesting trained agent...")
        test_scores = trainer.test(episodes=20, render=False)
        
        # Create visualizations
        print("\nCreating visualizations...")
        create_visualizations(results, test_scores)
        
        # Generate report
        print("\nGenerating training report...")
        generate_report(results, test_scores)
        
    finally:
        trainer.close()
    
    print("\nCartPole DQN demo completed successfully!")


def create_visualizations(results, test_scores):
    """Create and save visualizations."""
    # Create output directory
    os.makedirs("cartpole_results", exist_ok=True)
    
    # Initialize plotter
    plotter = DQNPlotter()
    
    # Plot learning curves
    plotter.plot_learning_curves(
        results['episode_rewards'],
        results['episode_lengths'],
        results['episode_losses'],
        [results['episode_rewards'][i] for i in range(len(results['episode_rewards']))],  # Placeholder for epsilons
        save_path="cartpole_results/learning_curves.png",
        show=False
    )
    
    # Plot evaluation comparison
    if results['eval_scores']:
        eval_episodes = list(range(0, len(results['episode_rewards']), 50))
        plotter.plot_evaluation_comparison(
            results['eval_scores'],
            eval_episodes,
            save_path="cartpole_results/evaluation_progress.png",
            show=False
        )
    
    # Plot agent performance
    plotter.plot_agent_performance(
        test_scores,
        [200] * len(test_scores),  # Placeholder for episode lengths
        save_path="cartpole_results/agent_performance.png",
        show=False
    )
    
    print("Visualizations saved to cartpole_results/")


def generate_report(results, test_scores):
    """Generate a comprehensive training report."""
    # Create monitor for report generation
    monitor = TrainingMonitor("logs")
    
    # Generate report
    report = {
        "training_summary": {
            "total_episodes": len(results['episode_rewards']),
            "final_avg_reward": np.mean(results['episode_rewards'][-10:]),
            "best_avg_reward": max(results['episode_rewards']),
            "final_avg_length": np.mean(results['episode_lengths'][-10:]),
            "convergence_episode": find_convergence_episode(results['episode_rewards'])
        },
        "test_results": {
            "mean_score": np.mean(test_scores),
            "std_score": np.std(test_scores),
            "min_score": min(test_scores),
            "max_score": max(test_scores),
            "success_rate": sum(1 for score in test_scores if score >= 195) / len(test_scores)
        },
        "learning_metrics": {
            "reward_improvement": results['episode_rewards'][-1] - results['episode_rewards'][0],
            "stability_score": calculate_stability_score(results['episode_rewards'])
        }
    }
    
    # Save report
    import json
    with open("cartpole_results/training_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total Episodes: {report['training_summary']['total_episodes']}")
    print(f"Final Average Reward: {report['training_summary']['final_avg_reward']:.2f}")
    print(f"Best Average Reward: {report['training_summary']['best_avg_reward']:.2f}")
    print(f"Convergence Episode: {report['training_summary']['convergence_episode']}")
    print(f"Test Success Rate: {report['test_results']['success_rate']:.1%}")
    print(f"Mean Test Score: {report['test_results']['mean_score']:.2f} ± {report['test_results']['std_score']:.2f}")


def find_convergence_episode(rewards, window=50):
    """Find the episode where learning converged."""
    if len(rewards) < window:
        return None
    
    # Calculate rolling standard deviation
    rolling_std = np.array([np.std(rewards[max(0, i-window):i+1]) for i in range(len(rewards))])
    
    # Find where standard deviation becomes small
    threshold = np.mean(rolling_std) * 0.1
    convergence_idx = np.where(rolling_std < threshold)[0]
    
    return convergence_idx[0] if len(convergence_idx) > 0 else None


def calculate_stability_score(rewards):
    """Calculate a stability score based on reward variance."""
    if len(rewards) < 10:
        return 0.0
    
    # Use the last 20% of episodes for stability calculation
    last_portion = rewards[-max(10, len(rewards) // 5):]
    stability_score = 1.0 / (1.0 + np.std(last_portion))
    
    return stability_score


def quick_demo():
    """Quick demo with minimal training."""
    print("Running quick CartPole demo...")
    
    # Quick configuration
    config = CART_POLE_CONFIG
    config.max_episodes = 100
    config.eval_frequency = 25
    config.log_frequency = 10
    
    # Train
    trainer = DQNTrainer(config)
    results = trainer.train()
    
    # Test
    test_scores = trainer.test(episodes=5)
    
    # Print results
    print(f"Quick demo completed!")
    print(f"Final average reward: {np.mean(results['episode_rewards'][-10:]):.2f}")
    print(f"Test scores: {test_scores}")
    
    trainer.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CartPole DQN Demo")
    parser.add_argument("--quick", action="store_true", help="Run quick demo")
    args = parser.parse_args()
    
    if args.quick:
        quick_demo()
    else:
        main()
