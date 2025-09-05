#!/usr/bin/env python3
"""
Main DQN Training Script

This script provides a command-line interface for training DQN agents
on various environments with different configurations.
"""

import argparse
import os
import sys
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqn.agent import create_agent
from environments.wrappers import make_env, get_env_info
from training.config import DQNConfig, get_config, CONFIG_REGISTRY
from training.trainer import DQNTrainer
from visualization.monitor import TrainingMonitor
from visualization.plotter import DQNPlotter


def main():
    """Main function for DQN training."""
    parser = argparse.ArgumentParser(description="DQN Training Script")
    
    # Environment arguments
    parser.add_argument("--env", type=str, default="CartPole-v1", 
                       help="Environment name")
    parser.add_argument("--config", type=str, choices=list(CONFIG_REGISTRY.keys()),
                       help="Use predefined configuration")
    
    # Training arguments
    parser.add_argument("--episodes", type=int, default=1000,
                       help="Number of training episodes")
    parser.add_argument("--eval-freq", type=int, default=100,
                       help="Evaluation frequency")
    parser.add_argument("--save-freq", type=int, default=200,
                       help="Model save frequency")
    
    # Agent arguments
    parser.add_argument("--agent-type", type=str, default="dqn",
                       choices=["dqn", "double", "dueling", "rainbow"],
                       help="Type of DQN agent")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0,
                       help="Initial epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.995,
                       help="Epsilon decay rate")
    parser.add_argument("--epsilon-min", type=float, default=0.01,
                       help="Minimum epsilon")
    
    # Network arguments
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[128, 64],
                       help="Hidden layer sizes")
    parser.add_argument("--network-type", type=str, default="dqn",
                       choices=["dqn", "dueling", "cnn"],
                       help="Network architecture type")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--memory-size", type=int, default=10000,
                       help="Replay buffer size")
    parser.add_argument("--target-update-freq", type=int, default=10,
                       help="Target network update frequency")
    
    # Device and logging
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device to use")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    
    # Visualization
    parser.add_argument("--render", action="store_true",
                       help="Render environment during training")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip generating plots")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("DQN TRAINING SCRIPT")
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"Agent Type: {args.agent_type}")
    print(f"Device: {device}")
    print(f"Episodes: {args.episodes}")
    print(f"Output Directory: {args.output_dir}")
    print("-" * 60)
    
    # Create configuration
    if args.config:
        config = get_config(args.config)
        config.env_name = args.env
    else:
        config = DQNConfig(
            env_name=args.env,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            epsilon_min=args.epsilon_min,
            network_type=args.network_type,
            hidden_layers=args.hidden_layers,
            batch_size=args.batch_size,
            memory_size=args.memory_size,
            target_update_frequency=args.target_update_freq,
            max_episodes=args.episodes,
            eval_frequency=args.eval_freq,
            save_frequency=args.save_freq,
            device=device,
            seed=args.seed,
            save_path=os.path.join(args.output_dir, "models"),
            log_path=os.path.join(args.output_dir, "logs")
        )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(config.save_path, exist_ok=True)
    os.makedirs(config.log_path, exist_ok=True)
    
    # Save configuration
    config.save(os.path.join(args.output_dir, "config.json"))
    
    # Create trainer
    trainer = DQNTrainer(config)
    
    # Set agent type
    if args.agent_type != "dqn":
        trainer.agent = create_agent(args.agent_type, **config.to_dict())
    
    try:
        # Resume from checkpoint if specified
        if args.resume:
            trainer.load_model(args.resume)
            print(f"Resumed training from {args.resume}")
        
        # Train the agent
        print("Starting training...")
        results = trainer.train()
        
        # Test the trained agent
        print("\nTesting trained agent...")
        test_scores = trainer.test(episodes=20, render=args.render)
        
        # Save results
        results_file = os.path.join(args.output_dir, "training_results.json")
        trainer.save_training_results(results_file)
        
        # Generate visualizations
        if not args.no_plots:
            print("\nGenerating visualizations...")
            create_visualizations(results, test_scores, args.output_dir)
        
        # Generate report
        print("\nGenerating training report...")
        generate_report(results, test_scores, args.output_dir, config)
        
        print(f"\nTraining completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_model("interrupted_model.pth")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    finally:
        trainer.close()


def create_visualizations(results, test_scores, output_dir):
    """Create and save visualizations."""
    plotter = DQNPlotter()
    
    # Plot learning curves
    plotter.plot_learning_curves(
        results['episode_rewards'],
        results['episode_lengths'],
        results['episode_losses'],
        [results['episode_rewards'][i] for i in range(len(results['episode_rewards']))],  # Placeholder for epsilons
        save_path=os.path.join(output_dir, "learning_curves.png"),
        show=False
    )
    
    # Plot evaluation comparison
    if results['eval_scores']:
        eval_episodes = list(range(0, len(results['episode_rewards']), 50))
        plotter.plot_evaluation_comparison(
            results['eval_scores'],
            eval_episodes,
            save_path=os.path.join(output_dir, "evaluation_progress.png"),
            show=False
        )
    
    # Plot agent performance
    plotter.plot_agent_performance(
        test_scores,
        [100] * len(test_scores),  # Placeholder for episode lengths
        save_path=os.path.join(output_dir, "agent_performance.png"),
        show=False
    )
    
    print("Visualizations saved to output directory")


def generate_report(results, test_scores, output_dir, config):
    """Generate a comprehensive training report."""
    import numpy as np
    
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
            "success_rate": calculate_success_rate(test_scores, config.env_name)
        },
        "learning_metrics": {
            "reward_improvement": results['episode_rewards'][-1] - results['episode_rewards'][0],
            "stability_score": calculate_stability_score(results['episode_rewards'])
        },
        "configuration": config.to_dict()
    }
    
    # Save report
    report_file = os.path.join(output_dir, "training_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Environment: {config.env_name}")
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
    
    import numpy as np
    
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
    
    import numpy as np
    
    # Use the last 20% of episodes for stability calculation
    last_portion = rewards[-max(10, len(rewards) // 5):]
    stability_score = 1.0 / (1.0 + np.std(last_portion))
    
    return stability_score


def calculate_success_rate(test_scores, env_name):
    """Calculate success rate based on environment type."""
    if "CartPole" in env_name:
        # Success if score > 195
        return sum(1 for score in test_scores if score >= 195) / len(test_scores)
    elif "MountainCar" in env_name:
        # Success if score > -110
        return sum(1 for score in test_scores if score >= -110) / len(test_scores)
    elif "LunarLander" in env_name:
        # Success if score > 200
        return sum(1 for score in test_scores if score >= 200) / len(test_scores)
    else:
        return 0.0


if __name__ == "__main__":
    main()
