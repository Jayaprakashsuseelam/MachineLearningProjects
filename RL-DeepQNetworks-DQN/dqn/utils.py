"""
Utility Functions for DQN

This module provides utility functions for DQN training and analysis.
"""

import numpy as np
import torch
import random
import os
import json
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
from datetime import datetime


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_training_results(
    results: Dict[str, List[float]],
    filepath: str
) -> None:
    """
    Save training results to JSON file.
    
    Args:
        results: Dictionary containing training results
        filepath: Path to save the results
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, list):
            serializable_results[key] = value
        else:
            serializable_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def load_training_results(filepath: str) -> Dict[str, List[float]]:
    """
    Load training results from JSON file.
    
    Args:
        filepath: Path to load the results from
        
    Returns:
        Dictionary containing training results
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results


def calculate_moving_average(
    data: List[float],
    window: int = 100
) -> List[float]:
    """
    Calculate moving average of data.
    
    Args:
        data: Input data
        window: Window size for moving average
        
    Returns:
        List of moving averages
    """
    if len(data) < window:
        return data
    
    moving_avg = []
    for i in range(len(data)):
        start_idx = max(0, i - window + 1)
        moving_avg.append(np.mean(data[start_idx:i+1]))
    
    return moving_avg


def calculate_convergence_episode(
    rewards: List[float],
    window: int = 100,
    threshold: float = 0.1
) -> Optional[int]:
    """
    Calculate the episode where learning converged.
    
    Args:
        rewards: List of episode rewards
        window: Window size for rolling statistics
        threshold: Threshold for convergence
        
    Returns:
        Episode number where convergence occurred, or None
    """
    if len(rewards) < window:
        return None
    
    # Calculate rolling standard deviation
    rolling_std = []
    for i in range(len(rewards)):
        start_idx = max(0, i - window + 1)
        rolling_std.append(np.std(rewards[start_idx:i+1]))
    
    # Find where standard deviation becomes small
    mean_std = np.mean(rolling_std)
    convergence_threshold = mean_std * threshold
    
    for i, std_val in enumerate(rolling_std):
        if std_val < convergence_threshold:
            return i
    
    return None


def calculate_stability_score(
    rewards: List[float],
    last_portion: float = 0.2
) -> float:
    """
    Calculate stability score based on reward variance.
    
    Args:
        rewards: List of episode rewards
        last_portion: Portion of episodes to use for stability calculation
        
    Returns:
        Stability score (higher is more stable)
    """
    if len(rewards) < 10:
        return 0.0
    
    # Use the last portion of episodes
    last_episodes = int(len(rewards) * last_portion)
    last_rewards = rewards[-last_episodes:]
    
    # Calculate stability score
    stability_score = 1.0 / (1.0 + np.std(last_rewards))
    
    return stability_score


def calculate_learning_efficiency(
    rewards: List[float],
    target_reward: float
) -> float:
    """
    Calculate learning efficiency (how quickly target reward was reached).
    
    Args:
        rewards: List of episode rewards
        target_reward: Target reward to reach
        
    Returns:
        Learning efficiency score
    """
    # Find first episode where target reward was reached
    for i, reward in enumerate(rewards):
        if reward >= target_reward:
            return 1.0 / (i + 1)  # Higher efficiency for earlier convergence
    
    return 0.0  # Target never reached


def analyze_training_progress(
    results: Dict[str, List[float]]
) -> Dict[str, Any]:
    """
    Analyze training progress and return comprehensive statistics.
    
    Args:
        results: Dictionary containing training results
        
    Returns:
        Dictionary containing analysis results
    """
    analysis = {}
    
    # Episode rewards analysis
    if 'episode_rewards' in results:
        rewards = results['episode_rewards']
        analysis['rewards'] = {
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'min': np.min(rewards),
            'max': np.max(rewards),
            'final': rewards[-1] if rewards else 0,
            'improvement': rewards[-1] - rewards[0] if len(rewards) > 1 else 0,
            'convergence_episode': calculate_convergence_episode(rewards),
            'stability_score': calculate_stability_score(rewards)
        }
    
    # Episode lengths analysis
    if 'episode_lengths' in results:
        lengths = results['episode_lengths']
        analysis['lengths'] = {
            'mean': np.mean(lengths),
            'std': np.std(lengths),
            'min': np.min(lengths),
            'max': np.max(lengths),
            'final': lengths[-1] if lengths else 0
        }
    
    # Loss analysis
    if 'episode_losses' in results:
        losses = results['episode_losses']
        analysis['losses'] = {
            'mean': np.mean(losses),
            'std': np.std(losses),
            'min': np.min(losses),
            'max': np.max(losses),
            'final': losses[-1] if losses else 0,
            'trend': 'decreasing' if len(losses) > 1 and losses[-1] < losses[0] else 'increasing'
        }
    
    # Evaluation scores analysis
    if 'eval_scores' in results:
        eval_scores = results['eval_scores']
        analysis['evaluation'] = {
            'mean': np.mean(eval_scores),
            'std': np.std(eval_scores),
            'min': np.min(eval_scores),
            'max': np.max(eval_scores),
            'final': eval_scores[-1] if eval_scores else 0,
            'improvement': eval_scores[-1] - eval_scores[0] if len(eval_scores) > 1 else 0
        }
    
    return analysis


def create_training_summary(
    results: Dict[str, List[float]],
    config: Dict[str, Any]
) -> str:
    """
    Create a text summary of training results.
    
    Args:
        results: Dictionary containing training results
        config: Training configuration
        
    Returns:
        Formatted text summary
    """
    analysis = analyze_training_progress(results)
    
    summary = f"""
DQN Training Summary
{'=' * 50}

Configuration:
- Environment: {config.get('env_name', 'Unknown')}
- Max Episodes: {config.get('max_episodes', 'Unknown')}
- Learning Rate: {config.get('learning_rate', 'Unknown')}
- Gamma: {config.get('gamma', 'Unknown')}
- Epsilon: {config.get('epsilon', 'Unknown')} -> {config.get('epsilon_min', 'Unknown')}

Training Results:
"""
    
    if 'rewards' in analysis:
        rewards_analysis = analysis['rewards']
        summary += f"""
Episode Rewards:
- Mean: {rewards_analysis['mean']:.2f}
- Std: {rewards_analysis['std']:.2f}
- Min: {rewards_analysis['min']:.2f}
- Max: {rewards_analysis['max']:.2f}
- Final: {rewards_analysis['final']:.2f}
- Improvement: {rewards_analysis['improvement']:.2f}
- Convergence Episode: {rewards_analysis['convergence_episode']}
- Stability Score: {rewards_analysis['stability_score']:.3f}
"""
    
    if 'evaluation' in analysis:
        eval_analysis = analysis['evaluation']
        summary += f"""
Evaluation Scores:
- Mean: {eval_analysis['mean']:.2f}
- Std: {eval_analysis['std']:.2f}
- Min: {eval_analysis['min']:.2f}
- Max: {eval_analysis['max']:.2f}
- Final: {eval_analysis['final']:.2f}
- Improvement: {eval_analysis['improvement']:.2f}
"""
    
    if 'losses' in analysis:
        losses_analysis = analysis['losses']
        summary += f"""
Training Losses:
- Mean: {losses_analysis['mean']:.4f}
- Std: {losses_analysis['std']:.4f}
- Min: {losses_analysis['min']:.4f}
- Max: {losses_analysis['max']:.4f}
- Final: {losses_analysis['final']:.4f}
- Trend: {losses_analysis['trend']}
"""
    
    return summary


def plot_training_summary(
    results: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Create a comprehensive training summary plot.
    
    Args:
        results: Dictionary containing training results
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DQN Training Summary', fontsize=16)
    
    # Plot episode rewards
    if 'episode_rewards' in results:
        rewards = results['episode_rewards']
        episodes = range(len(rewards))
        
        axes[0, 0].plot(episodes, rewards, 'b-', alpha=0.7, linewidth=1)
        
        # Add moving average
        if len(rewards) > 10:
            window = min(50, len(rewards) // 10)
            moving_avg = calculate_moving_average(rewards, window)
            axes[0, 0].plot(episodes, moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window})')
            axes[0, 0].legend()
        
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot episode lengths
    if 'episode_lengths' in results:
        lengths = results['episode_lengths']
        episodes = range(len(lengths))
        
        axes[0, 1].plot(episodes, lengths, 'g-', alpha=0.7, linewidth=1)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Length')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot losses
    if 'episode_losses' in results:
        losses = results['episode_losses']
        episodes = range(len(losses))
        
        axes[1, 0].plot(episodes, losses, 'r-', alpha=0.7, linewidth=1)
        axes[1, 0].set_title('Training Losses')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot evaluation scores
    if 'eval_scores' in results:
        eval_scores = results['eval_scores']
        eval_episodes = list(range(0, len(results.get('episode_rewards', [])), 50))
        
        if len(eval_episodes) == len(eval_scores):
            axes[1, 1].plot(eval_episodes, eval_scores, 'm-o', linewidth=2, markersize=4)
            axes[1, 1].set_title('Evaluation Scores')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()


def compare_agents(
    results_list: List[Dict[str, List[float]]],
    agent_names: List[str],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Compare multiple agents' training results.
    
    Args:
        results_list: List of training results dictionaries
        agent_names: List of agent names
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Agent Comparison', fontsize=16)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(agent_names)))
    
    # Plot episode rewards
    for i, (results, name) in enumerate(zip(results_list, agent_names)):
        if 'episode_rewards' in results:
            rewards = results['episode_rewards']
            episodes = range(len(rewards))
            
            # Plot moving average
            if len(rewards) > 10:
                window = min(50, len(rewards) // 10)
                moving_avg = calculate_moving_average(rewards, window)
                axes[0, 0].plot(episodes, moving_avg, color=colors[i], 
                               linewidth=2, label=name)
    
    axes[0, 0].set_title('Episode Rewards Comparison')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot evaluation scores
    for i, (results, name) in enumerate(zip(results_list, agent_names)):
        if 'eval_scores' in results:
            eval_scores = results['eval_scores']
            eval_episodes = list(range(0, len(results.get('episode_rewards', [])), 50))
            
            if len(eval_episodes) == len(eval_scores):
                axes[0, 1].plot(eval_episodes, eval_scores, color=colors[i], 
                               linewidth=2, marker='o', markersize=4, label=name)
    
    axes[0, 1].set_title('Evaluation Scores Comparison')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot final performance comparison
    final_rewards = []
    final_eval_scores = []
    
    for results in results_list:
        if 'episode_rewards' in results:
            final_rewards.append(np.mean(results['episode_rewards'][-10:]))
        else:
            final_rewards.append(0)
        
        if 'eval_scores' in results:
            final_eval_scores.append(results['eval_scores'][-1] if results['eval_scores'] else 0)
        else:
            final_eval_scores.append(0)
    
    x_pos = np.arange(len(agent_names))
    axes[1, 0].bar(x_pos, final_rewards, color=colors, alpha=0.7)
    axes[1, 0].set_title('Final Episode Rewards')
    axes[1, 0].set_xlabel('Agent')
    axes[1, 0].set_ylabel('Final Reward')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(agent_names, rotation=45)
    
    axes[1, 1].bar(x_pos, final_eval_scores, color=colors, alpha=0.7)
    axes[1, 1].set_title('Final Evaluation Scores')
    axes[1, 1].set_xlabel('Agent')
    axes[1, 1].set_ylabel('Final Score')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(agent_names, rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()


if __name__ == "__main__":
    # Test utility functions
    print("Testing DQN Utility Functions...")
    
    # Generate sample data
    np.random.seed(42)
    episodes = 200
    
    sample_results = {
        'episode_rewards': [10 + 5 * np.sin(i / 20) + np.random.normal(0, 2) for i in range(episodes)],
        'episode_lengths': [200 + 50 * np.sin(i / 30) + np.random.normal(0, 20) for i in range(episodes)],
        'episode_losses': [0.5 * np.exp(-i / 50) + np.random.normal(0, 0.01) for i in range(episodes)],
        'eval_scores': [max(10 + 5 * np.sin(i / 20) + np.random.normal(0, 2) for i in range(j+1)) 
                       for j in range(0, episodes, 20)]
    }
    
    # Test analysis
    analysis = analyze_training_progress(sample_results)
    print(f"Analysis completed: {len(analysis)} metrics analyzed")
    
    # Test summary
    config = {'env_name': 'TestEnv', 'max_episodes': episodes}
    summary = create_training_summary(sample_results, config)
    print("Training summary generated")
    
    # Test plotting
    plot_training_summary(sample_results, save_path="test_summary.png", show=False)
    print("Training summary plot created")
    
    # Clean up
    import os
    if os.path.exists("test_summary.png"):
        os.remove("test_summary.png")
    
    print("DQN Utility Functions tested successfully!")
