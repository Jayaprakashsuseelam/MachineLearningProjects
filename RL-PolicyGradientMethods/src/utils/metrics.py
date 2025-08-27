"""
Metrics and evaluation utilities for policy gradient methods
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import torch
from collections import deque


def calculate_metrics(episode_rewards: List[float], episode_lengths: List[float],
                     window_size: int = 100) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics
    
    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        window_size: Window size for moving averages
    
    Returns:
        Dictionary of calculated metrics
    """
    if not episode_rewards:
        return {}
    
    metrics = {}
    
    # Basic statistics
    metrics['total_episodes'] = len(episode_rewards)
    metrics['mean_reward'] = np.mean(episode_rewards)
    metrics['std_reward'] = np.std(episode_rewards)
    metrics['min_reward'] = np.min(episode_rewards)
    metrics['max_reward'] = np.max(episode_rewards)
    
    metrics['mean_length'] = np.mean(episode_lengths)
    metrics['std_length'] = np.std(episode_lengths)
    metrics['min_length'] = np.min(episode_lengths)
    metrics['max_length'] = np.max(episode_lengths)
    
    # Learning progress metrics
    if len(episode_rewards) >= window_size:
        # Recent performance
        recent_rewards = episode_rewards[-window_size:]
        metrics['recent_mean_reward'] = np.mean(recent_rewards)
        metrics['recent_std_reward'] = np.std(recent_rewards)
        
        # Learning trend
        first_half = episode_rewards[:len(episode_rewards)//2]
        second_half = episode_rewards[len(episode_rewards)//2:]
        metrics['learning_improvement'] = np.mean(second_half) - np.mean(first_half)
        
        # Consistency
        metrics['consistency'] = 1.0 / (1.0 + metrics['std_reward'])
    
    # Success rate (episodes above threshold)
    reward_threshold = np.percentile(episode_rewards, 75)  # Top 25% threshold
    metrics['success_rate'] = np.mean(np.array(episode_rewards) >= reward_threshold)
    
    # Stability metrics
    if len(episode_rewards) > 1:
        reward_changes = np.diff(episode_rewards)
        metrics['reward_volatility'] = np.std(reward_changes)
        metrics['reward_trend'] = np.mean(reward_changes)
    
    return metrics


def evaluate_policy(agent, env, num_episodes: int = 10, max_steps: int = 1000,
                   deterministic: bool = True) -> Dict[str, float]:
    """
    Evaluate a trained policy
    
    Args:
        agent: Trained agent
        env: Environment to evaluate on
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        deterministic: Whether to use deterministic policy
    
    Returns:
        Dictionary of evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            action = agent.select_action(state, deterministic=deterministic)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
    
    # Calculate evaluation metrics
    metrics = calculate_metrics(episode_rewards, episode_lengths)
    metrics['evaluation_episodes'] = num_episodes
    
    return metrics


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio for trading performance
    
    Args:
        returns: List of returns
        risk_free_rate: Risk-free rate
    
    Returns:
        Sharpe ratio
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    returns = np.array(returns)
    excess_returns = returns - risk_free_rate
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(excess_returns)


def calculate_max_drawdown(portfolio_values: List[float]) -> float:
    """
    Calculate maximum drawdown
    
    Args:
        portfolio_values: List of portfolio values over time
    
    Returns:
        Maximum drawdown as a percentage
    """
    if not portfolio_values:
        return 0.0
    
    portfolio_values = np.array(portfolio_values)
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    
    return np.min(drawdown)


def calculate_win_rate(trades: List[Dict[str, Any]]) -> float:
    """
    Calculate win rate for trading
    
    Args:
        trades: List of trade dictionaries
    
    Returns:
        Win rate as a percentage
    """
    if not trades:
        return 0.0
    
    # Count winning trades (simplified - assumes sell trades are profitable)
    winning_trades = 0
    total_trades = 0
    
    for trade in trades:
        if trade.get('action') == 'sell':
            total_trades += 1
            # Simple profitability check
            if trade.get('proceeds', 0) > trade.get('cost', 0):
                winning_trades += 1
    
    return winning_trades / total_trades if total_trades > 0 else 0.0


def calculate_episode_statistics(episode_rewards: List[float], 
                               episode_lengths: List[float]) -> Dict[str, Any]:
    """
    Calculate detailed episode statistics
    
    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
    
    Returns:
        Dictionary of episode statistics
    """
    if not episode_rewards:
        return {}
    
    stats = {}
    
    # Reward statistics
    stats['reward_stats'] = {
        'mean': np.mean(episode_rewards),
        'median': np.median(episode_rewards),
        'std': np.std(episode_rewards),
        'min': np.min(episode_rewards),
        'max': np.max(episode_rewards),
        'q25': np.percentile(episode_rewards, 25),
        'q75': np.percentile(episode_rewards, 75)
    }
    
    # Length statistics
    stats['length_stats'] = {
        'mean': np.mean(episode_lengths),
        'median': np.median(episode_lengths),
        'std': np.std(episode_lengths),
        'min': np.min(episode_lengths),
        'max': np.max(episode_lengths),
        'q25': np.percentile(episode_lengths, 25),
        'q75': np.percentile(episode_lengths, 75)
    }
    
    # Correlation
    if len(episode_rewards) > 1:
        stats['reward_length_correlation'] = np.corrcoef(episode_rewards, episode_lengths)[0, 1]
    
    return stats


def calculate_learning_metrics(episode_rewards: List[float], 
                             window_size: int = 100) -> Dict[str, float]:
    """
    Calculate learning-specific metrics
    
    Args:
        episode_rewards: List of episode rewards
        window_size: Window size for analysis
    
    Returns:
        Dictionary of learning metrics
    """
    if len(episode_rewards) < window_size:
        return {}
    
    metrics = {}
    
    # Learning rate (improvement over time)
    first_window = episode_rewards[:window_size]
    last_window = episode_rewards[-window_size:]
    metrics['learning_rate'] = np.mean(last_window) - np.mean(first_window)
    
    # Convergence metrics
    recent_rewards = episode_rewards[-window_size:]
    metrics['convergence_std'] = np.std(recent_rewards)
    metrics['convergence_mean'] = np.mean(recent_rewards)
    
    # Stability (how much the performance varies)
    if len(episode_rewards) > window_size:
        rolling_means = []
        for i in range(window_size, len(episode_rewards)):
            rolling_means.append(np.mean(episode_rewards[i-window_size:i]))
        
        metrics['stability'] = 1.0 / (1.0 + np.std(rolling_means))
    
    # Sample efficiency (how quickly it learns)
    target_performance = np.percentile(episode_rewards, 90)  # Top 10% performance
    sample_efficiency = 0
    for i, reward in enumerate(episode_rewards):
        if reward >= target_performance:
            sample_efficiency = i
            break
    
    metrics['sample_efficiency'] = sample_efficiency
    
    return metrics


def compare_algorithms(results: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
    """
    Compare multiple algorithms
    
    Args:
        results: Dictionary with algorithm names as keys and metrics as values
    
    Returns:
        Dictionary of comparison results
    """
    comparison = {}
    
    for algo_name, metrics in results.items():
        if 'episode_rewards' in metrics:
            comparison[algo_name] = calculate_metrics(
                metrics['episode_rewards'], 
                metrics.get('episode_lengths', [])
            )
    
    # Find best algorithm for each metric
    if comparison:
        best_algorithms = {}
        for metric in ['mean_reward', 'recent_mean_reward', 'success_rate', 'consistency']:
            best_algo = max(comparison.keys(), 
                          key=lambda x: comparison[x].get(metric, -np.inf))
            best_algorithms[metric] = best_algo
        
        comparison['best_algorithms'] = best_algorithms
    
    return comparison


def print_metrics(metrics: Dict[str, float], title: str = "Performance Metrics"):
    """
    Print metrics in a formatted way
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the metrics
    """
    print(f"\n{title}")
    print("=" * len(title))
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print()


def save_metrics(metrics: Dict[str, float], filepath: str):
    """
    Save metrics to a file
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save the metrics
    """
    import json
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {filepath}")
