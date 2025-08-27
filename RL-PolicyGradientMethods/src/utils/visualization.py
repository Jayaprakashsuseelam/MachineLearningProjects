"""
Visualization utilities for policy gradient methods
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_training_progress(episode_rewards: List[float], episode_lengths: List[float], 
                          losses: Optional[List[float]] = None, 
                          title: str = "Training Progress", 
                          window_size: int = 100):
    """
    Plot training progress for policy gradient methods
    
    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        losses: Optional list of losses
        title: Plot title
        window_size: Window size for moving average
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.3, color='blue')
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        axes[0, 0].plot(range(window_size-1, len(episode_rewards)), moving_avg, 
                       color='red', linewidth=2, label=f'Moving Average ({window_size})')
        axes[0, 0].legend()
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Plot episode lengths
    axes[0, 1].plot(episode_lengths, alpha=0.3, color='green')
    if len(episode_lengths) >= window_size:
        moving_avg = np.convolve(episode_lengths, np.ones(window_size)/window_size, mode='valid')
        axes[0, 1].plot(range(window_size-1, len(episode_lengths)), moving_avg, 
                       color='red', linewidth=2, label=f'Moving Average ({window_size})')
        axes[0, 1].legend()
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True)
    
    # Plot losses if provided
    if losses is not None:
        axes[1, 0].plot(losses, color='orange')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
    else:
        axes[1, 0].text(0.5, 0.5, 'No loss data available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Training Loss')
    
    # Plot reward distribution
    axes[1, 1].hist(episode_rewards, bins=50, alpha=0.7, color='purple')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_policy_heatmap(policy_network, state_space: np.ndarray, 
                       action_space: np.ndarray, title: str = "Policy Heatmap"):
    """
    Plot policy heatmap for 2D state spaces
    
    Args:
        policy_network: Trained policy network
        state_space: 2D array of state values
        action_space: Array of possible actions
        title: Plot title
    """
    if state_space.shape[1] != 2:
        print("Policy heatmap only supports 2D state spaces")
        return
    
    # Create meshgrid
    x_min, x_max = state_space[:, 0].min(), state_space[:, 0].max()
    y_min, y_max = state_space[:, 1].min(), state_space[:, 1].max()
    
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    
    # Get policy probabilities
    states = np.column_stack([X.ravel(), Y.ravel()])
    states_tensor = torch.FloatTensor(states)
    
    with torch.no_grad():
        log_probs = policy_network(states_tensor)
        probs = torch.exp(log_probs)
    
    # Plot for each action
    fig, axes = plt.subplots(1, len(action_space), figsize=(5*len(action_space), 4))
    if len(action_space) == 1:
        axes = [axes]
    
    for i, action in enumerate(action_space):
        action_probs = probs[:, action].numpy().reshape(X.shape)
        
        im = axes[i].contourf(X, Y, action_probs, levels=20, cmap='viridis')
        axes[i].set_title(f'Action {action} Probability')
        axes[i].set_xlabel('State Dimension 1')
        axes[i].set_ylabel('State Dimension 2')
        plt.colorbar(im, ax=axes[i])
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_value_function(value_network, state_space: np.ndarray, 
                       title: str = "Value Function"):
    """
    Plot value function for 2D state spaces
    
    Args:
        value_network: Trained value network
        state_space: 2D array of state values
        title: Plot title
    """
    if state_space.shape[1] != 2:
        print("Value function plot only supports 2D state spaces")
        return
    
    # Create meshgrid
    x_min, x_max = state_space[:, 0].min(), state_space[:, 0].max()
    y_min, y_max = state_space[:, 1].min(), state_space[:, 1].max()
    
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    
    # Get value estimates
    states = np.column_stack([X.ravel(), Y.ravel()])
    states_tensor = torch.FloatTensor(states)
    
    with torch.no_grad():
        values = value_network(states_tensor).numpy().reshape(X.shape)
    
    # Plot value function
    plt.figure(figsize=(10, 8))
    im = plt.contourf(X, Y, values, levels=20, cmap='viridis')
    plt.colorbar(im, label='Value')
    plt.title(title)
    plt.xlabel('State Dimension 1')
    plt.ylabel('State Dimension 2')
    plt.show()


def plot_algorithm_comparison(results: Dict[str, Dict[str, List[float]]], 
                            title: str = "Algorithm Comparison"):
    """
    Plot comparison of different algorithms
    
    Args:
        results: Dictionary with algorithm names as keys and metrics as values
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot episode rewards
    for algo_name, metrics in results.items():
        if 'episode_rewards' in metrics:
            axes[0, 0].plot(metrics['episode_rewards'], label=algo_name, alpha=0.7)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot episode lengths
    for algo_name, metrics in results.items():
        if 'episode_lengths' in metrics:
            axes[0, 1].plot(metrics['episode_lengths'], label=algo_name, alpha=0.7)
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot losses
    for algo_name, metrics in results.items():
        if 'losses' in metrics:
            axes[1, 0].plot(metrics['losses'], label=algo_name, alpha=0.7)
    axes[1, 0].set_title('Training Losses')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot final performance comparison
    final_rewards = []
    algo_names = []
    for algo_name, metrics in results.items():
        if 'episode_rewards' in metrics and len(metrics['episode_rewards']) > 0:
            final_rewards.append(metrics['episode_rewards'][-1])
            algo_names.append(algo_name)
    
    if final_rewards:
        axes[1, 1].bar(algo_names, final_rewards)
        axes[1, 1].set_title('Final Performance')
        axes[1, 1].set_ylabel('Final Reward')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_trading_performance(portfolio_history: List[float], prices: List[float], 
                           trades: List[Dict], title: str = "Trading Performance"):
    """
    Plot trading performance
    
    Args:
        portfolio_history: List of portfolio values over time
        prices: List of stock prices over time
        trades: List of trade dictionaries
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot portfolio value
    ax1.plot(portfolio_history, label='Portfolio Value', linewidth=2)
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot stock price with trades
    ax2.plot(prices[:len(portfolio_history)], label='Stock Price', alpha=0.7)
    
    # Mark trades
    for trade in trades:
        if trade['action'] == 'buy':
            ax2.scatter(trade['step'], trade['price'], color='green', 
                       marker='^', s=100, label='Buy' if trade == trades[0] else "")
        else:
            ax2.scatter(trade['step'], trade['price'], color='red', 
                       marker='v', s=100, label='Sell' if trade == trades[0] else "")
    
    ax2.set_title('Stock Price with Trading Actions')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def create_interactive_plot(episode_rewards: List[float], episode_lengths: List[float],
                          title: str = "Interactive Training Progress"):
    """
    Create interactive plot using Plotly
    
    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        title: Plot title
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Episode Rewards', 'Episode Lengths'),
        vertical_spacing=0.1
    )
    
    # Add episode rewards
    fig.add_trace(
        go.Scatter(y=episode_rewards, mode='lines', name='Rewards', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Add episode lengths
    fig.add_trace(
        go.Scatter(y=episode_lengths, mode='lines', name='Lengths', line=dict(color='green')),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=600,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Episode", row=2, col=1)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="Steps", row=2, col=1)
    
    fig.show()


def plot_learning_curves(episode_rewards: List[float], window_size: int = 100,
                        title: str = "Learning Curves"):
    """
    Plot learning curves with confidence intervals
    
    Args:
        episode_rewards: List of episode rewards
        window_size: Window size for moving average
        title: Plot title
    """
    if len(episode_rewards) < window_size:
        print(f"Not enough data for window size {window_size}")
        return
    
    # Calculate moving average and standard deviation
    moving_avg = []
    moving_std = []
    
    for i in range(window_size, len(episode_rewards) + 1):
        window_data = episode_rewards[i-window_size:i]
        moving_avg.append(np.mean(window_data))
        moving_std.append(np.std(window_data))
    
    x = range(window_size, len(episode_rewards) + 1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(x, moving_avg, label=f'Moving Average ({window_size})', linewidth=2)
    plt.fill_between(x, 
                     np.array(moving_avg) - np.array(moving_std),
                     np.array(moving_avg) + np.array(moving_std),
                     alpha=0.3, label=f'±1 Std Dev')
    
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.show()
