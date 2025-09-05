"""
Plotting Utilities for DQN Analysis

This module provides specialized plotting functions for analyzing
DQN training results and agent performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class DQNPlotter:
    """
    Specialized plotting class for DQN analysis and visualization.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize the DQN plotter.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        sns.set_palette("husl")
        
        # Set default figure size
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def plot_learning_curves(
        self,
        episode_rewards: List[float],
        episode_lengths: List[float],
        episode_losses: List[float],
        epsilons: List[float],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot comprehensive learning curves.
        
        Args:
            episode_rewards: List of episode rewards
            episode_lengths: List of episode lengths
            episode_losses: List of episode losses
            epsilons: List of epsilon values
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        episodes = range(len(episode_rewards))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DQN Learning Curves', fontsize=16)
        
        # Plot episode rewards
        axes[0, 0].plot(episodes, episode_rewards, 'b-', linewidth=2, alpha=0.7)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add moving average
        if len(episode_rewards) > 10:
            window = min(50, len(episode_rewards) // 10)
            moving_avg = pd.Series(episode_rewards).rolling(window=window).mean()
            axes[0, 0].plot(episodes, moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window})')
            axes[0, 0].legend()
        
        # Plot episode lengths
        axes[0, 1].plot(episodes, episode_lengths, 'g-', linewidth=2, alpha=0.7)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Length')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot losses
        axes[1, 0].plot(episodes, episode_losses, 'r-', linewidth=2, alpha=0.7)
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # Plot epsilon decay
        axes[1, 1].plot(episodes, epsilons, 'm-', linewidth=2, alpha=0.7)
        axes[1, 1].set_title('Epsilon Decay')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
    
    def plot_evaluation_comparison(
        self,
        eval_scores: List[float],
        eval_episodes: List[int],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot evaluation scores over time.
        
        Args:
            eval_scores: List of evaluation scores
            eval_episodes: List of evaluation episode numbers
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot evaluation scores
        ax.plot(eval_episodes, eval_scores, 'b-o', linewidth=2, markersize=6, label='Evaluation Score')
        
        # Add trend line
        if len(eval_scores) > 2:
            z = np.polyfit(eval_episodes, eval_scores, 1)
            p = np.poly1d(z)
            ax.plot(eval_episodes, p(eval_episodes), 'r--', alpha=0.7, label='Trend Line')
        
        # Highlight best score
        best_idx = np.argmax(eval_scores)
        ax.plot(eval_episodes[best_idx], eval_scores[best_idx], 'ro', markersize=10, label='Best Score')
        
        ax.set_title('DQN Evaluation Progress', fontsize=14)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Evaluation Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
    
    def plot_hyperparameter_sensitivity(
        self,
        results: Dict[str, Dict[str, List[float]]],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot hyperparameter sensitivity analysis.
        
        Args:
            results: Dictionary with hyperparameter values as keys and results as values
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Hyperparameter Sensitivity Analysis', fontsize=16)
        
        # Extract hyperparameters and results
        hyperparams = list(results.keys())
        final_scores = [max(results[hp]['eval_scores']) for hp in hyperparams]
        convergence_episodes = [self._find_convergence_episode(results[hp]['episode_rewards']) for hp in hyperparams]
        stability_scores = [self._calculate_stability_score(results[hp]['episode_rewards']) for hp in hyperparams]
        
        # Plot final scores
        axes[0, 0].bar(range(len(hyperparams)), final_scores, alpha=0.7)
        axes[0, 0].set_title('Final Evaluation Scores')
        axes[0, 0].set_xlabel('Hyperparameter Configuration')
        axes[0, 0].set_ylabel('Final Score')
        axes[0, 0].set_xticks(range(len(hyperparams)))
        axes[0, 0].set_xticklabels(hyperparams, rotation=45)
        
        # Plot convergence episodes
        axes[0, 1].bar(range(len(hyperparams)), convergence_episodes, alpha=0.7, color='green')
        axes[0, 1].set_title('Convergence Episodes')
        axes[0, 1].set_xlabel('Hyperparameter Configuration')
        axes[0, 1].set_ylabel('Convergence Episode')
        axes[0, 1].set_xticks(range(len(hyperparams)))
        axes[0, 1].set_xticklabels(hyperparams, rotation=45)
        
        # Plot stability scores
        axes[1, 0].bar(range(len(hyperparams)), stability_scores, alpha=0.7, color='red')
        axes[1, 0].set_title('Stability Scores')
        axes[1, 0].set_xlabel('Hyperparameter Configuration')
        axes[1, 0].set_ylabel('Stability Score')
        axes[1, 0].set_xticks(range(len(hyperparams)))
        axes[1, 0].set_xticklabels(hyperparams, rotation=45)
        
        # Plot learning curves comparison
        for i, hp in enumerate(hyperparams):
            episodes = range(len(results[hp]['episode_rewards']))
            axes[1, 1].plot(episodes, results[hp]['episode_rewards'], 
                           alpha=0.7, label=f'{hp}', linewidth=2)
        
        axes[1, 1].set_title('Learning Curves Comparison')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Episode Reward')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
    
    def plot_agent_performance(
        self,
        test_scores: List[float],
        episode_lengths: List[int],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot agent performance analysis.
        
        Args:
            test_scores: List of test episode scores
            episode_lengths: List of test episode lengths
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Agent Performance Analysis', fontsize=16)
        
        # Score distribution
        axes[0, 0].hist(test_scores, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Score Distribution')
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(test_scores), color='red', linestyle='--', label=f'Mean: {np.mean(test_scores):.2f}')
        axes[0, 0].legend()
        
        # Length distribution
        axes[0, 1].hist(episode_lengths, bins=20, alpha=0.7, edgecolor='black', color='green')
        axes[0, 1].set_title('Episode Length Distribution')
        axes[0, 1].set_xlabel('Length')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(episode_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(episode_lengths):.1f}')
        axes[0, 1].legend()
        
        # Score vs Length scatter
        axes[1, 0].scatter(episode_lengths, test_scores, alpha=0.6)
        axes[1, 0].set_title('Score vs Episode Length')
        axes[1, 0].set_xlabel('Episode Length')
        axes[1, 0].set_ylabel('Score')
        
        # Add correlation
        correlation = np.corrcoef(episode_lengths, test_scores)[0, 1]
        axes[1, 0].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=axes[1, 0].transAxes, verticalalignment='top')
        
        # Performance statistics
        stats_text = f"""
        Mean Score: {np.mean(test_scores):.2f}
        Std Score: {np.std(test_scores):.2f}
        Min Score: {np.min(test_scores):.2f}
        Max Score: {np.max(test_scores):.2f}
        
        Mean Length: {np.mean(episode_lengths):.1f}
        Std Length: {np.std(episode_lengths):.1f}
        """
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Performance Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
    
    def plot_q_value_analysis(
        self,
        q_values_history: List[float],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot Q-value analysis.
        
        Args:
            q_values_history: List of Q-values over time
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Q-Value Analysis', fontsize=16)
        
        episodes = range(len(q_values_history))
        
        # Q-values over time
        axes[0, 0].plot(episodes, q_values_history, 'b-', linewidth=2, alpha=0.7)
        axes[0, 0].set_title('Q-Values Over Time')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Q-Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-value distribution
        axes[0, 1].hist(q_values_history, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Q-Value Distribution')
        axes[0, 1].set_xlabel('Q-Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(q_values_history), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(q_values_history):.3f}')
        axes[0, 1].legend()
        
        # Q-value rolling statistics
        if len(q_values_history) > 10:
            window = min(50, len(q_values_history) // 10)
            rolling_mean = pd.Series(q_values_history).rolling(window=window).mean()
            rolling_std = pd.Series(q_values_history).rolling(window=window).std()
            
            axes[1, 0].plot(episodes, q_values_history, 'b-', alpha=0.3, linewidth=1)
            axes[1, 0].plot(episodes, rolling_mean, 'r-', linewidth=2, label=f'Rolling Mean (window={window})')
            axes[1, 0].fill_between(episodes, rolling_mean - rolling_std, rolling_mean + rolling_std, 
                                  alpha=0.2, color='red', label='±1 Std')
            axes[1, 0].set_title('Q-Value Rolling Statistics')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Q-Value')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Q-value statistics
        stats_text = f"""
        Mean Q-Value: {np.mean(q_values_history):.4f}
        Std Q-Value: {np.std(q_values_history):.4f}
        Min Q-Value: {np.min(q_values_history):.4f}
        Max Q-Value: {np.max(q_values_history):.4f}
        
        Q-Value Range: {np.max(q_values_history) - np.min(q_values_history):.4f}
        Q-Value CV: {np.std(q_values_history) / np.mean(q_values_history):.4f}
        """
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Q-Value Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
    
    def create_interactive_dashboard(
        self,
        training_data: Dict[str, List[float]],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Create an interactive dashboard using Plotly.
        
        Args:
            training_data: Dictionary containing training data
            save_path: Path to save the dashboard
            show: Whether to display the dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Episode Rewards', 'Episode Lengths',
                'Training Loss', 'Epsilon Decay',
                'Q-Values', 'Memory Usage'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        episodes = range(len(training_data.get('episode_rewards', [])))
        
        # Plot episode rewards
        if 'episode_rewards' in training_data:
            fig.add_trace(
                go.Scatter(x=list(episodes), y=training_data['episode_rewards'],
                          mode='lines', name='Episode Rewards', line=dict(color='blue')),
                row=1, col=1
            )
        
        # Plot episode lengths
        if 'episode_lengths' in training_data:
            fig.add_trace(
                go.Scatter(x=list(episodes), y=training_data['episode_lengths'],
                          mode='lines', name='Episode Lengths', line=dict(color='green')),
                row=1, col=2
            )
        
        # Plot losses
        if 'episode_losses' in training_data:
            fig.add_trace(
                go.Scatter(x=list(episodes), y=training_data['episode_losses'],
                          mode='lines', name='Training Loss', line=dict(color='red')),
                row=2, col=1
            )
        
        # Plot epsilon decay
        if 'epsilons' in training_data:
            fig.add_trace(
                go.Scatter(x=list(episodes), y=training_data['epsilons'],
                          mode='lines', name='Epsilon', line=dict(color='purple')),
                row=2, col=2
            )
        
        # Plot Q-values
        if 'q_values' in training_data:
            fig.add_trace(
                go.Scatter(x=list(episodes), y=training_data['q_values'],
                          mode='lines', name='Q-Values', line=dict(color='orange')),
                row=3, col=1
            )
        
        # Plot memory usage
        if 'memory_sizes' in training_data:
            fig.add_trace(
                go.Scatter(x=list(episodes), y=training_data['memory_sizes'],
                          mode='lines', name='Memory Size', line=dict(color='brown')),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="DQN Training Interactive Dashboard",
            title_x=0.5,
            height=900,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Episode", row=3, col=1)
        fig.update_xaxes(title_text="Episode", row=3, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        if show:
            fig.show()
    
    def _find_convergence_episode(self, rewards: List[float], window: int = 50) -> Optional[int]:
        """Find the episode where learning converged."""
        if len(rewards) < window:
            return None
        
        # Calculate rolling standard deviation
        rolling_std = pd.Series(rewards).rolling(window=window).std()
        
        # Find where standard deviation becomes small
        threshold = rolling_std.mean() * 0.1
        convergence_idx = rolling_std[rolling_std < threshold].first_valid_index()
        
        return convergence_idx
    
    def _calculate_stability_score(self, rewards: List[float]) -> float:
        """Calculate a stability score based on reward variance."""
        if len(rewards) < 10:
            return 0.0
        
        # Use the last 20% of episodes for stability calculation
        last_portion = rewards[-max(10, len(rewards) // 5):]
        stability_score = 1.0 / (1.0 + np.std(last_portion))
        
        return stability_score


if __name__ == "__main__":
    # Test the plotting utilities
    print("Testing DQN Plotting Utilities...")
    
    # Generate sample data
    np.random.seed(42)
    episodes = 200
    
    # Sample training data
    episode_rewards = [10 + 5 * np.sin(i / 20) + np.random.normal(0, 2) for i in range(episodes)]
    episode_lengths = [200 + 50 * np.sin(i / 30) + np.random.normal(0, 20) for i in range(episodes)]
    episode_losses = [0.5 * np.exp(-i / 50) + np.random.normal(0, 0.01) for i in range(episodes)]
    epsilons = [max(0.01, 1.0 * (0.995 ** i)) for i in range(episodes)]
    q_values = [0.1 + 0.05 * np.sin(i / 25) + np.random.normal(0, 0.01) for i in range(episodes)]
    
    # Sample evaluation data
    eval_episodes = list(range(0, episodes, 20))
    eval_scores = [max(episode_rewards[:i+1]) + np.random.normal(0, 1) for i in eval_episodes]
    
    # Test plotter
    plotter = DQNPlotter()
    
    # Test learning curves
    plotter.plot_learning_curves(
        episode_rewards, episode_lengths, episode_losses, epsilons,
        save_path="test_learning_curves.png", show=False
    )
    
    # Test evaluation comparison
    plotter.plot_evaluation_comparison(
        eval_scores, eval_episodes,
        save_path="test_evaluation.png", show=False
    )
    
    # Test Q-value analysis
    plotter.plot_q_value_analysis(
        q_values,
        save_path="test_q_values.png", show=False
    )
    
    # Test interactive dashboard
    training_data = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_losses': episode_losses,
        'epsilons': epsilons,
        'q_values': q_values,
        'memory_sizes': [min(10000, i * 10) for i in range(episodes)]
    }
    
    plotter.create_interactive_dashboard(
        training_data,
        save_path="test_dashboard.html", show=False
    )
    
    # Clean up
    import os
    for file in ["test_learning_curves.png", "test_evaluation.png", 
                 "test_q_values.png", "test_dashboard.html"]:
        if os.path.exists(file):
            os.remove(file)
    
    print("DQN Plotting Utilities tested successfully!")
