"""
Visualization and Monitoring Tools for DQN Training

This module provides comprehensive visualization and monitoring tools
for tracking DQN training progress and analyzing results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class TrainingMonitor:
    """
    Training monitor for tracking and visualizing DQN training progress.
    """
    
    def __init__(self, log_path: str = "logs"):
        """
        Initialize the training monitor.
        
        Args:
            log_path: Path to training logs
        """
        self.log_path = log_path
        self.training_data = None
        self.eval_data = None
        self.config = None
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_logs(self) -> None:
        """Load training logs from JSON files."""
        # Load training log
        training_log_path = os.path.join(self.log_path, "training_log.json")
        if os.path.exists(training_log_path):
            with open(training_log_path, 'r') as f:
                self.training_data = json.load(f)
        
        # Load evaluation log
        eval_log_path = os.path.join(self.log_path, "eval_log.json")
        if os.path.exists(eval_log_path):
            with open(eval_log_path, 'r') as f:
                self.eval_data = json.load(f)
        
        # Load configuration
        config_path = os.path.join(self.log_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
    
    def plot_training_curves(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot training curves including rewards, losses, and epsilon.
        
        Args:
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        if self.training_data is None:
            self.load_logs()
        
        if self.training_data is None:
            print("No training data found!")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.training_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DQN Training Progress', fontsize=16)
        
        # Plot episode rewards
        axes[0, 0].plot(df['episode'], df['avg_reward'], 'b-', linewidth=2)
        axes[0, 0].set_title('Average Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot episode lengths
        axes[0, 1].plot(df['episode'], df['avg_length'], 'g-', linewidth=2)
        axes[0, 1].set_title('Average Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Length')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot losses
        axes[1, 0].plot(df['episode'], df['avg_loss'], 'r-', linewidth=2)
        axes[1, 0].set_title('Average Training Loss')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Average Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot epsilon decay
        axes[1, 1].plot(df['episode'], df['epsilon'], 'm-', linewidth=2)
        axes[1, 1].set_title('Epsilon Decay')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
    
    def plot_evaluation_curves(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot evaluation curves.
        
        Args:
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        if self.eval_data is None:
            self.load_logs()
        
        if self.eval_data is None:
            print("No evaluation data found!")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.eval_data)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot evaluation scores
        ax.plot(df['episode'], df['eval_score'], 'b-o', linewidth=2, markersize=4)
        ax.plot(df['episode'], df['best_eval_score'], 'r--', linewidth=2, alpha=0.7)
        
        ax.set_title('DQN Evaluation Progress', fontsize=14)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Evaluation Score')
        ax.legend(['Evaluation Score', 'Best Score'])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
    
    def plot_comprehensive_analysis(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Create a comprehensive analysis plot with multiple metrics.
        
        Args:
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        if self.training_data is None or self.eval_data is None:
            self.load_logs()
        
        if self.training_data is None:
            print("No training data found!")
            return
        
        # Convert to DataFrames
        train_df = pd.DataFrame(self.training_data)
        eval_df = pd.DataFrame(self.eval_data) if self.eval_data else pd.DataFrame()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Episode Rewards', 'Episode Lengths',
                'Training Loss', 'Epsilon Decay',
                'Evaluation Scores', 'Memory Usage'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot episode rewards
        fig.add_trace(
            go.Scatter(x=train_df['episode'], y=train_df['avg_reward'],
                      mode='lines', name='Avg Reward', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Plot episode lengths
        fig.add_trace(
            go.Scatter(x=train_df['episode'], y=train_df['avg_length'],
                      mode='lines', name='Avg Length', line=dict(color='green')),
            row=1, col=2
        )
        
        # Plot losses
        fig.add_trace(
            go.Scatter(x=train_df['episode'], y=train_df['avg_loss'],
                      mode='lines', name='Avg Loss', line=dict(color='red')),
            row=2, col=1
        )
        
        # Plot epsilon decay
        fig.add_trace(
            go.Scatter(x=train_df['episode'], y=train_df['epsilon'],
                      mode='lines', name='Epsilon', line=dict(color='purple')),
            row=2, col=2
        )
        
        # Plot evaluation scores
        if not eval_df.empty:
            fig.add_trace(
                go.Scatter(x=eval_df['episode'], y=eval_df['eval_score'],
                          mode='lines+markers', name='Eval Score', line=dict(color='orange')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=eval_df['episode'], y=eval_df['best_eval_score'],
                          mode='lines', name='Best Score', line=dict(color='red', dash='dash')),
                row=3, col=1
            )
        
        # Plot memory usage
        fig.add_trace(
            go.Scatter(x=train_df['episode'], y=train_df['memory_size'],
                      mode='lines', name='Memory Size', line=dict(color='brown')),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="DQN Training Comprehensive Analysis",
            title_x=0.5,
            height=900,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Episode", row=3, col=1)
        fig.update_xaxes(title_text="Episode", row=3, col=2)
        fig.update_yaxes(title_text="Score", row=3, col=1)
        fig.update_yaxes(title_text="Memory Size", row=3, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        if show:
            fig.show()
    
    def plot_learning_curves_smooth(
        self,
        window_size: int = 50,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot smoothed learning curves.
        
        Args:
            window_size: Window size for smoothing
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        if self.training_data is None:
            self.load_logs()
        
        if self.training_data is None:
            print("No training data found!")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.training_data)
        
        # Smooth the data
        df['smoothed_reward'] = df['avg_reward'].rolling(window=window_size, center=True).mean()
        df['smoothed_length'] = df['avg_length'].rolling(window=window_size, center=True).mean()
        df['smoothed_loss'] = df['avg_loss'].rolling(window=window_size, center=True).mean()
        
        # Create plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot smoothed rewards
        axes[0].plot(df['episode'], df['avg_reward'], 'b-', alpha=0.3, linewidth=1)
        axes[0].plot(df['episode'], df['smoothed_reward'], 'b-', linewidth=2)
        axes[0].set_title(f'Smoothed Episode Rewards (window={window_size})')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Average Reward')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(['Raw', 'Smoothed'])
        
        # Plot smoothed lengths
        axes[1].plot(df['episode'], df['avg_length'], 'g-', alpha=0.3, linewidth=1)
        axes[1].plot(df['episode'], df['smoothed_length'], 'g-', linewidth=2)
        axes[1].set_title(f'Smoothed Episode Lengths (window={window_size})')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Average Length')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(['Raw', 'Smoothed'])
        
        # Plot smoothed losses
        axes[2].plot(df['episode'], df['avg_loss'], 'r-', alpha=0.3, linewidth=1)
        axes[2].plot(df['episode'], df['smoothed_loss'], 'r-', linewidth=2)
        axes[2].set_title(f'Smoothed Training Loss (window={window_size})')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Average Loss')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(['Raw', 'Smoothed'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
    
    def generate_training_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive training report.
        
        Returns:
            Dictionary containing training statistics
        """
        if self.training_data is None or self.eval_data is None:
            self.load_logs()
        
        if self.training_data is None:
            return {"error": "No training data found!"}
        
        # Convert to DataFrames
        train_df = pd.DataFrame(self.training_data)
        eval_df = pd.DataFrame(self.eval_data) if self.eval_data else pd.DataFrame()
        
        # Calculate statistics
        report = {
            "training_summary": {
                "total_episodes": len(train_df),
                "total_steps": train_df['total_steps'].iloc[-1] if 'total_steps' in train_df.columns else 0,
                "final_avg_reward": train_df['avg_reward'].iloc[-1],
                "best_avg_reward": train_df['avg_reward'].max(),
                "final_avg_length": train_df['avg_length'].iloc[-1],
                "final_epsilon": train_df['epsilon'].iloc[-1],
                "final_memory_size": train_df['memory_size'].iloc[-1] if 'memory_size' in train_df.columns else 0
            },
            "learning_metrics": {
                "reward_improvement": train_df['avg_reward'].iloc[-1] - train_df['avg_reward'].iloc[0],
                "length_improvement": train_df['avg_length'].iloc[-1] - train_df['avg_length'].iloc[0],
                "convergence_episode": self._find_convergence_episode(train_df['avg_reward']),
                "stability_score": self._calculate_stability_score(train_df['avg_reward'])
            }
        }
        
        if not eval_df.empty:
            report["evaluation_summary"] = {
                "total_evaluations": len(eval_df),
                "final_eval_score": eval_df['eval_score'].iloc[-1],
                "best_eval_score": eval_df['best_eval_score'].iloc[-1],
                "eval_improvement": eval_df['eval_score'].iloc[-1] - eval_df['eval_score'].iloc[0]
            }
        
        if self.config:
            report["configuration"] = self.config
        
        return report
    
    def _find_convergence_episode(self, rewards: pd.Series, window: int = 50) -> Optional[int]:
        """Find the episode where learning converged."""
        if len(rewards) < window:
            return None
        
        # Calculate rolling standard deviation
        rolling_std = rewards.rolling(window=window).std()
        
        # Find where standard deviation becomes small
        threshold = rolling_std.mean() * 0.1
        convergence_idx = rolling_std[rolling_std < threshold].first_valid_index()
        
        return convergence_idx
    
    def _calculate_stability_score(self, rewards: pd.Series) -> float:
        """Calculate a stability score based on reward variance."""
        if len(rewards) < 10:
            return 0.0
        
        # Use the last 20% of episodes for stability calculation
        last_portion = rewards.tail(max(10, len(rewards) // 5))
        stability_score = 1.0 / (1.0 + last_portion.std())
        
        return stability_score
    
    def save_report(self, filepath: str) -> None:
        """Save training report to JSON file."""
        report = self.generate_training_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def print_summary(self) -> None:
        """Print a summary of training results."""
        report = self.generate_training_report()
        
        if "error" in report:
            print(report["error"])
            return
        
        print("=" * 60)
        print("DQN TRAINING SUMMARY")
        print("=" * 60)
        
        training_summary = report["training_summary"]
        print(f"Total Episodes: {training_summary['total_episodes']}")
        print(f"Total Steps: {training_summary['total_steps']}")
        print(f"Final Average Reward: {training_summary['final_avg_reward']:.2f}")
        print(f"Best Average Reward: {training_summary['best_avg_reward']:.2f}")
        print(f"Final Average Length: {training_summary['final_avg_length']:.1f}")
        print(f"Final Epsilon: {training_summary['final_epsilon']:.3f}")
        
        learning_metrics = report["learning_metrics"]
        print(f"Reward Improvement: {learning_metrics['reward_improvement']:.2f}")
        print(f"Length Improvement: {learning_metrics['length_improvement']:.1f}")
        print(f"Stability Score: {learning_metrics['stability_score']:.3f}")
        
        if "evaluation_summary" in report:
            eval_summary = report["evaluation_summary"]
            print(f"Final Evaluation Score: {eval_summary['final_eval_score']:.2f}")
            print(f"Best Evaluation Score: {eval_summary['best_eval_score']:.2f}")
        
        print("=" * 60)


class RealTimeMonitor:
    """
    Real-time monitoring for DQN training.
    """
    
    def __init__(self, update_frequency: int = 10):
        """
        Initialize real-time monitor.
        
        Args:
            update_frequency: Frequency of updates in episodes
        """
        self.update_frequency = update_frequency
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []
        self.episodes = []
        
        # Initialize plot
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Real-time DQN Training Monitor')
        
        # Initialize lines
        self.reward_line, = self.axes[0, 0].plot([], [], 'b-')
        self.length_line, = self.axes[0, 1].plot([], [], 'g-')
        self.loss_line, = self.axes[1, 0].plot([], [], 'r-')
        self.epsilon_line, = self.axes[1, 1].plot([], [], 'm-')
        
        # Set labels
        self.axes[0, 0].set_title('Episode Rewards')
        self.axes[0, 0].set_xlabel('Episode')
        self.axes[0, 0].set_ylabel('Reward')
        self.axes[0, 0].grid(True)
        
        self.axes[0, 1].set_title('Episode Lengths')
        self.axes[0, 1].set_xlabel('Episode')
        self.axes[0, 1].set_ylabel('Length')
        self.axes[0, 1].grid(True)
        
        self.axes[1, 0].set_title('Training Loss')
        self.axes[1, 0].set_xlabel('Episode')
        self.axes[1, 0].set_ylabel('Loss')
        self.axes[1, 0].grid(True)
        
        self.axes[1, 1].set_title('Epsilon Decay')
        self.axes[1, 1].set_xlabel('Episode')
        self.axes[1, 1].set_ylabel('Epsilon')
        self.axes[1, 1].grid(True)
    
    def update(self, episode: int, reward: float, length: int, loss: float, epsilon: float) -> None:
        """
        Update the monitor with new data.
        
        Args:
            episode: Current episode number
            reward: Episode reward
            length: Episode length
            loss: Training loss
            epsilon: Current epsilon value
        """
        self.episodes.append(episode)
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_losses.append(loss)
        
        # Update plots
        self.reward_line.set_data(self.episodes, self.episode_rewards)
        self.length_line.set_data(self.episodes, self.episode_lengths)
        self.loss_line.set_data(self.episodes, self.episode_losses)
        
        # Update epsilon plot
        epsilons = [epsilon] * len(self.episodes)
        self.epsilon_line.set_data(self.episodes, epsilons)
        
        # Update axes limits
        for ax in self.axes.flat:
            ax.relim()
            ax.autoscale_view()
        
        # Refresh plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def close(self) -> None:
        """Close the monitor."""
        plt.close(self.fig)


if __name__ == "__main__":
    # Test the monitoring tools
    print("Testing DQN Monitoring Tools...")
    
    # Create sample data
    monitor = TrainingMonitor()
    
    # Generate sample training data
    sample_data = []
    for i in range(100):
        sample_data.append({
            'episode': i,
            'avg_reward': 10 + 5 * np.sin(i / 10) + np.random.normal(0, 1),
            'avg_length': 200 + 50 * np.sin(i / 15) + np.random.normal(0, 10),
            'avg_loss': 0.5 * np.exp(-i / 50) + np.random.normal(0, 0.01),
            'epsilon': max(0.01, 1.0 * (0.995 ** i)),
            'memory_size': min(10000, i * 10),
            'total_steps': i * 200
        })
    
    # Save sample data
    os.makedirs("test_logs", exist_ok=True)
    with open("test_logs/training_log.json", 'w') as f:
        json.dump(sample_data, f)
    
    # Test monitor
    monitor.log_path = "test_logs"
    monitor.plot_training_curves(save_path="test_training_curves.png", show=False)
    monitor.plot_learning_curves_smooth(save_path="test_smooth_curves.png", show=False)
    
    # Generate report
    report = monitor.generate_training_report()
    monitor.save_report("test_logs/training_report.json")
    monitor.print_summary()
    
    # Clean up
    import shutil
    if os.path.exists("test_logs"):
        shutil.rmtree("test_logs")
    if os.path.exists("test_training_curves.png"):
        os.remove("test_training_curves.png")
    if os.path.exists("test_smooth_curves.png"):
        os.remove("test_smooth_curves.png")
    
    print("DQN Monitoring Tools tested successfully!")
