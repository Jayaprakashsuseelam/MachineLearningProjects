"""
Custom Environment DQN Demo

This script demonstrates how to create and train a DQN agent on custom environments.
It includes examples of different environment types and configurations.
"""

import os
import sys
import numpy as np
import torch
import gym
from gym import spaces
from typing import Tuple, Any
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqn.agent import DQNAgent
from environments.wrappers import make_env, get_env_info
from training.config import DQNConfig
from training.trainer import DQNTrainer
from visualization.monitor import TrainingMonitor
from visualization.plotter import DQNPlotter


class CustomGridWorld(gym.Env):
    """
    Custom Grid World environment for DQN demonstration.
    
    A simple grid world where the agent must navigate to a goal.
    """
    
    def __init__(self, size: int = 5):
        """
        Initialize the Grid World environment.
        
        Args:
            size: Size of the grid (size x size)
        """
        super().__init__()
        
        self.size = size
        self.agent_pos = [0, 0]
        self.goal_pos = [size-1, size-1]
        
        # Action space: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)
        
        # Observation space: agent position (x, y)
        self.observation_space = spaces.Box(
            low=0, high=size-1, shape=(2,), dtype=np.int32
        )
        
        # Action mappings
        self.actions = {
            0: [-1, 0],  # up
            1: [1, 0],   # down
            2: [0, -1],  # left
            3: [0, 1]    # right
        }
    
    def reset(self) -> np.ndarray:
        """Reset the environment."""
        self.agent_pos = [0, 0]
        return np.array(self.agent_pos, dtype=np.int32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Take a step in the environment."""
        # Move agent
        move = self.actions[action]
        new_pos = [self.agent_pos[0] + move[0], self.agent_pos[1] + move[1]]
        
        # Check bounds
        if (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            self.agent_pos = new_pos
        
        # Calculate reward
        if self.agent_pos == self.goal_pos:
            reward = 10.0
            done = True
        else:
            # Distance-based reward
            distance = np.sqrt((self.agent_pos[0] - self.goal_pos[0])**2 + 
                             (self.agent_pos[1] - self.goal_pos[1])**2)
            reward = -0.1 - distance * 0.1
            done = False
        
        return np.array(self.agent_pos, dtype=np.int32), reward, done, {}
    
    def render(self, mode: str = 'human') -> None:
        """Render the environment."""
        grid = np.zeros((self.size, self.size))
        grid[self.agent_pos[0], self.agent_pos[1]] = 1  # Agent
        grid[self.goal_pos[0], self.goal_pos[1]] = 2     # Goal
        
        print("Grid World:")
        for row in grid:
            print(" ".join(["A" if x == 1 else "G" if x == 2 else "." for x in row]))
        print()


class CustomMountainCar(gym.Env):
    """
    Custom Mountain Car environment with different reward structure.
    """
    
    def __init__(self):
        """Initialize the custom Mountain Car environment."""
        super().__init__()
        
        # State: [position, velocity]
        self.observation_space = spaces.Box(
            low=np.array([-1.2, -0.07]), 
            high=np.array([0.6, 0.07]), 
            dtype=np.float32
        )
        
        # Actions: 0=left, 1=nothing, 2=right
        self.action_space = spaces.Discrete(3)
        
        # Environment parameters
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = 0.0
        
        # Physics parameters
        self.force = 0.001
        self.gravity = 0.0025
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset the environment."""
        self.state = np.array([np.random.uniform(low=-0.6, high=-0.4), 0])
        return self.state.copy()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Take a step in the environment."""
        position, velocity = self.state
        
        # Apply force
        force = self.force if action == 2 else -self.force if action == 0 else 0
        
        # Update velocity
        velocity += force - self.gravity * np.cos(3 * position)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        
        # Update position
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        
        # Reset velocity if hit left wall
        if position == self.min_position and velocity < 0:
            velocity = 0
        
        self.state = np.array([position, velocity])
        
        # Check if goal reached
        done = position >= self.goal_position
        
        # Custom reward structure
        if done:
            reward = 100.0
        else:
            # Reward for getting closer to goal
            reward = (position - self.min_position) / (self.max_position - self.min_position)
            # Small penalty for each step
            reward -= 0.1
        
        return self.state.copy(), reward, done, {}
    
    def render(self, mode: str = 'human') -> None:
        """Render the environment."""
        position, velocity = self.state
        print(f"Position: {position:.3f}, Velocity: {velocity:.3f}")


class CustomLunarLander(gym.Env):
    """
    Custom Lunar Lander environment with simplified physics.
    """
    
    def __init__(self):
        """Initialize the custom Lunar Lander environment."""
        super().__init__()
        
        # State: [x, y, vx, vy, angle, angular_velocity, left_leg, right_leg]
        self.observation_space = spaces.Box(
            low=np.array([-1.5, -1.5, -5, -5, -np.pi, -5, 0, 0]),
            high=np.array([1.5, 1.5, 5, 5, np.pi, 5, 1, 1]),
            dtype=np.float32
        )
        
        # Actions: 0=nothing, 1=left, 2=main, 3=right
        self.action_space = spaces.Discrete(4)
        
        # Environment parameters
        self.gravity = 0.1
        self.thrust_power = 0.5
        self.rotation_power = 0.1
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset the environment."""
        # Random starting position
        self.state = np.array([
            np.random.uniform(-1, 1),  # x
            np.random.uniform(0.5, 1.5),  # y
            0,  # vx
            0,  # vy
            np.random.uniform(-0.2, 0.2),  # angle
            0,  # angular_velocity
            0,  # left_leg
            0   # right_leg
        ])
        return self.state.copy()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Take a step in the environment."""
        x, y, vx, vy, angle, angular_vel, left_leg, right_leg = self.state
        
        # Apply thrust
        if action == 1:  # left
            vx -= self.thrust_power * np.sin(angle)
            vy += self.thrust_power * np.cos(angle)
            angular_vel -= self.rotation_power
        elif action == 2:  # main
            vx -= self.thrust_power * np.sin(angle)
            vy += self.thrust_power * np.cos(angle)
        elif action == 3:  # right
            vx -= self.thrust_power * np.sin(angle)
            vy += self.thrust_power * np.cos(angle)
            angular_vel += self.rotation_power
        
        # Apply gravity
        vy -= self.gravity
        
        # Update position
        x += vx
        y += vy
        angle += angular_vel
        
        # Update state
        self.state = np.array([x, y, vx, vy, angle, angular_vel, left_leg, right_leg])
        
        # Check landing
        done = y <= 0
        
        # Calculate reward
        if done:
            # Landing reward based on landing conditions
            if abs(x) < 0.1 and abs(angle) < 0.1 and abs(vy) < 0.5:
                reward = 100.0
            else:
                reward = -100.0
        else:
            # Small penalty for each step
            reward = -0.1
        
        return self.state.copy(), reward, done, {}
    
    def render(self, mode: str = 'human') -> None:
        """Render the environment."""
        x, y, vx, vy, angle, angular_vel, left_leg, right_leg = self.state
        print(f"Position: ({x:.2f}, {y:.2f}), Velocity: ({vx:.2f}, {vy:.2f}), Angle: {angle:.2f}")


def main():
    """Main function to run the custom environment DQN demo."""
    print("=" * 60)
    print("CUSTOM ENVIRONMENT DQN DEMO")
    print("=" * 60)
    
    # Choose environment
    print("Available custom environments:")
    print("1. Grid World")
    print("2. Custom Mountain Car")
    print("3. Custom Lunar Lander")
    
    choice = input("Select environment (1-3): ").strip()
    
    if choice == "1":
        env_name = "GridWorld"
        env = CustomGridWorld(size=5)
        config = create_grid_world_config()
    elif choice == "2":
        env_name = "CustomMountainCar"
        env = CustomMountainCar()
        config = create_mountain_car_config()
    elif choice == "3":
        env_name = "CustomLunarLander"
        env = CustomLunarLander()
        config = create_lunar_lander_config()
    else:
        print("Invalid choice. Using Grid World.")
        env_name = "GridWorld"
        env = CustomGridWorld(size=5)
        config = create_grid_world_config()
    
    print(f"Selected environment: {env_name}")
    print("-" * 60)
    
    # Get environment info
    env_info = get_env_info(env)
    print(f"State size: {env_info['state_size']}")
    print(f"Action size: {env_info['n_actions']}")
    
    # Create trainer with custom environment
    trainer = DQNTrainer(config)
    trainer.env = env  # Replace with custom environment
    trainer.env_info = env_info
    
    try:
        # Train the agent
        print("Starting training...")
        results = trainer.train()
        
        # Test the trained agent
        print("\nTesting trained agent...")
        test_scores = trainer.test(episodes=10, render=False)
        
        # Create visualizations
        print("\nCreating visualizations...")
        create_visualizations(results, test_scores, env_name)
        
        # Generate report
        print("\nGenerating training report...")
        generate_report(results, test_scores, env_name)
        
    finally:
        trainer.close()
    
    print(f"\nCustom {env_name} DQN demo completed successfully!")


def create_grid_world_config():
    """Create configuration for Grid World environment."""
    config = DQNConfig(
        env_name="GridWorld",
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        hidden_layers=[64, 32],
        batch_size=32,
        memory_size=1000,
        target_update_frequency=10,
        max_episodes=500,
        max_steps_per_episode=100,
        eval_frequency=50,
        eval_episodes=10
    )
    return config


def create_mountain_car_config():
    """Create configuration for Custom Mountain Car environment."""
    config = DQNConfig(
        env_name="CustomMountainCar",
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.01,
        hidden_layers=[128, 64],
        batch_size=64,
        memory_size=10000,
        target_update_frequency=10,
        max_episodes=1000,
        max_steps_per_episode=200,
        eval_frequency=100,
        eval_episodes=10
    )
    return config


def create_lunar_lander_config():
    """Create configuration for Custom Lunar Lander environment."""
    config = DQNConfig(
        env_name="CustomLunarLander",
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        hidden_layers=[128, 64],
        batch_size=64,
        memory_size=10000,
        target_update_frequency=10,
        max_episodes=1000,
        max_steps_per_episode=1000,
        eval_frequency=100,
        eval_episodes=10
    )
    return config


def create_visualizations(results, test_scores, env_name):
    """Create and save visualizations."""
    # Create output directory
    os.makedirs(f"{env_name.lower()}_results", exist_ok=True)
    
    # Initialize plotter
    plotter = DQNPlotter()
    
    # Plot learning curves
    plotter.plot_learning_curves(
        results['episode_rewards'],
        results['episode_lengths'],
        results['episode_losses'],
        [results['episode_rewards'][i] for i in range(len(results['episode_rewards']))],  # Placeholder for epsilons
        save_path=f"{env_name.lower()}_results/learning_curves.png",
        show=False
    )
    
    # Plot evaluation comparison
    if results['eval_scores']:
        eval_episodes = list(range(0, len(results['episode_rewards']), 50))
        plotter.plot_evaluation_comparison(
            results['eval_scores'],
            eval_episodes,
            save_path=f"{env_name.lower()}_results/evaluation_progress.png",
            show=False
        )
    
    # Plot agent performance
    plotter.plot_agent_performance(
        test_scores,
        [100] * len(test_scores),  # Placeholder for episode lengths
        save_path=f"{env_name.lower()}_results/agent_performance.png",
        show=False
    )
    
    print(f"Visualizations saved to {env_name.lower()}_results/")


def generate_report(results, test_scores, env_name):
    """Generate a comprehensive training report."""
    # Generate report
    report = {
        "environment": env_name,
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
            "success_rate": calculate_success_rate(test_scores, env_name)
        },
        "learning_metrics": {
            "reward_improvement": results['episode_rewards'][-1] - results['episode_rewards'][0],
            "stability_score": calculate_stability_score(results['episode_rewards'])
        }
    }
    
    # Save report
    import json
    with open(f"{env_name.lower()}_results/training_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"CUSTOM {env_name.upper()} TRAINING SUMMARY")
    print("=" * 60)
    print(f"Environment: {env_name}")
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


def calculate_success_rate(test_scores, env_name):
    """Calculate success rate based on environment type."""
    if env_name == "GridWorld":
        # Success if score > 5 (reached goal)
        return sum(1 for score in test_scores if score > 5) / len(test_scores)
    elif env_name == "CustomMountainCar":
        # Success if score > 90 (reached goal)
        return sum(1 for score in test_scores if score > 90) / len(test_scores)
    elif env_name == "CustomLunarLander":
        # Success if score > 50 (good landing)
        return sum(1 for score in test_scores if score > 50) / len(test_scores)
    else:
        return 0.0


def quick_demo():
    """Quick demo with Grid World."""
    print("Running quick custom environment demo...")
    
    # Create Grid World environment
    env = CustomGridWorld(size=4)
    config = create_grid_world_config()
    config.max_episodes = 100
    config.eval_frequency = 25
    config.log_frequency = 10
    
    # Create trainer
    trainer = DQNTrainer(config)
    trainer.env = env
    trainer.env_info = get_env_info(env)
    
    # Train
    results = trainer.train()
    
    # Test
    test_scores = trainer.test(episodes=5)
    
    # Print results
    print(f"Quick custom environment demo completed!")
    print(f"Final average reward: {np.mean(results['episode_rewards'][-10:]):.2f}")
    print(f"Test scores: {test_scores}")
    
    trainer.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Custom Environment DQN Demo")
    parser.add_argument("--quick", action="store_true", help="Run quick demo")
    args = parser.parse_args()
    
    if args.quick:
        quick_demo()
    else:
        main()
