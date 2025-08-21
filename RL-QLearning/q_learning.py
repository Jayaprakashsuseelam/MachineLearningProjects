import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
import random

class GridWorld:
    """
    A simple grid world environment for Q-Learning demonstration.
    The agent must navigate from start (S) to goal (G) avoiding obstacles (X).
    """
    
    def __init__(self, size: int = 5):
        self.size = size
        self.grid = np.zeros((size, size))
        self.start_pos = (0, 0)
        self.goal_pos = (size-1, size-1)
        
        # Set start and goal positions
        self.grid[self.start_pos] = 1  # Start
        self.grid[self.goal_pos] = 2   # Goal
        
        # Add some obstacles
        self.add_obstacles()
        
        # Actions: 0=up, 1=right, 2=down, 3=left
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.action_names = ['↑', '→', '↓', '←']
        
        self.current_pos = self.start_pos
        
    def add_obstacles(self):
        """Add random obstacles to the grid"""
        num_obstacles = self.size
        for _ in range(num_obstacles):
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            if (x, y) not in [self.start_pos, self.goal_pos]:
                self.grid[x, y] = -1  # Obstacle
    
    def reset(self):
        """Reset the environment to start position"""
        self.current_pos = self.start_pos
        return self.current_pos
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Take a step in the environment
        
        Args:
            action: Action index (0-3)
            
        Returns:
            (new_position, reward, done)
        """
        dx, dy = self.actions[action]
        new_x = self.current_pos[0] + dx
        new_y = self.current_pos[1] + dy
        
        # Check boundaries
        if new_x < 0 or new_x >= self.size or new_y < 0 or new_y >= self.size:
            return self.current_pos, -1, False
        
        # Check obstacles
        if self.grid[new_x, new_y] == -1:
            return self.current_pos, -1, False
        
        # Update position
        self.current_pos = (new_x, new_y)
        
        # Check if goal reached
        if self.current_pos == self.goal_pos:
            return self.current_pos, 100, True
        
        # Small negative reward for each step to encourage efficiency
        return self.current_pos, -0.1, False
    
    def get_state(self) -> int:
        """Convert position to state index"""
        return self.current_pos[0] * self.size + self.current_pos[1]
    
    def render(self):
        """Render the current state of the grid"""
        display_grid = self.grid.copy()
        display_grid[self.current_pos] = 3  # Agent position
        
        # Create a more readable display
        symbols = {0: '.', 1: 'S', 2: 'G', -1: 'X', 3: 'A'}
        display = np.vectorize(symbols.get)(display_grid)
        
        print("Grid World Environment:")
        print("S = Start, G = Goal, X = Obstacle, A = Agent, . = Empty")
        print("-" * (self.size * 3 + 1))
        for row in display:
            print("|" + "|".join(f" {cell} " for cell in row) + "|")
        print("-" * (self.size * 3 + 1))
        print(f"Agent Position: {self.current_pos}")
        print()


class QLearningAgent:
    """
    Q-Learning agent implementation
    """
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.1, 
                 discount_factor: float = 0.95, epsilon: float = 0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((state_size, action_size))
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        
    def choose_action(self, state: int, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
            
        Returns:
            Action index
        """
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, self.action_size - 1)
        else:
            # Exploitation: best action based on Q-values
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state: int, action: int, reward: float, 
                      next_state: int, done: bool):
        """
        Update Q-value using Q-Learning update rule
        
        Q(s,a) = Q(s,a) + α[r + γ * max Q(s',a') - Q(s,a)]
        """
        if done:
            # Terminal state
            target = reward
        else:
            # Non-terminal state
            target = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        # Q-Learning update
        self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])
    
    def train(self, env: GridWorld, episodes: int = 1000):
        """
        Train the agent using Q-Learning
        
        Args:
            env: GridWorld environment
            episodes: Number of training episodes
        """
        print(f"Training Q-Learning agent for {episodes} episodes...")
        
        for episode in range(episodes):
            state = env.reset()
            state_idx = env.get_state()
            total_reward = 0
            steps = 0
            
            while True:
                # Choose action
                action = self.choose_action(state_idx)
                
                # Take action
                next_state, reward, done = env.step(action)
                next_state_idx = env.get_state()
                
                # Update Q-value
                self.update_q_value(state_idx, action, reward, next_state_idx, done)
                
                # Update state
                state = next_state
                state_idx = next_state_idx
                
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Record statistics
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            
            # Print progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_steps = np.mean(self.episode_lengths[-100:])
                print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.2f}, Avg Steps = {avg_steps:.2f}")
        
        print("Training completed!")
    
    def evaluate(self, env: GridWorld, episodes: int = 100) -> Tuple[float, float]:
        """
        Evaluate the trained agent
        
        Args:
            env: GridWorld environment
            episodes: Number of evaluation episodes
            
        Returns:
            (average_reward, success_rate)
        """
        total_rewards = []
        successes = 0
        
        for _ in range(episodes):
            state = env.reset()
            state_idx = env.get_state()
            total_reward = 0
            
            while True:
                # Choose best action (no exploration)
                action = self.choose_action(state_idx, training=False)
                
                # Take action
                next_state, reward, done = env.step(action)
                next_state_idx = env.get_state()
                
                state = next_state
                state_idx = next_state_idx
                total_reward += reward
                
                if done:
                    if reward > 0:  # Goal reached
                        successes += 1
                    break
            
            total_rewards.append(total_reward)
        
        avg_reward = np.mean(total_rewards)
        success_rate = successes / episodes
        
        return avg_reward, success_rate


def visualize_training(agent: QLearningAgent):
    """Visualize training progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot episode rewards
    ax1.plot(agent.episode_rewards)
    ax1.set_title('Episode Rewards During Training')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    # Plot episode lengths
    ax2.plot(agent.episode_lengths)
    ax2.set_title('Episode Lengths During Training')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def visualize_q_table(agent: QLearningAgent, env: GridWorld):
    """Visualize the learned Q-table"""
    # Reshape Q-table to grid format
    q_grid = agent.q_table.reshape(env.size, env.size, len(env.actions))
    
    # Create a heatmap for each action
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Q-Values for Each Action', fontsize=16)
    
    action_names = ['Up', 'Right', 'Down', 'Left']
    
    for i, (ax, action_name) in enumerate(zip(axes.flat, action_names)):
        sns.heatmap(q_grid[:, :, i], annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax)
        ax.set_title(f'{action_name} ({action_names[i]})')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
    
    plt.tight_layout()
    plt.show()


def demonstrate_agent(agent: QLearningAgent, env: GridWorld, max_steps: int = 20):
    """Demonstrate the trained agent's behavior"""
    print("Demonstrating trained agent:")
    print("=" * 40)
    
    state = env.reset()
    env.render()
    
    for step in range(max_steps):
        state_idx = env.get_state()
        action = agent.choose_action(state_idx, training=False)
        
        print(f"Step {step + 1}: Agent chooses action '{env.action_names[action]}'")
        
        next_state, reward, done = env.step(action)
        env.render()
        
        if done:
            if reward > 0:
                print("🎉 Goal reached! Agent successfully navigated to the goal.")
            else:
                print("❌ Episode ended without reaching goal.")
            break
        
        if step == max_steps - 1:
            print("⚠️  Maximum steps reached without finding goal.")


def main():
    """Main function to run the Q-Learning demonstration"""
    print("🚀 Q-Learning Reinforcement Learning Algorithm Demonstration")
    print("=" * 60)
    
    # Create environment and agent
    env = GridWorld(size=5)
    state_size = env.size * env.size
    action_size = len(env.actions)
    
    agent = QLearningAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1
    )
    
    # Show initial environment
    print("Initial Environment:")
    env.render()
    
    # Train the agent
    agent.train(env, episodes=500)
    
    # Evaluate the trained agent
    avg_reward, success_rate = agent.evaluate(env, episodes=100)
    print(f"\n📊 Evaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.2%}")
    
    # Visualize training progress
    visualize_training(agent)
    
    # Visualize learned Q-table
    visualize_q_table(agent, env)
    
    # Demonstrate the trained agent
    demonstrate_agent(agent, env)
    
    print("\n✅ Q-Learning demonstration completed!")


if __name__ == "__main__":
    main()
