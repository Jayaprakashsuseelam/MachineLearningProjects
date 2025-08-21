"""
Q-Learning Tutorial - Interactive Learning Script
This script provides an interactive introduction to Q-Learning concepts.
Run this script to learn Q-Learning step by step.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

def section_header(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def q_learning_tutorial():
    """Main tutorial function"""
    
    section_header("Q-LEARNING REINFORCEMENT LEARNING TUTORIAL")
    
    print("""
Welcome to the Q-Learning tutorial! This tutorial will teach you:
1. The Q-Learning update rule
2. How to implement Q-Learning
3. How to train an agent
4. How to interpret results
5. How parameters affect learning

Let's get started!
""")
    
    # Section 1: Understanding Q-Learning
    section_header("1. UNDERSTANDING Q-LEARNING")
    
    print("""
Q-Learning is a model-free reinforcement learning algorithm that learns 
the quality of actions, telling an agent what action to take under what circumstances.

Key Concepts:
- Q-Table: Stores Q(s,a) values for each state-action pair
- Learning Rate (α): Controls how much new information overrides old information
- Discount Factor (γ): Determines importance of future rewards vs immediate rewards
- Epsilon (ε): Controls exploration vs exploitation balance

The Q-Learning Update Rule:
Q(s,a) = Q(s,a) + α[r + γ * max Q(s',a') - Q(s,a)]

Where:
- s = current state, a = current action
- r = reward received, s' = next state
- α = learning rate, γ = discount factor
""")
    
    # Section 2: The Q-Learning Update Rule
    section_header("2. THE Q-LEARNING UPDATE RULE")
    
    def q_learning_update(q_table, state, action, reward, next_state, 
                         learning_rate=0.1, discount_factor=0.9):
        """Implement the Q-Learning update rule"""
        current_q = q_table[state, action]
        max_next_q = np.max(q_table[next_state])
        target_q = reward + discount_factor * max_next_q
        new_q = current_q + learning_rate * (target_q - current_q)
        return new_q
    
    print("✅ Q-Learning update function created!")
    print("Function: q_learning_update(q_table, state, action, reward, next_state, learning_rate, discount_factor)")
    
    # Section 3: Simple Example
    section_header("3. SIMPLE EXAMPLE: 2X2 GRID WORLD")
    
    print("""
Let's create a simple 2x2 grid world:
States: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right (goal)
Actions: 0=right, 1=down

The agent starts at top-left and must reach the bottom-right goal.
""")
    
    # Initialize Q-table
    q_table = np.zeros((4, 2))
    print("\nInitial Q-table:")
    print(q_table)
    
    # Section 4: Simulating One Step
    section_header("4. SIMULATING ONE Q-LEARNING STEP")
    
    print("""
Simulating one Q-Learning step:
Starting at state 0 (top-left)
Taking action 0 (right)
Moving to state 1 (top-right)
Reward: 0 (not goal)
""")
    
    # Update Q-value for state 0, action 0
    old_q = q_table[0, 0]
    new_q = q_learning_update(q_table, 0, 0, 0, 1)
    q_table[0, 0] = new_q
    
    print(f"\nQ-value updated:")
    print(f"Q(0,0) changed from {old_q:.3f} to {new_q:.3f}")
    print("\nUpdated Q-table:")
    print(q_table)
    
    # Section 5: Training the Agent
    section_header("5. TRAINING THE AGENT")
    
    print("Now let's train the agent for multiple episodes to see how it learns.")
    
    # Training parameters
    learning_rate = 0.1
    discount_factor = 0.9
    epsilon = 0.1
    episodes = 100
    
    # Reset Q-table
    q_table = np.zeros((4, 2))
    
    print(f"\nTraining for {episodes} episodes...")
    print(f"Learning rate: {learning_rate}")
    print(f"Discount factor: {discount_factor}")
    print(f"Epsilon: {epsilon}")
    
    # Training loop
    for episode in range(episodes):
        state = 0  # Start at top-left
        
        while state != 3:  # Continue until goal reached
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, 1)  # Random action
            else:
                action = np.argmax(q_table[state])  # Best action
            
            # Take action and get next state
            if action == 0:  # Right
                next_state = state + 1 if state % 2 == 0 else state
            else:  # Down
                next_state = state + 2 if state < 2 else state
            
            # Get reward
            reward = 1 if next_state == 3 else 0
            
            # Q-Learning update
            if next_state == 3:  # Terminal state
                target = reward
            else:
                target = reward + discount_factor * np.max(q_table[next_state])
            
            q_table[state, action] += learning_rate * (target - q_table[state, action])
            
            state = next_state
    
    print("Training completed!")
    print("\nFinal Q-table:")
    print(q_table)
    
    # Section 6: Testing the Learned Policy
    section_header("6. TESTING THE LEARNED POLICY")
    
    print("Let's test how well our trained agent performs.")
    
    state = 0
    path = [state]
    actions_taken = []
    
    while state != 3:
        action = np.argmax(q_table[state])
        actions_taken.append(action)
        
        if action == 0:  # Right
            next_state = state + 1 if state % 2 == 0 else state
        else:  # Down
            next_state = state + 2 if state < 2 else state
        
        state = next_state
        path.append(state)
    
    print(f"\nPath taken: {path}")
    print(f"Actions taken: {actions_taken}")
    print("✅ Goal reached!")
    
    # Section 7: Understanding Results
    section_header("7. UNDERSTANDING THE RESULTS")
    
    print("Analyzing the learned Q-table:")
    print("\nQ-values for each state-action pair:")
    for state in range(4):
        state_name = ['top-left', 'top-right', 'bottom-left', 'bottom-right'][state]
        for action in range(2):
            action_name = ['right', 'down'][action]
            q_value = q_table[state, action]
            print(f"Q({state_name}, {action_name}) = {q_value:.3f}")
    
    print("\nOptimal actions for each state:")
    for state in range(4):
        if state != 3:  # Skip goal state
            state_name = ['top-left', 'top-right', 'bottom-left', 'bottom-right'][state]
            best_action = np.argmax(q_table[state])
            action_name = ['right', 'down'][best_action]
            print(f"State {state_name}: Best action is {action_name}")
    
    # Section 8: Parameter Experimentation
    section_header("8. EXPERIMENTING WITH PARAMETERS")
    
    print("Let's see how different parameters affect learning.")
    
    def train_with_parameters(lr, df, eps, num_episodes=100):
        """Train with specific parameters and return final Q-value"""
        q_table = np.zeros((4, 2))
        
        for episode in range(num_episodes):
            state = 0
            
            while state != 3:
                if random.random() < eps:
                    action = random.randint(0, 1)
                else:
                    action = np.argmax(q_table[state])
                
                if action == 0:
                    next_state = state + 1 if state % 2 == 0 else state
                else:
                    next_state = state + 2 if state < 2 else state
                
                reward = 1 if next_state == 3 else 0
                
                if next_state == 3:
                    target = reward
                else:
                    target = reward + df * np.max(q_table[next_state])
                
                q_table[state, action] += lr * (target - q_table[state, action])
                state = next_state
        
        return q_table[0, 0]  # Return Q(0,0)
    
    print("\nTesting different learning rates:")
    for lr in [0.01, 0.1, 0.5]:
        final_q = train_with_parameters(lr, 0.9, 0.1, 100)
        print(f"Learning rate {lr}: Q(0,0) = {final_q:.3f}")
    
    print("\nTesting different discount factors:")
    for df in [0.5, 0.9, 0.99]:
        final_q = train_with_parameters(0.1, df, 0.1, 100)
        print(f"Discount factor {df}: Q(0,0) = {final_q:.3f}")
    
    # Section 9: Summary
    section_header("9. SUMMARY")
    
    print("""
Congratulations! You've completed the Q-Learning tutorial.

What you've learned:
1. ✅ Q-Learning Update Rule: The mathematical foundation
2. ✅ Q-Table: How to store and update Q-values  
3. ✅ Training Process: How to train an agent over multiple episodes
4. ✅ Policy Extraction: How to use learned Q-values to make decisions
5. ✅ Parameter Effects: How learning rate, discount factor, and epsilon affect learning

Key Takeaways:
- Q-Learning learns optimal policies without knowing environment dynamics
- The algorithm balances exploration and exploitation
- Parameters significantly affect learning speed and quality
- Q-values represent the expected future reward for taking actions in states

Next Steps:
- Try the main q_learning.py script for a more complex grid world
- Experiment with different environments and reward structures
- Explore other reinforcement learning algorithms like SARSA or Deep Q-Networks
- Apply Q-Learning to real-world problems like robotics or game AI

Happy learning! 🚀
""")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    try:
        q_learning_tutorial()
    except KeyboardInterrupt:
        print("\n\nTutorial interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check your Python environment and dependencies.")
