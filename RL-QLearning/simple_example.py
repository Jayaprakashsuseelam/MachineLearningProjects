"""
Simple Q-Learning Example
This script demonstrates the core concepts of Q-Learning with a basic example.
"""

import numpy as np
import random

def simple_q_learning_example():
    """
    A very simple Q-Learning example with a 2x2 grid world
    """
    print("🎯 Simple Q-Learning Example")
    print("=" * 40)
    
    # Simple 2x2 grid world
    # States: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right (goal)
    # Actions: 0=right, 1=down
    
    # Initialize Q-table
    q_table = np.zeros((4, 2))
    
    # Learning parameters
    learning_rate = 0.1
    discount_factor = 0.9
    epsilon = 0.1
    
    print("Initial Q-table:")
    print(q_table)
    print()
    
    # Training loop
    print("Training the agent...")
    for episode in range(100):
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
    print("Final Q-table:")
    print(q_table)
    print()
    
    # Test the learned policy
    print("Testing learned policy:")
    state = 0
    path = [state]
    
    while state != 3:
        action = np.argmax(q_table[state])
        if action == 0:  # Right
            next_state = state + 1 if state % 2 == 0 else state
        else:  # Down
            next_state = state + 2 if state < 2 else state
        
        state = next_state
        path.append(state)
    
    print(f"Path taken: {path}")
    print("✅ Goal reached!")


def explain_q_learning():
    """
    Explain the key concepts of Q-Learning
    """
    print("\n📚 Q-Learning Concepts Explained")
    print("=" * 50)
    
    print("""
1. **Q-Learning Algorithm**
   - Q-Learning is a model-free reinforcement learning algorithm
   - It learns the quality of actions (Q-values) without knowing the environment dynamics
   - Uses temporal difference learning to update Q-values

2. **Key Components**
   - Q-Table: Stores Q(s,a) values for each state-action pair
   - Learning Rate (α): Controls how much new information overrides old information
   - Discount Factor (γ): Determines importance of future rewards vs immediate rewards
   - Epsilon (ε): Controls exploration vs exploitation balance

3. **Q-Learning Update Rule**
   Q(s,a) = Q(s,a) + α[r + γ * max Q(s',a') - Q(s,a)]
   
   Where:
   - s = current state
   - a = current action
   - r = reward received
   - s' = next state
   - a' = next action
   - α = learning rate
   - γ = discount factor

4. **Exploration vs Exploitation**
   - Exploration: Try random actions to discover new strategies
   - Exploitation: Use learned Q-values to choose best actions
   - Epsilon-greedy policy balances both

5. **Convergence**
   - Q-Learning converges to optimal Q-values under certain conditions
   - All state-action pairs must be visited infinitely often
   - Learning rate must decrease over time
""")

if __name__ == "__main__":
    simple_q_learning_example()
    explain_q_learning()
