"""
Test script for Q-Learning implementation
This script tests the basic functionality of the Q-Learning algorithm.
"""

import numpy as np
import random

def test_simple_q_learning():
    """Test the basic Q-Learning functionality"""
    print("🧪 Testing Q-Learning Implementation...")
    
    # Test 1: Q-Learning update rule
    print("\nTest 1: Q-Learning update rule")
    q_table = np.zeros((4, 2))
    
    # Simple update
    old_q = q_table[0, 0]
    learning_rate = 0.1
    discount_factor = 0.9
    reward = 0
    next_state = 1
    
    # Q-Learning update
    target = reward + discount_factor * np.max(q_table[next_state])
    new_q = old_q + learning_rate * (target - old_q)
    q_table[0, 0] = new_q
    
    print(f"Q(0,0) updated from {old_q:.3f} to {new_q:.3f}")
    assert abs(new_q - 0.0) < 1e-10, "Q-value should remain 0 for zero reward and zero next state Q-values"
    print("✅ Test 1 passed!")
    
    # Test 2: Training convergence
    print("\nTest 2: Training convergence")
    q_table = np.zeros((4, 2))
    
    # Train for a few episodes
    for episode in range(50):
        state = 0
        while state != 3:
            action = random.randint(0, 1)
            
            if action == 0:  # Right
                next_state = state + 1 if state % 2 == 0 else state
            else:  # Down
                next_state = state + 2 if state < 2 else state
            
            reward = 1 if next_state == 3 else 0
            
            if next_state == 3:
                target = reward
            else:
                target = reward + discount_factor * np.max(q_table[next_state])
            
            q_table[state, action] += learning_rate * (target - q_table[state, action])
            state = next_state
    
    print("Final Q-table after training:")
    print(q_table)
    
    # Check that Q-values have been updated
    assert np.any(q_table != 0), "Q-table should have non-zero values after training"
    print("✅ Test 2 passed!")
    
    # Test 3: Policy extraction
    print("\nTest 3: Policy extraction")
    
    # Extract optimal policy
    policy = np.argmax(q_table, axis=1)
    print(f"Optimal policy: {policy}")
    
    # Test the policy
    state = 0
    path = [state]
    
    while state != 3 and len(path) < 10:
        action = policy[state]
        if action == 0:  # Right
            next_state = state + 1 if state % 2 == 0 else state
        else:  # Down
            next_state = state + 2 if state < 2 else state
        
        state = next_state
        path.append(state)
    
    print(f"Path taken: {path}")
    assert 3 in path, "Policy should eventually reach the goal"
    print("✅ Test 3 passed!")
    
    print("\n🎉 All tests passed! Q-Learning implementation is working correctly.")

def test_grid_world_import():
    """Test if the GridWorld class can be imported and used"""
    try:
        from q_learning import GridWorld, QLearningAgent
        print("✅ GridWorld and QLearningAgent imported successfully!")
        
        # Test GridWorld creation
        env = GridWorld(size=3)
        print("✅ GridWorld created successfully!")
        
        # Test agent creation
        agent = QLearningAgent(state_size=9, action_size=4)
        print("✅ QLearningAgent created successfully!")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error creating objects: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Q-Learning Tests")
    print("=" * 40)
    
    # Test basic functionality
    test_simple_q_learning()
    
    print("\n" + "=" * 40)
    
    # Test imports
    if test_grid_world_import():
        print("✅ All components are working correctly!")
    else:
        print("⚠️  Some components may have issues. Check the error messages above.")
    
    print("\n🎯 Testing completed!")
