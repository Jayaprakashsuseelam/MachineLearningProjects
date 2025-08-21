# Q-Learning Reinforcement Learning Algorithm

A comprehensive implementation and tutorial for Q-Learning, a fundamental reinforcement learning algorithm. This project provides both theoretical understanding and practical implementation with a grid world navigation case study.

## 🎯 What is Q-Learning?

Q-Learning is a model-free reinforcement learning algorithm that learns the quality of actions, telling an agent what action to take under what circumstances. It's one of the most important algorithms in reinforcement learning and serves as the foundation for more advanced techniques like Deep Q-Networks (DQN).

### Key Concepts

- **Q-Table**: Stores Q(s,a) values for each state-action pair
- **Learning Rate (α)**: Controls how much new information overrides old information
- **Discount Factor (γ)**: Determines importance of future rewards vs immediate rewards
- **Epsilon (ε)**: Controls exploration vs exploitation balance

### The Q-Learning Update Rule

```
Q(s,a) = Q(s,a) + α[r + γ * max Q(s',a') - Q(s,a)]
```

Where:
- `s` = current state, `a` = current action
- `r` = reward received, `s'` = next state
- `α` = learning rate, `γ` = discount factor

## 🚀 Case Study: Grid World Navigation

The main case study demonstrates Q-Learning in a **5x5 grid world environment** where an agent must navigate from start to goal while avoiding obstacles.

### Environment Features

- **Grid Size**: 5x5 grid world
- **Start Position**: Top-left corner (0,0)
- **Goal Position**: Bottom-right corner (4,4)
- **Obstacles**: Randomly placed obstacles that the agent must avoid
- **Actions**: 4 possible actions (up, right, down, left)
- **Rewards**: 
  - Goal reached: +100
  - Obstacle hit: -1
  - Each step: -0.1 (encourages efficiency)

### Learning Process

1. **Exploration Phase**: Agent explores the environment using epsilon-greedy policy
2. **Q-Value Updates**: Q-values are updated using the Q-Learning update rule
3. **Policy Improvement**: Agent gradually learns the optimal path to the goal
4. **Convergence**: Q-values converge to optimal values after sufficient training

## 📁 Project Structure

```
RL-QLearning/
├── README.md                 # This file - project documentation
├── requirements.txt          # Python dependencies
├── q_learning.py            # Main implementation with grid world case study
├── simple_example.py        # Simplified Q-Learning example for learning
├── q_learning_tutorial.py   # Interactive tutorial script
└── test_q_learning.py       # Test script to verify implementation
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation Steps

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd RL-QLearning
   ```

2. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python test_q_learning.py
   ```

## 📚 Learning Path

### 1. Start with the Tutorial
Begin your learning journey with the interactive tutorial:

```bash
python q_learning_tutorial.py
```

This tutorial covers:
- Q-Learning concepts and mathematics
- Step-by-step implementation
- Parameter effects on learning
- Hands-on examples

### 2. Understand the Simple Example
Run the simplified example to see Q-Learning in action:

```bash
python simple_example.py
```

This demonstrates:
- Basic Q-Learning implementation
- 2x2 grid world navigation
- Training and evaluation process

### 3. Explore the Main Case Study
Dive into the comprehensive grid world implementation:

```bash
python q_learning.py
```

This showcases:
- Full 5x5 grid world environment
- Advanced visualization and analysis
- Training progress monitoring
- Q-table visualization

### 4. Run Tests
Verify everything works correctly:

```bash
python test_q_learning.py
```

## 🔧 Usage Examples

### Basic Q-Learning Implementation

```python
from q_learning import GridWorld, QLearningAgent

# Create environment and agent
env = GridWorld(size=5)
agent = QLearningAgent(
    state_size=25,  # 5x5 grid
    action_size=4,  # 4 actions
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon=0.1
)

# Train the agent
agent.train(env, episodes=500)

# Evaluate performance
avg_reward, success_rate = agent.evaluate(env, episodes=100)
print(f"Success Rate: {success_rate:.2%}")
```

### Custom Environment

```python
# Create custom grid world
env = GridWorld(size=8)  # 8x8 grid
env.start_pos = (1, 1)   # Custom start position
env.goal_pos = (6, 6)    # Custom goal position

# Train with different parameters
agent = QLearningAgent(
    state_size=64,
    action_size=4,
    learning_rate=0.2,    # Higher learning rate
    discount_factor=0.9,  # Lower discount factor
    epsilon=0.05          # Less exploration
)
```

## 📊 Understanding the Results

### Q-Table Interpretation

The Q-table shows the expected future reward for taking each action in each state:
- **Higher values** indicate better actions
- **Lower values** indicate worse actions
- **Optimal policy** is extracted by choosing actions with highest Q-values

### Training Metrics

- **Episode Rewards**: Total reward per episode (should increase over time)
- **Episode Lengths**: Steps per episode (should decrease over time)
- **Success Rate**: Percentage of episodes that reach the goal

### Visualization

The project includes several visualization tools:
- Training progress plots
- Q-table heatmaps for each action
- Grid world rendering
- Agent behavior demonstration

## 🎓 Key Learning Outcomes

After completing this tutorial, you will understand:

1. **Q-Learning Algorithm**: Mathematical foundation and implementation
2. **Reinforcement Learning Concepts**: States, actions, rewards, policies
3. **Exploration vs Exploitation**: Balancing learning and performance
4. **Parameter Tuning**: How hyperparameters affect learning
5. **Environment Design**: Creating custom RL environments
6. **Evaluation Methods**: Assessing agent performance

## 🔬 Advanced Topics

### Extensions to Explore

- **Deep Q-Networks (DQN)**: Neural network-based Q-Learning
- **SARSA**: On-policy temporal difference learning
- **Multi-Agent Systems**: Multiple agents learning together
- **Continuous State Spaces**: Handling continuous environments
- **Hierarchical Learning**: Learning at multiple abstraction levels

### Real-World Applications

- **Robotics**: Navigation and manipulation tasks
- **Game AI**: Strategy games and simulations
- **Autonomous Systems**: Self-driving cars, drones
- **Resource Management**: Optimization problems
- **Recommendation Systems**: Personalized content delivery

## 🤝 Contributing

This project is designed for learning purposes. Feel free to:
- Experiment with different parameters
- Modify the environment design
- Add new features or algorithms
- Share your findings and improvements

## 📖 Additional Resources

### Books
- "Reinforcement Learning: An Introduction" by Sutton & Barto
- "Deep Reinforcement Learning" by Pieter Abbeel

### Online Courses
- Coursera: Reinforcement Learning Specialization
- Udacity: Deep Reinforcement Learning Nanodegree
- MIT OpenCourseWare: Introduction to Machine Learning

### Research Papers
- "Q-Learning" by Watkins (1989)
- "Playing Atari with Deep Reinforcement Learning" by Mnih et al. (2013)

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Visualization Issues**: Check if matplotlib backend is working
3. **Memory Issues**: Reduce grid size or number of episodes
4. **Slow Training**: Adjust learning parameters or use smaller environment

### Getting Help

- Check the test script output for error messages
- Verify Python version compatibility
- Ensure all required packages are installed correctly

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Richard S. Sutton and Andrew G. Barto for foundational RL concepts
- The reinforcement learning research community
- Open source contributors to numpy, matplotlib, and seaborn

---

**Happy Learning! 🚀**

Start with the tutorial and gradually explore more complex implementations. Q-Learning is a powerful algorithm that forms the foundation for many advanced reinforcement learning techniques.
