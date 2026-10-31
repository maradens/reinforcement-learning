"""
build_notebook.py
-----------------
Helper script to generate reinforcement_learning_for_beginners.ipynb cleanly without shell escaping issues.
"""

import json

nb = {
    'cells': [],
    'metadata': {
        'language_info': {
            'name': 'python',
            'version': '3.13.14'
        },
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

def add_md(text):
    nb['cells'].append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in text.strip().split('\n')]
    })

def add_code(text):
    nb['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in text.strip().split('\n')]
    })

# --- CELL 1: Introduction ---
add_md(r"""# 🤖 Introduction to Reinforcement Learning (Q-Learning)

Welcome to this interactive tutorial on **Reinforcement Learning (RL)**! 

Unlike Supervised Learning (where models learn from labeled examples) or Unsupervised Learning (where models find hidden patterns in unlabeled data), **Reinforcement Learning is learning by trial and error**:
- An **Agent** interacts with an **Environment**.
- At each step, the agent observes the current **State** ($s$).
- The agent selects an **Action** ($a$).
- The environment provides feedback: a scalar **Reward** ($r$) and transitions to a **Next State** ($s'$).
- **The Ultimate Goal**: Learn an optimal **Policy** ($\pi^*$) that maximizes cumulative rewards over time.

---

## The Reinforcement Learning Feedback Loop

```
            +----------------------------------+
            |                                  |
            v                                  |
      +------------+       Action (a)     +-----------------+
      |            | -------------------> |                 |
      |   Agent    |                      |   Environment   |
      |            | <------------------- |                 |
      +------------+   Reward (r)         +-----------------+
                         & State (s')
```
""")

# --- CELL 2: Mathematics ---
add_md(r"""## 1. The Core Mathematics: The Bellman Equation

How does an agent evaluate whether an action is "good" if the reward is delayed?

We define the **Q-value** $Q(s, a)$ as the expected total future discounted reward of taking action $a$ in state $s$:

$$Q(s, a) = \mathbb{E} \left[ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \mid S_t = s, A_t = a \right]$$

### The Q-Learning Update Rule (Temporal Difference Learning)
When the agent takes action $a$ from state $s$, receives reward $r$, and lands in state $s'$, it updates its belief:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \cdot \underbrace{\Big( \overbrace{r + \gamma \max_{a'} Q(s', a')}^{\text{TD Target}} - Q(s, a) \Big)}_{\text{TD Error (Surprise)}}$$

Where:
- $\alpha$ (**Learning Rate**, $0 < \alpha \le 1$): How quickly the agent updates its beliefs with new experience.
- $\gamma$ (**Discount Factor**, $0 \le \gamma \le 1$): How much the agent cares about long-term future rewards vs. immediate rewards.
- $\epsilon$ (**Exploration Rate** in $\epsilon$-greedy): Balance between trying unknown actions (**Exploration**) vs. picking the currently known best action (**Exploitation**).
""")

# --- CELL 3: Imports ---
add_code(r"""# Setup imports
import numpy as np
import matplotlib.pyplot as plt
import random

# Import our pedagogical GridWorld and Q-Learning agent
from gridworld import GridWorld, ACTION_NAMES, ACTIONS
from q_agent import QLearningAgent

print("All modules loaded successfully!")
""")

# --- CELL 4: Environment intro ---
add_md(r"""## 2. Inspecting the GridWorld Environment

Let's initialize our 4x4 GridWorld environment.
- **S**: Start position `(0, 0)`
- **G**: Goal position `(3, 3)` with reward **+10.0**
- **T**: Trap positions with penalty **-10.0**
- **W**: Impassable Wall/Obstacle
- Each regular move has a step penalty of **-0.1** to encourage the agent to find the shortest path.
""")

# --- CELL 5: Environment setup ---
add_code(r"""# Initialize environment
env = GridWorld(preset_name="Classic 4x4", step_penalty=-0.1)

print(f"Grid Size: {env.rows} rows x {env.cols} cols")
print(f"Start Position: {env.start_pos}")
print(f"Goal Positions: {env.goal_positions}")
print(f"Trap Positions: {env.trap_positions}")
print(f"Wall Positions: {env.wall_positions}")
""")

# --- CELL 6: Training agent intro ---
add_md(r"""## 3. Creating and Training the Q-Learning Agent

We will create an agent and train it for 250 episodes.
Watch how the agent transitions from wandering randomly (high $\epsilon$) to consistently reaching the goal via the optimal path!
""")

# --- CELL 7: Training Loop ---
add_code(r"""# Instantiate our Q-learning agent
agent = QLearningAgent(
    learning_rate=0.2,     # alpha
    discount_factor=0.9,   # gamma
    epsilon=0.4,           # initial exploration probability
    epsilon_min=0.01,
    epsilon_decay=0.985
)

# Training loop
num_episodes = 250
rewards_history = []
steps_history = []

for ep in range(1, num_episodes + 1):
    state = env.reset()
    ep_reward = 0.0
    steps = 0
    done = False
    
    while not done and steps < 100:
        action, _ = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        
        state = next_state
        ep_reward += reward
        steps += 1
        
    agent.decay_epsilon()
    rewards_history.append(ep_reward)
    steps_history.append(steps)
    
    if ep % 50 == 0 or ep == 1:
        avg_rew = np.mean(rewards_history[-20:])
        avg_step = np.mean(steps_history[-20:])
        print(f"Episode {ep:3d} | Avg Reward (last 20): {avg_rew:+.2f} | Avg Steps: {avg_step:.1f} | Epsilon: {agent.epsilon:.3f}")
""")

# --- CELL 8: Plotting curves intro ---
add_md(r"""## 4. Visualizing Training Curves

Let's plot:
1. **Total Reward per Episode**: Shows the agent transitioning from negative/low rewards to the maximum possible reward.
2. **Steps Taken per Episode**: Shows the agent finding the most efficient, shortest path.
""")

# --- CELL 9: Plotting curves code ---
add_code(r"""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

# Rewards plot
ax1.plot(rewards_history, color="#93C5FD", alpha=0.6, label="Episode Reward")
window = 15
if len(rewards_history) >= window:
    rolling = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
    ax1.plot(range(window-1, len(rewards_history)), rolling, color="#1D4ED8", linewidth=2, label=f"{window}-Ep Moving Avg")
ax1.set_title("Learning Curve: Total Reward", fontweight="bold")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Reward")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend()

# Steps plot
ax2.plot(steps_history, color="#FCA5A5", alpha=0.6, label="Steps Taken")
if len(steps_history) >= window:
    rolling_steps = np.convolve(steps_history, np.ones(window)/window, mode='valid')
    ax2.plot(range(window-1, len(steps_history)), rolling_steps, color="#DC2626", linewidth=2, label=f"{window}-Ep Moving Avg")
ax2.set_title("Path Efficiency: Steps to Terminal", fontweight="bold")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Number of Steps")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.show()
""")

# --- CELL 10: Policy visualization intro ---
add_md(r"""## 5. Visualizing the Learned Policy & Value Function

Now let's examine what the agent actually learned. 
We can plot the **State Value Function** $V(s) = \max_a Q(s, a)$ as a heatmap and overlay arrows for the optimal policy $\pi^*(s) = \arg\max_a Q(s, a)$.
""")

# --- CELL 11: Policy visualization code ---
add_code(r"""fig, ax = plt.subplots(figsize=(6, 6))

value_grid = np.zeros((env.rows, env.cols))
arrow_map = {0: (0, -0.25), 1: (0, 0.25), 2: (-0.25, 0), 3: (0.25, 0)}

for r in range(env.rows):
    for c in range(env.cols):
        pos = (r, c)
        if pos in env.wall_positions:
            value_grid[r, c] = np.nan
        elif pos in env.goal_positions:
            value_grid[r, c] = env.goal_reward
        elif pos in env.trap_positions:
            value_grid[r, c] = env.trap_penalty
        else:
            q_vals = agent.get_q_values(pos)
            value_grid[r, c] = max(q_vals.values())

cmap = plt.cm.RdYlGn
cmap.set_bad(color="#334155")
im = ax.imshow(value_grid, cmap=cmap, vmin=-10, vmax=10)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="State Value V(s)")

for r in range(env.rows):
    for c in range(env.cols):
        pos = (r, c)
        if pos in env.wall_positions:
            ax.text(c, r, "WALL", ha="center", va="center", color="white", fontweight="bold")
        elif pos in env.goal_positions:
            ax.text(c, r, "GOAL\n+10", ha="center", va="center", color="black", fontweight="bold")
        elif pos in env.trap_positions:
            ax.text(c, r, "TRAP\n-10", ha="center", va="center", color="black", fontweight="bold")
        else:
            best_a = agent.get_best_action(pos)
            dx, dy = arrow_map[best_a]
            ax.arrow(c, r, dx, dy, head_width=0.15, head_length=0.15, fc="#0F172A", ec="#0F172A", linewidth=1.5)
            ax.text(c, r + 0.35, f"{value_grid[r, c]:.2f}", ha="center", va="center", fontsize=9, fontweight="bold")

ax.set_title("Learned Policy & Value Function", fontweight="bold")
ax.set_xticks(range(env.cols))
ax.set_yticks(range(env.rows))
ax.grid(color="white", linestyle="-", linewidth=1.5)
plt.show()
""")

# --- CELL 12: Rollout intro ---
add_md(r"""## 6. Demonstration Rollout: Watching the Agent Navigate

Let's execute the greedy policy step-by-step from Start to Goal.
""")

# --- CELL 13: Rollout code ---
add_code(r"""state = env.reset()
done = False
step = 0
path = [state]

print("Executing Optimal Greedy Path:")
while not done and step < 15:
    action = agent.get_best_action(state)
    next_state, reward, done, _ = env.step(action)
    print(f"Step {step+1}: State {state} -> Action {ACTION_NAMES[action]:<5} -> Next State {next_state} (Reward {reward:+.1f})")
    state = next_state
    path.append(state)
    step += 1

print(f"\nFull Path: {' -> '.join(str(s) for s in path)}")
if state in env.goal_positions:
    print("Agent successfully navigated to the GOAL!")
""")

# --- CELL 14: Discussion & Exercises ---
add_md(r"""## 7. Classroom Discussion Questions & Hands-On Exercises

### Exercise 1: What happens if $\gamma = 0$ (Myopic Agent)?
- Set `discount_factor=0.0`.
- Retrain the agent.
- **Question**: Why does the agent fail to reach the goal when $\gamma=0$? 
  *(Hint: Without discount factor, future rewards at the goal never propagate backward to earlier states!)*

### Exercise 2: What happens if $\epsilon = 0$ from the beginning?
- Set `epsilon=0.0`.
- **Question**: Does the agent ever find the goal? Why is exploration crucial in unknown environments?

### Exercise 3: Try Cliff Walking!
- Switch the environment to `GridWorld(preset_name="Cliff Walking (4x8)")`.
- Observe how Q-learning learns to steer clear of the cliff edge.
""")

with open('reinforcement_learning_for_beginners.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print('Notebook reinforcement_learning_for_beginners.ipynb generated successfully!')
