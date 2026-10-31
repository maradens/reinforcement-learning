"""
q_learning_demo.py
------------------
Headless / Scripted Demonstration of Q-Learning with Matplotlib Visualizations.

Generates:
1. Training curves (Episode Reward vs Episode, Steps to Goal vs Episode).
2. Learned Value Function Heatmap (V(s) = max_a Q(s, a)).
3. Visualized Greedy Policy Grid with arrows.

Run with:
    python q_learning_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from gridworld import (
    GridWorld, ACTIONS, ACTION_NAMES,
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT
)
from q_agent import QLearningAgent


def train_agent(
    env: GridWorld,
    agent: QLearningAgent,
    num_episodes: int = 300,
    max_steps_per_episode: int = 100
) -> Tuple[List[float], List[int]]:
    """
    Trains the Q-learning agent on the GridWorld environment.
    
    Returns:
        rewards_history: Cumulative reward per episode.
        steps_history: Steps taken per episode.
    """
    rewards_history = []
    steps_history = []

    print(f"Training Q-Learning Agent for {num_episodes} episodes...")
    print(f"Initial Epsilon: {agent.epsilon:.2f}, Alpha: {agent.alpha:.2f}, Gamma: {agent.gamma:.2f}")

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        steps = 0
        done = False

        while not done and steps < max_steps_per_episode:
            action, _ = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            steps += 1

        agent.decay_epsilon()
        rewards_history.append(episode_reward)
        steps_history.append(steps)

        if ep % 50 == 0 or ep == num_episodes:
            recent_avg_rew = np.mean(rewards_history[-20:])
            recent_avg_step = np.mean(steps_history[-20:])
            print(f"Episode {ep:3d}/{num_episodes} | Avg Reward (last 20): {recent_avg_rew:+.2f} | Avg Steps: {recent_avg_step:.1f} | Epsilon: {agent.epsilon:.3f}")

    return rewards_history, steps_history


def plot_results(
    env: GridWorld,
    agent: QLearningAgent,
    rewards: List[float],
    steps: List[int],
    save_path: str = "q_learning_results.png"
):
    """Generates a 3-panel figure showing training curves and the learned policy/value heatmap."""
    fig = plt.figure(figsize=(15, 5))

    # --- 1. Episode Rewards Curve ---
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(rewards, color="#93C5FD", alpha=0.6, label="Episode Reward")
    # Rolling average
    window = 15
    if len(rewards) >= window:
        rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), rolling_mean, color="#1D4ED8", linewidth=2, label=f"{window}-Ep Moving Avg")
    ax1.set_title("Training Performance: Total Reward", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative Reward")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # --- 2. Steps to Goal Curve ---
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(steps, color="#FCA5A5", alpha=0.6, label="Steps Taken")
    if len(steps) >= window:
        rolling_steps = np.convolve(steps, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(steps)), rolling_steps, color="#DC2626", linewidth=2, label=f"{window}-Ep Moving Avg")
    ax2.set_title("Efficiency: Steps per Episode", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Number of Steps")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    # --- 3. Value Heatmap & Learned Greedy Policy Arrows ---
    ax3 = fig.add_subplot(1, 3, 3)
    value_grid = np.zeros((env.rows, env.cols))

    arrow_map = {
        ACTION_UP: (0, -0.28),
        ACTION_DOWN: (0, 0.28),
        ACTION_LEFT: (-0.28, 0),
        ACTION_RIGHT: (0.28, 0)
    }

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

    # Plot value heatmap
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="#334155")  # Dark color for walls
    im = ax3.imshow(value_grid, cmap=cmap, vmin=-10, vmax=10)
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label="State Value V(s) = max_a Q(s,a)")

    # Overlay labels and policy arrows
    for r in range(env.rows):
        for c in range(env.cols):
            pos = (r, c)
            if pos in env.wall_positions:
                ax3.text(c, r, "WALL", ha="center", va="center", color="white", fontweight="bold")
            elif pos in env.goal_positions:
                ax3.text(c, r, "GOAL\n+10", ha="center", va="center", color="black", fontweight="bold")
            elif pos in env.trap_positions:
                ax3.text(c, r, "TRAP\n-10", ha="center", va="center", color="black", fontweight="bold")
            else:
                best_a = agent.get_best_action(pos)
                dx, dy = arrow_map[best_a]
                # Arrow pointing in optimal direction
                ax3.arrow(
                    c, r, dx, dy,
                    head_width=0.15, head_length=0.15,
                    fc="#1E293B", ec="#1E293B", linewidth=1.5
                )
                # Value label
                ax3.text(c, r + 0.35, f"{value_grid[r, c]:.2f}", ha="center", va="center", fontsize=8, color="#0F172A", fontweight="semibold")

    ax3.set_title("Learned Value Function & Policy (π*)", fontsize=11, fontweight="bold")
    ax3.set_xticks(range(env.cols))
    ax3.set_yticks(range(env.rows))
    ax3.grid(color="white", linestyle="-", linewidth=1.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"\nVisualization saved successfully to: {save_path}")
    plt.close()


def main():
    # 1. Initialize environment (Classic 4x4)
    env = GridWorld(preset_name="Classic 4x4")

    # 2. Initialize Q-Learning agent
    agent = QLearningAgent(
        learning_rate=0.2,
        discount_factor=0.9,
        epsilon=0.3,
        epsilon_min=0.01,
        epsilon_decay=0.99
    )

    # 3. Train agent
    rewards, steps = train_agent(env, agent, num_episodes=250)

    # 4. Generate and save plots
    plot_results(env, agent, rewards, steps, save_path="q_learning_results.png")

    # 5. Print a test rollout demonstrating the learned optimal path
    print("\n--- Demonstration Rollout (Greedy Policy) ---")
    state = env.reset()
    done = False
    step_count = 0
    path = [state]

    while not done and step_count < 20:
        action = agent.get_best_action(state)
        next_state, reward, done, info = env.step(action)
        print(f"Step {step_count+1}: State {state} -> Action: {ACTION_NAMES[action]:<5} -> Next: {next_state} (Reward: {reward:+.1f})")
        state = next_state
        path.append(state)
        step_count += 1

    print(f"\nFinal Path Taken: {' -> '.join(str(p) for p in path)}")
    if state in env.goal_positions:
        print("Success! Agent reached the GOAL along the optimal shortest path.")
    elif state in env.trap_positions:
        print("Agent fell into a TRAP.")


if __name__ == "__main__":
    main()
