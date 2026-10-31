"""
q_agent.py
----------
Pedagogical Q-Learning Agent.

Implements the fundamental Temporal Difference (TD) control algorithm:
    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

Features:
- Epsilon-greedy action selection (balances exploration & exploitation).
- Detailed arithmetic breakdown on every update for teaching/classroom inspection.
- Clean state-action table inspection.
"""

import random
from typing import Tuple, Dict, Any, List
from gridworld import ACTIONS, ACTION_NAMES


class QLearningAgent:
    """
    Q-Learning Agent with transparent Bellman TD calculations.
    """

    def __init__(
        self,
        learning_rate: float = 0.2,
        discount_factor: float = 0.9,
        epsilon: float = 0.2,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        """
        Args:
            learning_rate (alpha): How much new information overrides old Q-value (0.0 to 1.0).
            discount_factor (gamma): Importance of future rewards (0.0 = myopic, 1.0 = far-sighted).
            epsilon (eps): Probability of choosing a random exploratory action.
            epsilon_min: Lower bound for epsilon decay.
            epsilon_decay: Multiplicative decay factor per episode.
        """
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: mapping (row, col) -> {0: q_up, 1: q_down, 2: q_left, 3: q_right}
        self.q_table: Dict[Tuple[int, int], Dict[int, float]] = {}

    def get_q_values(self, state: Tuple[int, int]) -> Dict[int, float]:
        """Returns Q-values for all actions in a state. Initializes with 0.0 if unseen."""
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in ACTIONS}
        return self.q_table[state]

    def get_q(self, state: Tuple[int, int], action: int) -> float:
        """Returns Q(state, action)."""
        return self.get_q_values(state)[action]

    def choose_action(self, state: Tuple[int, int], force_greedy: bool = False) -> Tuple[int, bool]:
        """
        Selects an action using the epsilon-greedy strategy.
        
        Returns:
            (action, is_exploratory): Tuple indicating action taken and whether it was an exploration.
        """
        q_values = self.get_q_values(state)

        # Exploration: random action with probability epsilon
        if not force_greedy and random.random() < self.epsilon:
            return random.choice(ACTIONS), True

        # Exploitation: pick action with maximum Q-value
        max_q = max(q_values.values())
        # Break ties randomly among best actions
        best_actions = [a for a, q in q_values.items() if abs(q - max_q) < 1e-7]
        return random.choice(best_actions), False

    def get_best_action(self, state: Tuple[int, int]) -> int:
        """Returns the purely greedy action for a state."""
        action, _ = self.choose_action(state, force_greedy=True)
        return action

    def update(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
        done: bool
    ) -> Dict[str, Any]:
        """
        Performs one Bellman update:
            TD Target = reward + gamma * max_a' Q(next_state, a')   (if not done)
            TD Target = reward                                     (if done)
            TD Error  = TD Target - Q(state, action)
            Q_new     = Q_old + alpha * TD Error

        Returns:
            math_breakdown: Detailed step-by-step arithmetic for pedagogical display.
        """
        q_old = self.get_q(state, action)

        if done:
            max_q_next = 0.0
            td_target = reward
        else:
            next_q_values = self.get_q_values(next_state)
            max_q_next = max(next_q_values.values())
            td_target = reward + self.gamma * max_q_next

        td_error = td_target - q_old
        q_new = q_old + self.alpha * td_error

        # Update Q-table
        self.q_table[state][action] = q_new

        # Return full calculation trace for classroom explanation
        return {
            "state": state,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "q_old": q_old,
            "max_q_next": max_q_next,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "td_target": td_target,
            "td_error": td_error,
            "q_new": q_new
        }

    def decay_epsilon(self) -> float:
        """Decays epsilon by decay factor down to epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def reset_q_table(self):
        """Clears all learned values."""
        self.q_table.clear()
