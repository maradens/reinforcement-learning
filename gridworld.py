"""
gridworld.py
------------
A clean, pedagogical GridWorld environment for teaching Reinforcement Learning.
Features:
- Discrete states: (row, col) coordinates.
- Discrete actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT.
- Customizable cell types: Start (S), Empty (.), Goal (G), Trap (T), Wall (W).
- Standard RL interface similar to Gymnasium: reset(), step(action).
"""

from typing import Tuple, List, Dict, Optional
import copy


# Action mappings
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
ACTION_NAMES = {
    ACTION_UP: "UP",
    ACTION_DOWN: "DOWN",
    ACTION_LEFT: "LEFT",
    ACTION_RIGHT: "RIGHT"
}
ACTION_DELTAS = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1)
}


# Pre-defined classroom map layouts
PRESET_MAPS = {
    "Classic 4x4": [
        "S . . .",
        ". W . T",
        ". . . T",
        "W . . G"
    ],
    "Cliff Walking (4x8)": [
        ". . . . . . . .",
        ". . . . . . . .",
        ". . . . . . . .",
        "S T T T T T T G"
    ],
    "Maze (5x5)": [
        "S . W . .",
        ". W . W .",
        ". . . . .",
        "W W W . W",
        ". . . . G"
    ]
}


class GridWorld:
    """
    GridWorld Environment for Reinforcement Learning.
    
    Rewards:
    - Reaching Goal: +10.0
    - Falling into Trap: -10.0
    - Every normal step: -0.1 (living penalty, incentivizes finding shortest path)
    """

    def __init__(
        self,
        grid_layout: Optional[List[str]] = None,
        preset_name: str = "Classic 4x4",
        step_penalty: float = -0.1,
        goal_reward: float = 10.0,
        trap_penalty: float = -10.0
    ):
        if grid_layout is None:
            raw_layout = PRESET_MAPS.get(preset_name, PRESET_MAPS["Classic 4x4"])
        else:
            raw_layout = grid_layout

        # Parse string rows into 2D grid
        self.grid = [row.split() for row in raw_layout]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.trap_penalty = trap_penalty

        # Locate key positions
        self.start_pos: Tuple[int, int] = (0, 0)
        self.goal_positions: List[Tuple[int, int]] = []
        self.trap_positions: List[Tuple[int, int]] = []
        self.wall_positions: List[Tuple[int, int]] = []

        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.grid[r][c]
                if cell == 'S':
                    self.start_pos = (r, c)
                elif cell == 'G':
                    self.goal_positions.append((r, c))
                elif cell == 'T':
                    self.trap_positions.append((r, c))
                elif cell == 'W':
                    self.wall_positions.append((r, c))

        self.agent_pos = self.start_pos

    def reset(self) -> Tuple[int, int]:
        """Resets agent to start position."""
        self.agent_pos = self.start_pos
        return self.agent_pos

    def is_terminal(self, state: Tuple[int, int]) -> bool:
        """Returns True if state is Goal or Trap."""
        return (state in self.goal_positions) or (state in self.trap_positions)

    def is_wall(self, state: Tuple[int, int]) -> bool:
        """Returns True if cell is an impassable wall."""
        return state in self.wall_positions

    def get_valid_states(self) -> List[Tuple[int, int]]:
        """Returns all walkable states in the grid (excluding walls)."""
        states = []
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in self.wall_positions:
                    states.append((r, c))
        return states

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, dict]:
        """
        Executes one action.
        
        Returns:
            next_state: (row, col)
            reward: float
            done: bool (True if terminal state reached)
            info: dict with transition details
        """
        if self.is_terminal(self.agent_pos):
            # Already in terminal state
            return self.agent_pos, 0.0, True, {"msg": "Already terminal"}

        dr, dc = ACTION_DELTAS[action]
        candidate_r = self.agent_pos[0] + dr
        candidate_c = self.agent_pos[1] + dc

        # Check grid boundaries and walls
        if (
            0 <= candidate_r < self.rows
            and 0 <= candidate_c < self.cols
            and (candidate_r, candidate_c) not in self.wall_positions
        ):
            next_state = (candidate_r, candidate_c)
        else:
            # Hit wall or boundary: bounce back / stay in place
            next_state = self.agent_pos

        self.agent_pos = next_state

        # Calculate reward and terminal condition
        if next_state in self.goal_positions:
            reward = self.goal_reward
            done = True
        elif next_state in self.trap_positions:
            reward = self.trap_penalty
            done = True
        else:
            reward = self.step_penalty
            done = False

        info = {
            "action_name": ACTION_NAMES[action],
            "is_goal": next_state in self.goal_positions,
            "is_trap": next_state in self.trap_positions
        }

        return next_state, reward, done, info
