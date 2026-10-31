"""
rl_classroom_gui.py
-------------------
Interactive Reinforcement Learning Classroom Simulation GUI (Tkinter).

Designed specifically for teachers and instructors to demonstrate:
1. Environment dynamics (States, Actions, Rewards, Terminal states).
2. Q-Table intuition (Live directional arrows & numeric Q-values per cell).
3. Epsilon-Greedy action selection (Live Exploration vs Exploitation indicator).
4. The Bellman Equation / TD Error step-by-step arithmetic.
5. Hyperparameter sensitivity (Learning rate alpha, Discount factor gamma, Epsilon).

Run with:
    python rl_classroom_gui.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from typing import Tuple, Optional

from gridworld import (
    GridWorld, PRESET_MAPS, ACTIONS, ACTION_NAMES,
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT
)
from q_agent import QLearningAgent


class RLClassroomGUI:
    """Classroom Demonstration GUI for Q-Learning."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Reinforcement Learning Classroom Simulation — Q-Learning & Bellman Equation")
        self.root.geometry("1280x820")
        self.root.minsize(1050, 720)

        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Core RL Components
        self.current_preset = "Classic 4x4"
        self.env = GridWorld(preset_name=self.current_preset)
        self.agent = QLearningAgent(
            learning_rate=0.2,
            discount_factor=0.9,
            epsilon=0.2,
            epsilon_decay=0.995
        )

        # Training & Animation state
        self.is_running = False
        self.is_training_fast = False
        self.current_episode = 0
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.last_action_info = "Ready to start"
        self.last_choice_type = "None"

        # Build UI Layout
        self._build_ui()
        self.render_grid()

    # =========================================================================
    # UI CONSTRUCTION
    # =========================================================================
    def _build_ui(self):
        # Main container: Left panel (Controls & Inspector) | Right panel (Grid View)
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ------------------ LEFT SIDEBAR ------------------
        left_frame = ttk.Frame(main_pane, padding=10)
        main_pane.add(left_frame, weight=1)

        # Title Banner
        title_lbl = tk.Label(
            left_frame,
            text="Reinforcement Learning Demonstration",
            font=("Segoe UI", 14, "bold"),
            fg="#1E293B"
        )
        title_lbl.pack(anchor="w", pady=(0, 4))
        subtitle_lbl = tk.Label(
            left_frame,
            text="Interactive Bellman Update & Policy Visualization",
            font=("Segoe UI", 9),
            fg="#64748B"
        )
        subtitle_lbl.pack(anchor="w", pady=(0, 10))

        # 1. Preset Environment Selection
        preset_group = ttk.LabelFrame(left_frame, text="1. Environment Map", padding=8)
        preset_group.pack(fill=tk.X, pady=(0, 8))

        preset_row = ttk.Frame(preset_group)
        preset_row.pack(fill=tk.X)
        ttk.Label(preset_row, text="Preset:").pack(side=tk.LEFT, padx=(0, 6))
        self.preset_combo = ttk.Combobox(
            preset_row,
            values=list(PRESET_MAPS.keys()),
            state="readonly",
            width=20
        )
        self.preset_combo.set(self.current_preset)
        self.preset_combo.pack(side=tk.LEFT, padx=(0, 6))
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_change)

        btn_reset_env = ttk.Button(preset_row, text="Reload Map", command=self.reload_environment)
        btn_reset_env.pack(side=tk.LEFT)

        # 2. Classroom Simulation Controls
        ctrl_group = ttk.LabelFrame(left_frame, text="2. Simulation Controls", padding=8)
        ctrl_group.pack(fill=tk.X, pady=(0, 8))

        row1 = ttk.Frame(ctrl_group)
        row1.pack(fill=tk.X, pady=2)
        self.btn_step = ttk.Button(row1, text="▶ Step (1 Action)", command=self.step_one_action)
        self.btn_step.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.btn_episode = ttk.Button(row1, text="⏩ Run 1 Episode", command=self.run_one_episode)
        self.btn_episode.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        row2 = ttk.Frame(ctrl_group)
        row2.pack(fill=tk.X, pady=2)
        self.btn_train_50 = ttk.Button(row2, text="⚡ Fast Train (50)", command=lambda: self.fast_train(50))
        self.btn_train_50.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.btn_train_200 = ttk.Button(row2, text="⚡ Fast Train (200)", command=lambda: self.fast_train(200))
        self.btn_train_200.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        row3 = ttk.Frame(ctrl_group)
        row3.pack(fill=tk.X, pady=2)
        self.btn_stop = ttk.Button(row3, text="⏸ Stop / Pause", command=self.stop_execution, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.btn_reset_agent = ttk.Button(row3, text="↺ Reset Agent Pos", command=self.reset_agent_position)
        self.btn_reset_agent.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.btn_clear_q = ttk.Button(row3, text="🗑 Clear Q-Table", command=self.clear_q_table)
        self.btn_clear_q.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # 3. Hyperparameter Sliders
        param_group = ttk.LabelFrame(left_frame, text="3. Hyperparameters (Live Tuning)", padding=8)
        param_group.pack(fill=tk.X, pady=(0, 8))

        # Alpha (Learning Rate)
        alpha_row = ttk.Frame(param_group)
        alpha_row.pack(fill=tk.X, pady=2)
        ttk.Label(alpha_row, text="Learning Rate (α):", width=18).pack(side=tk.LEFT)
        self.alpha_var = tk.DoubleVar(value=self.agent.alpha)
        self.alpha_slider = ttk.Scale(
            alpha_row, from_=0.01, to=1.0, variable=self.alpha_var, command=self._on_alpha_change
        )
        self.alpha_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.alpha_lbl = ttk.Label(alpha_row, text=f"{self.agent.alpha:.2f}", width=5)
        self.alpha_lbl.pack(side=tk.RIGHT)

        # Gamma (Discount Factor)
        gamma_row = ttk.Frame(param_group)
        gamma_row.pack(fill=tk.X, pady=2)
        ttk.Label(gamma_row, text="Discount Factor (γ):", width=18).pack(side=tk.LEFT)
        self.gamma_var = tk.DoubleVar(value=self.agent.gamma)
        self.gamma_slider = ttk.Scale(
            gamma_row, from_=0.0, to=1.0, variable=self.gamma_var, command=self._on_gamma_change
        )
        self.gamma_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.gamma_lbl = ttk.Label(gamma_row, text=f"{self.agent.gamma:.2f}", width=5)
        self.gamma_lbl.pack(side=tk.RIGHT)

        # Epsilon (Exploration Rate)
        eps_row = ttk.Frame(param_group)
        eps_row.pack(fill=tk.X, pady=2)
        ttk.Label(eps_row, text="Exploration Rate (ε):", width=18).pack(side=tk.LEFT)
        self.eps_var = tk.DoubleVar(value=self.agent.epsilon)
        self.eps_slider = ttk.Scale(
            eps_row, from_=0.0, to=1.0, variable=self.eps_var, command=self._on_eps_change
        )
        self.eps_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.eps_lbl = ttk.Label(eps_row, text=f"{self.agent.epsilon:.2f}", width=5)
        self.eps_lbl.pack(side=tk.RIGHT)

        # Speed (Step delay ms)
        spd_row = ttk.Frame(param_group)
        spd_row.pack(fill=tk.X, pady=2)
        ttk.Label(spd_row, text="Anim Delay (ms):", width=18).pack(side=tk.LEFT)
        self.speed_var = tk.IntVar(value=120)
        self.speed_slider = ttk.Scale(
            spd_row, from_=10, to=600, variable=self.speed_var, command=self._on_speed_change
        )
        self.speed_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.speed_lbl = ttk.Label(spd_row, text=f"{self.speed_var.get()}ms", width=5)
        self.speed_lbl.pack(side=tk.RIGHT)

        # 4. Live Bellman Arithmetic Inspector
        inspector_group = ttk.LabelFrame(left_frame, text="4. Live Bellman Math Inspector", padding=8)
        inspector_group.pack(fill=tk.BOTH, expand=True)

        self.inspector_text = tk.Text(
            inspector_group,
            wrap=tk.WORD,
            font=("Consolas", 9),
            height=10,
            bg="#0F172A",
            fg="#38BDF8",
            relief=tk.FLAT
        )
        self.inspector_text.pack(fill=tk.BOTH, expand=True)
        self._write_inspector_initial()

        # ------------------ RIGHT CANVAS & STATS ------------------
        right_frame = ttk.Frame(main_pane, padding=10)
        main_pane.add(right_frame, weight=3)

        # Top Stats Bar
        stats_bar = ttk.Frame(right_frame)
        stats_bar.pack(fill=tk.X, pady=(0, 8))

        self.stat_ep_lbl = ttk.Label(stats_bar, text="Episode: 0", font=("Segoe UI", 10, "bold"))
        self.stat_ep_lbl.pack(side=tk.LEFT, padx=(0, 15))

        self.stat_step_lbl = ttk.Label(stats_bar, text="Step: 0", font=("Segoe UI", 10))
        self.stat_step_lbl.pack(side=tk.LEFT, padx=(0, 15))

        self.stat_rew_lbl = ttk.Label(stats_bar, text="Reward: 0.0", font=("Segoe UI", 10))
        self.stat_rew_lbl.pack(side=tk.LEFT, padx=(0, 15))

        self.stat_mode_badge = tk.Label(
            stats_bar,
            text="IDLE",
            font=("Segoe UI", 9, "bold"),
            bg="#94A3B8",
            fg="white",
            padx=8,
            pady=2
        )
        self.stat_mode_badge.pack(side=tk.RIGHT)

        # Main Canvas for Grid Visualization
        self.canvas_frame = ttk.Frame(right_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#F8FAFC", highlightthickness=1, highlightbackground="#CBD5E1")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.render_grid())

        # Legend Bar
        legend_bar = ttk.Frame(right_frame)
        legend_bar.pack(fill=tk.X, pady=(6, 0))

        self._create_legend_item(legend_bar, "#3B82F6", "Start (S)")
        self._create_legend_item(legend_bar, "#10B981", "Goal (+10)")
        self._create_legend_item(legend_bar, "#EF4444", "Trap (-10)")
        self._create_legend_item(legend_bar, "#475569", "Wall (Impassable)")
        self._create_legend_item(legend_bar, "#F59E0B", "Agent (Current Pos)")
        self._create_legend_item(legend_bar, "#6366F1", "Greedy Policy Arrow")

    def _create_legend_item(self, parent, color: str, text: str):
        item = ttk.Frame(parent)
        item.pack(side=tk.LEFT, padx=8)
        c = tk.Canvas(item, width=14, height=14, bg=color, highlightthickness=0)
        c.pack(side=tk.LEFT, padx=(0, 4))
        lbl = ttk.Label(item, text=text, font=("Segoe UI", 8))
        lbl.pack(side=tk.LEFT)

    def _write_inspector_initial(self):
        self.inspector_text.delete("1.0", tk.END)
        self.inspector_text.insert(
            tk.END,
            "=== BELLMAN EQUATION INSPECTOR ===\n\n"
            "Formula:\n"
            "  Q(s, a) <- Q(s, a) + α · TD_Error\n\n"
            "Where:\n"
            "  TD_Target = r + γ · max Q(s', a')\n"
            "  TD_Error  = TD_Target - Q(s, a)\n\n"
            "Click [Step (1 Action)] to watch this\n"
            "update live with exact numbers!"
        )

    # =========================================================================
    # EVENT HANDLERS & SLIDERS
    # =========================================================================
    def _on_preset_change(self, event=None):
        self.current_preset = self.preset_combo.get()
        self.reload_environment()

    def reload_environment(self):
        self.stop_execution()
        self.env = GridWorld(preset_name=self.current_preset)
        self.agent.reset_q_table()
        self.current_episode = 0
        self.episode_steps = 0
        self.episode_reward = 0.0
        self._write_inspector_initial()
        self._update_stats_bar()
        self.render_grid()

    def _on_alpha_change(self, val):
        v = float(val)
        self.agent.alpha = v
        self.alpha_lbl.config(text=f"{v:.2f}")

    def _on_gamma_change(self, val):
        v = float(val)
        self.agent.gamma = v
        self.gamma_lbl.config(text=f"{v:.2f}")

    def _on_eps_change(self, val):
        v = float(val)
        self.agent.epsilon = v
        self.eps_lbl.config(text=f"{v:.2f}")

    def _on_speed_change(self, val):
        v = int(float(val))
        self.speed_lbl.config(text=f"{v}ms")

    def reset_agent_position(self):
        self.env.reset()
        self.episode_steps = 0
        self.episode_reward = 0.0
        self._update_stats_bar()
        self.render_grid()

    def clear_q_table(self):
        self.agent.reset_q_table()
        self.current_episode = 0
        self.episode_steps = 0
        self.episode_reward = 0.0
        self._write_inspector_initial()
        self._update_stats_bar()
        self.render_grid()

    def stop_execution(self):
        self.is_running = False
        self.is_training_fast = False
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_step.config(state=tk.NORMAL)
        self.btn_episode.config(state=tk.NORMAL)
        self.btn_train_50.config(state=tk.NORMAL)
        self.btn_train_200.config(state=tk.NORMAL)
        self.stat_mode_badge.config(text="PAUSED", bg="#EAB308")

    # =========================================================================
    # SIMULATION EXECUTION LOGIC
    # =========================================================================
    def step_one_action(self):
        """Executes a single step, performs Bellman update, and updates inspector."""
        if self.env.is_terminal(self.env.agent_pos):
            self.env.reset()
            self.current_episode += 1
            self.episode_steps = 0
            self.episode_reward = 0.0

        state = self.env.agent_pos
        action, is_exploratory = self.agent.choose_action(state)
        next_state, reward, done, info = self.env.step(action)

        self.episode_steps += 1
        self.episode_reward += reward

        # Bellman update
        breakdown = self.agent.update(state, action, reward, next_state, done)

        # Update Live Math Inspector
        self._display_breakdown(breakdown, is_exploratory)

        # Update Badge
        if is_exploratory:
            self.stat_mode_badge.config(text="EXPLORE (Random)", bg="#8B5CF6")
        else:
            self.stat_mode_badge.config(text="EXPLOIT (Greedy)", bg="#10B981")

        self._update_stats_bar()
        self.render_grid()

        if done:
            self.agent.decay_epsilon()
            self.eps_var.set(self.agent.epsilon)
            self.eps_lbl.config(text=f"{self.agent.epsilon:.2f}")

    def _display_breakdown(self, b: dict, is_exploratory: bool):
        self.inspector_text.delete("1.0", tk.END)
        act_type = "EXPLORATION (Random Action)" if is_exploratory else "EXPLOITATION (Greedy Action)"

        next_state_str = str(b['next_state'])
        state_str = str(b['state'])

        text = (
            f"=== STEP UPDATE TRACE ===\n"
            f"Action Selected : {b['action_name']} [{act_type}]\n"
            f"Transition      : {state_str} -> {next_state_str}\n"
            f"Reward Received : {b['reward']:+.2f}\n"
            f"Terminal State? : {b['done']}\n\n"
            f"--- 1. Current State Evaluation ---\n"
            f"Old Q({state_str}, {b['action_name']}) = {b['q_old']:.4f}\n\n"
            f"--- 2. Next State Max Q ---\n"
            f"max_a' Q({next_state_str}, a') = {b['max_q_next']:.4f}\n\n"
            f"--- 3. Bellman TD Target & Error ---\n"
            f"TD Target = r + gamma * max Q(s')\n"
            f"          = {b['reward']:.2f} + ({b['gamma']:.2f} x {b['max_q_next']:.4f})\n"
            f"          = {b['td_target']:.4f}\n\n"
            f"TD Error  = TD Target - Old Q\n"
            f"          = {b['td_target']:.4f} - {b['q_old']:.4f}\n"
            f"          = {b['td_error']:+.4f}\n\n"
            f"--- 4. Updated Q-Value ---\n"
            f"New Q     = Old Q + alpha * TD Error\n"
            f"          = {b['q_old']:.4f} + ({b['alpha']:.2f} x {b['td_error']:+.4f})\n"
            f"          = {b['q_new']:.4f}\n"
        )
        self.inspector_text.insert(tk.END, text)

    def run_one_episode(self):
        """Runs one full episode with visual animations."""
        if self.is_running:
            return

        self.is_running = True
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_step.config(state=tk.DISABLED)
        self.btn_episode.config(state=tk.DISABLED)
        self.btn_train_50.config(state=tk.DISABLED)
        self.btn_train_200.config(state=tk.DISABLED)

        threading.Thread(target=self._episode_worker, daemon=True).start()

    def _episode_worker(self):
        self.env.reset()
        self.current_episode += 1
        self.episode_steps = 0
        self.episode_reward = 0.0

        done = False
        max_steps = 150

        while self.is_running and not done and self.episode_steps < max_steps:
            state = self.env.agent_pos
            action, is_exploratory = self.agent.choose_action(state)
            next_state, reward, done, info = self.env.step(action)

            self.episode_steps += 1
            self.episode_reward += reward

            breakdown = self.agent.update(state, action, reward, next_state, done)

            # Schedule UI update on main thread
            self.root.after(0, self._display_breakdown, breakdown, is_exploratory)
            self.root.after(0, self._update_stats_bar)
            self.root.after(0, self.render_grid)

            delay_sec = max(0.01, self.speed_var.get() / 1000.0)
            time.sleep(delay_sec)

        self.agent.decay_epsilon()
        self.root.after(0, self._finish_episode_run)

    def _finish_episode_run(self):
        self.is_running = False
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_step.config(state=tk.NORMAL)
        self.btn_episode.config(state=tk.NORMAL)
        self.btn_train_50.config(state=tk.NORMAL)
        self.btn_train_200.config(state=tk.NORMAL)
        self.eps_var.set(self.agent.epsilon)
        self.eps_lbl.config(text=f"{self.agent.epsilon:.2f}")
        self.stat_mode_badge.config(text="FINISHED", bg="#10B981")
        self.render_grid()

    def fast_train(self, num_episodes: int):
        """Trains multiple episodes rapidly without animation delay."""
        if self.is_running:
            return

        self.is_running = True
        self.is_training_fast = True
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_step.config(state=tk.DISABLED)
        self.btn_episode.config(state=tk.DISABLED)
        self.btn_train_50.config(state=tk.DISABLED)
        self.btn_train_200.config(state=tk.DISABLED)
        self.stat_mode_badge.config(text=f"TRAINING ({num_episodes} eps)", bg="#F97316")

        threading.Thread(target=self._fast_train_worker, args=(num_episodes,), daemon=True).start()

    def _fast_train_worker(self, num_episodes: int):
        for _ in range(num_episodes):
            if not self.is_running:
                break

            state = self.env.reset()
            self.current_episode += 1
            self.episode_steps = 0
            self.episode_reward = 0.0
            done = False
            step_count = 0

            while not done and step_count < 200:
                action, _ = self.agent.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.update(state, action, reward, next_state, done)
                state = next_state
                step_count += 1
                self.episode_reward += reward

            self.episode_steps = step_count
            self.agent.decay_epsilon()

        self.root.after(0, self._finish_fast_train)

    def _finish_fast_train(self):
        self.is_running = False
        self.is_training_fast = False
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_step.config(state=tk.NORMAL)
        self.btn_episode.config(state=tk.NORMAL)
        self.btn_train_50.config(state=tk.NORMAL)
        self.btn_train_200.config(state=tk.NORMAL)
        self.eps_var.set(self.agent.epsilon)
        self.eps_lbl.config(text=f"{self.agent.epsilon:.2f}")
        self.stat_mode_badge.config(text="TRAINED", bg="#10B981")
        self._update_stats_bar()
        self.render_grid()

    def _update_stats_bar(self):
        self.stat_ep_lbl.config(text=f"Episode: {self.current_episode}")
        self.stat_step_lbl.config(text=f"Step: {self.episode_steps}")
        self.stat_rew_lbl.config(text=f"Reward: {self.episode_reward:+.1f}")

    # =========================================================================
    # GRID RENDERING & VECTOR FIELD ARROWS
    # =========================================================================
    def render_grid(self):
        """Draws the grid, cell backgrounds, Q-values, policy arrows, and agent."""
        self.canvas.delete("all")

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width < 50 or height < 50:
            return

        rows = self.env.rows
        cols = self.env.cols

        cell_w = width / cols
        cell_h = height / rows

        for r in range(rows):
            for c in range(cols):
                x1 = c * cell_w
                y1 = r * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h
                pos = (r, c)

                # Cell background color
                if pos in self.env.wall_positions:
                    bg_color = "#334155"  # Wall
                elif pos in self.env.goal_positions:
                    bg_color = "#DCFCE7"  # Soft green
                elif pos in self.env.trap_positions:
                    bg_color = "#FEE2E2"  # Soft red
                elif pos == self.env.start_pos:
                    bg_color = "#DBEAFE"  # Soft blue
                else:
                    bg_color = "#FFFFFF"

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=bg_color, outline="#CBD5E1", width=1)

                # Special cell label badges
                if pos in self.env.wall_positions:
                    self.canvas.create_text(
                        (x1 + x2) / 2, (y1 + y2) / 2,
                        text="WALL", fill="#94A3B8", font=("Segoe UI", 11, "bold")
                    )
                    continue

                if pos in self.env.goal_positions:
                    self.canvas.create_text(
                        (x1 + x2) / 2, (y1 + y2) / 2,
                        text="★ GOAL\n(+10)", fill="#166534", font=("Segoe UI", 11, "bold"), justify=tk.CENTER
                    )
                    continue

                if pos in self.env.trap_positions:
                    self.canvas.create_text(
                        (x1 + x2) / 2, (y1 + y2) / 2,
                        text="☠ TRAP\n(-10)", fill="#991B1B", font=("Segoe UI", 11, "bold"), justify=tk.CENTER
                    )
                    continue

                # Coordinate label in top-left
                coord_text = f"({r},{c})"
                if pos == self.env.start_pos:
                    coord_text += " [START]"
                self.canvas.create_text(x1 + 6, y1 + 8, text=coord_text, anchor="nw", fill="#94A3B8", font=("Segoe UI", 7))

                # Draw Q-values and directional policy arrows
                self._draw_cell_q_vectors(x1, y1, x2, y2, pos)

        # Draw the Agent Avatar
        ar, ac = self.env.agent_pos
        ax1 = ac * cell_w + cell_w * 0.28
        ay1 = ar * cell_h + cell_h * 0.28
        ax2 = ac * cell_w + cell_w * 0.72
        ay2 = ar * cell_h + cell_h * 0.72

        self.canvas.create_oval(ax1, ay1, ax2, ay2, fill="#F59E0B", outline="#B45309", width=3)
        self.canvas.create_text(
            (ax1 + ax2) / 2, (ay1 + ay2) / 2,
            text="AGENT", fill="#FFFFFF", font=("Segoe UI", 8, "bold")
        )

    def _draw_cell_q_vectors(self, x1: float, y1: float, x2: float, y2: float, pos: Tuple[int, int]):
        """Draws 4 small arrows with Q-values and highlights the greedy best direction."""
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        q_vals = self.agent.get_q_values(pos)

        # Find best action for highlighting
        max_q = max(q_vals.values())
        has_learned = any(abs(q) > 1e-4 for q in q_vals.values())

        # Vector offsets
        offset = min((x2 - x1), (y2 - y1)) * 0.32
        arrow_len = offset * 0.55

        directions = {
            ACTION_UP: (cx, cy - offset * 0.45, cx, cy - offset * 0.45 - arrow_len, f"{q_vals[ACTION_UP]:.2f}", cx, cy - offset * 0.9),
            ACTION_DOWN: (cx, cy + offset * 0.45, cx, cy + offset * 0.45 + arrow_len, f"{q_vals[ACTION_DOWN]:.2f}", cx, cy + offset * 0.9),
            ACTION_LEFT: (cx - offset * 0.45, cy, cx - offset * 0.45 - arrow_len, cy, f"{q_vals[ACTION_LEFT]:.2f}", cx - offset * 0.9, cy),
            ACTION_RIGHT: (cx + offset * 0.45, cy, cx + offset * 0.45 + arrow_len, cy, f"{q_vals[ACTION_RIGHT]:.2f}", cx + offset * 0.9, cy)
        }

        for action, (sx, sy, ex, ey, q_str, tx, ty) in directions.items():
            q = q_vals[action]
            is_best = has_learned and abs(q - max_q) < 1e-7

            if is_best:
                color = "#4F46E5"  # Bright indigo/violet for the greedy policy
                width = 3
                arrow_shape = (10, 12, 5)
            elif q > 0:
                color = "#10B981"  # Positive Q
                width = 1
                arrow_shape = (6, 8, 3)
            elif q < 0:
                color = "#F87171"  # Negative Q
                width = 1
                arrow_shape = (6, 8, 3)
            else:
                color = "#CBD5E1"  # Unvisited
                width = 1
                arrow_shape = (5, 6, 2)

            # Draw directional arrow
            self.canvas.create_line(sx, sy, ex, ey, fill=color, width=width, arrow=tk.LAST, arrowshape=arrow_shape)

            # Draw numeric Q-value label
            text_color = "#1E293B" if is_best else "#64748B"
            font_weight = "bold" if is_best else "normal"
            self.canvas.create_text(
                tx, ty,
                text=q_str,
                fill=text_color,
                font=("Consolas", 7, font_weight)
            )


def main():
    root = tk.Tk()
    app = RLClassroomGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
