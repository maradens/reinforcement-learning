"""
AI 100m Dash - Humanoid Reinforcement Learning Desktop Demo
Inspired by AI Warehouse: "AI vs AI in 100m Dash (deep reinforcement learning)"

Built with standard Python + Tkinter + Numpy (zero heavy C-dependencies, runs on any laptop).
Includes model retention (auto-saving/loading checkpoints to 'humanoid_champion.json').
"""

import json
import math
import os
import random
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np


# =====================================================================
# ARTICULATED BIPED HUMANOID RUNNER PHYSICS & NEURAL POLICY
# =====================================================================

class HumanoidAgent:
    """
    Articulated Biped Ragdoll with 7 continuous segments:
    - Torso & Head
    - Left Thigh, Left Shin, Left Foot
    - Right Thigh, Right Shin, Right Foot
    - Counterbalancing Arm Swing
    """
    def __init__(self, name, color, lane_idx, brain_type="neural"):
        self.name = name
        self.color = color
        self.lane_idx = lane_idx
        self.brain_type = brain_type  # 'gen0', 'gen12', 'gen45', 'gen120', 'neural', 'custom_champion'
        
        # Dimensions & physical constants
        self.pixels_per_meter = 40.0
        self.track_length_meters = 100.0
        self.thigh_len = 28.0
        self.shin_len = 26.0
        self.torso_len = 32.0
        self.arm_len = 22.0
        
        # Neural network policy architecture (12 inputs -> 16 hidden -> 4 torques)
        self.num_inputs = 12
        self.num_hidden = 16
        self.num_outputs = 4
        
        self.w1 = (np.random.randn(self.num_hidden, self.num_inputs) * 0.4).astype(np.float32)
        self.b1 = np.zeros(self.num_hidden, dtype=np.float32)
        self.w2 = (np.random.randn(self.num_outputs, self.num_hidden) * 0.4).astype(np.float32)
        self.b2 = np.zeros(self.num_outputs, dtype=np.float32)
        
        self.fitness = 0.0
        self.generation = 0
        self.reset(lane_base_y=260.0)

    def reset(self, lane_base_y=260.0):
        self.ground_y = lane_base_y + self.lane_idx * 34.0
        self.x = 20.0
        self.y = self.ground_y - 65.0
        self.vx = 0.0
        self.vy = 0.0
        
        self.tilt = 0.05
        self.tilt_vel = 0.0
        
        # Joint angles in radians
        self.l_hip = 0.1
        self.l_knee = 0.2
        self.r_hip = -0.1
        self.r_knee = 0.2
        self.arm_swing = 0.0
        
        self.l_hip_vel = 0.0
        self.l_knee_vel = 0.0
        self.r_hip_vel = 0.0
        self.r_knee_vel = 0.0
        
        self.l_contact = False
        self.r_contact = False
        
        self.distance_m = 0.0
        self.top_speed = 0.0
        self.finished = False
        self.finish_time = None
        self.fallen = False
        
        self.last_torques = [0.0, 0.0, 0.0, 0.0]

    def get_motor_torques(self, race_time):
        if self.fallen:
            return [0.0, 0.0, 0.0, 0.0]

        # Stage 0: Random flailing (toddler)
        if self.brain_type == "gen0":
            return [
                math.sin(race_time * 11.0 + self.lane_idx) * 1.5,
                math.cos(race_time * 8.0) * 1.2,
                math.cos(race_time * 13.0) * 1.5,
                math.sin(race_time * 9.0) * 1.2
            ]

        # Stage 1: One-legged hopping / knee dragging
        elif self.brain_type == "gen12":
            p = race_time * 7.5
            return [
                math.sin(p) * 1.4 + 0.4,
                max(0.0, math.sin(p + 1.2) * 1.6),
                0.2,
                0.5
            ]

        # Stage 2: Alternating run, high body wobble
        elif self.brain_type == "gen45":
            p = race_time * 12.0
            balance = -self.tilt * 2.2 - self.tilt_vel * 0.8
            return [
                math.sin(p) * 1.5 + balance,
                max(0.0, math.sin(p - 1.2) * 1.8),
                -math.sin(p) * 1.5 + balance,
                max(0.0, -math.sin(p - 1.2) * 1.8)
            ]

        # Stage 3: Olympic Sprinter (PPO-optimized stride, forward lean & arm swing)
        elif self.brain_type == "gen120":
            p = race_time * 18.5
            forward_lean = 0.22
            balance = (forward_lean - self.tilt) * 3.5 - self.tilt_vel * 1.1
            return [
                math.sin(p) * 1.9 + balance + 0.3,
                max(-0.2, math.sin(p - 1.0) * 2.1),
                -math.sin(p) * 1.9 + balance + 0.3,
                max(-0.2, -math.sin(p - 1.0) * 2.1)
            ]

        # Generic / Custom Trained Neural Policy
        obs = np.array([
            (self.ground_y - self.y) / 70.0,
            self.tilt,
            self.vx / 10.0,
            self.vy / 10.0,
            self.tilt_vel,
            self.l_hip, self.l_knee,
            self.r_hip, self.r_knee,
            1.0 if self.l_contact else 0.0,
            1.0 if self.r_contact else 0.0,
            math.sin(race_time * 10.0)
        ], dtype=np.float32)

        hidden = np.tanh(np.dot(self.w1, obs) + self.b1)
        torques = np.tanh(np.dot(self.w2, hidden) + self.b2) * 2.0
        return torques.tolist()

    def update(self, dt, race_time):
        if self.finished:
            return

        if self.fallen:
            self.vx *= 0.94
            self.x += self.vx * dt
            self.distance_m = max(0.0, (self.x - 20.0) / self.pixels_per_meter)
            return

        torques = self.get_motor_torques(race_time)
        self.last_torques = torques
        t_l_hip, t_l_knee, t_r_hip, t_r_knee = torques

        damping = 8.0
        spring = 35.0

        # Left Leg
        self.l_hip_vel += (t_l_hip * 30.0 - self.l_hip * spring - self.l_hip_vel * damping) * dt
        self.l_hip += self.l_hip_vel * dt
        self.l_hip = max(-0.9, min(1.2, self.l_hip))

        self.l_knee_vel += (t_l_knee * 35.0 - self.l_knee * spring - self.l_knee_vel * damping) * dt
        self.l_knee += self.l_knee_vel * dt
        self.l_knee = max(0.0, min(2.1, self.l_knee))

        # Right Leg
        self.r_hip_vel += (t_r_hip * 30.0 - self.r_hip * spring - self.r_hip_vel * damping) * dt
        self.r_hip += self.r_hip_vel * dt
        self.r_hip = max(-0.9, min(1.2, self.r_hip))

        self.r_knee_vel += (t_r_knee * 35.0 - self.r_knee * spring - self.r_knee_vel * damping) * dt
        self.r_knee += self.r_knee_vel * dt
        self.r_knee = max(0.0, min(2.1, self.r_knee))

        # Arms swing counter to leg cycle
        self.arm_swing = (self.l_hip - self.r_hip) * 0.75

        # Foot positions
        l_knee_x = self.x + math.sin(self.tilt + self.l_hip) * self.thigh_len
        l_knee_y = self.y + math.cos(self.tilt + self.l_hip) * self.thigh_len
        l_foot_x = l_knee_x + math.sin(self.tilt + self.l_hip - self.l_knee) * self.shin_len
        l_foot_y = l_knee_y + math.cos(self.tilt + self.l_hip - self.l_knee) * self.shin_len

        r_knee_x = self.x + math.sin(self.tilt + self.r_hip) * self.thigh_len
        r_knee_y = self.y + math.cos(self.tilt + self.r_hip) * self.thigh_len
        r_foot_x = r_knee_x + math.sin(self.tilt + self.r_hip - self.r_knee) * self.shin_len
        r_foot_y = r_knee_y + math.cos(self.tilt + self.r_hip - self.r_knee) * self.shin_len

        self.l_contact = (l_foot_y >= self.ground_y)
        self.r_contact = (r_foot_y >= self.ground_y)

        gravity = 9.81 * 40.0
        ground_push_x = 0.0
        ground_push_y = 0.0
        ground_torque = 0.0

        if self.l_contact:
            penetration = l_foot_y - self.ground_y
            ground_push_y -= penetration * 280.0
            push_vel = (self.l_hip_vel - self.l_knee_vel) * 18.0
            ground_push_x += max(-30.0, min(180.0, push_vel))
            ground_torque += (l_foot_x - self.x) * 0.8

        if self.r_contact:
            penetration = r_foot_y - self.ground_y
            ground_push_y -= penetration * 280.0
            push_vel = (self.r_hip_vel - self.r_knee_vel) * 18.0
            ground_push_x += max(-30.0, min(180.0, push_vel))
            ground_torque += (r_foot_x - self.x) * 0.8

        self.vy += (gravity + ground_push_y) * dt
        self.vx += ground_push_x * dt
        self.vx *= 0.985
        self.vy *= 0.980

        self.tilt_vel += (ground_torque - self.tilt * 25.0 - self.tilt_vel * 5.0) * dt
        self.tilt += self.tilt_vel * dt

        self.x += self.vx * dt
        self.y += self.vy * dt

        if self.y >= self.ground_y - 24.0 or abs(self.tilt) > 1.25:
            self.fallen = True
            self.vx *= 0.4
            self.y = self.ground_y - 16.0
            self.tilt = 1.57 if self.tilt > 0 else -1.57

        self.distance_m = max(0.0, (self.x - 20.0) / self.pixels_per_meter)
        current_speed = self.vx / self.pixels_per_meter
        if current_speed > self.top_speed:
            self.top_speed = current_speed

        if self.distance_m >= self.track_length_meters and not self.finished:
            self.finished = True
            self.finish_time = race_time
            self.distance_m = self.track_length_meters

    def to_dict(self):
        return {
            "name": self.name,
            "generation": self.generation,
            "best_distance": float(self.distance_m),
            "top_speed": float(self.top_speed),
            "fitness": float(self.fitness),
            "weights1": self.w1.tolist(),
            "weights2": self.w2.tolist(),
            "b1": self.b1.tolist(),
            "b2": self.b2.tolist(),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

    def from_dict(self, data):
        self.generation = data.get("generation", 0)
        self.distance_m = data.get("best_distance", 0.0)
        self.top_speed = data.get("top_speed", 0.0)
        self.fitness = data.get("fitness", 0.0)
        
        # Support both weights1/weights2 format from Web and NumPy array formats
        if "weights1" in data:
            self.w1 = np.array(data["weights1"], dtype=np.float32)
        if "weights2" in data:
            self.w2 = np.array(data["weights2"], dtype=np.float32)
        if "b1" in data:
            self.b1 = np.array(data["b1"], dtype=np.float32)
        if "b2" in data:
            self.b2 = np.array(data["b2"], dtype=np.float32)


# =====================================================================
# TKINTER GRAPHICAL SIMULATION INTERFACE
# =====================================================================

class HumanoidSprintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI 100m Dash - Humanoid Reinforcement Learning Simulation")
        self.root.geometry("1200x780")
        self.root.minsize(980, 660)
        self.root.configure(bg="#0d1117")

        self.sim_speed = 1.0
        self.race_running = True
        self.race_time = 0.0
        self.camera_x = -40.0
        self.checkpoint_file = "humanoid_champion.json"
        
        # Build 4 benchmark competitors
        self.runners = [
            HumanoidAgent("Gen 0: Flailer", "#f85149", 0, "gen0"),
            HumanoidAgent("Gen 12: Hopper", "#d29922", 1, "gen12"),
            HumanoidAgent("Gen 45: Galloper", "#58a6ff", 2, "gen45"),
            HumanoidAgent("Gen 120: Sprinter", "#3fb950", 3, "gen120")
        ]
        
        self.custom_champion = None
        self.current_gen = 0
        self.all_time_best = 0.0
        
        self.create_ui()
        self.auto_load_checkpoint()
        self.last_tick = time.time()
        self.update_simulation()

    def create_ui(self):
        # 1. Header Bar
        header = tk.Frame(self.root, bg="#161b22", height=50, padx=16, pady=8)
        header.pack(fill=tk.X, side=tk.TOP)
        
        title_lbl = tk.Label(
            header, text="🏃‍♂️ AI 100m Dash — Humanoid RL Lab",
            font=("Segoe UI", 14, "bold"), fg="#e6edf3", bg="#161b22"
        )
        title_lbl.pack(side=tk.LEFT)
        
        self.lbl_retention_badge = tk.Label(
            header, text="💾 Model Checkpoint: Auto-Save Active",
            font=("Segoe UI", 9, "bold"), fg="#3fb950", bg="#101a14",
            padx=10, pady=2, relief=tk.SOLID, borderwidth=1
        )
        self.lbl_retention_badge.pack(side=tk.RIGHT)

        # 2. Main Layout (Canvas + Sidebar)
        body = tk.Frame(self.root, bg="#0d1117")
        body.pack(fill=tk.BOTH, expand=True)

        view_frame = tk.Frame(body, bg="#090d13")
        view_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(view_frame, bg="#090d13", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        hud_bar = tk.Frame(view_frame, bg="#161b22", padx=12, pady=6)
        hud_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.lbl_clock = tk.Label(hud_bar, text="Clock: 00:00.00", font=("Consolas", 11, "bold"), fg="#fff", bg="#161b22")
        self.lbl_clock.pack(side=tk.LEFT, padx=10)

        self.lbl_lead_dist = tk.Label(hud_bar, text="Leader Dist: 0.0 m", font=("Consolas", 11), fg="#58a6ff", bg="#161b22")
        self.lbl_lead_dist.pack(side=tk.LEFT, padx=10)

        self.lbl_lead_speed = tk.Label(hud_bar, text="Top Speed: 0.0 m/s", font=("Consolas", 11), fg="#3fb950", bg="#161b22")
        self.lbl_lead_speed.pack(side=tk.LEFT, padx=10)

        # Control Sidebar (Right)
        side = tk.Frame(body, bg="#161b22", width=330, padx=16, pady=16)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)

        # Section: Race Controls
        sec1 = tk.Label(side, text="🎮 RACE CONTROLS", font=("Segoe UI", 10, "bold"), fg="#8b949e", bg="#161b22")
        sec1.pack(anchor="w", pady=(0, 6))

        btn_box = tk.Frame(side, bg="#161b22")
        btn_box.pack(fill=tk.X, pady=(0, 8))

        self.btn_toggle = tk.Button(
            btn_box, text="⏸ Pause", font=("Segoe UI", 10, "bold"),
            bg="#238636", fg="#fff", relief=tk.FLAT, padx=10, pady=6,
            command=self.toggle_race
        )
        self.btn_toggle.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_reset = tk.Button(
            btn_box, text="↺ Reset", font=("Segoe UI", 10),
            bg="#21262d", fg="#e6edf3", relief=tk.FLAT, padx=10, pady=6,
            command=self.reset_race
        )
        self.btn_reset.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        # Speed Slider
        speed_box = tk.Frame(side, bg="#161b22")
        speed_box.pack(fill=tk.X, pady=(2, 10))
        
        lbl_sp = tk.Label(speed_box, text="Sim Speed (1.0x - 4.0x):", font=("Segoe UI", 9), fg="#8b949e", bg="#161b22")
        lbl_sp.pack(anchor="w")

        self.speed_slider = tk.Scale(
            speed_box, from_=0.5, to=4.0, resolution=0.1, orient=tk.HORIZONTAL,
            bg="#161b22", fg="#e6edf3", highlightthickness=0, command=self.on_speed_change
        )
        self.speed_slider.set(1.0)
        self.speed_slider.pack(fill=tk.X)

        # Section: Model Checkpoint Retention & Evolution
        sec_ckpt = tk.Label(side, text="💾 MODEL RETENTION & TRAINING", font=("Segoe UI", 10, "bold"), fg="#00f2fe", bg="#161b22")
        sec_ckpt.pack(anchor="w", pady=(8, 4))

        ckpt_box = tk.Frame(side, bg="#10141a", padx=8, pady=8, highlightbackground="#00f2fe", highlightthickness=1)
        ckpt_box.pack(fill=tk.X, pady=(0, 10))

        self.lbl_ckpt_status = tk.Label(
            ckpt_box, text="Status: Ready to train/load",
            font=("Segoe UI", 8), fg="#c9d1d9", bg="#10141a", justify=tk.LEFT, wraplength=280
        )
        self.lbl_ckpt_status.pack(anchor="w", pady=(0, 6))

        btn_ckpt_box = tk.Frame(ckpt_box, bg="#10141a")
        btn_ckpt_box.pack(fill=tk.X, pady=(0, 4))

        self.btn_fast_train = tk.Button(
            btn_ckpt_box, text="⚡ Fast Evolve 10 Gens", font=("Segoe UI", 9, "bold"),
            bg="#007791", fg="#fff", relief=tk.FLAT, padx=6, pady=4,
            command=self.fast_evolve_generations
        )
        self.btn_fast_train.pack(fill=tk.X, expand=True, pady=2)

        file_btn_box = tk.Frame(ckpt_box, bg="#10141a")
        file_btn_box.pack(fill=tk.X)

        self.btn_save = tk.Button(
            file_btn_box, text="💾 Save JSON", font=("Segoe UI", 8),
            bg="#21262d", fg="#e6edf3", relief=tk.FLAT, padx=6, pady=3,
            command=self.save_checkpoint_dialog
        )
        self.btn_save.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.btn_load = tk.Button(
            file_btn_box, text="📂 Load JSON", font=("Segoe UI", 8),
            bg="#21262d", fg="#e6edf3", relief=tk.FLAT, padx=6, pady=3,
            command=self.load_checkpoint_dialog
        )
        self.btn_load.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        # Section: Live Leaderboard
        sec2 = tk.Label(side, text="🏆 100m LEADERBOARD", font=("Segoe UI", 10, "bold"), fg="#8b949e", bg="#161b22")
        sec2.pack(anchor="w", pady=(6, 4))

        self.leader_frame = tk.Frame(side, bg="#10141a", padx=8, pady=8)
        self.leader_frame.pack(fill=tk.X, pady=(0, 10))

        self.leader_labels = []
        for i in range(5):
            lbl = tk.Label(
                self.leader_frame, text=f"#{i+1} Loading...",
                font=("Segoe UI", 8), fg="#e6edf3", bg="#10141a", anchor="w"
            )
            lbl.pack(fill=tk.X, pady=1)
            self.leader_labels.append(lbl)

        # Section: Educational Explanations
        sec3 = tk.Label(side, text="💡 RL EVOLUTION STAGES", font=("Segoe UI", 10, "bold"), fg="#8b949e", bg="#161b22")
        sec3.pack(anchor="w", pady=(6, 4))

        info_box = tk.Frame(side, bg="#10141a", padx=8, pady=8)
        info_box.pack(fill=tk.BOTH, expand=True)

        explanation = (
            "• Gen 0 (Red): Random flailing.\n"
            "• Gen 12 (Yellow): One-legged hop.\n"
            "• Gen 45 (Blue): Alternating run.\n"
            "• Gen 120 (Green): Olympic sprint!\n"
            "• Cyan (#00f2fe): Your Trained Champion."
        )
        info_lbl = tk.Label(
            info_box, text=explanation, font=("Segoe UI", 8),
            fg="#c9d1d9", bg="#10141a", justify=tk.LEFT, anchor="nw"
        )
        info_lbl.pack(fill=tk.BOTH, expand=True)

    def on_speed_change(self, val):
        self.sim_speed = float(val)

    def toggle_race(self):
        self.race_running = not self.race_running
        self.btn_toggle.config(text="⏸ Pause" if self.race_running else "▶ Resume")

    def reset_race(self):
        self.race_time = 0.0
        for r in self.runners:
            r.reset(lane_base_y=240.0)
        self.race_running = True
        self.btn_toggle.config(text="⏸ Pause")

    # =================================================================
    # MODEL RETENTION & CHECKPOINTING
    # =================================================================

    def auto_load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, "r") as f:
                    data = json.load(f)
                self.inject_custom_champion(data)
                self.lbl_ckpt_status.config(
                    text=f"Loaded from {self.checkpoint_file}\nGen: {data.get('generation', 0)} | Best: {data.get('best_distance', 0):.1f}m"
                )
            except Exception as e:
                self.lbl_ckpt_status.config(text=f"Checkpoint load error: {e}")

    def inject_custom_champion(self, data):
        gen = data.get("generation", 0)
        dist = data.get("best_distance", 0.0)
        
        # Check if champion runner already exists in runners
        if self.custom_champion is None:
            self.custom_champion = HumanoidAgent(
                f"Gen {gen}: Champion", "#00f2fe", 4, "custom_champion"
            )
            self.runners.append(self.custom_champion)
        else:
            self.custom_champion.name = f"Gen {gen}: Champion"
        
        self.custom_champion.from_dict(data)
        self.custom_champion.reset(lane_base_y=240.0)
        self.reset_race()

    def save_checkpoint(self, filepath=None):
        if self.custom_champion is None and len(self.runners) > 3:
            champ = self.runners[3] # fallback to Sprinter
        else:
            champ = self.custom_champion or self.runners[3]

        filepath = filepath or self.checkpoint_file
        data = champ.to_dict()
        data["generation"] = self.current_gen
        data["best_distance"] = max(self.all_time_best, champ.distance_m)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        self.lbl_ckpt_status.config(
            text=f"Saved to {os.path.basename(filepath)}\nGen {self.current_gen} | Best: {data['best_distance']:.1f}m"
        )
        messagebox.showinfo("Checkpoint Saved", f"Successfully saved trained champion to:\n{os.path.abspath(filepath)}")

    def save_checkpoint_dialog(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON Files", "*.json")], initialfile="humanoid_champion.json"
        )
        if path:
            self.save_checkpoint(path)

    def load_checkpoint_dialog(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON Files", "*.json")]
        )
        if path and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                self.inject_custom_champion(data)
                self.lbl_ckpt_status.config(
                    text=f"Loaded from {os.path.basename(path)}\nGen {data.get('generation', 0)} | Best: {data.get('best_distance', 0):.1f}m"
                )
            except Exception as e:
                messagebox.showerror("Error Loading", str(e))

    def fast_evolve_generations(self, num_gens=10):
        """
        Runs headless continuous policy evolution right in Python.
        """
        self.lbl_ckpt_status.config(text="Evolving generations in background...")
        self.root.update()

        # Create population of 8 candidates
        pop = [HumanoidAgent(f"Cand #{i+1}", "#00f2fe", 4, "neural") for i in range(8)]
        if self.custom_champion:
            pop[0].w1 = self.custom_champion.w1.copy()
            pop[0].w2 = self.custom_champion.w2.copy()

        for g in range(num_gens):
            self.current_gen += 1
            # Simulate 5.0 seconds
            dt = 0.03
            sim_steps = int(5.0 / dt)
            for p in pop:
                p.reset(lane_base_y=240.0)

            for step in range(sim_steps):
                sim_t = step * dt
                for p in pop:
                    p.update(dt, sim_t)

            # Fitness
            for p in pop:
                fit = p.distance_m * 2.5 + p.top_speed * 4.0
                if not p.fallen:
                    fit += 25.0
                p.fitness = fit

            pop.sort(key=lambda x: x.fitness, reverse=True)
            champ = pop[0]
            if champ.distance_m > self.all_time_best:
                self.all_time_best = champ.distance_m

            # Mutate rest from champion
            for i in range(1, 8):
                child = pop[i]
                child.w1 = champ.w1 + np.random.randn(*champ.w1.shape).astype(np.float32) * 0.15
                child.w2 = champ.w2 + np.random.randn(*champ.w2.shape).astype(np.float32) * 0.15

        # Inject resulting champion and auto-save
        champ_data = pop[0].to_dict()
        champ_data["generation"] = self.current_gen
        champ_data["best_distance"] = self.all_time_best
        self.inject_custom_champion(champ_data)

        # Auto-save to humanoid_champion.json
        with open(self.checkpoint_file, "w") as f:
            json.dump(champ_data, f, indent=2)

        self.lbl_ckpt_status.config(
            text=f"Auto-Saved: Gen {self.current_gen}\nBest Distance: {self.all_time_best:.1f}m"
        )
        self.reset_race()

    def update_simulation(self):
        now = time.time()
        dt = min(0.05, now - self.last_tick) * self.sim_speed
        self.last_tick = now

        if self.race_running:
            self.race_time += dt
            for r in self.runners:
                r.update(dt, self.race_time)

            sorted_runners = sorted(self.runners, key=lambda x: x.distance_m, reverse=True)
            leader = sorted_runners[0]

            c_width = self.canvas.winfo_width() or 800
            target_cam = leader.x - c_width * 0.35
            self.camera_x += (target_cam - self.camera_x) * 0.1

            mins = int(self.race_time // 60)
            secs = self.race_time % 60
            self.lbl_clock.config(text=f"Clock: {mins:02d}:{secs:05.2f}")
            self.lbl_lead_dist.config(text=f"Leader: {leader.distance_m:.1f} m")
            self.lbl_lead_speed.config(text=f"Top Speed: {leader.top_speed:.1f} m/s")

            for idx, r in enumerate(sorted_runners):
                if idx < len(self.leader_labels):
                    status_str = f"🏁 {r.finish_time:.2f}s" if r.finished else ("💥 Fall" if r.fallen else f"{r.distance_m:.1f}m")
                    self.leader_labels[idx].config(
                        text=f"#{idx+1} {r.name}: {status_str} ({r.vx/40.0:.1f} m/s)",
                        fg=r.color
                    )

        self.draw_scene()
        self.root.after(16, self.update_simulation)

    def draw_scene(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width() or 800
        h = self.canvas.winfo_height() or 500

        track_y = 200
        num_lanes = len(self.runners)
        lane_height = 34
        track_h = num_lanes * lane_height + 20

        # 1. Background Sky & Stadium Wall
        self.canvas.create_rectangle(0, 0, w, track_y, fill="#0f1622", outline="")
        self.canvas.create_rectangle(0, track_y - 20, w, track_y, fill="#1c2536", outline="")

        # 2. Running Track (Synthetic Brick Red)
        self.canvas.create_rectangle(0, track_y, w, track_y + track_h, fill="#a93226", outline="")

        # Lanes
        for i in range(num_lanes + 1):
            ly = track_y + i * lane_height
            self.canvas.create_line(0, ly, w, ly, fill="#e6b0aa", width=1, dash=(8, 4))

        # Start Line (0m)
        start_scr_x = 20.0 - self.camera_x
        self.canvas.create_line(start_scr_x, track_y, start_scr_x, track_y + track_h, fill="#ffffff", width=4)
        self.canvas.create_text(start_scr_x, track_y + track_h + 16, text="START (0m)", fill="#ffffff", font=("Segoe UI", 9, "bold"))

        # Distance Markers
        for m in range(10, 101, 10):
            mx = 20.0 + m * 40.0 - self.camera_x
            if -50 <= mx <= w + 50:
                color = "#ffea00" if m == 100 else "#ffffff"
                width = 3 if m == 100 else 1
                self.canvas.create_line(mx, track_y, mx, track_y + track_h, fill=color, width=width)
                self.canvas.create_text(mx, track_y + track_h + 14, text=f"{m}m", fill=color, font=("Consolas", 9, "bold"))

        # 100m Finish Arch
        finish_scr_x = 20.0 + 100.0 * 40.0 - self.camera_x
        if -100 <= finish_scr_x <= w + 100:
            self.canvas.create_line(finish_scr_x, track_y, finish_scr_x, track_y - 45, fill="#ffea00", width=5)
            self.canvas.create_rectangle(finish_scr_x - 30, track_y - 65, finish_scr_x + 30, track_y - 45, fill="#d29922", outline="")
            self.canvas.create_text(finish_scr_x, track_y - 55, text="🏁 100m FINISH", fill="#000", font=("Segoe UI", 8, "bold"))

        # 3. Draw Humanoid Agents
        for r in self.runners:
            sx = r.x - self.camera_x
            sy = r.y

            # Shadow
            self.canvas.create_oval(sx - 14, r.ground_y - 3, sx + 14, r.ground_y + 3, fill="#5a1812", outline="")

            # Torso & Head
            neck_x = sx - math.sin(r.tilt) * r.torso_len
            neck_y = sy - math.cos(r.tilt) * r.torso_len
            head_x = neck_x - math.sin(r.tilt) * 12.0
            head_y = neck_y - math.cos(r.tilt) * 12.0

            # Left Leg (Back)
            l_knee_x = sx + math.sin(r.tilt + r.l_hip) * r.thigh_len
            l_knee_y = sy + math.cos(r.tilt + r.l_hip) * r.thigh_len
            l_foot_x = l_knee_x + math.sin(r.tilt + r.l_hip - r.l_knee) * r.shin_len
            l_foot_y = min(r.ground_y, l_knee_y + math.cos(r.tilt + r.l_hip - r.l_knee) * r.shin_len)

            # Right Leg (Front)
            r_knee_x = sx + math.sin(r.tilt + r.r_hip) * r.thigh_len
            r_knee_y = sy + math.cos(r.tilt + r.r_hip) * r.thigh_len
            r_foot_x = r_knee_x + math.sin(r.tilt + r.r_hip - r.r_knee) * r.shin_len
            r_foot_y = min(r.ground_y, r_knee_y + math.cos(r.tilt + r.r_hip - r.r_knee) * r.shin_len)

            # Left Leg
            self.canvas.create_line(sx, sy, l_knee_x, l_knee_y, l_foot_x, l_foot_y, fill="#555", width=4, capstyle=tk.ROUND)
            
            # Torso
            self.canvas.create_line(sx, sy, neck_x, neck_y, fill=r.color, width=6, capstyle=tk.ROUND)
            
            # Head
            self.canvas.create_oval(head_x - 7, head_y - 7, head_x + 7, head_y + 7, fill=r.color, outline="#fff", width=1)

            # Right Leg
            self.canvas.create_line(sx, sy, r_knee_x, r_knee_y, r_foot_x, r_foot_y, fill=r.color, width=5, capstyle=tk.ROUND)

            # Arms
            arm_x = neck_x + math.sin(r.tilt - r.arm_swing) * r.arm_len
            arm_y = neck_y + math.cos(r.tilt - r.arm_swing) * r.arm_len
            self.canvas.create_line(neck_x, neck_y, arm_x, arm_y, fill=r.color, width=4, capstyle=tk.ROUND)

            # Name Tag
            self.canvas.create_text(sx, head_y - 12, text=r.name.split(":")[0], fill="#ffffff", font=("Segoe UI", 8, "bold"))


# =====================================================================
# MAIN ENTRYPOINT
# =====================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = HumanoidSprintApp(root)
    root.mainloop()
