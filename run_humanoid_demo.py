"""
AI 100m Dash - Humanoid Reinforcement Learning Demo Launcher
Select whether to launch the Interactive Web Simulation or the Native Tkinter GUI.
"""

import os
import sys
import webbrowser

def main():
    print("=" * 65)
    print(" 🏃 AI 100m Dash — Humanoid Reinforcement Learning Simulation")
    print(" Inspired by AI Warehouse: 'AI vs AI in 100m Dash'")
    print("=" * 65)
    print("\nSelect an interface to launch:")
    print(" [1] Interactive Web Simulation (Recommended — 60 FPS, 3D track,")
    print("     camera controls, neural weight visualizer & live evolution lab)")
    print(" [2] Native Desktop Tkinter GUI (Pure Python + Tkinter)")
    print(" [3] Exit\n")

    choice = input("Enter choice [1, 2, or 3] (default is 1): ").strip()
    
    html_path = os.path.abspath("humanoid_dash_simulation.html")
    
    if choice == "2":
        print("\nLaunching Native Tkinter Desktop GUI...")
        import humanoid_rl_demo
        import tkinter as tk
        root = tk.Tk()
        app = humanoid_rl_demo.HumanoidSprintApp(root)
        root.mainloop()
    elif choice == "3":
        print("Goodbye!")
        sys.exit(0)
    else:
        print(f"\nOpening Interactive Simulation in your default browser: file://{html_path}")
        webbrowser.open(f"file://{html_path}")
        print("Done! You can switch tabs between '100m Race', 'Live Train', and 'Compute FAQ'.")

if __name__ == "__main__":
    main()
