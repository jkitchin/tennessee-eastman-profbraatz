"""
GUI Dashboard for the Tennessee Eastman Process Simulator.

This module provides an interactive graphical interface for:
- Controlling manipulated variables
- Enabling/disabling process disturbances
- Real-time visualization of process measurements
- Simulation control (start, stop, reset)

Requirements:
    - tkinter (usually included with Python)
    - matplotlib

Usage:
    python -m tep.dashboard

    Or from Python:
        from tep.dashboard import TEPDashboard
        app = TEPDashboard()
        app.run()
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from typing import Dict, List, Optional
import threading
import time

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Install with: pip install matplotlib")

from .simulator import TEPSimulator, ControlMode
from .constants import (
    MEASUREMENT_NAMES, MANIPULATED_VAR_NAMES, DISTURBANCE_NAMES,
    NUM_MEASUREMENTS, NUM_MANIPULATED_VARS, NUM_DISTURBANCES, INITIAL_STATES
)


class TEPDashboard:
    """
    Interactive GUI dashboard for the Tennessee Eastman Process.

    Features:
    - Real-time plotting of key process measurements
    - Slider controls for all 12 manipulated variables
    - Checkbox controls for all 20 disturbances
    - Start/Stop/Reset simulation controls
    - Adjustable simulation speed
    """

    def __init__(self):
        """Initialize the dashboard."""
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for the dashboard. Install with: pip install matplotlib")

        # Create main window
        self.root = tk.Tk()
        self.root.title("Tennessee Eastman Process Simulator")
        self.root.geometry("1400x900")

        # Simulation state
        self.simulator = TEPSimulator(control_mode=ControlMode.CLOSED_LOOP)
        self.running = False
        self.sim_thread = None
        self.update_interval = 50  # ms between GUI updates
        self.steps_per_update = 10  # simulation steps per GUI update

        # Data history for plotting
        self.history_length = 500
        self.time_history = []
        self.meas_history = {i: [] for i in range(NUM_MEASUREMENTS)}
        self.mv_history = {i: [] for i in range(NUM_MANIPULATED_VARS)}

        # Control variables
        self.mv_vars = []
        self.idv_vars = []
        self.control_mode_var = tk.StringVar(value="closed_loop")
        self.speed_var = tk.IntVar(value=10)

        # Build the GUI
        self._create_layout()
        self._initialize_simulator()

    def _create_layout(self):
        """Create the main GUI layout."""
        # Main container with three columns
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel - Controls
        left_frame = ttk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        left_frame.pack_propagate(False)

        # Center panel - Plots
        center_frame = ttk.Frame(main_frame)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Right panel - Disturbances
        right_frame = ttk.Frame(main_frame, width=250)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        right_frame.pack_propagate(False)

        # Build each panel
        self._create_control_panel(left_frame)
        self._create_plot_panel(center_frame)
        self._create_disturbance_panel(right_frame)

    def _create_control_panel(self, parent):
        """Create the control panel with MVs and simulation controls."""
        # Title
        ttk.Label(parent, text="Control Panel", font=('Helvetica', 12, 'bold')).pack(pady=5)

        # Simulation controls frame
        sim_frame = ttk.LabelFrame(parent, text="Simulation", padding=5)
        sim_frame.pack(fill=tk.X, pady=5)

        # Control mode
        mode_frame = ttk.Frame(sim_frame)
        mode_frame.pack(fill=tk.X, pady=2)
        ttk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="Closed Loop", variable=self.control_mode_var,
                        value="closed_loop", command=self._on_mode_change).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="Manual", variable=self.control_mode_var,
                        value="manual", command=self._on_mode_change).pack(side=tk.LEFT)

        # Speed control
        speed_frame = ttk.Frame(sim_frame)
        speed_frame.pack(fill=tk.X, pady=2)
        ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT)
        ttk.Scale(speed_frame, from_=1, to=50, variable=self.speed_var,
                  orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Buttons
        btn_frame = ttk.Frame(sim_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        self.start_btn = ttk.Button(btn_frame, text="Start", command=self._start_simulation)
        self.start_btn.pack(side=tk.LEFT, padx=2)

        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self._stop_simulation, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)

        self.reset_btn = ttk.Button(btn_frame, text="Reset", command=self._reset_simulation)
        self.reset_btn.pack(side=tk.LEFT, padx=2)

        # Status
        self.status_var = tk.StringVar(value="Ready")
        self.time_var = tk.StringVar(value="Time: 0.00 hr")
        ttk.Label(sim_frame, textvariable=self.status_var).pack(anchor=tk.W)
        ttk.Label(sim_frame, textvariable=self.time_var).pack(anchor=tk.W)

        # MV controls frame with scrollbar
        mv_container = ttk.LabelFrame(parent, text="Manipulated Variables", padding=5)
        mv_container.pack(fill=tk.BOTH, expand=True, pady=5)

        # Canvas and scrollbar for MVs
        mv_canvas = tk.Canvas(mv_container, highlightthickness=0)
        mv_scrollbar = ttk.Scrollbar(mv_container, orient=tk.VERTICAL, command=mv_canvas.yview)
        mv_scroll_frame = ttk.Frame(mv_canvas)

        mv_scroll_frame.bind("<Configure>", lambda e: mv_canvas.configure(scrollregion=mv_canvas.bbox("all")))
        mv_canvas.create_window((0, 0), window=mv_scroll_frame, anchor=tk.NW)
        mv_canvas.configure(yscrollcommand=mv_scrollbar.set)

        mv_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        mv_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create MV sliders
        mv_short_names = [
            "D Feed", "E Feed", "A Feed", "A+C Feed",
            "Recycle", "Purge", "Sep Liq", "Strip Liq",
            "Steam", "React CW", "Cond CW", "Agitator"
        ]

        initial_mvs = INITIAL_STATES[38:50]

        for i in range(NUM_MANIPULATED_VARS):
            mv_frame = ttk.Frame(mv_scroll_frame)
            mv_frame.pack(fill=tk.X, pady=1)

            var = tk.DoubleVar(value=initial_mvs[i])
            self.mv_vars.append(var)

            ttk.Label(mv_frame, text=f"{i+1}. {mv_short_names[i]}", width=12).pack(side=tk.LEFT)

            scale = ttk.Scale(mv_frame, from_=0, to=100, variable=var,
                             orient=tk.HORIZONTAL, command=lambda v, idx=i: self._on_mv_change(idx))
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

            val_label = ttk.Label(mv_frame, textvariable=var, width=6)
            val_label.pack(side=tk.RIGHT)

    def _create_plot_panel(self, parent):
        """Create the plotting panel with real-time charts."""
        # Create figure with subplots
        self.fig = Figure(figsize=(10, 8), dpi=80)
        self.fig.set_facecolor('#f0f0f0')

        # Create 6 subplots in a 3x2 grid
        self.axes = []
        plot_configs = [
            ("Reactor", [("Temp (°C)", 8), ("Level (%)", 7), ("Pressure (kPa)", 6)]),
            ("Separator", [("Temp (°C)", 10), ("Level (%)", 11), ("Pressure (kPa)", 12)]),
            ("Stripper", [("Temp (°C)", 17), ("Level (%)", 14), ("Pressure (kPa)", 15)]),
            ("Flows", [("A Feed", 0), ("Purge", 9), ("Product", 16)]),
            ("Compositions", [("Feed A%", 22), ("Feed D%", 25), ("Prod E%", 37)]),
            ("Utilities", [("React CW (°C)", 20), ("Sep CW (°C)", 21), ("Comp Work", 19)]),
        ]

        self.plot_lines = {}
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for idx, (title, signals) in enumerate(plot_configs):
            ax = self.fig.add_subplot(3, 2, idx + 1)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Time (min)", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            self.axes.append(ax)

            for color_idx, (label, meas_idx) in enumerate(signals):
                line, = ax.plot([], [], label=label, color=colors[color_idx], linewidth=1)
                self.plot_lines[(idx, meas_idx)] = line

            ax.legend(loc='upper left', fontsize=7)

        self.fig.tight_layout()

        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_disturbance_panel(self, parent):
        """Create the disturbance control panel."""
        ttk.Label(parent, text="Disturbances", font=('Helvetica', 12, 'bold')).pack(pady=5)

        # Canvas and scrollbar for disturbances
        idv_container = ttk.Frame(parent)
        idv_container.pack(fill=tk.BOTH, expand=True)

        idv_canvas = tk.Canvas(idv_container, highlightthickness=0)
        idv_scrollbar = ttk.Scrollbar(idv_container, orient=tk.VERTICAL, command=idv_canvas.yview)
        idv_scroll_frame = ttk.Frame(idv_canvas)

        idv_scroll_frame.bind("<Configure>", lambda e: idv_canvas.configure(scrollregion=idv_canvas.bbox("all")))
        idv_canvas.create_window((0, 0), window=idv_scroll_frame, anchor=tk.NW)
        idv_canvas.configure(yscrollcommand=idv_scrollbar.set)

        idv_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        idv_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Short disturbance names
        idv_short_names = [
            "A/C Ratio Step", "B Comp Step", "D Feed Temp",
            "React CW Temp", "Cond CW Temp", "A Feed Loss",
            "C Header Loss", "A,B,C Random", "D Temp Random",
            "C Temp Random", "React CW Rand", "Cond CW Rand",
            "Kinetics Drift", "React Valve Stick", "Cond Valve Stick",
            "Unknown 16", "Unknown 17", "Unknown 18",
            "Unknown 19", "Unknown 20"
        ]

        for i in range(NUM_DISTURBANCES):
            var = tk.BooleanVar(value=False)
            self.idv_vars.append(var)

            cb = ttk.Checkbutton(
                idv_scroll_frame,
                text=f"{i+1}. {idv_short_names[i]}",
                variable=var,
                command=lambda idx=i: self._on_idv_change(idx)
            )
            cb.pack(anchor=tk.W, pady=1)

        # Quick buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Clear All", command=self._clear_disturbances).pack(side=tk.LEFT, padx=2)

    def _initialize_simulator(self):
        """Initialize the simulator and history."""
        self.simulator.initialize()
        self._clear_history()
        self._update_plots()

    def _clear_history(self):
        """Clear all history data."""
        self.time_history = []
        for i in range(NUM_MEASUREMENTS):
            self.meas_history[i] = []
        for i in range(NUM_MANIPULATED_VARS):
            self.mv_history[i] = []

    def _on_mode_change(self):
        """Handle control mode change."""
        mode = self.control_mode_var.get()
        if mode == "closed_loop":
            self.simulator.control_mode = ControlMode.CLOSED_LOOP
            self.simulator._init_controller()
        else:
            self.simulator.control_mode = ControlMode.MANUAL
            self.simulator._init_controller()
            # Set current MV values
            for i, var in enumerate(self.mv_vars):
                self.simulator.set_mv(i + 1, var.get())

    def _on_mv_change(self, idx):
        """Handle MV slider change."""
        if self.control_mode_var.get() == "manual":
            value = self.mv_vars[idx].get()
            self.simulator.set_mv(idx + 1, value)

    def _on_idv_change(self, idx):
        """Handle disturbance checkbox change."""
        value = 1 if self.idv_vars[idx].get() else 0
        self.simulator.set_disturbance(idx + 1, value)

    def _clear_disturbances(self):
        """Clear all disturbances."""
        for i, var in enumerate(self.idv_vars):
            var.set(False)
        self.simulator.clear_disturbances()

    def _start_simulation(self):
        """Start the simulation."""
        if not self.running:
            self.running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_var.set("Running")
            self._simulation_loop()

    def _stop_simulation(self):
        """Stop the simulation."""
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Stopped")

    def _reset_simulation(self):
        """Reset the simulation."""
        self._stop_simulation()
        self._initialize_simulator()

        # Reset MV sliders
        initial_mvs = INITIAL_STATES[38:50]
        for i, var in enumerate(self.mv_vars):
            var.set(initial_mvs[i])

        # Clear disturbances
        self._clear_disturbances()

        self.status_var.set("Ready")
        self.time_var.set("Time: 0.00 hr")

    def _simulation_loop(self):
        """Main simulation loop."""
        if not self.running:
            return

        # Get speed setting
        steps = self.speed_var.get()

        # Run simulation steps
        for _ in range(steps):
            if not self.simulator.step():
                self._stop_simulation()
                self.status_var.set("SHUTDOWN")
                return

        # Record data
        self.time_history.append(self.simulator.time * 60)  # Convert to minutes

        meas = self.simulator.get_measurements()
        for i in range(NUM_MEASUREMENTS):
            self.meas_history[i].append(meas[i])

        mvs = self.simulator.get_manipulated_vars()
        for i in range(NUM_MANIPULATED_VARS):
            self.mv_history[i].append(mvs[i])
            # Update slider if in closed-loop mode
            if self.control_mode_var.get() == "closed_loop":
                self.mv_vars[i].set(round(mvs[i], 1))

        # Trim history
        if len(self.time_history) > self.history_length:
            self.time_history = self.time_history[-self.history_length:]
            for i in range(NUM_MEASUREMENTS):
                self.meas_history[i] = self.meas_history[i][-self.history_length:]
            for i in range(NUM_MANIPULATED_VARS):
                self.mv_history[i] = self.mv_history[i][-self.history_length:]

        # Update display
        self.time_var.set(f"Time: {self.simulator.time:.2f} hr ({self.simulator.time*60:.1f} min)")

        # Update plots periodically
        self._update_plots()

        # Schedule next iteration
        self.root.after(self.update_interval, self._simulation_loop)

    def _update_plots(self):
        """Update all plots with current data."""
        if not self.time_history:
            return

        time_arr = np.array(self.time_history)

        # Update each subplot
        plot_configs = [
            [8, 7, 6],    # Reactor
            [10, 11, 12], # Separator
            [17, 14, 15], # Stripper
            [0, 9, 16],   # Flows
            [22, 25, 37], # Compositions
            [20, 21, 19], # Utilities
        ]

        for ax_idx, meas_indices in enumerate(plot_configs):
            ax = self.axes[ax_idx]

            y_min, y_max = float('inf'), float('-inf')

            for meas_idx in meas_indices:
                line = self.plot_lines.get((ax_idx, meas_idx))
                if line and self.meas_history[meas_idx]:
                    y_data = np.array(self.meas_history[meas_idx])
                    line.set_data(time_arr, y_data)

                    if len(y_data) > 0:
                        y_min = min(y_min, np.min(y_data))
                        y_max = max(y_max, np.max(y_data))

            # Adjust axes
            if len(time_arr) > 0:
                ax.set_xlim(time_arr[0], max(time_arr[-1], time_arr[0] + 1))

            if y_min < y_max:
                margin = (y_max - y_min) * 0.1 + 0.1
                ax.set_ylim(y_min - margin, y_max + margin)

        self.canvas.draw_idle()

    def run(self):
        """Run the dashboard application."""
        self.root.mainloop()


def main():
    """Entry point for the dashboard."""
    try:
        app = TEPDashboard()
        app.run()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nTo run the dashboard, install matplotlib:")
        print("  pip install matplotlib")


if __name__ == "__main__":
    main()
