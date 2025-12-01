"""
Process controllers for the Tennessee Eastman Process.

This module implements the PI/PID controllers used in the closed-loop
control scheme from temain_mod.f.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum

from .controller_base import BaseController, ControllerRegistry, register_controller
from .constants import OPERATING_MODES, DEFAULT_OPERATING_MODE


class ControllerType(Enum):
    """Types of controllers available."""
    P = "proportional"
    PI = "proportional-integral"
    PID = "proportional-integral-derivative"


@dataclass
class PIController:
    """
    Proportional-Integral controller using velocity form.

    The velocity form calculates the change in output rather than
    the absolute output, which prevents integral windup.
    """
    setpoint: float
    gain: float
    taui: float = 0.0  # Reset time (hours). 0 = P-only control
    err_old: float = 0.0
    output_min: float = 0.0
    output_max: float = 100.0
    scale: float = 1.0  # Scaling factor for error normalization

    def calculate(
        self,
        measurement: float,
        current_output: float,
        dt: float
    ) -> float:
        """
        Calculate new controller output.

        Args:
            measurement: Current process measurement
            current_output: Current valve position (%)
            dt: Time step since last calculation (hours)

        Returns:
            New valve position (%)
        """
        # Calculate error (normalized)
        error = (self.setpoint - measurement) * self.scale

        # P-only or PI control
        if self.taui > 0:
            # PI controller (velocity form)
            dxmv = self.gain * ((error - self.err_old) + error * dt / self.taui)
        else:
            # P-only controller
            dxmv = self.gain * (error - self.err_old)

        # Update output
        new_output = current_output + dxmv

        # Apply limits
        new_output = np.clip(new_output, self.output_min, self.output_max)

        # Store error for next iteration
        self.err_old = error

        return new_output

    def calculate_change(
        self,
        measurement: float,
        dt: float
    ) -> float:
        """
        Calculate controller output change (dxmv) for cascade control.

        This method returns just the change in output (velocity form)
        without applying limits, for use in cascade controllers where
        the output adjusts a setpoint rather than a valve position.

        Args:
            measurement: Current process measurement
            dt: Time step since last calculation (hours)

        Returns:
            Change in output (dxmv)
        """
        # Calculate error (normalized)
        error = (self.setpoint - measurement) * self.scale

        # P-only or PI control
        if self.taui > 0:
            # PI controller (velocity form)
            dxmv = self.gain * ((error - self.err_old) + error * dt / self.taui)
        else:
            # P-only controller
            dxmv = self.gain * (error - self.err_old)

        # Store error for next iteration
        self.err_old = error

        return dxmv

    def reset(self):
        """Reset controller state."""
        self.err_old = 0.0


@dataclass
class CascadeController:
    """
    Cascade controller where primary controller adjusts secondary setpoint.

    Used for composition control loops in TEP.
    """
    primary: PIController
    secondary_setpoint_index: int  # Which setpoint to adjust
    secondary_scale: float = 1.0  # Scaling between primary output and secondary setpoint


@register_controller(
    name="decentralized",
    description="Decentralized multi-loop PI control system (18+ loops)"
)
class DecentralizedController(BaseController):
    """
    Decentralized multi-loop control system for TEP.

    Implements the control scheme from temain_mod.f with 18+ loops.
    """

    name = "decentralized"
    description = "Decentralized multi-loop PI control system (18+ loops)"
    version = "1.0.0"
    controlled_mvs = list(range(1, 12))  # Controls MVs 1-11

    def __init__(self, mode: int = DEFAULT_OPERATING_MODE):
        """
        Initialize the decentralized controller.

        Args:
            mode: Operating mode (1-6). Default is mode 1 (50/50 G/H, base rate).
        """
        # Store operating mode
        self._mode = mode

        # Initialize all controller loops
        self._init_controllers()

        # Control timing
        self.dt = 1.0 / 3600.0  # 1 second in hours
        self.step_count = 0

        # Setpoints array (for cascade controllers)
        self.setpoints = np.zeros(20)
        self._init_setpoints()

        # Purge valve flag for override control
        self.purge_flag = 0

    def _init_setpoints(self):
        """Initialize setpoint values based on operating mode."""
        # Get mode configuration
        mode_config = OPERATING_MODES.get(self._mode)

        if mode_config is not None:
            # Use optimal setpoints from Ricker (1995)
            sp = mode_config.xmeas_setpoints
            self.setpoints[0] = sp.get(1, 3664.0)     # D Feed flow (XMEAS 2)
            self.setpoints[1] = sp.get(2, 4509.3)     # E Feed flow (XMEAS 3)
            self.setpoints[2] = sp.get(0, 0.25052)    # A Feed flow (XMEAS 1)
            self.setpoints[3] = sp.get(3, 9.3477)     # A+C Feed flow (XMEAS 4)
            self.setpoints[4] = 26.902                 # Recycle flow (not in mode spec)
            self.setpoints[5] = sp.get(9, 0.33712)    # Purge rate (XMEAS 10)
            self.setpoints[6] = sp.get(11, 50.0)      # Separator level (XMEAS 12)
            self.setpoints[7] = sp.get(14, 50.0)      # Stripper level (XMEAS 15)
            self.setpoints[8] = 230.31                 # Steam flow (varies by mode, keep default)
            self.setpoints[9] = 94.599                 # Reactor CW outlet temp
            self.setpoints[10] = 22.949                # Stripper underflow
            self.setpoints[11] = 2633.7                # Separator pressure
            self.setpoints[12] = 32.188                # Reactor feed A composition
            self.setpoints[13] = 6.8820                # Reactor feed D composition
            self.setpoints[14] = 18.776                # Reactor feed E composition
            self.setpoints[15] = sp.get(17, 65.731)   # Stripper temperature (XMEAS 18)
            self.setpoints[16] = sp.get(7, 75.000)    # Reactor level (XMEAS 8)
            self.setpoints[17] = sp.get(8, 120.40)    # Reactor temperature (XMEAS 9)
            self.setpoints[18] = 13.823                # Purge B composition
            self.setpoints[19] = 0.83570               # Product E composition
        else:
            # Default setpoints (Downs & Vogel base case)
            self.setpoints[0] = 3664.0    # D Feed flow
            self.setpoints[1] = 4509.3    # E Feed flow
            self.setpoints[2] = 0.25052   # A Feed flow
            self.setpoints[3] = 9.3477    # A+C Feed flow
            self.setpoints[4] = 26.902    # Recycle flow
            self.setpoints[5] = 0.33712   # Purge rate
            self.setpoints[6] = 50.0      # Separator level
            self.setpoints[7] = 50.0      # Stripper level
            self.setpoints[8] = 230.31    # Steam flow
            self.setpoints[9] = 94.599    # Reactor CW outlet temp
            self.setpoints[10] = 22.949   # Stripper underflow
            self.setpoints[11] = 2633.7   # Separator pressure
            self.setpoints[12] = 32.188   # Reactor feed A composition
            self.setpoints[13] = 6.8820   # Reactor feed D composition
            self.setpoints[14] = 18.776   # Reactor feed E composition
            self.setpoints[15] = 65.731   # Stripper temperature
            self.setpoints[16] = 75.000   # Reactor level
            self.setpoints[17] = 120.40   # Reactor temperature
            self.setpoints[18] = 13.823   # Purge B composition
            self.setpoints[19] = 0.83570  # Product E composition

    @property
    def mode(self) -> int:
        """Get current operating mode."""
        return self._mode

    def set_mode(self, mode: int):
        """
        Change the operating mode.

        This updates the controller setpoints to match the new mode.
        The process will transition to the new operating point over time.

        Args:
            mode: Operating mode (1-6)
                1: 50/50 G/H, Base Rate
                2: 10/90 G/H, Base Rate
                3: 90/10 G/H, Base Rate
                4: 50/50 G/H, Max Rate
                5: 10/90 G/H, Max Rate
                6: 90/10 G/H, Max Rate
        """
        if mode not in OPERATING_MODES:
            raise ValueError(f"Invalid mode {mode}. Must be 1-6.")

        self._mode = mode
        self._init_setpoints()

        # Reset controller error states for smooth transition
        for ctrl in [self.ctrl1, self.ctrl2, self.ctrl3, self.ctrl4, self.ctrl5,
                     self.ctrl6, self.ctrl7, self.ctrl8, self.ctrl9, self.ctrl10,
                     self.ctrl11, self.ctrl13, self.ctrl14, self.ctrl15, self.ctrl16,
                     self.ctrl17, self.ctrl18, self.ctrl19, self.ctrl20, self.ctrl22]:
            ctrl.err_old = 0.0

    def get_mode_info(self) -> dict:
        """Get information about the current operating mode."""
        mode_config = OPERATING_MODES.get(self._mode)
        if mode_config:
            return {
                "mode": self._mode,
                "name": mode_config.name,
                "g_h_ratio": mode_config.g_h_ratio,
                "production": mode_config.production,
            }
        return {"mode": self._mode, "name": "Unknown"}

    def _init_controllers(self):
        """Initialize all PI controllers."""
        # Controller 1: D Feed Flow (XMEAS 2 -> XMV 1)
        self.ctrl1 = PIController(
            setpoint=3664.0, gain=1.0, taui=0.0,
            scale=100.0/5811.0
        )

        # Controller 2: E Feed Flow (XMEAS 3 -> XMV 2)
        self.ctrl2 = PIController(
            setpoint=4509.3, gain=1.0, taui=0.0,
            scale=100.0/8354.0
        )

        # Controller 3: A Feed Flow (XMEAS 1 -> XMV 3)
        self.ctrl3 = PIController(
            setpoint=0.25052, gain=1.0, taui=0.0,
            scale=100.0/1.017
        )

        # Controller 4: A+C Feed Flow (XMEAS 4 -> XMV 4)
        self.ctrl4 = PIController(
            setpoint=9.3477, gain=1.0, taui=0.0,
            scale=100.0/15.25
        )

        # Controller 5: Recycle Flow (XMEAS 5 -> XMV 5)
        self.ctrl5 = PIController(
            setpoint=26.902, gain=-0.083, taui=1.0/3600.0,
            scale=100.0/53.0
        )

        # Controller 6: Purge Rate (XMEAS 10 -> XMV 6)
        self.ctrl6 = PIController(
            setpoint=0.33712, gain=1.22, taui=0.0,
            scale=100.0/1.0
        )

        # Controller 7: Separator Level (XMEAS 12 -> XMV 7)
        self.ctrl7 = PIController(
            setpoint=50.0, gain=-2.06, taui=0.0,
            scale=100.0/70.0
        )

        # Controller 8: Stripper Level (XMEAS 15 -> XMV 8)
        self.ctrl8 = PIController(
            setpoint=50.0, gain=-1.62, taui=0.0,
            scale=100.0/70.0
        )

        # Controller 9: Steam Flow (XMEAS 19 -> XMV 9)
        self.ctrl9 = PIController(
            setpoint=230.31, gain=0.41, taui=0.0,
            scale=100.0/460.0
        )

        # Controller 10: Reactor CW Temp (XMEAS 21 -> XMV 10)
        self.ctrl10 = PIController(
            setpoint=94.599, gain=-0.156*10, taui=1452.0/3600.0,
            scale=100.0/150.0
        )

        # Controller 11: Stripper Underflow (XMEAS 17 -> XMV 11)
        self.ctrl11 = PIController(
            setpoint=22.949, gain=1.09, taui=2600.0/3600.0,
            scale=100.0/46.0
        )

        # Controller 13: Reactor Feed A (cascade to setpoint 3)
        self.ctrl13 = PIController(
            setpoint=32.188, gain=18.0, taui=3168.0/3600.0,
            scale=100.0/100.0
        )

        # Controller 14: Reactor Feed D (cascade to setpoint 1)
        self.ctrl14 = PIController(
            setpoint=6.8820, gain=8.3, taui=3168.0/3600.0,
            scale=100.0/100.0
        )

        # Controller 15: Reactor Feed E (cascade to setpoint 2)
        self.ctrl15 = PIController(
            setpoint=18.776, gain=2.37, taui=5069.0/3600.0,
            scale=100.0/100.0
        )

        # Controller 16: Stripper Temp (cascade to setpoint 9)
        self.ctrl16 = PIController(
            setpoint=65.731, gain=1.69/10.0, taui=236.0/3600.0,
            scale=100.0/130.0
        )

        # Controller 17: Reactor Level (cascade to setpoint 4)
        self.ctrl17 = PIController(
            setpoint=75.000, gain=11.1/10.0, taui=3168.0/3600.0,
            scale=100.0/50.0
        )

        # Controller 18: Reactor Temp (cascade to setpoint 10)
        self.ctrl18 = PIController(
            setpoint=120.40, gain=2.83*10.0, taui=982.0/3600.0,
            scale=100.0/150.0
        )

        # Controller 19: Purge B (cascade to setpoint 6)
        self.ctrl19 = PIController(
            setpoint=13.823, gain=-83.2/5.0/3.0, taui=6336.0/3600.0,
            scale=100.0/26.0
        )

        # Controller 20: Product E (cascade to setpoint 16)
        self.ctrl20 = PIController(
            setpoint=0.83570, gain=-16.3/5.0, taui=12408.0/3600.0,
            scale=100.0/1.6
        )

        # Controller 22: Separator Pressure (backup for purge)
        self.ctrl22 = PIController(
            setpoint=2633.7, gain=-1.0*5.0, taui=1000.0/3600.0,
            scale=1.0
        )

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        time_step: int
    ) -> np.ndarray:
        """
        Calculate new manipulated variable values.

        Args:
            xmeas: Current measurements (41 elements, 0-indexed)
            xmv: Current manipulated variables (12 elements, 0-indexed)
            time_step: Current simulation step number

        Returns:
            Updated manipulated variables
        """
        xmv_new = xmv.copy()
        dt3 = 3.0 / 3600.0  # 3 seconds in hours
        dt360 = 360.0 / 3600.0  # 360 seconds in hours
        dt900 = 900.0 / 3600.0  # 900 seconds in hours

        # Fast loops (every 3 seconds)
        if time_step % 3 == 0:
            # Update setpoints for cascade controllers
            self.ctrl1.setpoint = self.setpoints[0]
            self.ctrl2.setpoint = self.setpoints[1]
            self.ctrl3.setpoint = self.setpoints[2]
            self.ctrl4.setpoint = self.setpoints[3]
            self.ctrl9.setpoint = self.setpoints[8]

            # Controller 1: D Feed Flow
            xmv_new[0] = self.ctrl1.calculate(xmeas[1], xmv_new[0], dt3)

            # Controller 2: E Feed Flow
            xmv_new[1] = self.ctrl2.calculate(xmeas[2], xmv_new[1], dt3)

            # Controller 3: A Feed Flow
            xmv_new[2] = self.ctrl3.calculate(xmeas[0], xmv_new[2], dt3)

            # Controller 4: A+C Feed Flow
            xmv_new[3] = self.ctrl4.calculate(xmeas[3], xmv_new[3], dt3)

            # Controller 5: Recycle Flow
            xmv_new[4] = self.ctrl5.calculate(xmeas[4], xmv_new[4], dt3)

            # Controller 6: Purge Rate (with override logic)
            xmv_new[5] = self._purge_control(xmeas, xmv_new[5], dt3)

            # Controller 7: Separator Level
            xmv_new[6] = self.ctrl7.calculate(xmeas[11], xmv_new[6], dt3)

            # Controller 8: Stripper Level
            xmv_new[7] = self.ctrl8.calculate(xmeas[14], xmv_new[7], dt3)

            # Controller 9: Steam Flow
            xmv_new[8] = self.ctrl9.calculate(xmeas[18], xmv_new[8], dt3)

            # Controller 10: Reactor CW Temp
            xmv_new[9] = self.ctrl10.calculate(xmeas[20], xmv_new[9], dt3)

            # Controller 11: Stripper Underflow
            xmv_new[10] = self.ctrl11.calculate(xmeas[16], xmv_new[10], dt3)

            # Controller 16: Stripper Temp -> Steam setpoint (cascade)
            dxmv = self.ctrl16.calculate_change(xmeas[17], dt3)
            self.setpoints[8] += dxmv * 460.0 / 100.0

            # Controller 17: Reactor Level -> A+C Feed setpoint (cascade)
            dxmv = self.ctrl17.calculate_change(xmeas[7], dt3)
            self.setpoints[3] += dxmv * 15.25 / 100.0

            # Controller 18: Reactor Temp -> Reactor CW setpoint (cascade)
            dxmv = self.ctrl18.calculate_change(xmeas[8], dt3)
            self.setpoints[9] += dxmv * 150.0 / 100.0
            self.ctrl10.setpoint = self.setpoints[9]

        # Medium loops (every 360 seconds = 6 minutes)
        if time_step % 360 == 0:
            # Controller 13: Reactor Feed A -> A Feed setpoint (cascade)
            dxmv = self.ctrl13.calculate_change(xmeas[22], dt360)
            self.setpoints[2] += dxmv * 1.017 / 100.0

            # Controller 14: Reactor Feed D -> D Feed setpoint (cascade)
            dxmv = self.ctrl14.calculate_change(xmeas[25], dt360)
            self.setpoints[0] += dxmv * 5811.0 / 100.0

            # Controller 15: Reactor Feed E -> E Feed setpoint (cascade)
            dxmv = self.ctrl15.calculate_change(xmeas[26], dt360)
            self.setpoints[1] += dxmv * 8354.0 / 100.0

            # Controller 19: Purge B -> Purge Rate setpoint (cascade)
            dxmv = self.ctrl19.calculate_change(xmeas[29], dt360)
            self.setpoints[5] += dxmv * 1.0 / 100.0
            self.ctrl6.setpoint = self.setpoints[5]

        # Slow loops (every 900 seconds = 15 minutes)
        if time_step % 900 == 0:
            # Controller 20: Product E -> Stripper Temp setpoint (cascade)
            dxmv = self.ctrl20.calculate_change(xmeas[37], dt900)
            self.setpoints[15] += dxmv * 130.0 / 100.0
            self.ctrl16.setpoint = self.setpoints[15]

        # Apply constraints
        for i in range(11):
            xmv_new[i] = np.clip(xmv_new[i], 0.0, 100.0)

        self.step_count = time_step

        return xmv_new

    def _purge_control(self, xmeas: np.ndarray, current_xmv6: float, dt: float) -> float:
        """
        Purge valve control with override logic.

        Implements the complex purge control from CONTRL6 in temain_mod.f.
        """
        sep_pressure = xmeas[12]  # Separator pressure

        # High pressure override
        if sep_pressure >= 2950.0:
            self.purge_flag = 1
            return 100.0
        elif self.purge_flag == 1 and sep_pressure >= 2633.7:
            return 100.0
        elif self.purge_flag == 1 and sep_pressure < 2633.7:
            self.purge_flag = 0
            self.ctrl6.err_old = 0.0
            return 40.060

        # Low pressure override
        if sep_pressure <= 2300.0:
            self.purge_flag = 2
            return 0.0
        elif self.purge_flag == 2 and sep_pressure <= 2633.7:
            return 0.0
        elif self.purge_flag == 2 and sep_pressure > 2633.7:
            self.purge_flag = 0
            self.ctrl6.err_old = 0.0
            return 40.060

        # Normal control
        self.purge_flag = 0
        return self.ctrl6.calculate(xmeas[9], current_xmv6, dt)

    def reset(self):
        """Reset all controllers to initial state."""
        self._init_setpoints()
        self._init_controllers()
        self.purge_flag = 0
        self.step_count = 0

    def get_parameters(self) -> Dict[str, Any]:
        """Get controller parameters."""
        return {
            "mode": self._mode,
            "mode_info": self.get_mode_info(),
            "setpoints": self.setpoints.copy(),
            "purge_flag": self.purge_flag,
        }

    def set_setpoint(self, index: int, value: float):
        """
        Set a setpoint value.

        Args:
            index: Setpoint index (0-19)
            value: New setpoint value
        """
        if 0 <= index < len(self.setpoints):
            self.setpoints[index] = value


@register_controller(
    name="manual",
    description="Manual control - holds MVs at specified values"
)
class ManualController(BaseController):
    """
    Manual control mode - holds MVs at specified values.

    Useful for open-loop testing or when manual intervention is needed.
    """

    name = "manual"
    description = "Manual control - holds MVs at specified values"
    version = "1.0.0"
    controlled_mvs = list(range(1, 13))  # Can control all MVs

    def __init__(self, initial_values: np.ndarray = None):
        """
        Initialize manual controller.

        Args:
            initial_values: Initial MV values (default from steady state)
        """
        if initial_values is None:
            from .constants import INITIAL_STATES
            self.mv_values = INITIAL_STATES[38:50].copy()
        else:
            self.mv_values = initial_values.copy()

    def set_mv(self, index: int, value: float):
        """
        Set a manipulated variable.

        Args:
            index: MV index (1-12)
            value: Valve position (0-100%)
        """
        idx = index - 1 if index >= 1 else index
        if 0 <= idx < 12:
            self.mv_values[idx] = np.clip(value, 0.0, 100.0)

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        time_step: int
    ) -> np.ndarray:
        """
        Return the manually set MV values.

        Args:
            xmeas: Current measurements (ignored)
            xmv: Current MVs (ignored)
            time_step: Current step (ignored)

        Returns:
            Manual MV values
        """
        return self.mv_values.copy()

    def reset(self):
        """Reset to initial MV values."""
        from .constants import INITIAL_STATES
        self.mv_values = INITIAL_STATES[38:50].copy()

    def get_parameters(self) -> Dict[str, Any]:
        """Get controller parameters."""
        return {
            "mv_values": self.mv_values.copy(),
        }
