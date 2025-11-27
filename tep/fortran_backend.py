"""Fortran backend wrapper for TEP.

This module provides a Python interface to the original Fortran code,
enabling exact reproduction of the original simulation results.
"""

import numpy as np
from typing import Optional


class FortranTEProcess:
    """Wraps Fortran TEINIT/TEFUNC with common block access.

    This class provides an interface compatible with TEProcess but uses
    the original Fortran implementation for exact numerical parity.
    """

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize the Fortran TEP wrapper.

        Parameters
        ----------
        random_seed : int, optional
            Random seed for reproducibility.
        """
        from tep._fortran import teprob
        self._teprob = teprob
        self._nn = 50  # Number of state variables
        self._initialized = False
        self.time = 0.0

        # State vectors (Fortran-contiguous arrays)
        self.yy = np.zeros(self._nn, dtype=np.float64, order='F')
        self.yp = np.zeros(self._nn, dtype=np.float64, order='F')

        # Create state wrapper for API compatibility with TEProcess
        self.state = FortranTEProcessState(self)

        # Create disturbances wrapper for API compatibility
        self.disturbances = FortranDisturbanceManager(self)

        # Set random seed if provided (there's no randsd in the wrapped module)
        # The seed is set during teinit via the RANDSD common block

    def _initialize(self):
        """Call TEINIT to initialize the process (internal method for TEPSimulator)."""
        self.initialize()

    def initialize(self):
        """Call TEINIT to initialize the process."""
        self._teprob.teinit(0.0, self.yy, self.yp)
        self._initialized = True
        self.time = 0.0

    def step(self, dt: float = 1.0/3600.0):
        """Single Euler integration step.

        Parameters
        ----------
        dt : float
            Time step in hours. Default is 1 second (1/3600 hours).
        """
        if not self._initialized:
            raise RuntimeError("Process not initialized. Call initialize() first.")

        # Call tefunc to get derivatives
        self._teprob.tefunc(self.time, self.yy, self.yp)

        # Euler integration: yy = yy + yp * dt
        self.yy[:] = self.yy + self.yp * dt
        self.time += dt

        # Apply valve constraints (same as CONSHAND)
        self._apply_constraints()

    def _apply_constraints(self):
        """Apply valve constraints (Python implementation of CONSHAND)."""
        # XMV values are constrained to 0-100%
        xmv = self._teprob.pv.xmv
        for i in range(11):  # XMV 1-11 have 0-100 limits
            xmv[i] = np.clip(xmv[i], 0.0, 100.0)
        # XMV(12) might have different limits in some versions

    @property
    def xmeas(self) -> np.ndarray:
        """Get current measurement values (XMEAS common block)."""
        return self._teprob.pv.xmeas.copy()

    @property
    def xmv(self) -> np.ndarray:
        """Get current manipulated variables (XMV common block)."""
        return self._teprob.pv.xmv.copy()

    @property
    def idv(self) -> np.ndarray:
        """Get current disturbance vector (IDV common block)."""
        return self._teprob.dvec.idv.copy()

    def set_xmv(self, index: int, value: float):
        """Set a manipulated variable.

        Parameters
        ----------
        index : int
            1-based index of the manipulated variable (1-12).
        value : float
            Value to set (will be clipped to 0-100).
        """
        if not 1 <= index <= 12:
            raise ValueError(f"XMV index must be 1-12, got {index}")
        self._teprob.pv.xmv[index - 1] = np.clip(value, 0.0, 100.0)

    def set_idv(self, index: int, value: int):
        """Set a disturbance variable.

        Parameters
        ----------
        index : int
            1-based index of the disturbance (1-20).
        value : int
            0 to disable, 1 to enable the disturbance.
        """
        if not 1 <= index <= 20:
            raise ValueError(f"IDV index must be 1-20, got {index}")
        self._teprob.dvec.idv[index - 1] = value

    def clear_disturbances(self):
        """Clear all disturbances (set IDV to 0)."""
        self._teprob.dvec.idv[:] = 0

    def get_xmeas(self) -> np.ndarray:
        """Get current measurement values (for TEPSimulator compatibility)."""
        return self._teprob.pv.xmeas.copy()

    def get_xmv(self) -> np.ndarray:
        """Get current manipulated variables (for TEPSimulator compatibility)."""
        return self._teprob.pv.xmv.copy()

    def evaluate(self, time: float, yy: np.ndarray) -> np.ndarray:
        """Evaluate derivatives using TEFUNC (for TEPSimulator compatibility).

        Parameters
        ----------
        time : float
            Current time in hours.
        yy : np.ndarray
            Current state vector.

        Returns
        -------
        np.ndarray
            Derivative vector.
        """
        # Update internal state
        self.yy[:] = yy

        # Call TEFUNC to compute derivatives
        self._teprob.tefunc(time, self.yy, self.yp)

        return self.yp.copy()

    def is_shutdown(self) -> bool:
        """Check if process is in shutdown state.

        The TEP process shuts down if certain measurements exceed limits.
        This is a simplified check - the actual Fortran code sets flag6.
        """
        # Check flag6 if available
        if hasattr(self._teprob, 'flag6'):
            return self._teprob.flag6.flag != 0

        # Fallback: check for extreme conditions
        xmeas = self._teprob.pv.xmeas
        # Reactor pressure limits
        if xmeas[6] < 2500 or xmeas[6] > 3200:
            return True
        # Reactor level limits
        if xmeas[7] < 1 or xmeas[7] > 99:
            return True
        return False

    def run_controllers(self, step_count: int):
        """Placeholder for controller execution.

        Note: The Fortran controller subroutines (CONTRL1-CONTRL22) are not
        exposed by f2py in the current build. Use the Python DecentralizedController
        instead when running closed-loop simulations.

        This method is kept for API compatibility but currently does nothing.
        For exact Fortran behavior, use the Python controllers which implement
        the same PI control logic.

        Parameters
        ----------
        step_count : int
            Current step count (unused in this placeholder).
        """
        # The Fortran controller subroutines are not exposed by f2py
        # Use Python DecentralizedController for closed-loop control
        pass

    def get_state(self) -> np.ndarray:
        """Get current state vector."""
        return self.yy.copy()

    def set_state(self, state: np.ndarray):
        """Set state vector."""
        if len(state) != self._nn:
            raise ValueError(f"State must have {self._nn} elements")
        self.yy[:] = state


class FortranTEProcessState:
    """State wrapper that mimics TEProcess state interface."""

    def __init__(self, process: 'FortranTEProcess'):
        self._process = process

    @property
    def yy(self) -> np.ndarray:
        return self._process.yy

    @yy.setter
    def yy(self, value: np.ndarray):
        self._process.yy[:] = value

    @property
    def xmeas(self) -> np.ndarray:
        return self._process.xmeas

    @property
    def xmv(self) -> np.ndarray:
        return self._process.xmv


class FortranDisturbanceManager:
    """Disturbance manager wrapper for API compatibility with TEProcess."""

    def __init__(self, process: 'FortranTEProcess'):
        self._process = process

    def clear_all_disturbances(self):
        """Clear all disturbances (set IDV to 0)."""
        self._process.clear_disturbances()

    def set_disturbance(self, index: int, value: int):
        """Set a disturbance variable."""
        self._process.set_idv(index, value)
