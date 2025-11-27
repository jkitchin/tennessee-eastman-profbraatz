"""
Main Tennessee Eastman Process Simulator.

This module provides the high-level TEPSimulator class that integrates
all components and provides an easy-to-use interface for simulation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple, Union
from enum import Enum
from .process import TEProcess
from .controllers import DecentralizedController, ManualController, PIController
from .integrators import Integrator, IntegratorType
from .disturbances import DisturbanceManager
from .constants import (
    NUM_STATES, NUM_MEASUREMENTS, NUM_MANIPULATED_VARS,
    INITIAL_STATES, DEFAULT_RANDOM_SEED
)


class ControlMode(Enum):
    """Simulation control modes."""
    OPEN_LOOP = "open_loop"
    CLOSED_LOOP = "closed_loop"
    MANUAL = "manual"


@dataclass
class SimulationResult:
    """Container for simulation results."""
    time: np.ndarray  # Time points (hours)
    states: np.ndarray  # State trajectories (n_steps x 50)
    measurements: np.ndarray  # Measurement trajectories (n_steps x 41)
    manipulated_vars: np.ndarray  # MV trajectories (n_steps x 12)
    shutdown: bool = False  # Whether simulation ended in shutdown
    shutdown_time: float = None  # Time of shutdown (if any)

    @property
    def time_seconds(self) -> np.ndarray:
        """Get time in seconds."""
        return self.time * 3600.0

    @property
    def time_minutes(self) -> np.ndarray:
        """Get time in minutes."""
        return self.time * 60.0


class TEPSimulator:
    """
    Tennessee Eastman Process Simulator.

    This class provides a high-level interface for running TEP simulations
    with various control modes and disturbance scenarios.

    Example usage:
        >>> sim = TEPSimulator()
        >>> result = sim.simulate(duration_hours=1.0)
        >>> print(result.measurements.shape)
        (3601, 41)

    For real-time dashboard integration:
        >>> sim = TEPSimulator()
        >>> sim.initialize()
        >>> while running:
        ...     sim.step()
        ...     measurements = sim.get_measurements()
        ...     # Update dashboard
    """

    def __init__(
        self,
        random_seed: int = None,
        control_mode: ControlMode = ControlMode.CLOSED_LOOP,
        integrator: IntegratorType = IntegratorType.EULER
    ):
        """
        Initialize the TEP simulator.

        Args:
            random_seed: Random seed for reproducibility
            control_mode: Control mode (open_loop, closed_loop, or manual)
            integrator: Integration method to use
        """
        if random_seed is None:
            random_seed = DEFAULT_RANDOM_SEED

        self.random_seed = random_seed
        self.control_mode = control_mode

        # Initialize process
        self.process = TEProcess(random_seed)

        # Initialize controller based on mode
        self._init_controller()

        # Initialize integrator
        self.integrator = Integrator(integrator)

        # Simulation state
        self.time = 0.0  # Current time (hours)
        self.step_count = 0  # Step counter
        self.dt = 1.0 / 3600.0  # Time step (1 second in hours)
        self.initialized = False

        # History buffers for streaming/dashboard use
        self._history_size = 0
        self._time_history: List[float] = []
        self._state_history: List[np.ndarray] = []
        self._meas_history: List[np.ndarray] = []
        self._mv_history: List[np.ndarray] = []

    def _init_controller(self):
        """Initialize controller based on control mode."""
        if self.control_mode == ControlMode.CLOSED_LOOP:
            self.controller = DecentralizedController()
        elif self.control_mode == ControlMode.MANUAL:
            self.controller = ManualController()
        else:  # OPEN_LOOP
            self.controller = ManualController()

    def initialize(self):
        """
        Initialize or reset the simulator to steady-state conditions.

        This must be called before running a simulation or stepping.
        """
        self.process._initialize()
        self._init_controller()

        self.time = 0.0
        self.step_count = 0
        self.initialized = True

        # Clear history
        self._time_history = [self.time]
        self._state_history = [self.process.state.yy.copy()]
        self._meas_history = [self.process.state.xmeas.copy()]
        self._mv_history = [self.process.state.xmv.copy()]

        # Calculate initial measurements
        _ = self.process.evaluate(self.time, self.process.state.yy)

    def set_disturbance(self, idv_index: int, value: int = 1):
        """
        Set a process disturbance.

        Args:
            idv_index: Disturbance index (1-20)
            value: 0 = off, 1 = on
        """
        self.process.set_idv(idv_index, value)

    def clear_disturbances(self):
        """Turn off all disturbances."""
        self.process.disturbances.clear_all_disturbances()

    def set_mv(self, index: int, value: float):
        """
        Set a manipulated variable (for manual control).

        Args:
            index: MV index (1-12)
            value: Valve position (0-100%)
        """
        self.process.set_xmv(index, value)
        if isinstance(self.controller, ManualController):
            self.controller.set_mv(index, value)

    def get_measurements(self) -> np.ndarray:
        """Get current process measurements (41 values)."""
        return self.process.get_xmeas()

    def get_manipulated_vars(self) -> np.ndarray:
        """Get current manipulated variable values (12 values)."""
        return self.process.get_xmv()

    def get_states(self) -> np.ndarray:
        """Get current state vector (50 values)."""
        return self.process.state.yy.copy()

    def is_shutdown(self) -> bool:
        """Check if process is in shutdown state."""
        return self.process.is_shutdown()

    def step(self, n_steps: int = 1) -> bool:
        """
        Advance simulation by n steps.

        Args:
            n_steps: Number of integration steps to take

        Returns:
            True if simulation is still running, False if shutdown
        """
        if not self.initialized:
            self.initialize()

        for _ in range(n_steps):
            # Execute controller
            if self.control_mode != ControlMode.OPEN_LOOP:
                xmeas = self.process.get_xmeas()
                xmv = self.process.get_xmv()
                new_xmv = self.controller.calculate(xmeas, xmv, self.step_count)

                # Update process MVs
                for i in range(NUM_MANIPULATED_VARS):
                    self.process.set_xmv(i + 1, new_xmv[i])

            # Integrate one step
            yp = self.process.evaluate(self.time, self.process.state.yy)
            self.time += self.dt
            self.process.state.yy = self.process.state.yy + yp * self.dt

            self.step_count += 1

            # Check for shutdown
            if self.process.is_shutdown():
                return False

        return True

    def simulate(
        self,
        duration_hours: float = 1.0,
        dt_hours: float = None,
        disturbances: Dict[int, Tuple[float, int]] = None,
        record_interval: int = 1,
        progress_callback: Callable[[float], None] = None
    ) -> SimulationResult:
        """
        Run a complete simulation.

        Args:
            duration_hours: Simulation duration in hours
            dt_hours: Time step in hours (default 1 second)
            disturbances: Dict mapping IDV index to (time_hours, value)
                          e.g., {1: (0.5, 1)} activates IDV(1) at 0.5 hours
            record_interval: Record every N steps (default 1 = all steps)
            progress_callback: Optional callback called with progress (0-1)

        Returns:
            SimulationResult containing time series data
        """
        if not self.initialized:
            self.initialize()

        if dt_hours is None:
            dt_hours = self.dt

        # Calculate number of steps
        n_steps = int(np.ceil(duration_hours / dt_hours))

        # Initialize recording arrays
        record_steps = (n_steps // record_interval) + 1
        times = np.zeros(record_steps)
        states = np.zeros((record_steps, NUM_STATES))
        measurements = np.zeros((record_steps, NUM_MEASUREMENTS))
        mvs = np.zeros((record_steps, NUM_MANIPULATED_VARS))

        # Record initial state
        times[0] = self.time
        states[0] = self.process.state.yy.copy()
        measurements[0] = self.process.get_xmeas()
        mvs[0] = self.process.get_xmv()

        record_idx = 1
        shutdown = False
        shutdown_time = None

        # Process disturbance schedule
        disturbance_times = {}
        if disturbances:
            for idv_idx, (time_hr, value) in disturbances.items():
                disturbance_times[time_hr] = (idv_idx, value)

        # Main simulation loop
        for step in range(1, n_steps + 1):
            # Check for scheduled disturbances
            for dist_time, (idv_idx, value) in list(disturbance_times.items()):
                if self.time >= dist_time:
                    self.set_disturbance(idv_idx, value)
                    del disturbance_times[dist_time]

            # Execute one step
            if not self.step():
                shutdown = True
                shutdown_time = self.time
                break

            # Record if at interval
            if step % record_interval == 0 and record_idx < record_steps:
                times[record_idx] = self.time
                states[record_idx] = self.process.state.yy.copy()
                measurements[record_idx] = self.process.get_xmeas()
                mvs[record_idx] = self.process.get_xmv()
                record_idx += 1

            # Progress callback
            if progress_callback and step % 1000 == 0:
                progress_callback(step / n_steps)

        # Trim arrays if shutdown occurred early
        if record_idx < record_steps:
            times = times[:record_idx]
            states = states[:record_idx]
            measurements = measurements[:record_idx]
            mvs = mvs[:record_idx]

        return SimulationResult(
            time=times,
            states=states,
            measurements=measurements,
            manipulated_vars=mvs,
            shutdown=shutdown,
            shutdown_time=shutdown_time
        )

    def simulate_with_controller(
        self,
        duration_hours: float,
        controller: Union[DecentralizedController, PIController, Callable],
        disturbances: Dict[int, Tuple[float, int]] = None,
        record_interval: int = 1
    ) -> SimulationResult:
        """
        Run simulation with a custom controller.

        Args:
            duration_hours: Simulation duration
            controller: Controller object or callable(xmeas, xmv, step) -> new_xmv
            disturbances: Disturbance schedule
            record_interval: Recording interval

        Returns:
            SimulationResult
        """
        # Temporarily replace controller
        old_controller = self.controller
        old_mode = self.control_mode

        if callable(controller) and not hasattr(controller, 'calculate'):
            # Wrap callable in a simple class
            class FunctionController:
                def __init__(self, func):
                    self.func = func
                def calculate(self, xmeas, xmv, step):
                    return self.func(xmeas, xmv, step)

            self.controller = FunctionController(controller)
        else:
            self.controller = controller

        self.control_mode = ControlMode.CLOSED_LOOP

        try:
            result = self.simulate(
                duration_hours=duration_hours,
                disturbances=disturbances,
                record_interval=record_interval
            )
        finally:
            self.controller = old_controller
            self.control_mode = old_mode

        return result

    def get_measurement_names(self) -> List[str]:
        """Get list of measurement variable names."""
        from .constants import MEASUREMENT_NAMES
        return MEASUREMENT_NAMES.copy()

    def get_mv_names(self) -> List[str]:
        """Get list of manipulated variable names."""
        from .constants import MANIPULATED_VAR_NAMES
        return MANIPULATED_VAR_NAMES.copy()

    def get_disturbance_names(self) -> List[str]:
        """Get list of disturbance names."""
        from .constants import DISTURBANCE_NAMES
        return DISTURBANCE_NAMES.copy()

    # Real-time streaming interface
    def start_stream(self, history_size: int = 1000):
        """
        Start streaming mode for real-time dashboard.

        Args:
            history_size: Number of historical points to maintain
        """
        self._history_size = history_size
        self.initialize()

    def stream_step(self) -> Dict:
        """
        Take one step and return current state for dashboard.

        Returns:
            Dict with 'time', 'measurements', 'mvs', 'shutdown' keys
        """
        running = self.step()

        # Update history
        self._time_history.append(self.time)
        self._meas_history.append(self.process.get_xmeas())
        self._mv_history.append(self.process.get_xmv())

        # Trim history if needed
        if len(self._time_history) > self._history_size:
            self._time_history = self._time_history[-self._history_size:]
            self._meas_history = self._meas_history[-self._history_size:]
            self._mv_history = self._mv_history[-self._history_size:]

        return {
            'time': self.time,
            'time_seconds': self.time * 3600,
            'measurements': self.process.get_xmeas(),
            'mvs': self.process.get_xmv(),
            'shutdown': not running
        }

    def get_stream_history(self) -> Dict:
        """
        Get historical data for dashboard plotting.

        Returns:
            Dict with 'time', 'measurements', 'mvs' arrays
        """
        return {
            'time': np.array(self._time_history),
            'time_seconds': np.array(self._time_history) * 3600,
            'measurements': np.array(self._meas_history),
            'mvs': np.array(self._mv_history)
        }
