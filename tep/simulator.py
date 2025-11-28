"""
Main Tennessee Eastman Process Simulator.

This module provides the high-level TEPSimulator class that integrates
all components and provides an easy-to-use interface for simulation.

This version uses the Fortran backend exclusively via f2py for exact
reproduction of the original simulation results.

Features:
    - Batch and streaming simulation modes
    - Configurable control modes (open loop, closed loop, manual)
    - Disturbance injection with scheduling
    - Real-time fault detection with pluggable detectors
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple, Union, TYPE_CHECKING
from enum import Enum
from .fortran_backend import FortranTEProcess
from .controllers import DecentralizedController, ManualController, PIController
from .constants import (
    NUM_STATES, NUM_MEASUREMENTS, NUM_MANIPULATED_VARS,
    DEFAULT_RANDOM_SEED
)

# Import detector types for type checking (avoid circular imports)
if TYPE_CHECKING:
    from .detector_base import BaseFaultDetector, DetectionResult


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
    detection_results: Dict[str, List] = field(default_factory=dict)  # Detector results

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
    with various control modes and disturbance scenarios. It uses the original
    Fortran code via f2py for exact reproduction of simulation results.

    Features:
        - Batch simulation with configurable duration and recording
        - Real-time streaming mode for dashboards
        - Multiple control modes (open loop, closed loop, manual)
        - Disturbance injection with time-based scheduling
        - Pluggable fault detector support with metrics tracking

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

    With fault detection:
        >>> from tep.detector_base import FaultDetectorRegistry
        >>> sim = TEPSimulator()
        >>> detector = FaultDetectorRegistry.create("pca", window_size=200)
        >>> sim.add_detector(detector)
        >>> sim.initialize()
        >>> sim.set_ground_truth(0)  # Normal operation
        >>> result = sim.simulate(duration_hours=1.0)
        >>> print(detector.metrics)
    """

    def __init__(
        self,
        random_seed: int = None,
        control_mode: ControlMode = ControlMode.CLOSED_LOOP,
    ):
        """
        Initialize the TEP simulator.

        Args:
            random_seed: Random seed for reproducibility
            control_mode: Control mode (open_loop, closed_loop, or manual)

        Raises:
            ImportError: If Fortran extension is not available
        """
        if random_seed is None:
            random_seed = DEFAULT_RANDOM_SEED

        self.random_seed = random_seed
        self.control_mode = control_mode

        # Initialize process with Fortran backend
        self.process = FortranTEProcess(random_seed)

        # Initialize controller based on mode
        self._init_controller()

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

        # Fault detection
        self._detectors: List['BaseFaultDetector'] = []
        self._detection_results: Dict[str, List['DetectionResult']] = {}
        self._ground_truth: int = 0  # Current true fault class (0=normal)
        self._fault_onset_step: Optional[int] = None

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
        Resets the process, controller, and all registered detectors.
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

        # Reset detectors
        self._ground_truth = 0
        self._fault_onset_step = None
        self._detection_results = {d.name: [] for d in self._detectors}
        for detector in self._detectors:
            detector.reset()

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

    def get_disturbances(self) -> np.ndarray:
        """Get current disturbance vector (20 values, 0 or 1)."""
        return self.process.idv

    def get_active_disturbances(self) -> list:
        """Get list of active disturbance indices (1-based)."""
        idv = self.process.idv
        return [i + 1 for i in range(len(idv)) if idv[i] != 0]

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

    @property
    def backend(self) -> str:
        """Get the current simulation backend (always 'fortran')."""
        return "fortran"

    def step(self, n_steps: int = 1) -> bool:
        """
        Advance simulation by n steps.

        Each step:
        1. Executes the controller (if not open loop)
        2. Integrates the process equations
        3. Processes all registered fault detectors

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

            # Process fault detectors
            if self._detectors:
                xmeas = self.process.get_xmeas()
                for detector in self._detectors:
                    result = detector.process(xmeas, self.step_count)
                    if detector.name in self._detection_results:
                        self._detection_results[detector.name].append(result)

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
                    # Update ground truth for detectors
                    if value != 0:
                        self.set_ground_truth(idv_idx)
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
            shutdown_time=shutdown_time,
            detection_results=dict(self._detection_results)
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

    # =========================================================================
    # Fault Detection Interface
    # =========================================================================

    def add_detector(self, detector: 'BaseFaultDetector'):
        """
        Add a fault detector to the simulation.

        The detector will be called after each simulation step with the
        current measurements. Detection results are accumulated and can
        be retrieved via get_detection_results().

        Args:
            detector: A fault detector instance (must inherit from BaseFaultDetector)

        Example:
            >>> from tep.detector_base import FaultDetectorRegistry
            >>> detector = FaultDetectorRegistry.create("pca", window_size=200)
            >>> sim.add_detector(detector)
        """
        self._detectors.append(detector)
        self._detection_results[detector.name] = []
        # Set current ground truth
        detector.set_ground_truth(self._ground_truth, self._fault_onset_step)

    def remove_detector(self, name: str) -> bool:
        """
        Remove a detector by name.

        Args:
            name: Name of the detector to remove

        Returns:
            True if detector was found and removed, False otherwise
        """
        for i, detector in enumerate(self._detectors):
            if detector.name == name:
                self._detectors.pop(i)
                self._detection_results.pop(name, None)
                return True
        return False

    def get_detector(self, name: str) -> Optional['BaseFaultDetector']:
        """
        Get a detector by name.

        Args:
            name: Name of the detector

        Returns:
            Detector instance or None if not found
        """
        for detector in self._detectors:
            if detector.name == name:
                return detector
        return None

    def list_detectors(self) -> List[str]:
        """
        List names of all registered detectors.

        Returns:
            List of detector names
        """
        return [d.name for d in self._detectors]

    def clear_detectors(self):
        """Remove all registered detectors."""
        for detector in self._detectors:
            detector.shutdown()
        self._detectors.clear()
        self._detection_results.clear()

    def set_ground_truth(self, fault_class: int):
        """
        Set the current ground truth fault class for all detectors.

        This enables detectors to track their accuracy metrics. Call this
        when the true process state changes (e.g., when a fault is introduced).

        Args:
            fault_class: True fault class (0=normal, 1-20=IDV fault index)

        Example:
            >>> sim.set_ground_truth(0)  # Normal operation
            >>> sim.set_disturbance(4, 1)  # Introduce fault
            >>> sim.set_ground_truth(4)  # Tell detectors the truth
        """
        old_truth = self._ground_truth
        self._ground_truth = fault_class

        # Track when fault started
        if fault_class > 0 and old_truth == 0:
            self._fault_onset_step = self.step_count
        elif fault_class == 0:
            self._fault_onset_step = None

        # Update all detectors
        for detector in self._detectors:
            detector.set_ground_truth(fault_class, self._fault_onset_step)

    def get_ground_truth(self) -> int:
        """
        Get the current ground truth fault class.

        Returns:
            Current fault class (0=normal, 1-20=fault)
        """
        return self._ground_truth

    def get_detection_results(self, detector_name: str = None) -> Dict[str, List]:
        """
        Get detection results from registered detectors.

        Args:
            detector_name: Optional name to get results for specific detector

        Returns:
            Dict mapping detector name to list of DetectionResult objects.
            If detector_name is specified, returns only that detector's results.
        """
        if detector_name:
            return {detector_name: self._detection_results.get(detector_name, [])}
        return dict(self._detection_results)

    def get_latest_detection(self, detector_name: str = None) -> Dict[str, 'DetectionResult']:
        """
        Get the most recent detection result from each detector.

        Args:
            detector_name: Optional name to get result for specific detector

        Returns:
            Dict mapping detector name to latest DetectionResult
        """
        results = {}
        for name, result_list in self._detection_results.items():
            if detector_name and name != detector_name:
                continue
            if result_list:
                results[name] = result_list[-1]
        return results

    def get_detector_metrics(self, detector_name: str = None) -> Dict[str, Dict]:
        """
        Get performance metrics from detectors.

        Args:
            detector_name: Optional name to get metrics for specific detector

        Returns:
            Dict mapping detector name to metrics summary dict
        """
        metrics = {}
        for detector in self._detectors:
            if detector_name and detector.name != detector_name:
                continue
            metrics[detector.name] = detector.metrics.summary()
        return metrics

    def reset_detector_metrics(self):
        """Reset metrics for all detectors without resetting detector state."""
        for detector in self._detectors:
            detector.reset_metrics()
