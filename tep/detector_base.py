"""
Base classes and plugin registry for TEP fault detectors.

This module provides the foundation for a real-time fault detection system,
allowing users to create and register custom detectors that analyze
process measurements and identify fault conditions.

Key Classes:
    - DetectionResult: Return value from detectors with fault class and confidence
    - DetectionMetrics: Accumulative performance metrics (accuracy, F1, etc.)
    - BaseFaultDetector: Abstract base class for all detectors
    - FaultDetectorRegistry: Plugin discovery and instantiation

Example:
    >>> from tep.detector_base import BaseFaultDetector, register_detector
    >>>
    >>> @register_detector(name="my_detector")
    ... class MyDetector(BaseFaultDetector):
    ...     window_size = 100
    ...
    ...     def detect(self, xmeas, step):
    ...         if not self.window_ready:
    ...             return DetectionResult(-1, 0.0, step)
    ...         # Detection logic here
    ...         return DetectionResult(0, 0.9, step)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Optional, Type, Any, Tuple, Callable
import numpy as np
import threading
import time

from .constants import NUM_MEASUREMENTS, NUM_DISTURBANCES


# =============================================================================
# Detection Result
# =============================================================================

@dataclass
class DetectionResult:
    """
    Result from a fault detector.

    This is the return type for all detector.detect() calls. It provides
    both a simple interface (fault_class, confidence) and rich optional
    data for sophisticated analysis.

    Attributes:
        fault_class: Predicted class. -1=unknown/not ready, 0=normal, 1-20=fault
        confidence: Confidence in prediction, 0.0 to 1.0
        step: Simulation step when this detection was made
        timestamp: Wall clock time of detection
        latency_steps: For async detectors, how many steps old this result is
        alternatives: Other likely classes as [(class, confidence), ...]
        contributing_sensors: XMEAS indices that drove the detection
        statistics: Detector-specific stats (e.g., T2, SPE values)

    Example:
        >>> result = DetectionResult(fault_class=4, confidence=0.85, step=1000)
        >>> if result.is_fault and result.confidence > 0.8:
        ...     print(f"High confidence fault {result.fault_class} detected")
    """

    # Core prediction
    fault_class: int
    confidence: float
    step: int

    # Timing
    timestamp: float = field(default_factory=time.time)
    latency_steps: int = 0

    # Extended information
    alternatives: Optional[List[Tuple[int, float]]] = None
    contributing_sensors: Optional[List[int]] = None
    statistics: Optional[Dict[str, float]] = None

    @property
    def is_ready(self) -> bool:
        """True if detector has enough data to make predictions."""
        return self.fault_class != -1

    @property
    def is_normal(self) -> bool:
        """True if no fault detected."""
        return self.fault_class == 0

    @property
    def is_fault(self) -> bool:
        """True if a fault is detected (class 1-20)."""
        return self.fault_class > 0

    def top_k(self, k: int = 3) -> List[Tuple[int, float]]:
        """
        Get top k predictions including primary.

        Args:
            k: Number of predictions to return

        Returns:
            List of (class, confidence) tuples sorted by confidence
        """
        result = [(self.fault_class, self.confidence)]
        if self.alternatives:
            result.extend(self.alternatives[:k-1])
        return result

    def above_threshold(self, threshold: float = 0.5) -> List[int]:
        """
        Get all fault classes with confidence above threshold.

        Useful for detecting multiple simultaneous faults.

        Args:
            threshold: Minimum confidence threshold

        Returns:
            List of fault classes (excludes class 0/normal)
        """
        result = []
        if self.fault_class > 0 and self.confidence >= threshold:
            result.append(self.fault_class)
        if self.alternatives:
            for cls, conf in self.alternatives:
                if cls > 0 and conf >= threshold:
                    result.append(cls)
        return result

    def __repr__(self) -> str:
        status = "unknown" if self.fault_class == -1 else (
            "normal" if self.fault_class == 0 else f"fault_{self.fault_class}"
        )
        return f"DetectionResult({status}, conf={self.confidence:.2f}, step={self.step})"


# =============================================================================
# Detection Metrics
# =============================================================================

@dataclass
class DetectionMetrics:
    """
    Accumulative metrics for fault detector evaluation.

    Tracks a confusion matrix and computes standard classification metrics.
    Supports 21 classes: 0=normal, 1-20=fault types (IDV indices).

    The metrics update incrementally as detections are recorded, making
    this suitable for both batch evaluation and real-time monitoring.

    Example:
        >>> metrics = DetectionMetrics()
        >>> metrics.update(actual=4, predicted=4, step=1000)  # Correct
        >>> metrics.update(actual=4, predicted=0, step=1001)  # Missed
        >>> print(metrics.accuracy)
        0.5
        >>> print(metrics.recall(4))
        0.5
    """

    n_classes: int = 21  # 0=normal + 20 faults

    # Confusion matrix: rows=actual, cols=predicted
    confusion_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((21, 21), dtype=np.int64)
    )

    # Track "unknown" predictions (-1)
    unknown_count: int = 0
    unknown_by_actual: Dict[int, int] = field(default_factory=dict)

    # Temporal tracking
    total_predictions: int = 0
    detection_delays: Dict[int, List[int]] = field(default_factory=dict)

    def update(self, actual: int, predicted: int, step: int = None,
               fault_onset_step: int = None):
        """
        Record a prediction.

        Args:
            actual: True class (0=normal, 1-20=fault)
            predicted: Predicted class from detector (-1, 0, or 1-20)
            step: Current simulation step
            fault_onset_step: When the fault started (for delay tracking)
        """
        self.total_predictions += 1

        if predicted == -1:
            self.unknown_count += 1
            self.unknown_by_actual[actual] = self.unknown_by_actual.get(actual, 0) + 1
            return

        if 0 <= actual < self.n_classes and 0 <= predicted < self.n_classes:
            self.confusion_matrix[actual, predicted] += 1

        # Track first correct detection delay for faults
        if (actual > 0 and predicted == actual and
            step is not None and fault_onset_step is not None):
            delay = step - fault_onset_step
            if actual not in self.detection_delays:
                self.detection_delays[actual] = []
            # Only record if this is a new/earlier detection
            delays = self.detection_delays[actual]
            if not delays or delay not in delays:
                delays.append(delay)

    def reset(self):
        """Clear all accumulated metrics."""
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)
        self.unknown_count = 0
        self.unknown_by_actual.clear()
        self.total_predictions = 0
        self.detection_delays.clear()

    # --- Aggregate Metrics ---

    @property
    def total_samples(self) -> int:
        """Total samples with known predictions (excludes unknown)."""
        return int(self.confusion_matrix.sum())

    @property
    def accuracy(self) -> float:
        """Overall classification accuracy."""
        if self.total_samples == 0:
            return 0.0
        correct = np.trace(self.confusion_matrix)
        return float(correct / self.total_samples)

    @property
    def fault_detection_rate(self) -> float:
        """
        Proportion of faults correctly identified as *some* fault.

        This measures whether faults are detected at all, regardless
        of whether the specific fault type is correctly identified.
        """
        actual_faults = self.confusion_matrix[1:, :].sum()
        detected_as_fault = self.confusion_matrix[1:, 1:].sum()
        if actual_faults == 0:
            return 0.0
        return float(detected_as_fault / actual_faults)

    @property
    def false_alarm_rate(self) -> float:
        """Proportion of normal samples incorrectly flagged as faults."""
        actual_normal = self.confusion_matrix[0, :].sum()
        false_alarms = self.confusion_matrix[0, 1:].sum()
        if actual_normal == 0:
            return 0.0
        return float(false_alarms / actual_normal)

    @property
    def missed_detection_rate(self) -> float:
        """Proportion of faults missed (classified as normal)."""
        actual_faults = self.confusion_matrix[1:, :].sum()
        missed = self.confusion_matrix[1:, 0].sum()
        if actual_faults == 0:
            return 0.0
        return float(missed / actual_faults)

    # --- Per-Class Metrics ---

    def precision(self, fault_class: int) -> float:
        """
        Precision for a specific class.

        Precision = TP / (TP + FP)
        """
        if fault_class < 0 or fault_class >= self.n_classes:
            return 0.0
        predicted_as_class = self.confusion_matrix[:, fault_class].sum()
        if predicted_as_class == 0:
            return 0.0
        true_positives = self.confusion_matrix[fault_class, fault_class]
        return float(true_positives / predicted_as_class)

    def recall(self, fault_class: int) -> float:
        """
        Recall (sensitivity) for a specific class.

        Recall = TP / (TP + FN)
        """
        if fault_class < 0 or fault_class >= self.n_classes:
            return 0.0
        actual_class = self.confusion_matrix[fault_class, :].sum()
        if actual_class == 0:
            return 0.0
        true_positives = self.confusion_matrix[fault_class, fault_class]
        return float(true_positives / actual_class)

    def f1_score(self, fault_class: int) -> float:
        """F1 score for a specific class."""
        p = self.precision(fault_class)
        r = self.recall(fault_class)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def support(self, fault_class: int) -> int:
        """Number of actual samples for this class."""
        if fault_class < 0 or fault_class >= self.n_classes:
            return 0
        return int(self.confusion_matrix[fault_class, :].sum())

    # --- Aggregate Class Metrics ---

    def macro_precision(self) -> float:
        """Macro-averaged precision across all classes with support > 0."""
        precisions = []
        for i in range(self.n_classes):
            if self.support(i) > 0:
                precisions.append(self.precision(i))
        return float(np.mean(precisions)) if precisions else 0.0

    def macro_recall(self) -> float:
        """Macro-averaged recall across all classes with support > 0."""
        recalls = []
        for i in range(self.n_classes):
            if self.support(i) > 0:
                recalls.append(self.recall(i))
        return float(np.mean(recalls)) if recalls else 0.0

    def macro_f1(self) -> float:
        """Macro-averaged F1 score across all classes with support > 0."""
        f1s = []
        for i in range(self.n_classes):
            if self.support(i) > 0:
                f1s.append(self.f1_score(i))
        return float(np.mean(f1s)) if f1s else 0.0

    def weighted_f1(self) -> float:
        """F1 score weighted by class support."""
        total = self.total_samples
        if total == 0:
            return 0.0
        weighted = sum(
            self.f1_score(i) * self.support(i)
            for i in range(self.n_classes)
        )
        return float(weighted / total)

    # --- Detection Delay Metrics ---

    def mean_detection_delay(self, fault_class: int = None) -> Optional[float]:
        """
        Mean steps between fault onset and correct detection.

        Args:
            fault_class: Specific fault class, or None for all faults

        Returns:
            Mean delay in steps, or None if no delays recorded
        """
        if fault_class is not None:
            delays = self.detection_delays.get(fault_class, [])
            if not delays:
                return None
            return float(np.mean(delays))

        all_delays = []
        for delays in self.detection_delays.values():
            all_delays.extend(delays)
        return float(np.mean(all_delays)) if all_delays else None

    def min_detection_delay(self, fault_class: int = None) -> Optional[int]:
        """Minimum detection delay (first correct detection)."""
        if fault_class is not None:
            delays = self.detection_delays.get(fault_class, [])
            return min(delays) if delays else None

        all_delays = []
        for delays in self.detection_delays.values():
            all_delays.extend(delays)
        return min(all_delays) if all_delays else None

    # --- Reporting ---

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics as a dictionary."""
        return {
            "total_samples": self.total_samples,
            "unknown_count": self.unknown_count,
            "accuracy": self.accuracy,
            "fault_detection_rate": self.fault_detection_rate,
            "false_alarm_rate": self.false_alarm_rate,
            "missed_detection_rate": self.missed_detection_rate,
            "macro_precision": self.macro_precision(),
            "macro_recall": self.macro_recall(),
            "macro_f1": self.macro_f1(),
            "weighted_f1": self.weighted_f1(),
            "mean_detection_delay": self.mean_detection_delay(),
        }

    def per_class_report(self) -> List[Dict[str, Any]]:
        """Get per-class metrics as a list of dictionaries."""
        report = []
        for i in range(self.n_classes):
            sup = self.support(i)
            pred_count = int(self.confusion_matrix[:, i].sum())
            if sup > 0 or pred_count > 0:
                report.append({
                    "class": i,
                    "name": "Normal" if i == 0 else f"IDV({i})",
                    "precision": self.precision(i),
                    "recall": self.recall(i),
                    "f1": self.f1_score(i),
                    "support": sup,
                    "predictions": pred_count,
                    "mean_delay": self.mean_detection_delay(i),
                    "min_delay": self.min_detection_delay(i),
                })
        return report

    def __str__(self) -> str:
        """Human-readable summary."""
        s = self.summary()
        lines = [
            f"DetectionMetrics ({s['total_samples']} samples, {s['unknown_count']} unknown)",
            f"  Accuracy:             {s['accuracy']:.3f}",
            f"  Fault Detection Rate: {s['fault_detection_rate']:.3f}",
            f"  False Alarm Rate:     {s['false_alarm_rate']:.3f}",
            f"  Missed Detection:     {s['missed_detection_rate']:.3f}",
            f"  Macro Precision:      {s['macro_precision']:.3f}",
            f"  Macro Recall:         {s['macro_recall']:.3f}",
            f"  Macro F1:             {s['macro_f1']:.3f}",
            f"  Weighted F1:          {s['weighted_f1']:.3f}",
        ]
        if s['mean_detection_delay'] is not None:
            lines.append(f"  Mean Detection Delay: {s['mean_detection_delay']:.1f} steps")
        return "\n".join(lines)


# =============================================================================
# Base Fault Detector
# =============================================================================

class BaseFaultDetector(ABC):
    """
    Abstract base class for real-time fault detectors.

    Subclasses implement the detect() method with their detection logic.
    The framework handles window management, sampling, async execution,
    and metrics tracking automatically.

    Class Attributes (override in subclasses):
        name: Unique identifier for the detector
        description: Human-readable description
        version: Version string
        window_size: Number of points to keep in the sliding window
        window_sample_interval: Store every Nth measurement point
        detect_interval: Run detection every N steps
        async_mode: If True, run detection in a background thread

    Example:
        >>> class MyDetector(BaseFaultDetector):
        ...     name = "my_detector"
        ...     description = "Example detector"
        ...     window_size = 100
        ...     detect_interval = 10
        ...
        ...     def detect(self, xmeas, step):
        ...         if not self.window_ready:
        ...             return DetectionResult(-1, 0.0, step)
        ...
        ...         # Analyze self.window (100 x 41 array)
        ...         mean_pressure = self.window[:, 6].mean()
        ...         if mean_pressure > 2800:
        ...             return DetectionResult(4, 0.8, step)
        ...         return DetectionResult(0, 0.9, step)
        ...
        ...     def _reset_impl(self):
        ...         pass  # Reset any custom state
    """

    # --- Class attributes (override in subclasses) ---
    name: str = "base"
    description: str = "Base fault detector"
    version: str = "1.0.0"

    # Window configuration
    window_size: int = 100
    window_sample_interval: int = 1

    # Detection frequency
    detect_interval: int = 1

    # Async mode
    async_mode: bool = False

    def __init__(self, **kwargs):
        """
        Initialize detector.

        Args:
            **kwargs: Override any class attribute (window_size, detect_interval, etc.)
        """
        # Apply parameter overrides
        for key, value in kwargs.items():
            if hasattr(self.__class__, key) or hasattr(self, key):
                setattr(self, key, value)

        # Window buffer
        self._buffer: List[np.ndarray] = []
        self._buffer_steps: List[int] = []
        self._buffer_lock = threading.Lock()

        # Latest result for async mode
        self._latest_result = DetectionResult(-1, 0.0, step=0)

        # Async execution
        self._executor: Optional[ThreadPoolExecutor] = None
        self._pending: Optional[Future] = None
        if self.async_mode:
            self._executor = ThreadPoolExecutor(max_workers=1)

        # Metrics tracking
        self._metrics = DetectionMetrics()
        self._ground_truth: Optional[int] = None
        self._fault_onset_step: Optional[int] = None

        # Result callback (optional)
        self._result_callback: Optional[Callable[[DetectionResult], None]] = None

    # --- Window Management ---

    def _accumulate(self, xmeas: np.ndarray, step: int):
        """
        Add measurement to window buffer.

        Called by process() every step. Respects window_sample_interval.
        """
        if step % self.window_sample_interval != 0:
            return

        with self._buffer_lock:
            self._buffer.append(xmeas.copy())
            self._buffer_steps.append(step)

            while len(self._buffer) > self.window_size:
                self._buffer.pop(0)
                self._buffer_steps.pop(0)

    @property
    def window(self) -> Optional[np.ndarray]:
        """
        Get current window as (window_size, NUM_MEASUREMENTS) array.

        Returns:
            NumPy array of shape (window_size, 41), or None if not ready
        """
        with self._buffer_lock:
            if len(self._buffer) < self.window_size:
                return None
            return np.array(self._buffer)

    @property
    def window_ready(self) -> bool:
        """True if window has accumulated enough data."""
        return len(self._buffer) >= self.window_size

    @property
    def window_fill(self) -> float:
        """Fraction of window filled (0.0 to 1.0)."""
        return min(1.0, len(self._buffer) / self.window_size)

    @property
    def window_steps(self) -> Optional[np.ndarray]:
        """Step numbers corresponding to window rows."""
        with self._buffer_lock:
            if len(self._buffer) < self.window_size:
                return None
            return np.array(self._buffer_steps)

    @property
    def window_span_seconds(self) -> int:
        """Time span covered by a full window in seconds."""
        return self.window_size * self.window_sample_interval

    # --- Detection Execution ---

    def process(self, xmeas: np.ndarray, step: int) -> DetectionResult:
        """
        Process a measurement point.

        This is called by the simulator every step. It handles:
        - Accumulating measurements into the window
        - Checking if detection should run this step
        - Dispatching to sync or async detection
        - Recording metrics

        Args:
            xmeas: Current measurement vector (41 elements)
            step: Current simulation step

        Returns:
            DetectionResult (may be from previous detection if async)
        """
        self._accumulate(xmeas, step)

        # Check if we should run detection this step
        if step % self.detect_interval != 0:
            return self._latest_result

        if self.async_mode:
            return self._process_async(xmeas, step)
        else:
            return self._process_sync(xmeas, step)

    def _process_sync(self, xmeas: np.ndarray, step: int) -> DetectionResult:
        """Synchronous detection."""
        result = self.detect(xmeas, step)
        self._latest_result = result
        self._record_metrics(result, step)

        if self._result_callback:
            self._result_callback(result)

        return result

    def _process_async(self, xmeas: np.ndarray, step: int) -> DetectionResult:
        """Asynchronous detection with non-blocking semantics."""
        # Check for completed work
        if self._pending is not None and self._pending.done():
            try:
                result = self._pending.result()
                self._latest_result = result
                self._record_metrics(result, result.step)

                if self._result_callback:
                    self._result_callback(result)
            except Exception:
                pass  # Keep last good result
            self._pending = None

        # Submit new work if idle and ready
        if self._pending is None and self.window_ready:
            with self._buffer_lock:
                window_copy = np.array(self._buffer)
            xmeas_copy = xmeas.copy()

            self._pending = self._executor.submit(
                self._async_detect, xmeas_copy, step, window_copy
            )

        # Return latest result with latency info
        return DetectionResult(
            fault_class=self._latest_result.fault_class,
            confidence=self._latest_result.confidence,
            step=self._latest_result.step,
            latency_steps=step - self._latest_result.step,
            alternatives=self._latest_result.alternatives,
            contributing_sensors=self._latest_result.contributing_sensors,
            statistics=self._latest_result.statistics,
        )

    def _async_detect(self, xmeas: np.ndarray, step: int,
                      window: np.ndarray) -> DetectionResult:
        """Run detection in thread with provided window snapshot."""
        old_buffer = None
        try:
            with self._buffer_lock:
                old_buffer = self._buffer
                self._buffer = list(window)
            return self.detect(xmeas, step)
        finally:
            if old_buffer is not None:
                with self._buffer_lock:
                    self._buffer = old_buffer

    # --- Metrics ---

    def set_ground_truth(self, fault_class: int, onset_step: int = None):
        """
        Set current ground truth for metrics tracking.

        Call this when the true fault state changes. The detector will
        use this to compute accuracy, detection delays, etc.

        Args:
            fault_class: True fault class (0=normal, 1-20=fault IDV index)
            onset_step: Step when fault started (for delay tracking)
        """
        self._ground_truth = fault_class
        if onset_step is not None:
            self._fault_onset_step = onset_step
        elif fault_class > 0 and self._fault_onset_step is None:
            # Auto-set onset if transitioning to fault
            self._fault_onset_step = None  # Will be set by simulator

    def _record_metrics(self, result: DetectionResult, step: int):
        """Record prediction in metrics if ground truth is set."""
        if self._ground_truth is not None:
            self._metrics.update(
                actual=self._ground_truth,
                predicted=result.fault_class,
                step=step,
                fault_onset_step=self._fault_onset_step
            )

    @property
    def metrics(self) -> DetectionMetrics:
        """Get accumulated performance metrics."""
        return self._metrics

    def reset_metrics(self):
        """Reset metrics without resetting detector state."""
        self._metrics.reset()

    # --- Callbacks ---

    def set_result_callback(self, callback: Callable[[DetectionResult], None]):
        """
        Set a callback to be invoked on each detection result.

        Args:
            callback: Function called with DetectionResult after each detection
        """
        self._result_callback = callback

    # --- Lifecycle ---

    def reset(self):
        """
        Reset detector state for a new simulation.

        Clears the window buffer and resets to initial state.
        Metrics are preserved (use reset_metrics() to clear those).
        """
        with self._buffer_lock:
            self._buffer.clear()
            self._buffer_steps.clear()

        self._latest_result = DetectionResult(-1, 0.0, step=0)
        self._ground_truth = None
        self._fault_onset_step = None

        # Cancel pending async work
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None

        # Call subclass reset
        self._reset_impl()

    def _reset_impl(self):
        """
        Override for subclass-specific reset logic.

        Called by reset() after clearing the window buffer.
        Use this to reset any custom state (e.g., running statistics).
        """
        pass

    def shutdown(self):
        """Clean up resources (thread pool, etc.)."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.shutdown()

    # --- Info ---

    def get_info(self) -> Dict[str, Any]:
        """Get detector configuration and info."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "window_size": self.window_size,
            "window_sample_interval": self.window_sample_interval,
            "detect_interval": self.detect_interval,
            "async_mode": self.async_mode,
            "window_span_seconds": self.window_span_seconds,
        }

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get tunable parameters.

        Override to expose detector-specific parameters.
        """
        return {}

    def set_parameter(self, name: str, value: Any):
        """
        Set a tunable parameter.

        Args:
            name: Parameter name
            value: New value
        """
        if hasattr(self, name):
            setattr(self, name, value)
        else:
            raise AttributeError(f"Unknown parameter: {name}")

    # --- Abstract Method ---

    @abstractmethod
    def detect(self, xmeas: np.ndarray, step: int) -> DetectionResult:
        """
        Perform fault detection on current measurement.

        Implement your detection logic here. You have access to:
        - xmeas: Current measurement vector (41 elements)
        - self.window: Historical measurements (window_size x 41), or None
        - self.window_ready: Whether window is full
        - self.window_steps: Step numbers for window rows

        Args:
            xmeas: Current measurement vector (41 elements)
            step: Current simulation step (1 step = 1 second)

        Returns:
            DetectionResult with:
            - fault_class: -1 (unknown), 0 (normal), or 1-20 (fault)
            - confidence: 0.0 to 1.0
            - step: The step number
            - Optional: alternatives, contributing_sensors, statistics
        """
        pass


# =============================================================================
# Detector Registry
# =============================================================================

@dataclass
class DetectorConfig:
    """Configuration for a registered detector."""
    detector_class: Type[BaseFaultDetector]
    name: str
    description: str
    default_params: Dict[str, Any] = field(default_factory=dict)


class FaultDetectorRegistry:
    """
    Registry for fault detector plugins.

    Provides discovery, registration, and instantiation of detectors.
    Similar to ControllerRegistry but for fault detection.

    Example:
        >>> # Register a detector
        >>> FaultDetectorRegistry.register(MyDetector)
        >>>
        >>> # List available detectors
        >>> print(FaultDetectorRegistry.list_available())
        ['threshold', 'pca', 'my_detector']
        >>>
        >>> # Create an instance
        >>> detector = FaultDetectorRegistry.create('pca', window_size=200)
    """

    _detectors: Dict[str, DetectorConfig] = {}

    @classmethod
    def register(cls, detector_class: Type[BaseFaultDetector],
                 name: str = None, description: str = None,
                 default_params: Dict[str, Any] = None):
        """
        Register a detector class.

        Args:
            detector_class: Detector class (must inherit from BaseFaultDetector)
            name: Optional name override (defaults to class.name)
            description: Optional description override
            default_params: Default parameters for instantiation
        """
        if not issubclass(detector_class, BaseFaultDetector):
            raise TypeError(
                f"Detector must inherit from BaseFaultDetector, "
                f"got {detector_class.__bases__}"
            )

        reg_name = name or detector_class.name
        reg_desc = description or detector_class.description

        cls._detectors[reg_name] = DetectorConfig(
            detector_class=detector_class,
            name=reg_name,
            description=reg_desc,
            default_params=default_params or {}
        )

    @classmethod
    def unregister(cls, name: str):
        """Remove a detector from the registry."""
        if name in cls._detectors:
            del cls._detectors[name]

    @classmethod
    def get(cls, name: str) -> Type[BaseFaultDetector]:
        """
        Get a detector class by name.

        Args:
            name: Detector name

        Returns:
            Detector class

        Raises:
            KeyError: If detector not found
        """
        if name not in cls._detectors:
            available = ", ".join(cls._detectors.keys())
            raise KeyError(f"Detector '{name}' not found. Available: {available}")
        return cls._detectors[name].detector_class

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseFaultDetector:
        """
        Create a detector instance.

        Args:
            name: Detector name
            **kwargs: Parameters passed to detector constructor

        Returns:
            Detector instance
        """
        config = cls._detectors.get(name)
        if config is None:
            available = ", ".join(cls._detectors.keys())
            raise KeyError(f"Detector '{name}' not found. Available: {available}")

        # Merge default params with provided kwargs
        params = {**config.default_params, **kwargs}
        return config.detector_class(**params)

    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered detector names."""
        return list(cls._detectors.keys())

    @classmethod
    def get_info(cls, name: str) -> Dict[str, Any]:
        """Get information about a registered detector."""
        if name not in cls._detectors:
            raise KeyError(f"Detector '{name}' not found")

        config = cls._detectors[name]
        return {
            "name": config.name,
            "description": config.description,
            "class": config.detector_class.__name__,
            "default_params": config.default_params,
        }

    @classmethod
    def list_all_info(cls) -> List[Dict[str, Any]]:
        """Get information about all registered detectors."""
        return [cls.get_info(name) for name in cls._detectors]

    @classmethod
    def clear(cls):
        """Clear all registered detectors (mainly for testing)."""
        cls._detectors.clear()


def register_detector(name: str = None, description: str = None,
                      default_params: Dict[str, Any] = None):
    """
    Decorator to register a detector class.

    Example:
        >>> @register_detector(name="my_detector", description="My custom detector")
        ... class MyDetector(BaseFaultDetector):
        ...     pass
    """
    def decorator(cls: Type[BaseFaultDetector]) -> Type[BaseFaultDetector]:
        FaultDetectorRegistry.register(
            cls,
            name=name,
            description=description,
            default_params=default_params
        )
        return cls
    return decorator
