"""
Built-in fault detector implementations for the Tennessee Eastman Process.

This module provides several ready-to-use fault detectors ranging from
simple threshold checks to statistical methods. These serve as both
practical tools and examples for implementing custom detectors.

Available Detectors:
    - ThresholdDetector: Fast safety limit checking
    - EWMADetector: Exponentially weighted moving average change detection
    - CUSUMDetector: Cumulative sum control chart for drift detection
    - PCADetector: Principal Component Analysis with T² and SPE statistics
    - StatisticalDetector: Multi-statistic ensemble detector
    - SlidingWindowDetector: Simple sliding window comparison

Example:
    >>> from tep.detector_base import FaultDetectorRegistry
    >>>
    >>> # Create a detector
    >>> detector = FaultDetectorRegistry.create("threshold")
    >>>
    >>> # Or with custom parameters
    >>> detector = FaultDetectorRegistry.create("pca", window_size=300)
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from .detector_base import (
    BaseFaultDetector,
    DetectionResult,
    FaultDetectorRegistry,
    register_detector,
)
from .constants import SAFETY_LIMITS, NUM_MEASUREMENTS


# =============================================================================
# Threshold Detector
# =============================================================================

@register_detector(
    name="threshold",
    description="Fast threshold-based detection using process safety limits"
)
class ThresholdDetector(BaseFaultDetector):
    """
    Simple threshold detector based on process safety limits or statistical bounds.

    Checks measurements against known safe operating ranges and flags
    violations. Very fast (no window needed) and useful as a first
    line of defense.

    Can operate in two modes:
    1. Safety limits mode (default): Uses fixed process safety limits
    2. Statistical mode: Uses mean ± n_sigma * std for each variable

    Attributes:
        limits: Dict mapping XMEAS index to (low, high) thresholds
        fault_mapping: Dict mapping violated sensor to likely fault class
        mean: Optional array of means for statistical mode
        std: Optional array of standard deviations for statistical mode
        n_sigma: Number of standard deviations for threshold (default 3.0)
    """

    name = "threshold"
    description = "Threshold-based safety limit detector"
    window_size = 1
    detect_interval = 1
    async_mode = False

    def __init__(self, mean: np.ndarray = None, std: np.ndarray = None,
                 n_sigma: float = 3.0, monitored_indices: List[int] = None, **kwargs):
        """
        Initialize threshold detector.

        Args:
            mean: Array of baseline means for each variable. If provided with std,
                  uses statistical mode instead of safety limits.
            std: Array of baseline standard deviations for each variable.
            n_sigma: Number of standard deviations for threshold bounds (default 3.0).
            monitored_indices: List of XMEAS indices to monitor (0-based).
                              If None, monitors all variables in statistical mode
                              or default safety-critical variables in limits mode.
        """
        super().__init__(**kwargs)

        self.n_sigma = n_sigma
        self._mean = mean
        self._std = std
        self._monitored_indices = monitored_indices
        self._statistical_mode = mean is not None and std is not None

        if self._statistical_mode:
            # Statistical mode: use mean ± n_sigma * std
            self._setup_statistical_limits()
        else:
            # Default thresholds from process safety limits
            # Index is 0-based XMEAS index
            self.limits: Dict[int, Tuple[Optional[float], Optional[float]]] = {
                6: (None, 2900.0),      # Reactor pressure max (kPa)
                7: (3.0, 22.0),         # Reactor level (%)
                8: (None, 170.0),       # Reactor temperature max (deg C)
                11: (2.0, 11.0),        # Separator level (%)
                14: (2.0, 7.0),         # Stripper level (%)
            }

        # Map sensor violations to likely fault classes
        self.fault_mapping: Dict[int, int] = {
            6: 4,   # Pressure issues -> IDV(4) cooling water
            7: 1,   # Reactor level -> IDV(1) feed ratio
            8: 4,   # Temperature -> IDV(4) cooling water
            11: 7,  # Separator level -> IDV(7) header pressure
            14: 6,  # Stripper level -> IDV(6) A feed loss
        }

    def _setup_statistical_limits(self):
        """Set up limits from mean and std arrays."""
        self.limits = {}

        if self._monitored_indices is not None:
            indices = self._monitored_indices
        else:
            # Monitor all variables
            indices = range(len(self._mean))

        for idx in indices:
            if idx < len(self._mean) and idx < len(self._std):
                low = self._mean[idx] - self.n_sigma * self._std[idx]
                high = self._mean[idx] + self.n_sigma * self._std[idx]
                self.limits[idx] = (low, high)

    def set_baseline(self, mean: np.ndarray, std: np.ndarray,
                     monitored_indices: List[int] = None):
        """
        Set baseline statistics for statistical threshold detection.

        Args:
            mean: Array of baseline means for each variable.
            std: Array of baseline standard deviations for each variable.
            monitored_indices: List of XMEAS indices to monitor (0-based).
                              If None, monitors all variables.
        """
        self._mean = mean
        self._std = std
        self._monitored_indices = monitored_indices
        self._statistical_mode = True
        self._setup_statistical_limits()

    def detect(self, xmeas: np.ndarray, step: int) -> DetectionResult:
        """Check measurements against thresholds."""
        violations = []
        max_severity = 0.0

        for idx, (low, high) in self.limits.items():
            val = xmeas[idx]
            severity = 0.0

            if low is not None and val < low:
                severity = (low - val) / max(abs(low), 1.0)
                violations.append((idx, severity))
            elif high is not None and val > high:
                severity = (val - high) / max(abs(high), 1.0)
                violations.append((idx, severity))

            max_severity = max(max_severity, severity)

        if violations:
            # Sort by severity
            violations.sort(key=lambda x: -x[1])
            top_sensor = violations[0][0]
            fault_class = self.fault_mapping.get(top_sensor, 1)

            # Confidence based on severity
            confidence = min(0.5 + max_severity * 0.5, 0.99)

            return DetectionResult(
                fault_class=fault_class,
                confidence=confidence,
                step=step,
                contributing_sensors=[v[0] for v in violations[:3]],
                statistics={"max_severity": max_severity}
            )

        return DetectionResult(
            fault_class=0,
            confidence=0.9,
            step=step
        )

    def get_parameters(self) -> Dict[str, Any]:
        params = {
            "limits": self.limits,
            "fault_mapping": self.fault_mapping,
            "n_sigma": self.n_sigma,
            "statistical_mode": self._statistical_mode,
        }
        if self._statistical_mode:
            params["monitored_indices"] = self._monitored_indices
        return params


# =============================================================================
# EWMA Detector
# =============================================================================

@register_detector(
    name="ewma",
    description="Exponentially Weighted Moving Average change detector"
)
class EWMADetector(BaseFaultDetector):
    """
    EWMA-based change detection.

    Tracks exponentially weighted moving averages and variances of
    all measurements. Detects anomalies when current values deviate
    significantly from the running statistics.

    Good for detecting gradual changes and persistent deviations.

    Attributes:
        alpha: Smoothing factor (0-1). Higher = faster response.
        threshold: Z-score threshold for anomaly detection.
        warmup_steps: Steps before detection begins.
    """

    name = "ewma"
    description = "EWMA-based change detection"
    window_size = 1  # Stateful, doesn't need window buffer
    detect_interval = 1
    async_mode = False

    def __init__(self, alpha: float = 0.1, threshold: float = 3.0,
                 warmup_steps: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.threshold = threshold
        self.warmup_steps = warmup_steps

        self._ewma: Optional[np.ndarray] = None
        self._ewma_var: Optional[np.ndarray] = None
        self._step_count = 0

    def detect(self, xmeas: np.ndarray, step: int) -> DetectionResult:
        """Detect using EWMA statistics."""
        self._step_count += 1

        # Initialize on first call
        if self._ewma is None:
            self._ewma = xmeas.copy()
            self._ewma_var = np.ones(NUM_MEASUREMENTS) * 0.01
            return DetectionResult(-1, 0.0, step)

        # Compute deviation before updating
        diff = xmeas - self._ewma
        std = np.sqrt(self._ewma_var + 1e-8)
        z_scores = np.abs(diff) / std

        # Update EWMA statistics
        self._ewma = self.alpha * xmeas + (1 - self.alpha) * self._ewma
        self._ewma_var = self.alpha * diff**2 + (1 - self.alpha) * self._ewma_var

        # Don't flag during warmup
        if self._step_count < self.warmup_steps:
            return DetectionResult(-1, 0.0, step)

        # Find anomalies
        max_z = float(np.max(z_scores))
        top_sensors = np.argsort(z_scores)[-5:][::-1].tolist()

        if max_z > self.threshold:
            # Confidence scales with how far above threshold
            confidence = min(0.5 + (max_z - self.threshold) * 0.1, 0.99)

            return DetectionResult(
                fault_class=1,  # Generic fault indication
                confidence=confidence,
                step=step,
                contributing_sensors=top_sensors[:3],
                statistics={
                    "max_z_score": max_z,
                    "mean_z_score": float(np.mean(z_scores)),
                }
            )

        return DetectionResult(
            fault_class=0,
            confidence=min(0.5 + (self.threshold - max_z) * 0.1, 0.95),
            step=step,
            statistics={"max_z_score": max_z}
        )

    def _reset_impl(self):
        """Reset EWMA state."""
        self._ewma = None
        self._ewma_var = None
        self._step_count = 0

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "threshold": self.threshold,
            "warmup_steps": self.warmup_steps,
        }


# =============================================================================
# CUSUM Detector
# =============================================================================

@register_detector(
    name="cusum",
    description="Cumulative Sum control chart for drift detection"
)
class CUSUMDetector(BaseFaultDetector):
    """
    CUSUM (Cumulative Sum) control chart detector.

    Accumulates deviations from expected values to detect persistent
    shifts. Very effective for detecting slow drifts that might be
    missed by instantaneous checks.

    Attributes:
        k: Slack parameter (allowance). Typically 0.5 * expected shift.
        h: Decision threshold. Higher = fewer false alarms.
        target: Target values (learned from initial data or provided).
    """

    name = "cusum"
    description = "CUSUM control chart for drift detection"
    window_size = 100  # For learning baseline
    detect_interval = 1
    async_mode = False

    def __init__(self, k: float = 0.5, h: float = 5.0, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.h = h

        self._target: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._cusum_pos: Optional[np.ndarray] = None
        self._cusum_neg: Optional[np.ndarray] = None
        self._baseline_learned = False

    def detect(self, xmeas: np.ndarray, step: int) -> DetectionResult:
        """Detect using CUSUM statistics."""
        # Learn baseline from window
        if not self._baseline_learned:
            if not self.window_ready:
                return DetectionResult(-1, 0.0, step)

            window = self.window
            self._target = np.mean(window, axis=0)
            self._std = np.std(window, axis=0) + 1e-8
            self._cusum_pos = np.zeros(NUM_MEASUREMENTS)
            self._cusum_neg = np.zeros(NUM_MEASUREMENTS)
            self._baseline_learned = True

        # Normalized deviation
        z = (xmeas - self._target) / self._std

        # Update CUSUM
        self._cusum_pos = np.maximum(0, self._cusum_pos + z - self.k)
        self._cusum_neg = np.maximum(0, self._cusum_neg - z - self.k)

        # Check for violations
        max_pos = float(np.max(self._cusum_pos))
        max_neg = float(np.max(self._cusum_neg))
        max_cusum = max(max_pos, max_neg)

        top_pos = np.argsort(self._cusum_pos)[-3:][::-1].tolist()
        top_neg = np.argsort(self._cusum_neg)[-3:][::-1].tolist()

        if max_cusum > self.h:
            confidence = min(0.5 + (max_cusum - self.h) / self.h * 0.3, 0.99)

            # Combine top contributors
            contributing = list(set(top_pos + top_neg))[:5]

            return DetectionResult(
                fault_class=1,  # Generic drift detected
                confidence=confidence,
                step=step,
                contributing_sensors=contributing,
                statistics={
                    "max_cusum_pos": max_pos,
                    "max_cusum_neg": max_neg,
                }
            )

        return DetectionResult(
            fault_class=0,
            confidence=0.9,
            step=step,
            statistics={
                "max_cusum_pos": max_pos,
                "max_cusum_neg": max_neg,
            }
        )

    def _reset_impl(self):
        """Reset CUSUM state."""
        self._target = None
        self._std = None
        self._cusum_pos = None
        self._cusum_neg = None
        self._baseline_learned = False

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "h": self.h,
        }


# =============================================================================
# PCA Detector
# =============================================================================

@register_detector(
    name="pca",
    description="PCA-based detection with T² and SPE statistics"
)
class PCADetector(BaseFaultDetector):
    """
    Principal Component Analysis detector.

    Projects measurements onto principal components learned from normal
    operation. Uses two statistics for detection:
    - T² (Hotelling's): Variation within the principal component space
    - SPE (Q): Residual variation not captured by the model

    This is a classic multivariate statistical process monitoring technique.

    Attributes:
        n_components: Number of principal components to retain
        t2_threshold: Threshold for T² statistic
        spe_threshold: Threshold for SPE statistic
        auto_train: If True, train on first full window
    """

    name = "pca"
    description = "PCA-based T² and SPE fault detection"
    window_size = 200
    window_sample_interval = 1
    detect_interval = 10
    async_mode = False

    def __init__(self, n_components: int = 10, t2_threshold: float = 15.0,
                 spe_threshold: float = 25.0, auto_train: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.t2_threshold = t2_threshold
        self.spe_threshold = spe_threshold
        self.auto_train = auto_train

        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._components: Optional[np.ndarray] = None
        self._eigenvalues: Optional[np.ndarray] = None
        self._trained = False

    def train(self, data: np.ndarray):
        """
        Train PCA model on normal operation data.

        Args:
            data: Training data of shape (n_samples, 41)
        """
        self._mean = np.mean(data, axis=0)
        self._std = np.std(data, axis=0)
        self._std[self._std < 1e-8] = 1.0  # Avoid division by zero

        # Normalize
        normalized = (data - self._mean) / self._std

        # SVD for PCA
        n_samples = len(data)
        U, S, Vt = np.linalg.svd(normalized, full_matrices=False)

        # Keep top components
        n_comp = min(self.n_components, len(S))
        self._components = Vt[:n_comp]
        self._eigenvalues = (S[:n_comp]**2) / (n_samples - 1)

        self._trained = True

    def detect(self, xmeas: np.ndarray, step: int) -> DetectionResult:
        """Detect using PCA statistics."""
        # Auto-train if needed
        if not self._trained:
            if self.auto_train and self.window_ready:
                self.train(self.window)
            else:
                return DetectionResult(-1, 0.0, step)

        # Normalize current measurement
        x = (xmeas - self._mean) / self._std

        # Project onto principal components
        scores = x @ self._components.T

        # T² statistic (Hotelling's T-squared)
        t2 = float(np.sum(scores**2 / self._eigenvalues))

        # SPE/Q statistic (squared prediction error)
        reconstruction = scores @ self._components
        residual = x - reconstruction
        spe = float(np.sum(residual**2))

        # Contribution analysis
        contributions = np.abs(residual)
        top_sensors = np.argsort(contributions)[-5:][::-1].tolist()

        # Check thresholds
        t2_violation = t2 > self.t2_threshold
        spe_violation = spe > self.spe_threshold

        if t2_violation or spe_violation:
            # Compute confidence based on how much thresholds are exceeded
            t2_ratio = t2 / self.t2_threshold if self.t2_threshold > 0 else 0
            spe_ratio = spe / self.spe_threshold if self.spe_threshold > 0 else 0
            max_ratio = max(t2_ratio, spe_ratio)

            confidence = min(0.5 + (max_ratio - 1) * 0.2, 0.99)

            return DetectionResult(
                fault_class=1,  # Anomaly detected
                confidence=confidence,
                step=step,
                contributing_sensors=top_sensors[:3],
                statistics={
                    "T2": t2,
                    "SPE": spe,
                    "T2_threshold": self.t2_threshold,
                    "SPE_threshold": self.spe_threshold,
                }
            )

        return DetectionResult(
            fault_class=0,
            confidence=0.85,
            step=step,
            statistics={"T2": t2, "SPE": spe}
        )

    def _reset_impl(self):
        """Reset PCA state (keeps trained model)."""
        pass  # Model is retained across resets

    def reset_model(self):
        """Clear the trained PCA model."""
        self._mean = None
        self._std = None
        self._components = None
        self._eigenvalues = None
        self._trained = False

    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self._trained

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "n_components": self.n_components,
            "t2_threshold": self.t2_threshold,
            "spe_threshold": self.spe_threshold,
            "auto_train": self.auto_train,
            "is_trained": self._trained,
        }


# =============================================================================
# Statistical Detector
# =============================================================================

@register_detector(
    name="statistical",
    description="Multi-statistic ensemble detector"
)
class StatisticalDetector(BaseFaultDetector):
    """
    Ensemble detector combining multiple statistical tests.

    Computes various statistics on the measurement window and flags
    anomalies when multiple tests indicate problems. More robust than
    single-statistic methods.

    Tests performed:
    - Mean shift detection
    - Variance change detection
    - Correlation change detection
    - Range (max-min) analysis
    """

    name = "statistical"
    description = "Multi-statistic ensemble detector"
    window_size = 120
    window_sample_interval = 1
    detect_interval = 30
    async_mode = False

    def __init__(self, mean_threshold: float = 2.5, var_threshold: float = 2.0,
                 votes_required: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.mean_threshold = mean_threshold
        self.var_threshold = var_threshold
        self.votes_required = votes_required

        self._baseline_mean: Optional[np.ndarray] = None
        self._baseline_std: Optional[np.ndarray] = None
        self._baseline_var: Optional[np.ndarray] = None
        self._baseline_learned = False

    def detect(self, xmeas: np.ndarray, step: int) -> DetectionResult:
        """Detect using multiple statistical tests."""
        if not self.window_ready:
            return DetectionResult(-1, 0.0, step)

        window = self.window

        # Learn baseline from first window
        if not self._baseline_learned:
            self._baseline_mean = np.mean(window, axis=0)
            self._baseline_std = np.std(window, axis=0) + 1e-8
            self._baseline_var = np.var(window, axis=0) + 1e-8
            self._baseline_learned = True
            return DetectionResult(-1, 0.0, step)

        # Current window statistics
        current_mean = np.mean(window, axis=0)
        current_var = np.var(window, axis=0) + 1e-8

        # Test 1: Mean shift
        mean_z = np.abs(current_mean - self._baseline_mean) / self._baseline_std
        mean_violation = np.any(mean_z > self.mean_threshold)
        mean_score = float(np.max(mean_z))

        # Test 2: Variance change (F-test approximation)
        var_ratio = np.maximum(current_var / self._baseline_var,
                               self._baseline_var / current_var)
        var_violation = np.any(var_ratio > self.var_threshold)
        var_score = float(np.max(var_ratio))

        # Test 3: Recent trend
        half = self.window_size // 2
        first_half_mean = np.mean(window[:half], axis=0)
        second_half_mean = np.mean(window[half:], axis=0)
        trend = np.abs(second_half_mean - first_half_mean) / self._baseline_std
        trend_violation = np.any(trend > self.mean_threshold)
        trend_score = float(np.max(trend))

        # Count votes
        votes = sum([mean_violation, var_violation, trend_violation])

        # Find contributing sensors
        all_scores = mean_z + var_ratio + trend
        top_sensors = np.argsort(all_scores)[-5:][::-1].tolist()

        if votes >= self.votes_required:
            confidence = min(0.4 + votes * 0.2, 0.95)

            return DetectionResult(
                fault_class=1,
                confidence=confidence,
                step=step,
                contributing_sensors=top_sensors[:3],
                statistics={
                    "mean_score": mean_score,
                    "var_score": var_score,
                    "trend_score": trend_score,
                    "votes": votes,
                }
            )

        return DetectionResult(
            fault_class=0,
            confidence=0.8,
            step=step,
            statistics={
                "mean_score": mean_score,
                "var_score": var_score,
                "trend_score": trend_score,
                "votes": votes,
            }
        )

    def _reset_impl(self):
        """Reset statistical baselines."""
        self._baseline_mean = None
        self._baseline_std = None
        self._baseline_var = None
        self._baseline_learned = False

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "mean_threshold": self.mean_threshold,
            "var_threshold": self.var_threshold,
            "votes_required": self.votes_required,
        }


# =============================================================================
# Sliding Window Detector
# =============================================================================

@register_detector(
    name="sliding_window",
    description="Simple sliding window comparison detector"
)
class SlidingWindowDetector(BaseFaultDetector):
    """
    Simple sliding window detector comparing recent vs older data.

    Splits the window into two halves and detects when the recent
    half differs significantly from the older half. Simple but
    effective for detecting step changes.
    """

    name = "sliding_window"
    description = "Sliding window comparison detector"
    window_size = 60
    detect_interval = 10
    async_mode = False

    def __init__(self, threshold: float = 3.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def detect(self, xmeas: np.ndarray, step: int) -> DetectionResult:
        """Detect by comparing window halves."""
        if not self.window_ready:
            return DetectionResult(-1, 0.0, step)

        window = self.window
        half = self.window_size // 2

        # Split window
        old_half = window[:half]
        new_half = window[half:]

        # Compare means
        old_mean = np.mean(old_half, axis=0)
        new_mean = np.mean(new_half, axis=0)
        old_std = np.std(old_half, axis=0) + 1e-8

        z_scores = np.abs(new_mean - old_mean) / old_std
        max_z = float(np.max(z_scores))
        top_sensors = np.argsort(z_scores)[-3:][::-1].tolist()

        if max_z > self.threshold:
            confidence = min(0.5 + (max_z - self.threshold) * 0.1, 0.95)

            return DetectionResult(
                fault_class=1,
                confidence=confidence,
                step=step,
                contributing_sensors=top_sensors,
                statistics={"max_z_score": max_z}
            )

        return DetectionResult(
            fault_class=0,
            confidence=0.85,
            step=step,
            statistics={"max_z_score": max_z}
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {"threshold": self.threshold}


# =============================================================================
# Composite Detector
# =============================================================================

@register_detector(
    name="composite",
    description="Combines multiple detectors with voting"
)
class CompositeDetector(BaseFaultDetector):
    """
    Combines multiple detectors using voting.

    Runs several detectors and aggregates their outputs. Can be
    configured to require unanimous agreement or majority voting.
    """

    name = "composite"
    description = "Multi-detector voting ensemble"
    window_size = 1  # Managed by sub-detectors
    detect_interval = 1
    async_mode = False

    def __init__(self, detectors: List[BaseFaultDetector] = None,
                 min_votes: int = 2, **kwargs):
        super().__init__(**kwargs)
        self._sub_detectors = detectors or []
        self.min_votes = min_votes

    def add_detector(self, detector: BaseFaultDetector):
        """Add a sub-detector."""
        self._sub_detectors.append(detector)

    def detect(self, xmeas: np.ndarray, step: int) -> DetectionResult:
        """Aggregate sub-detector results."""
        if not self._sub_detectors:
            return DetectionResult(-1, 0.0, step)

        results = []
        for detector in self._sub_detectors:
            result = detector.process(xmeas, step)
            if result.is_ready:
                results.append(result)

        if not results:
            return DetectionResult(-1, 0.0, step)

        # Count fault votes
        fault_votes = sum(1 for r in results if r.is_fault)
        total_votes = len(results)

        # Aggregate confidence
        if fault_votes >= self.min_votes:
            fault_results = [r for r in results if r.is_fault]
            avg_confidence = np.mean([r.confidence for r in fault_results])

            # Get most common fault class
            fault_classes = [r.fault_class for r in fault_results]
            most_common = max(set(fault_classes), key=fault_classes.count)

            # Aggregate contributing sensors
            all_sensors = []
            for r in fault_results:
                if r.contributing_sensors:
                    all_sensors.extend(r.contributing_sensors)

            return DetectionResult(
                fault_class=most_common,
                confidence=float(avg_confidence),
                step=step,
                contributing_sensors=list(set(all_sensors))[:5],
                statistics={
                    "fault_votes": fault_votes,
                    "total_votes": total_votes,
                }
            )

        # No fault consensus
        normal_results = [r for r in results if r.is_normal]
        if normal_results:
            avg_confidence = np.mean([r.confidence for r in normal_results])
        else:
            avg_confidence = 0.5

        return DetectionResult(
            fault_class=0,
            confidence=float(avg_confidence),
            step=step,
            statistics={
                "fault_votes": fault_votes,
                "total_votes": total_votes,
            }
        )

    def _reset_impl(self):
        """Reset all sub-detectors."""
        for detector in self._sub_detectors:
            detector.reset()

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "min_votes": self.min_votes,
            "n_detectors": len(self._sub_detectors),
            "detector_names": [d.name for d in self._sub_detectors],
        }


# =============================================================================
# Passthrough Detector (for testing/baseline)
# =============================================================================

@register_detector(
    name="passthrough",
    description="Always reports normal (baseline for comparison)"
)
class PassthroughDetector(BaseFaultDetector):
    """
    Baseline detector that always reports normal operation.

    Useful for:
    - Testing the detector framework
    - Establishing a false-alarm-free baseline
    - Placeholder when no detection is wanted
    """

    name = "passthrough"
    description = "Baseline detector (always normal)"
    window_size = 1
    detect_interval = 1
    async_mode = False

    def detect(self, xmeas: np.ndarray, step: int) -> DetectionResult:
        """Always return normal."""
        return DetectionResult(
            fault_class=0,
            confidence=1.0,
            step=step
        )
