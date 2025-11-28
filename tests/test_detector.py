"""
Tests for the fault detector plugin system.

This module tests:
1. DetectionResult data class
2. DetectionMetrics accumulation and computation
3. BaseFaultDetector abstract class
4. FaultDetectorRegistry
5. Built-in detector plugins
6. Integration with TEPSimulator
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from tep.detector_base import (
    BaseFaultDetector,
    FaultDetectorRegistry,
    DetectionResult,
    DetectionMetrics,
    register_detector,
)
from tep.detector_plugins import (
    ThresholdDetector,
    EWMADetector,
    CUSUMDetector,
    PCADetector,
    StatisticalDetector,
    SlidingWindowDetector,
    CompositeDetector,
    PassthroughDetector,
)
from tep.simulator import TEPSimulator, ControlMode
from tep.constants import NUM_MEASUREMENTS


# =============================================================================
# DETECTION RESULT TESTS
# =============================================================================

class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_basic_creation(self):
        """Can create a basic DetectionResult."""
        result = DetectionResult(fault_class=0, confidence=0.9, step=100)
        assert result.fault_class == 0
        assert result.confidence == 0.9
        assert result.step == 100

    def test_is_ready_unknown(self):
        """is_ready returns False for unknown (-1)."""
        result = DetectionResult(fault_class=-1, confidence=0.0, step=0)
        assert not result.is_ready

    def test_is_ready_known(self):
        """is_ready returns True for known classes."""
        assert DetectionResult(fault_class=0, confidence=0.9, step=0).is_ready
        assert DetectionResult(fault_class=5, confidence=0.8, step=0).is_ready

    def test_is_normal(self):
        """is_normal returns True only for class 0."""
        assert DetectionResult(fault_class=0, confidence=0.9, step=0).is_normal
        assert not DetectionResult(fault_class=1, confidence=0.9, step=0).is_normal
        assert not DetectionResult(fault_class=-1, confidence=0.0, step=0).is_normal

    def test_is_fault(self):
        """is_fault returns True for positive classes."""
        assert not DetectionResult(fault_class=0, confidence=0.9, step=0).is_fault
        assert not DetectionResult(fault_class=-1, confidence=0.0, step=0).is_fault
        assert DetectionResult(fault_class=1, confidence=0.9, step=0).is_fault
        assert DetectionResult(fault_class=20, confidence=0.9, step=0).is_fault

    def test_top_k(self):
        """top_k returns primary and alternatives."""
        result = DetectionResult(
            fault_class=4,
            confidence=0.8,
            step=100,
            alternatives=[(3, 0.6), (5, 0.4), (1, 0.2)]
        )
        top3 = result.top_k(3)
        assert len(top3) == 3
        assert top3[0] == (4, 0.8)
        assert top3[1] == (3, 0.6)
        assert top3[2] == (5, 0.4)

    def test_above_threshold(self):
        """above_threshold returns classes above confidence threshold."""
        result = DetectionResult(
            fault_class=4,
            confidence=0.8,
            step=100,
            alternatives=[(3, 0.6), (5, 0.4), (0, 0.9)]
        )
        above_05 = result.above_threshold(0.5)
        assert 4 in above_05
        assert 3 in above_05
        assert 5 not in above_05  # Below threshold
        assert 0 not in above_05  # Normal class excluded

    def test_latency_steps(self):
        """latency_steps is tracked for async results."""
        result = DetectionResult(fault_class=0, confidence=0.9, step=100, latency_steps=5)
        assert result.latency_steps == 5


# =============================================================================
# DETECTION METRICS TESTS
# =============================================================================

class TestDetectionMetrics:
    """Tests for DetectionMetrics class."""

    def test_initial_state(self):
        """Metrics start with zeros."""
        m = DetectionMetrics()
        assert m.total_samples == 0
        assert m.accuracy == 0.0
        assert m.unknown_count == 0

    def test_update_increments_counts(self):
        """update() increments confusion matrix."""
        m = DetectionMetrics()
        m.update(actual=0, predicted=0, step=1)
        m.update(actual=0, predicted=0, step=2)
        m.update(actual=1, predicted=1, step=3)

        assert m.total_samples == 3
        assert m.confusion_matrix[0, 0] == 2
        assert m.confusion_matrix[1, 1] == 1

    def test_update_unknown_prediction(self):
        """Unknown predictions (-1) are tracked separately."""
        m = DetectionMetrics()
        m.update(actual=0, predicted=-1, step=1)
        m.update(actual=4, predicted=-1, step=2)

        assert m.unknown_count == 2
        assert m.total_samples == 0  # Unknowns not in confusion matrix
        assert m.unknown_by_actual[0] == 1
        assert m.unknown_by_actual[4] == 1

    def test_accuracy(self):
        """Accuracy is computed correctly."""
        m = DetectionMetrics()
        # 3 correct, 1 wrong
        m.update(actual=0, predicted=0, step=1)
        m.update(actual=0, predicted=0, step=2)
        m.update(actual=1, predicted=1, step=3)
        m.update(actual=1, predicted=0, step=4)  # Wrong

        assert m.accuracy == 0.75

    def test_fault_detection_rate(self):
        """Fault detection rate measures if faults are detected at all."""
        m = DetectionMetrics()
        # 2 faults detected, 1 missed
        m.update(actual=1, predicted=1, step=1)  # Correct
        m.update(actual=1, predicted=2, step=2)  # Wrong class but detected
        m.update(actual=1, predicted=0, step=3)  # Missed

        assert m.fault_detection_rate == pytest.approx(2/3)

    def test_false_alarm_rate(self):
        """False alarm rate measures normal samples flagged as faults."""
        m = DetectionMetrics()
        m.update(actual=0, predicted=0, step=1)
        m.update(actual=0, predicted=0, step=2)
        m.update(actual=0, predicted=1, step=3)  # False alarm

        assert m.false_alarm_rate == pytest.approx(1/3)

    def test_missed_detection_rate(self):
        """Missed detection rate measures faults classified as normal."""
        m = DetectionMetrics()
        m.update(actual=1, predicted=1, step=1)
        m.update(actual=1, predicted=0, step=2)  # Missed

        assert m.missed_detection_rate == 0.5

    def test_precision(self):
        """Per-class precision is computed correctly."""
        m = DetectionMetrics()
        # For class 1: 2 TP, 1 FP
        m.update(actual=1, predicted=1, step=1)
        m.update(actual=1, predicted=1, step=2)
        m.update(actual=0, predicted=1, step=3)  # FP

        assert m.precision(1) == pytest.approx(2/3)

    def test_recall(self):
        """Per-class recall is computed correctly."""
        m = DetectionMetrics()
        # For class 1: 2 TP, 1 FN
        m.update(actual=1, predicted=1, step=1)
        m.update(actual=1, predicted=1, step=2)
        m.update(actual=1, predicted=0, step=3)  # FN

        assert m.recall(1) == pytest.approx(2/3)

    def test_f1_score(self):
        """F1 score is harmonic mean of precision and recall."""
        m = DetectionMetrics()
        m.update(actual=1, predicted=1, step=1)
        m.update(actual=1, predicted=0, step=2)
        m.update(actual=0, predicted=1, step=3)

        # precision = 1/2, recall = 1/2
        assert m.f1_score(1) == pytest.approx(0.5)

    def test_detection_delay_tracking(self):
        """Detection delays are tracked when fault onset is provided."""
        m = DetectionMetrics()
        m.update(actual=4, predicted=0, step=100, fault_onset_step=100)
        m.update(actual=4, predicted=0, step=101, fault_onset_step=100)
        m.update(actual=4, predicted=4, step=110, fault_onset_step=100)  # First detection

        delays = m.detection_delays.get(4)
        assert delays is not None
        assert 10 in delays

    def test_mean_detection_delay(self):
        """Mean detection delay is computed correctly."""
        m = DetectionMetrics()
        # First correct detection at step 50, onset at 40 -> delay 10
        m.update(actual=1, predicted=1, step=50, fault_onset_step=40)
        # Second detection at step 60 -> delay 20
        m.update(actual=1, predicted=1, step=60, fault_onset_step=40)

        # Mean of [10, 20] = 15
        assert m.mean_detection_delay(1) == pytest.approx(15.0)

    def test_reset(self):
        """reset() clears all metrics."""
        m = DetectionMetrics()
        m.update(actual=0, predicted=0, step=1)
        m.update(actual=1, predicted=1, step=2)

        m.reset()

        assert m.total_samples == 0
        assert m.unknown_count == 0
        assert m.confusion_matrix.sum() == 0

    def test_summary(self):
        """summary() returns dict with all metrics."""
        m = DetectionMetrics()
        m.update(actual=0, predicted=0, step=1)

        s = m.summary()
        assert "accuracy" in s
        assert "fault_detection_rate" in s
        assert "macro_f1" in s

    def test_per_class_report(self):
        """per_class_report() returns list of class metrics."""
        m = DetectionMetrics()
        m.update(actual=0, predicted=0, step=1)
        m.update(actual=4, predicted=4, step=2)

        report = m.per_class_report()
        classes = [r['class'] for r in report]
        assert 0 in classes
        assert 4 in classes


# =============================================================================
# BASE FAULT DETECTOR TESTS
# =============================================================================

class TestBaseFaultDetector:
    """Tests for BaseFaultDetector abstract class."""

    def test_cannot_instantiate_directly(self):
        """BaseFaultDetector cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseFaultDetector()

    def test_valid_subclass(self):
        """Valid subclass can be instantiated."""
        class SimpleDetector(BaseFaultDetector):
            name = "simple"

            def detect(self, xmeas, step):
                return DetectionResult(fault_class=0, confidence=1.0, step=step)

        detector = SimpleDetector()
        assert detector.name == "simple"

    def test_window_accumulation(self):
        """Window buffer accumulates measurements."""
        class TestDetector(BaseFaultDetector):
            name = "test"
            window_size = 5

            def detect(self, xmeas, step):
                return DetectionResult(fault_class=0, confidence=1.0, step=step)

        detector = TestDetector()
        assert not detector.window_ready

        for i in range(5):
            xmeas = np.ones(NUM_MEASUREMENTS) * i
            detector.process(xmeas, i)

        assert detector.window_ready
        assert detector.window.shape == (5, NUM_MEASUREMENTS)

    def test_window_sample_interval(self):
        """Window sampling respects interval."""
        class TestDetector(BaseFaultDetector):
            name = "test"
            window_size = 3
            window_sample_interval = 2

            def detect(self, xmeas, step):
                return DetectionResult(fault_class=0, confidence=1.0, step=step)

        detector = TestDetector()

        for i in range(6):
            xmeas = np.ones(NUM_MEASUREMENTS) * i
            detector.process(xmeas, i)

        # Only steps 0, 2, 4 should be stored
        assert detector.window_ready
        np.testing.assert_array_equal(detector.window[:, 0], [0, 2, 4])

    def test_detect_interval(self):
        """Detection only runs at specified interval."""
        class CountingDetector(BaseFaultDetector):
            name = "counting"
            detect_interval = 5
            window_size = 1

            def __init__(self):
                super().__init__()
                self.detect_count = 0

            def detect(self, xmeas, step):
                self.detect_count += 1
                return DetectionResult(fault_class=0, confidence=1.0, step=step)

        detector = CountingDetector()

        for i in range(15):
            detector.process(np.zeros(NUM_MEASUREMENTS), i)

        # Should detect at steps 0, 5, 10 = 3 times
        assert detector.detect_count == 3

    def test_reset_clears_window(self):
        """reset() clears window buffer."""
        class TestDetector(BaseFaultDetector):
            name = "test"
            window_size = 5

            def detect(self, xmeas, step):
                return DetectionResult(fault_class=0, confidence=1.0, step=step)

        detector = TestDetector()
        for i in range(5):
            detector.process(np.zeros(NUM_MEASUREMENTS), i)

        assert detector.window_ready
        detector.reset()
        assert not detector.window_ready

    def test_ground_truth_setting(self):
        """Ground truth can be set for metrics."""
        class TestDetector(BaseFaultDetector):
            name = "test"

            def detect(self, xmeas, step):
                return DetectionResult(fault_class=0, confidence=1.0, step=step)

        detector = TestDetector()
        detector.set_ground_truth(4, onset_step=100)

        assert detector._ground_truth == 4
        assert detector._fault_onset_step == 100

    def test_metrics_recorded_with_ground_truth(self):
        """Metrics are recorded when ground truth is set."""
        class TestDetector(BaseFaultDetector):
            name = "test"
            window_size = 1

            def detect(self, xmeas, step):
                return DetectionResult(fault_class=0, confidence=1.0, step=step)

        detector = TestDetector()
        detector.set_ground_truth(0)

        for i in range(10):
            detector.process(np.zeros(NUM_MEASUREMENTS), i)

        assert detector.metrics.total_samples == 10
        assert detector.metrics.accuracy == 1.0

    def test_get_info(self):
        """get_info() returns detector configuration."""
        class TestDetector(BaseFaultDetector):
            name = "test"
            description = "Test detector"
            window_size = 100
            detect_interval = 10

            def detect(self, xmeas, step):
                return DetectionResult(fault_class=0, confidence=1.0, step=step)

        detector = TestDetector()
        info = detector.get_info()

        assert info["name"] == "test"
        assert info["window_size"] == 100
        assert info["detect_interval"] == 10


# =============================================================================
# FAULT DETECTOR REGISTRY TESTS
# =============================================================================

class TestFaultDetectorRegistry:
    """Tests for FaultDetectorRegistry."""

    def test_list_available(self):
        """list_available() returns registered detector names."""
        available = FaultDetectorRegistry.list_available()
        assert "threshold" in available
        assert "pca" in available
        assert "ewma" in available

    def test_create_detector(self):
        """create() instantiates a detector."""
        detector = FaultDetectorRegistry.create("threshold")
        assert isinstance(detector, ThresholdDetector)

    def test_create_with_params(self):
        """create() passes parameters to constructor."""
        detector = FaultDetectorRegistry.create("ewma", alpha=0.2, threshold=4.0)
        assert detector.alpha == 0.2
        assert detector.threshold == 4.0

    def test_create_unknown_raises(self):
        """create() raises KeyError for unknown detector."""
        with pytest.raises(KeyError):
            FaultDetectorRegistry.create("nonexistent")

    def test_get_info(self):
        """get_info() returns detector metadata."""
        info = FaultDetectorRegistry.get_info("pca")
        assert info["name"] == "pca"
        assert "description" in info

    def test_register_custom_detector(self):
        """Custom detectors can be registered."""
        @register_detector(name="test_custom", description="Test custom detector")
        class CustomDetector(BaseFaultDetector):
            name = "test_custom"

            def detect(self, xmeas, step):
                return DetectionResult(fault_class=0, confidence=1.0, step=step)

        assert "test_custom" in FaultDetectorRegistry.list_available()
        detector = FaultDetectorRegistry.create("test_custom")
        assert isinstance(detector, CustomDetector)

        # Cleanup
        FaultDetectorRegistry.unregister("test_custom")


# =============================================================================
# BUILT-IN DETECTOR PLUGIN TESTS
# =============================================================================

class TestThresholdDetector:
    """Tests for ThresholdDetector."""

    def test_normal_operation(self):
        """Returns normal for values within limits."""
        detector = ThresholdDetector()
        xmeas = np.zeros(NUM_MEASUREMENTS)
        xmeas[6] = 2700   # Pressure within limit
        xmeas[7] = 10.0   # Reactor level within limit (3-22%)
        xmeas[8] = 120    # Temperature within limit
        xmeas[11] = 6.0   # Separator level within limit (2-11%)
        xmeas[14] = 4.0   # Stripper level within limit (2-7%)

        result = detector.detect(xmeas, 1)
        assert result.is_normal

    def test_detects_high_pressure(self):
        """Detects when pressure exceeds limit."""
        detector = ThresholdDetector()
        xmeas = np.zeros(NUM_MEASUREMENTS)
        xmeas[6] = 3000   # Pressure above limit (max is 2900)
        xmeas[7] = 10.0   # Reactor level within limit
        xmeas[8] = 120    # Temperature within limit
        xmeas[11] = 6.0   # Separator level within limit
        xmeas[14] = 4.0   # Stripper level within limit

        result = detector.detect(xmeas, 1)
        assert result.is_fault
        assert 6 in result.contributing_sensors


class TestEWMADetector:
    """Tests for EWMADetector."""

    def test_warmup_returns_unknown(self):
        """Returns unknown during warmup period."""
        detector = EWMADetector(warmup_steps=10)

        for i in range(5):
            result = detector.detect(np.zeros(NUM_MEASUREMENTS), i)

        assert not result.is_ready

    def test_detects_deviation(self):
        """Detects when values deviate from average."""
        detector = EWMADetector(alpha=0.5, threshold=2.0, warmup_steps=10)

        # Build up baseline
        for i in range(20):
            detector.process(np.ones(NUM_MEASUREMENTS), i)

        # Large deviation
        xmeas = np.ones(NUM_MEASUREMENTS) * 100
        result = detector.process(xmeas, 20)

        assert result.is_fault


class TestPCADetector:
    """Tests for PCADetector."""

    def test_not_trained_returns_unknown(self):
        """Returns unknown before training."""
        detector = PCADetector(auto_train=False)
        result = detector.detect(np.zeros(NUM_MEASUREMENTS), 1)
        assert not result.is_ready

    def test_auto_trains_on_window(self):
        """Auto-trains when window is full."""
        detector = PCADetector(window_size=10, auto_train=True)

        # Fill window
        for i in range(15):
            xmeas = np.random.randn(NUM_MEASUREMENTS) * 0.1
            result = detector.process(xmeas, i)

        assert detector.is_trained

    def test_train_method(self):
        """Manual training works."""
        detector = PCADetector(auto_train=False)
        data = np.random.randn(100, NUM_MEASUREMENTS)

        detector.train(data)

        assert detector.is_trained
        assert detector._mean is not None
        assert detector._components is not None


class TestPassthroughDetector:
    """Tests for PassthroughDetector."""

    def test_always_normal(self):
        """Always returns normal."""
        detector = PassthroughDetector()

        for i in range(10):
            xmeas = np.random.randn(NUM_MEASUREMENTS)
            result = detector.detect(xmeas, i)
            assert result.is_normal
            assert result.confidence == 1.0


class TestCompositeDetector:
    """Tests for CompositeDetector."""

    def test_votes_aggregation(self):
        """Aggregates votes from sub-detectors."""
        d1 = ThresholdDetector()
        d2 = PassthroughDetector()

        composite = CompositeDetector(detectors=[d1, d2], min_votes=1)

        # Trigger d1 to detect fault
        xmeas = np.zeros(NUM_MEASUREMENTS)
        xmeas[6] = 3000  # High pressure

        result = composite.detect(xmeas, 1)
        assert result.is_fault


# =============================================================================
# SIMULATOR INTEGRATION TESTS
# =============================================================================

class TestSimulatorDetectorIntegration:
    """Tests for detector integration with TEPSimulator."""

    def test_add_detector(self):
        """Can add detector to simulator."""
        sim = TEPSimulator(random_seed=42)
        detector = ThresholdDetector()

        sim.add_detector(detector)

        assert "threshold" in sim.list_detectors()

    def test_remove_detector(self):
        """Can remove detector from simulator."""
        sim = TEPSimulator(random_seed=42)
        detector = ThresholdDetector()

        sim.add_detector(detector)
        result = sim.remove_detector("threshold")

        assert result is True
        assert "threshold" not in sim.list_detectors()

    def test_get_detector(self):
        """Can retrieve detector by name."""
        sim = TEPSimulator(random_seed=42)
        detector = ThresholdDetector()

        sim.add_detector(detector)
        retrieved = sim.get_detector("threshold")

        assert retrieved is detector

    def test_detector_called_on_step(self):
        """Detector is called during step()."""
        sim = TEPSimulator(random_seed=42)
        detector = PassthroughDetector()

        sim.add_detector(detector)
        sim.initialize()

        for _ in range(10):
            sim.step()

        results = sim.get_detection_results()
        assert "passthrough" in results
        assert len(results["passthrough"]) == 10

    def test_ground_truth_propagates(self):
        """Ground truth is propagated to detectors."""
        sim = TEPSimulator(random_seed=42)
        detector = PassthroughDetector()

        sim.add_detector(detector)
        sim.initialize()

        sim.set_ground_truth(4)
        assert detector._ground_truth == 4

    def test_detector_reset_on_initialize(self):
        """Detectors are reset when simulator initializes."""
        sim = TEPSimulator(random_seed=42)
        detector = EWMADetector()

        sim.add_detector(detector)
        sim.initialize()

        for _ in range(100):
            sim.step()

        # Re-initialize should reset detector
        sim.initialize()
        assert not detector.window_ready

    def test_detection_results_in_simulate(self):
        """Detection results are included in SimulationResult."""
        sim = TEPSimulator(random_seed=42)
        detector = ThresholdDetector()

        sim.add_detector(detector)
        result = sim.simulate(duration_hours=0.1, record_interval=60)

        assert "threshold" in result.detection_results
        assert len(result.detection_results["threshold"]) > 0

    def test_get_latest_detection(self):
        """get_latest_detection returns most recent result."""
        sim = TEPSimulator(random_seed=42)
        detector = PassthroughDetector()

        sim.add_detector(detector)
        sim.initialize()

        for _ in range(10):
            sim.step()

        latest = sim.get_latest_detection()
        assert "passthrough" in latest
        assert latest["passthrough"].step == 10

    def test_get_detector_metrics(self):
        """get_detector_metrics returns metrics summary."""
        sim = TEPSimulator(random_seed=42)
        detector = PassthroughDetector()

        sim.add_detector(detector)
        sim.initialize()
        sim.set_ground_truth(0)

        for _ in range(100):
            sim.step()

        metrics = sim.get_detector_metrics()
        assert "passthrough" in metrics
        assert "accuracy" in metrics["passthrough"]

    def test_clear_detectors(self):
        """clear_detectors removes all detectors."""
        sim = TEPSimulator(random_seed=42)
        sim.add_detector(ThresholdDetector())
        sim.add_detector(EWMADetector())

        assert len(sim.list_detectors()) == 2

        sim.clear_detectors()

        assert len(sim.list_detectors()) == 0

    def test_disturbance_updates_ground_truth(self):
        """Disturbance in simulate() updates ground truth."""
        sim = TEPSimulator(random_seed=42)
        detector = PassthroughDetector()
        sim.add_detector(detector)

        # Simulate with disturbance at 0.01 hours
        sim.simulate(
            duration_hours=0.05,
            disturbances={4: (0.01, 1)}
        )

        # Ground truth should have been updated
        assert sim.get_ground_truth() == 4


# =============================================================================
# ASYNC DETECTOR TESTS
# =============================================================================

class TestAsyncDetector:
    """Tests for async detector mode."""

    def test_async_mode_creation(self):
        """Can create detector in async mode."""
        class AsyncTestDetector(BaseFaultDetector):
            name = "async_test"
            async_mode = True
            window_size = 5

            def detect(self, xmeas, step):
                return DetectionResult(fault_class=0, confidence=1.0, step=step)

        detector = AsyncTestDetector()
        assert detector.async_mode
        assert detector._executor is not None

    def test_async_returns_latest(self):
        """Async detector returns latest available result."""
        class AsyncTestDetector(BaseFaultDetector):
            name = "async_test"
            async_mode = True
            window_size = 3
            detect_interval = 1

            def detect(self, xmeas, step):
                return DetectionResult(fault_class=0, confidence=1.0, step=step)

        detector = AsyncTestDetector()

        # Process some steps
        for i in range(10):
            detector.process(np.zeros(NUM_MEASUREMENTS), i)

        # Result should be available (may have latency)
        result = detector._latest_result
        assert result is not None

    def test_async_shutdown(self):
        """Async detector cleans up on shutdown."""
        class AsyncTestDetector(BaseFaultDetector):
            name = "async_test"
            async_mode = True
            window_size = 3

            def detect(self, xmeas, step):
                return DetectionResult(fault_class=0, confidence=1.0, step=step)

        detector = AsyncTestDetector()
        detector.shutdown()
        assert detector._executor is None
