#!/usr/bin/env python3
"""
Example: Real-time Fault Detection with the TEP Simulator

This example demonstrates how to use the fault detection module to:
1. Register multiple detectors with the simulator
2. Run simulations with fault injection
3. Track detector performance metrics
4. Compare different detection approaches

The fault detection system provides:
- Pluggable detector architecture (similar to controller plugins)
- Automatic window management for time-series analysis
- Performance metrics (accuracy, F1, detection delay, etc.)
- Support for both sync and async detection
"""

import numpy as np
from tep import TEPSimulator, ControlMode
from tep.detector_base import FaultDetectorRegistry, DetectionResult


def list_available_detectors():
    """Show all registered fault detectors."""
    print("Available Fault Detectors")
    print("=" * 50)
    for info in FaultDetectorRegistry.list_all_info():
        print(f"  {info['name']:20s} - {info['description']}")
    print()


def basic_detection_example():
    """
    Basic example: Run simulation with a single detector.
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Fault Detection")
    print("=" * 60)

    # Create simulator
    sim = TEPSimulator(random_seed=42, control_mode=ControlMode.CLOSED_LOOP)

    # Create and add a detector
    detector = FaultDetectorRegistry.create("threshold")
    sim.add_detector(detector)

    print(f"Added detector: {detector.name}")
    print(f"Configuration: {detector.get_info()}")

    # Initialize and set ground truth
    sim.initialize()
    sim.set_ground_truth(0)  # Normal operation

    # Run for 30 minutes normally
    print("\nRunning 30 minutes of normal operation...")
    for _ in range(1800):  # 1800 seconds = 30 minutes
        sim.step()

    # Inject fault IDV(4) - Reactor cooling water inlet temperature
    print("Injecting fault IDV(4) at t=30 minutes...")
    sim.set_disturbance(4, 1)
    sim.set_ground_truth(4)

    # Run for 30 more minutes with fault
    print("Running 30 minutes with fault...")
    for _ in range(1800):
        sim.step()

    # Check results
    print("\n--- Detection Results ---")
    results = sim.get_detection_results()
    fault_detections = sum(1 for r in results['threshold'] if r.is_fault)
    total = len(results['threshold'])
    print(f"Total detections: {total}")
    print(f"Fault detections: {fault_detections}")

    # Show metrics
    print("\n--- Detector Metrics ---")
    print(detector.metrics)


def multiple_detectors_example():
    """
    Example: Compare multiple detection methods.
    """
    print("\n" + "=" * 60)
    print("Example 2: Comparing Multiple Detectors")
    print("=" * 60)

    sim = TEPSimulator(random_seed=123, control_mode=ControlMode.CLOSED_LOOP)

    # Add multiple detectors with different approaches
    detectors = [
        FaultDetectorRegistry.create("threshold"),
        FaultDetectorRegistry.create("ewma", alpha=0.1, threshold=3.0),
        FaultDetectorRegistry.create("cusum", k=0.5, h=5.0),
        FaultDetectorRegistry.create("sliding_window", threshold=3.0),
    ]

    for d in detectors:
        sim.add_detector(d)
        print(f"Added: {d.name}")

    # Initialize
    sim.initialize()
    sim.set_ground_truth(0)

    # Normal operation (20 minutes)
    print("\nPhase 1: Normal operation (20 min)...")
    for _ in range(1200):
        sim.step()

    # Introduce fault
    print("Phase 2: Fault IDV(1) active (40 min)...")
    sim.set_disturbance(1, 1)
    sim.set_ground_truth(1)
    for _ in range(2400):
        sim.step()

    # Compare results
    print("\n--- Detector Comparison ---")
    print(f"{'Detector':<20} {'Accuracy':>10} {'FDR':>10} {'FAR':>10} {'F1':>10}")
    print("-" * 60)

    for d in detectors:
        m = d.metrics
        print(f"{d.name:<20} {m.accuracy:>10.3f} {m.fault_detection_rate:>10.3f} "
              f"{m.false_alarm_rate:>10.3f} {m.macro_f1():>10.3f}")


def detection_with_simulation_example():
    """
    Example: Use simulate() with automatic ground truth tracking.
    """
    print("\n" + "=" * 60)
    print("Example 3: Using simulate() with Detectors")
    print("=" * 60)

    sim = TEPSimulator(random_seed=456)

    # Add PCA detector
    pca = FaultDetectorRegistry.create(
        "pca",
        n_components=10,
        window_size=200,
        detect_interval=10,
        t2_threshold=15.0,
        spe_threshold=25.0
    )
    sim.add_detector(pca)

    # Run simulation with scheduled disturbance
    # Ground truth is automatically updated when disturbance activates
    print("Running 1-hour simulation with fault at t=0.5 hours...")

    result = sim.simulate(
        duration_hours=1.0,
        disturbances={4: (0.5, 1)},  # IDV(4) at 0.5 hours
        record_interval=60
    )

    print(f"\nSimulation completed:")
    print(f"  Duration: {result.time[-1]*60:.1f} minutes")
    print(f"  Shutdown: {result.shutdown}")

    # Metrics are automatically tracked
    print("\n--- PCA Detector Performance ---")
    print(pca.metrics)

    # Detection results are included in SimulationResult
    print(f"\nDetection results stored: {len(result.detection_results.get('pca', []))} entries")


def custom_detector_example():
    """
    Example: Creating a custom detector.
    """
    print("\n" + "=" * 60)
    print("Example 4: Custom Detector Implementation")
    print("=" * 60)

    from tep.detector_base import BaseFaultDetector, register_detector

    # Define a custom detector
    @register_detector(name="reactor_monitor", description="Monitors reactor conditions")
    class ReactorMonitorDetector(BaseFaultDetector):
        """Custom detector focused on reactor measurements."""

        name = "reactor_monitor"
        window_size = 60  # 1 minute window
        detect_interval = 5  # Check every 5 seconds

        def __init__(self, pressure_limit=2850.0, temp_limit=165.0, **kwargs):
            super().__init__(**kwargs)
            self.pressure_limit = pressure_limit
            self.temp_limit = temp_limit

        def detect(self, xmeas, step):
            # XMEAS indices (0-based):
            # 6 = Reactor Pressure (kPa)
            # 8 = Reactor Temperature (deg C)
            # 7 = Reactor Level (%)

            reactor_pressure = xmeas[6]
            reactor_temp = xmeas[8]
            reactor_level = xmeas[7]

            issues = []
            severity = 0.0

            if reactor_pressure > self.pressure_limit:
                issues.append(6)
                severity = max(severity, (reactor_pressure - self.pressure_limit) / 100)

            if reactor_temp > self.temp_limit:
                issues.append(8)
                severity = max(severity, (reactor_temp - self.temp_limit) / 10)

            if reactor_level < 5 or reactor_level > 20:
                issues.append(7)
                severity = max(severity, 0.3)

            if issues:
                # High pressure/temp often indicates cooling issues (IDV 4, 5)
                if 6 in issues or 8 in issues:
                    fault_class = 4
                else:
                    fault_class = 1

                return DetectionResult(
                    fault_class=fault_class,
                    confidence=min(0.5 + severity, 0.95),
                    step=step,
                    contributing_sensors=issues,
                    statistics={"severity": severity}
                )

            return DetectionResult(fault_class=0, confidence=0.9, step=step)

        def _reset_impl(self):
            pass

    # Use the custom detector
    sim = TEPSimulator(random_seed=789)

    custom = FaultDetectorRegistry.create("reactor_monitor",
                                          pressure_limit=2800.0,
                                          temp_limit=160.0)
    sim.add_detector(custom)

    print(f"Custom detector: {custom.name}")
    print(f"Config: pressure_limit={custom.pressure_limit}, temp_limit={custom.temp_limit}")

    # Run simulation
    sim.initialize()
    sim.set_ground_truth(0)

    for _ in range(600):  # 10 minutes normal
        sim.step()

    sim.set_disturbance(4, 1)  # Cooling water fault
    sim.set_ground_truth(4)

    for _ in range(1200):  # 20 minutes with fault
        sim.step()

    print("\n--- Custom Detector Results ---")
    print(custom.metrics)


def streaming_detection_example():
    """
    Example: Real-time detection in streaming mode.
    """
    print("\n" + "=" * 60)
    print("Example 5: Streaming Mode Detection")
    print("=" * 60)

    sim = TEPSimulator(random_seed=321)

    # Add a fast detector for real-time use
    detector = FaultDetectorRegistry.create("ewma", alpha=0.05, threshold=2.5)
    sim.add_detector(detector)

    # Start streaming
    sim.start_stream(history_size=500)
    sim.set_ground_truth(0)

    print("Streaming simulation (simulating dashboard updates)...")

    alerts = []

    for step in range(1800):  # 30 minutes
        # Inject fault at 10 minutes
        if step == 600:
            sim.set_disturbance(1, 1)
            sim.set_ground_truth(1)
            print(f"  [Step {step}] Fault IDV(1) injected")

        # Take one step
        data = sim.stream_step()

        # Check for alerts (every 10 seconds in this example)
        if step % 10 == 0:
            latest = sim.get_latest_detection()
            if 'ewma' in latest and latest['ewma'].is_fault:
                result = latest['ewma']
                if not alerts or alerts[-1][0] < step - 30:  # Debounce
                    alerts.append((step, result.confidence))
                    print(f"  [Step {step}] ALERT: Fault detected "
                          f"(conf={result.confidence:.2f})")

        if data['shutdown']:
            print(f"  [Step {step}] Process shutdown!")
            break

    print(f"\nTotal alerts raised: {len(alerts)}")
    print("\n--- Final Metrics ---")
    print(detector.metrics)


def metrics_analysis_example():
    """
    Example: Detailed analysis of detection metrics.
    """
    print("\n" + "=" * 60)
    print("Example 6: Detailed Metrics Analysis")
    print("=" * 60)

    sim = TEPSimulator(random_seed=555)

    # Use statistical detector for this example
    detector = FaultDetectorRegistry.create("statistical",
                                            mean_threshold=2.0,
                                            var_threshold=2.0,
                                            votes_required=2)
    sim.add_detector(detector)

    # Simulate multiple fault scenarios
    scenarios = [
        (0, 600, 0, "Normal baseline"),
        (1, 600, 1, "IDV(1) - Feed ratio"),
        (0, 300, 0, "Recovery"),
        (4, 600, 4, "IDV(4) - Cooling water"),
    ]

    sim.initialize()

    for fault_class, duration, ground_truth, description in scenarios:
        print(f"\nScenario: {description} ({duration} steps)")

        if fault_class > 0:
            sim.set_disturbance(fault_class, 1)
        else:
            sim.clear_disturbances()

        sim.set_ground_truth(ground_truth)

        for _ in range(duration):
            sim.step()

    # Detailed metrics report
    print("\n" + "=" * 40)
    print("DETAILED METRICS REPORT")
    print("=" * 40)

    m = detector.metrics
    print(f"\nOverall Summary:")
    print(f"  Total samples:      {m.total_samples}")
    print(f"  Unknown:            {m.unknown_count}")
    print(f"  Accuracy:           {m.accuracy:.3f}")

    print(f"\nFault Detection:")
    print(f"  Detection rate:     {m.fault_detection_rate:.3f}")
    print(f"  False alarm rate:   {m.false_alarm_rate:.3f}")
    print(f"  Missed detection:   {m.missed_detection_rate:.3f}")

    print(f"\nAggregate Scores:")
    print(f"  Macro Precision:    {m.macro_precision():.3f}")
    print(f"  Macro Recall:       {m.macro_recall():.3f}")
    print(f"  Macro F1:           {m.macro_f1():.3f}")
    print(f"  Weighted F1:        {m.weighted_f1():.3f}")

    print(f"\nPer-Class Report:")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 52)
    for row in m.per_class_report():
        print(f"{row['name']:<12} {row['precision']:>10.3f} {row['recall']:>10.3f} "
              f"{row['f1']:>10.3f} {row['support']:>10d}")

    if m.mean_detection_delay() is not None:
        print(f"\nDetection Delays:")
        print(f"  Mean delay:         {m.mean_detection_delay():.1f} steps")
        print(f"  Min delay:          {m.min_detection_delay()} steps")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Tennessee Eastman Process - Fault Detection Examples")
    print("=" * 60)

    # List available detectors
    list_available_detectors()

    # Run examples
    basic_detection_example()
    multiple_detectors_example()
    detection_with_simulation_example()
    custom_detector_example()
    streaming_detection_example()
    metrics_analysis_example()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
