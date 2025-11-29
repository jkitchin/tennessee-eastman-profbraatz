# Fault Detection Framework

The TEP package includes a comprehensive fault detection system with built-in detectors and a plugin architecture for custom implementations.

## Quick Start

```python
from tep import TEPSimulator, FaultDetectorRegistry

# Create simulator and detector
sim = TEPSimulator()
detector = FaultDetectorRegistry.create("pca", window_size=200)

# Run with detection
sim.add_detector(detector)
sim.initialize()
sim.set_ground_truth(0)  # Normal operation initially

result = sim.simulate(
    duration_hours=2.0,
    disturbances={4: (1.0, 1)}  # Fault at t=1h
)

# Check metrics
print(detector.metrics)
```

## Built-in Detectors

| Name | Description | Key Parameters |
|------|-------------|----------------|
| `threshold` | Fast safety limit checking | `limits`, `fault_mapping` |
| `ewma` | Exponentially weighted moving average | `alpha=0.1`, `threshold=3.0`, `warmup_steps=100` |
| `cusum` | Cumulative sum control chart | `k=0.5`, `h=5.0` |
| `pca` | PCA with T² and SPE statistics | `n_components=10`, `t2_threshold=15.0`, `spe_threshold=25.0` |
| `statistical` | Multi-statistic ensemble | `mean_threshold=2.5`, `var_threshold=2.0`, `votes_required=2` |
| `sliding_window` | Window half comparison | `threshold=3.0` |
| `composite` | Combines multiple detectors | `min_votes=2` |
| `passthrough` | Always reports normal (baseline) | - |

### Creating Detectors

```python
from tep import FaultDetectorRegistry

# List available detectors
print(FaultDetectorRegistry.list_available())

# Create with defaults
detector = FaultDetectorRegistry.create("pca")

# Create with custom parameters
detector = FaultDetectorRegistry.create("pca",
    window_size=300,
    n_components=15,
    t2_threshold=20.0
)

# Create EWMA detector
ewma = FaultDetectorRegistry.create("ewma",
    alpha=0.05,       # Slower adaptation
    threshold=4.0,    # Higher threshold
    warmup_steps=200
)
```

## DetectionResult

Every detector returns a `DetectionResult`:

```python
result = detector.process(xmeas, step)

result.fault_class      # -1=unknown, 0=normal, 1-20=IDV fault
result.confidence       # 0.0 to 1.0
result.is_fault         # True if fault_class > 0
result.is_normal        # True if fault_class == 0
result.is_ready         # True if detector has enough data
result.contributing_sensors  # List of XMEAS indices (0-based)
result.statistics       # Dict of detector-specific stats

# For multi-class detectors
result.top_k(3)         # Top 3 predictions [(class, conf), ...]
result.above_threshold(0.5)  # All classes with conf > 0.5
```

## DetectionMetrics

Automatic performance tracking:

```python
# Set ground truth for metric computation
detector.set_ground_truth(fault_class=4, onset_step=3600)

# After running detection...
metrics = detector.metrics

# Aggregate metrics
metrics.accuracy               # Overall accuracy
metrics.fault_detection_rate   # % faults detected as some fault
metrics.false_alarm_rate       # % normal flagged as fault
metrics.missed_detection_rate  # % faults missed

# Per-class metrics
metrics.precision(4)           # Precision for IDV(4)
metrics.recall(4)              # Recall for IDV(4)
metrics.f1_score(4)            # F1 for IDV(4)
metrics.support(4)             # Sample count for IDV(4)

# Macro/weighted averages
metrics.macro_f1()
metrics.weighted_f1()

# Detection delay (steps from fault onset to first correct detection)
metrics.mean_detection_delay(4)
metrics.min_detection_delay(4)

# Reports
print(metrics)                 # Human-readable summary
metrics.summary()              # Dict of aggregate metrics
metrics.per_class_report()     # List of per-class dicts
```

## Custom Detector Implementation

```python
from tep import BaseFaultDetector, DetectionResult, register_detector
import numpy as np

@register_detector(name="pressure_monitor")
class PressureMonitor(BaseFaultDetector):
    """Monitors reactor pressure for cooling water faults."""

    name = "pressure_monitor"
    description = "Reactor pressure fault detector"
    window_size = 60        # 60 seconds of history
    detect_interval = 10    # Run every 10 steps

    def __init__(self, pressure_threshold=2750, **kwargs):
        super().__init__(**kwargs)
        self.pressure_threshold = pressure_threshold

    def detect(self, xmeas, step):
        if not self.window_ready:
            return DetectionResult(-1, 0.0, step)

        # Analyze pressure trend (XMEAS(7) = index 6)
        pressures = self.window[:, 6]
        mean_pressure = np.mean(pressures)
        trend = pressures[-1] - pressures[0]

        if mean_pressure > self.pressure_threshold and trend > 50:
            return DetectionResult(
                fault_class=4,  # IDV(4) cooling water fault
                confidence=min(0.5 + (mean_pressure - self.pressure_threshold) / 200, 0.95),
                step=step,
                contributing_sensors=[6],
                statistics={"mean_pressure": mean_pressure, "trend": trend}
            )

        return DetectionResult(fault_class=0, confidence=0.9, step=step)

    def _reset_impl(self):
        pass  # Reset custom state if needed

# Use custom detector
detector = FaultDetectorRegistry.create("pressure_monitor", pressure_threshold=2800)
```

### Key BaseFaultDetector Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | str | Unique identifier |
| `description` | str | Human-readable description |
| `window_size` | int | Points in sliding window |
| `window_sample_interval` | int | Store every Nth point |
| `detect_interval` | int | Run detection every N steps |
| `async_mode` | bool | Run in background thread |

### Available in detect() method

| Property | Description |
|----------|-------------|
| `self.window` | np.ndarray (window_size, 41) or None |
| `self.window_ready` | True if window is full |
| `self.window_fill` | Fraction filled (0-1) |
| `self.window_steps` | Step numbers for window rows |

## Composite Detector (Ensemble)

Combine multiple detectors with voting:

```python
from tep import FaultDetectorRegistry

# Create individual detectors
threshold = FaultDetectorRegistry.create("threshold")
pca = FaultDetectorRegistry.create("pca", window_size=200)
ewma = FaultDetectorRegistry.create("ewma", alpha=0.1)

# Create composite with majority voting
composite = FaultDetectorRegistry.create("composite", min_votes=2)
composite.add_detector(threshold)
composite.add_detector(pca)
composite.add_detector(ewma)

# Use like any other detector
sim.add_detector(composite)
```

## PCA Detector Training

For best results, train PCA on normal operation data:

```python
from tep import TEPSimulator, FaultDetectorRegistry

# Generate normal data
sim = TEPSimulator()
sim.initialize()
normal_result = sim.simulate(duration_hours=2.0)

# Train PCA detector
pca = FaultDetectorRegistry.create("pca", auto_train=False)
pca.train(normal_result.measurements)  # Shape: (n_samples, 41)

# Now use for fault detection
sim.initialize()
sim.add_detector(pca)
fault_result = sim.simulate(
    duration_hours=2.0,
    disturbances={4: (1.0, 1)}
)
```

## Detector Integration with Simulator

```python
sim = TEPSimulator()
sim.initialize()

# Add detector
detector = FaultDetectorRegistry.create("pca")
sim.add_detector(detector)

# Set ground truth (enables metrics)
sim.set_ground_truth(0)  # Normal

# Run normal operation
for _ in range(3600):  # 1 hour
    sim.step()

# Introduce fault
sim.set_disturbance(4, 1)
sim.set_ground_truth(4)

# Run with fault
for _ in range(3600):  # 1 hour
    sim.step()
    latest = sim.get_latest_detection()
    if latest.get("pca") and latest["pca"].is_fault:
        print(f"Fault detected at step {latest['pca'].step}")

# Get results
print(detector.metrics)
```
