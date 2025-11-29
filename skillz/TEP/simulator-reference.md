# TEPSimulator API Reference

## Class: TEPSimulator

The main simulation interface for the Tennessee Eastman Process.

### Constructor

```python
TEPSimulator(
    random_seed: int = None,           # Seed for reproducibility (default: 4651207995)
    control_mode: ControlMode = ControlMode.CLOSED_LOOP
)
```

**Control Modes:**
- `ControlMode.OPEN_LOOP` - No automatic control
- `ControlMode.CLOSED_LOOP` - Decentralized PI control (default)
- `ControlMode.MANUAL` - Manual valve manipulation

### Core Methods

#### initialize()
Reset simulator to steady-state. Must be called before simulation.

```python
sim = TEPSimulator()
sim.initialize()
```

#### simulate()
Run a batch simulation.

```python
result = sim.simulate(
    duration_hours: float = 1.0,       # Simulation length
    dt_hours: float = None,            # Time step (default: 1 second)
    disturbances: Dict[int, Tuple[float, int]] = None,  # {idv: (time_hr, value)}
    record_interval: int = 1,          # Record every N steps
    progress_callback: Callable[[float], None] = None
) -> SimulationResult
```

**Disturbance dict format:** `{idv_index: (start_time_hours, value)}`
- `idv_index`: 1-20 (IDV fault number)
- `start_time_hours`: When to activate
- `value`: 1 = on, 0 = off

```python
# Example: Apply IDV(4) at 1 hour, IDV(6) at 2 hours
result = sim.simulate(
    duration_hours=4.0,
    disturbances={4: (1.0, 1), 6: (2.0, 1)}
)
```

#### step()
Advance simulation by n steps (1 step = 1 second).

```python
running = sim.step(n_steps=1)  # Returns False if shutdown
```

### Disturbance Control

```python
sim.set_disturbance(idv_index: int, value: int = 1)  # Set IDV (1-20)
sim.clear_disturbances()                              # Turn off all
sim.get_disturbances() -> np.ndarray                  # Current IDV vector
sim.get_active_disturbances() -> list                 # Active IDV indices
```

### Manual Valve Control

```python
sim.set_mv(index: int, value: float)  # Set XMV (1-12), value 0-100%
```

### Data Access

```python
sim.get_measurements() -> np.ndarray      # Current XMEAS (41 values)
sim.get_manipulated_vars() -> np.ndarray  # Current XMV (12 values)
sim.get_states() -> np.ndarray            # Current state vector (50 values)
sim.is_shutdown() -> bool                 # Check if process shut down
```

### Variable Names

```python
sim.get_measurement_names() -> List[str]    # XMEAS(1-41) descriptions
sim.get_mv_names() -> List[str]             # XMV(1-12) descriptions
sim.get_disturbance_names() -> List[str]    # IDV(1-20) descriptions
```

### Streaming Interface (for dashboards)

```python
sim.start_stream(history_size=1000)  # Initialize streaming mode
data = sim.stream_step()             # Step and get current data dict
history = sim.get_stream_history()   # Get historical data dict
```

### Fault Detection Interface

```python
sim.add_detector(detector)                    # Add a BaseFaultDetector
sim.remove_detector(name: str) -> bool        # Remove by name
sim.get_detector(name: str) -> BaseFaultDetector
sim.list_detectors() -> List[str]
sim.clear_detectors()

sim.set_ground_truth(fault_class: int)        # Set true fault for metrics
sim.get_ground_truth() -> int

sim.get_detection_results(detector_name=None) # Get all DetectionResults
sim.get_latest_detection(detector_name=None)  # Get most recent
sim.get_detector_metrics(detector_name=None)  # Get performance metrics
sim.reset_detector_metrics()
```

### Custom Controller

```python
result = sim.simulate_with_controller(
    duration_hours=2.0,
    controller=my_controller,  # Object with calculate(xmeas, xmv, step) -> new_xmv
    disturbances={4: (1.0, 1)}
)
```

---

## Class: SimulationResult

Dataclass containing simulation output.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `time` | np.ndarray | Time points in hours |
| `states` | np.ndarray | State trajectories (n_steps, 50) |
| `measurements` | np.ndarray | XMEAS trajectories (n_steps, 41) |
| `manipulated_vars` | np.ndarray | XMV trajectories (n_steps, 12) |
| `shutdown` | bool | True if simulation ended in shutdown |
| `shutdown_time` | float | Time of shutdown (hours), or None |
| `detection_results` | Dict[str, List] | Detector results by name |

### Properties

```python
result.time_seconds -> np.ndarray   # Time in seconds
result.time_minutes -> np.ndarray   # Time in minutes
```

### Accessing Data

```python
# Get reactor temperature over time (XMEAS(9), 0-indexed = 8)
reactor_temp = result.measurements[:, 8]

# Get reactor cooling water valve position (XMV(10), 0-indexed = 9)
reactor_cw = result.manipulated_vars[:, 9]

# Get final pressure
final_pressure = result.measurements[-1, 6]  # XMEAS(7)
```

---

## Common Patterns

### Basic Fault Injection Experiment

```python
from tep import TEPSimulator

sim = TEPSimulator()
sim.initialize()

# 1 hour normal, then IDV(4) cooling water fault
result = sim.simulate(
    duration_hours=3.0,
    disturbances={4: (1.0, 1)}
)

# Plot reactor pressure response
import matplotlib.pyplot as plt
plt.plot(result.time, result.measurements[:, 6])
plt.xlabel('Time (hours)')
plt.ylabel('Reactor Pressure (kPa)')
plt.axvline(x=1.0, color='r', linestyle='--', label='Fault onset')
plt.legend()
plt.show()
```

### Multiple Runs with Different Seeds

```python
from tep import TEPSimulator

results = []
for seed in [1234, 5678, 9012]:
    sim = TEPSimulator(random_seed=seed)
    sim.initialize()
    result = sim.simulate(duration_hours=2.0)
    results.append(result)
```

### Real-time Dashboard Integration

```python
from tep import TEPSimulator

sim = TEPSimulator()
sim.start_stream(history_size=3600)  # 1 hour of history

while True:
    data = sim.stream_step()

    # Update dashboard with current values
    current_temp = data['measurements'][8]
    current_press = data['measurements'][6]

    if data['shutdown']:
        break
```
