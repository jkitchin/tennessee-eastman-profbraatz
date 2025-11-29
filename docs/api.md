# TEP Simulator API Reference

This document provides detailed API documentation for the Tennessee Eastman Process (TEP) Python simulator.

## Table of Contents

- [TEPSimulator](#tepsimulator)
- [SimulationResult](#simulationresult)
- [Controllers](#controllers)
- [FortranTEProcess](#fortranteprocess)
- [PythonTEProcess](#pythonteprocess)
- [Backend Selection](#backend-selection)
- [Constants](#constants)

---

## TEPSimulator

The main simulator class providing a high-level interface for running TEP simulations.

### Import

```python
from tep import TEPSimulator
from tep.simulator import ControlMode
```

### Constructor

```python
TEPSimulator(
    random_seed: int = None,
    control_mode: ControlMode = ControlMode.CLOSED_LOOP,
    backend: str = "fortran"
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_seed` | `int` | `1431655765` | Random seed for reproducible simulations |
| `control_mode` | `ControlMode` | `CLOSED_LOOP` | Control mode (see below) |
| `backend` | `str` | `"fortran"` | Simulation backend: `"fortran"` or `"python"` |

**Backend Options:**

| Backend | Description |
|---------|-------------|
| `"fortran"` | Original Fortran code via f2py (highest accuracy, requires gfortran) |
| `"python"` | Pure Python implementation (no compilation, statistically similar) |

**Note:** The default `"fortran"` backend requires a Fortran compiler (gfortran) during installation. Use `backend="python"` when Fortran is unavailable.

**Control Modes:**

| Mode | Description |
|------|-------------|
| `ControlMode.OPEN_LOOP` | No automatic control - MVs remain fixed |
| `ControlMode.CLOSED_LOOP` | Decentralized PI control active |
| `ControlMode.MANUAL` | User sets MVs directly via `set_mv()` |

### Methods

#### initialize()

Initialize or reset the simulator to steady-state conditions. **Must be called before simulation.**

```python
sim = TEPSimulator()
sim.initialize()
```

#### simulate()

Run a complete simulation and return results.

```python
result = sim.simulate(
    duration_hours: float = 1.0,
    dt_hours: float = None,
    disturbances: Dict[int, Tuple[float, int]] = None,
    record_interval: int = 1,
    progress_callback: Callable[[float], None] = None
) -> SimulationResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `duration_hours` | `float` | `1.0` | Simulation duration in hours |
| `dt_hours` | `float` | `1/3600` | Time step (default 1 second) |
| `disturbances` | `dict` | `None` | Scheduled disturbances `{idv: (time, value)}` |
| `record_interval` | `int` | `1` | Record every N steps |
| `progress_callback` | `callable` | `None` | Called with progress (0.0-1.0) |

**Returns:** `SimulationResult` object

**Example:**

```python
# Simple simulation
result = sim.simulate(duration_hours=2.0)

# With disturbance at t=0.5h
result = sim.simulate(
    duration_hours=4.0,
    disturbances={1: (0.5, 1)}  # IDV(1) at 0.5 hours
)

# Multiple disturbances
result = sim.simulate(
    duration_hours=8.0,
    disturbances={
        1: (1.0, 1),   # IDV(1) at 1 hour
        4: (2.0, 1),   # IDV(4) at 2 hours
    }
)
```

#### step()

Advance simulation by N steps.

```python
running = sim.step(n_steps: int = 1) -> bool
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_steps` | `int` | `1` | Number of integration steps |

**Returns:** `True` if running, `False` if shutdown occurred

#### set_disturbance()

Activate or deactivate a process disturbance.

```python
sim.set_disturbance(idv_index: int, value: int = 1)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `idv_index` | `int` | Disturbance index (1-20) |
| `value` | `int` | 0 = off, 1 = on |

#### clear_disturbances()

Turn off all active disturbances.

```python
sim.clear_disturbances()
```

#### set_mv()

Set a manipulated variable (for manual control mode).

```python
sim.set_mv(index: int, value: float)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `index` | `int` | MV index (1-12) |
| `value` | `float` | Valve position (0-100%) |

#### get_measurements()

Get current process measurements.

```python
measurements = sim.get_measurements() -> np.ndarray  # Shape: (41,)
```

#### get_manipulated_vars()

Get current manipulated variable values.

```python
mvs = sim.get_manipulated_vars() -> np.ndarray  # Shape: (12,)
```

#### get_states()

Get current state vector.

```python
states = sim.get_states() -> np.ndarray  # Shape: (50,)
```

#### is_shutdown()

Check if process is in safety shutdown state.

```python
shutdown = sim.is_shutdown() -> bool
```

### Streaming Interface

For real-time dashboard integration:

#### start_stream()

Initialize streaming mode with history buffer.

```python
sim.start_stream(history_size: int = 1000)
```

#### stream_step()

Take one step and return current state.

```python
data = sim.stream_step() -> dict
```

**Returns:**

```python
{
    'time': float,           # Current time (hours)
    'time_seconds': float,   # Current time (seconds)
    'measurements': np.ndarray,  # Shape: (41,)
    'mvs': np.ndarray,       # Shape: (12,)
    'shutdown': bool
}
```

#### get_stream_history()

Get historical data for plotting.

```python
history = sim.get_stream_history() -> dict
```

**Returns:**

```python
{
    'time': np.ndarray,          # Time points
    'time_seconds': np.ndarray,  # Time in seconds
    'measurements': np.ndarray,  # Shape: (n_points, 41)
    'mvs': np.ndarray            # Shape: (n_points, 12)
}
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `time` | `float` | Current simulation time (hours) |
| `step_count` | `int` | Number of steps taken |
| `initialized` | `bool` | Whether simulator is initialized |
| `control_mode` | `ControlMode` | Current control mode |

---

## SimulationResult

Container for simulation results returned by `simulate()`.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `time` | `np.ndarray` | Time points in hours |
| `states` | `np.ndarray` | State trajectories (n_steps, 50) |
| `measurements` | `np.ndarray` | Measurements (n_steps, 41) |
| `manipulated_vars` | `np.ndarray` | MVs (n_steps, 12) |
| `shutdown` | `bool` | Whether shutdown occurred |
| `shutdown_time` | `float` | Time of shutdown (if any) |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `time_seconds` | `np.ndarray` | Time in seconds |
| `time_minutes` | `np.ndarray` | Time in minutes |

### Example

```python
result = sim.simulate(duration_hours=1.0)

# Access data
print(f"Simulation ended at {result.time[-1]:.2f} hours")
print(f"Shutdown: {result.shutdown}")

# Plot reactor temperature
import matplotlib.pyplot as plt
plt.plot(result.time_minutes, result.measurements[:, 8])
plt.xlabel('Time (min)')
plt.ylabel('Reactor Temperature (C)')
plt.show()
```

---

## Controllers

### PIController

Proportional-Integral controller using velocity form.

```python
from tep.controllers import PIController

controller = PIController(
    setpoint: float,
    gain: float,
    taui: float = 0.0,      # Reset time (hours), 0 = P-only
    output_min: float = 0.0,
    output_max: float = 100.0,
    scale: float = 1.0
)

# Calculate new output
new_output = controller.calculate(
    measurement: float,
    current_output: float,
    dt: float  # Time step (hours)
)
```

### DecentralizedController

Multi-loop control system implementing the temain_mod.f control scheme.

```python
from tep.controllers import DecentralizedController

controller = DecentralizedController()

# Calculate all MV updates
new_xmv = controller.calculate(
    xmeas: np.ndarray,  # 41 measurements
    xmv: np.ndarray,    # 12 MVs
    time_step: int      # Step number
)

# Reset to initial state
controller.reset()
```

### ManualController

Holds MVs at user-specified values.

```python
from tep.controllers import ManualController

controller = ManualController()
controller.set_mv(1, 50.0)  # Set MV 1 to 50%

# Returns stored values
new_xmv = controller.calculate(xmeas, xmv, step)
```

### Custom Controllers

You can use any callable as a controller:

```python
def my_controller(xmeas, xmv, step):
    """Custom control law."""
    new_xmv = xmv.copy()
    # Your control logic here
    return new_xmv

result = sim.simulate_with_controller(
    duration_hours=1.0,
    controller=my_controller
)
```

---

## FortranTEProcess

Low-level Fortran process interface via f2py (direct TEINIT/TEFUNC wrapper).

```python
from tep import FortranTEProcess

process = FortranTEProcess(random_seed=12345)
process._initialize()

# Evaluate derivatives
yp = process.evaluate(time=0.0, yy=process.state.yy)

# Get/set measurements and MVs
xmeas = process.get_xmeas()
xmv = process.get_xmv()
process.set_xmv(1, 50.0)
process.set_idv(1, 1)
```

This class wraps the original Fortran subroutines `TEINIT` and `TEFUNC` via f2py, providing exact numerical results matching the original TEP benchmark.

---

## PythonTEProcess

Low-level pure Python process interface (reimplementation of Fortran TEINIT/TEFUNC).

```python
from tep import PythonTEProcess

process = PythonTEProcess(random_seed=12345)
process.teinit()

# Evaluate derivatives
process.tefunc(time=0.0)
yp = process.yp  # Derivative array

# Get/set measurements and MVs
xmeas = process.get_xmeas()
xmv = process.get_xmv()
process.set_xmv(1, 50.0)
process.set_idv(1, 1)
```

This class provides a pure Python implementation of the TEP simulator, matching the Fortran code's behavior without requiring compilation. See [Python Backend Documentation](python_backend.md) for details.

---

## Backend Selection

Utility functions for backend selection:

```python
from tep import (
    get_available_backends,  # Returns ['fortran', 'python'] or ['python']
    get_default_backend,     # Returns 'fortran' if available, else 'python'
    is_fortran_available,    # Returns True if Fortran backend was compiled
)

# Example usage
if is_fortran_available():
    sim = TEPSimulator(backend='fortran')
else:
    sim = TEPSimulator(backend='python')

# Or let the library choose
sim = TEPSimulator(backend=get_default_backend())
```

---

## Constants

### Dimensions

```python
from tep.constants import (
    NUM_STATES,           # 50
    NUM_MEASUREMENTS,     # 41
    NUM_MANIPULATED_VARS, # 12
    NUM_DISTURBANCES,     # 20
)
```

### Names

```python
from tep import (
    COMPONENT_NAMES,       # ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    MEASUREMENT_NAMES,     # List of 41 measurement descriptions
    MANIPULATED_VAR_NAMES, # List of 12 MV descriptions
    DISTURBANCE_NAMES,     # List of 20 disturbance descriptions
)
```

### Initial Conditions

```python
from tep.constants import INITIAL_STATES  # np.ndarray of 50 steady-state values
```

---

## Measurement Index Reference

| Index | Variable | Description | Units |
|-------|----------|-------------|-------|
| 0 | XMEAS(1) | A Feed Flow | kscmh |
| 1 | XMEAS(2) | D Feed Flow | kg/hr |
| 2 | XMEAS(3) | E Feed Flow | kg/hr |
| 3 | XMEAS(4) | A and C Feed Flow | kscmh |
| 4 | XMEAS(5) | Recycle Flow | kscmh |
| 5 | XMEAS(6) | Reactor Feed Rate | kscmh |
| 6 | XMEAS(7) | Reactor Pressure | kPa gauge |
| 7 | XMEAS(8) | Reactor Level | % |
| 8 | XMEAS(9) | Reactor Temperature | deg C |
| 9 | XMEAS(10) | Purge Rate | kscmh |
| 10 | XMEAS(11) | Separator Temperature | deg C |
| 11 | XMEAS(12) | Separator Level | % |
| 12 | XMEAS(13) | Separator Pressure | kPa gauge |
| 13 | XMEAS(14) | Separator Underflow | m3/hr |
| 14 | XMEAS(15) | Stripper Level | % |
| 15 | XMEAS(16) | Stripper Pressure | kPa gauge |
| 16 | XMEAS(17) | Stripper Underflow | m3/hr |
| 17 | XMEAS(18) | Stripper Temperature | deg C |
| 18 | XMEAS(19) | Stripper Steam Flow | kg/hr |
| 19 | XMEAS(20) | Compressor Work | kW |
| 20 | XMEAS(21) | Reactor CW Outlet Temp | deg C |
| 21 | XMEAS(22) | Separator CW Outlet Temp | deg C |
| 22-27 | XMEAS(23-28) | Reactor Feed Composition | mol% A-F |
| 28-35 | XMEAS(29-36) | Purge Gas Composition | mol% A-H |
| 36-40 | XMEAS(37-41) | Product Composition | mol% D-H |

## Manipulated Variable Index Reference

| Index | Variable | Description |
|-------|----------|-------------|
| 0 | XMV(1) | D Feed Flow Valve |
| 1 | XMV(2) | E Feed Flow Valve |
| 2 | XMV(3) | A Feed Flow Valve |
| 3 | XMV(4) | A and C Feed Flow Valve |
| 4 | XMV(5) | Compressor Recycle Valve |
| 5 | XMV(6) | Purge Valve |
| 6 | XMV(7) | Separator Liquid Flow Valve |
| 7 | XMV(8) | Stripper Product Flow Valve |
| 8 | XMV(9) | Stripper Steam Valve |
| 9 | XMV(10) | Reactor Cooling Water Valve |
| 10 | XMV(11) | Condenser Cooling Water Valve |
| 11 | XMV(12) | Agitator Speed |

## Disturbance Index Reference

| Index | Variable | Type | Description |
|-------|----------|------|-------------|
| 0 | IDV(1) | Step | A/C Feed Ratio Change |
| 1 | IDV(2) | Step | B Composition Change |
| 2 | IDV(3) | Step | D Feed Temperature |
| 3 | IDV(4) | Step | Reactor CW Inlet Temp |
| 4 | IDV(5) | Step | Condenser CW Inlet Temp |
| 5 | IDV(6) | Step | A Feed Loss |
| 6 | IDV(7) | Step | C Header Pressure Loss |
| 7 | IDV(8) | Random | A,B,C Feed Composition |
| 8 | IDV(9) | Random | D Feed Temperature |
| 9 | IDV(10) | Random | C Feed Temperature |
| 10 | IDV(11) | Random | Reactor CW Inlet Temp |
| 11 | IDV(12) | Random | Condenser CW Inlet Temp |
| 12 | IDV(13) | Drift | Reaction Kinetics |
| 13 | IDV(14) | Sticking | Reactor CW Valve |
| 14 | IDV(15) | Sticking | Condenser CW Valve |
| 15-19 | IDV(16-20) | Unknown | Reserved |
