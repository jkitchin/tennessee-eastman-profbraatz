---
name: tep-simulator
description: "Expert guidance for Tennessee Eastman Process (TEP) simulator - Python/Fortran chemical process simulation, fault injection with 20 IDV disturbances, fault detection with PCA/EWMA/CUSUM detectors, PI control systems, batch simulation CLI (tep-sim), and real-time web dashboard (tep-web). Use when working with TEP simulations, process control, or fault detection."
---

# Tennessee Eastman Process (TEP) Simulator

The TEP simulator is a Python interface to the classic Tennessee Eastman Process benchmark. It includes a pure Python implementation (default) and optional Fortran acceleration via f2py. It simulates an industrial chemical plant with reactor, separator, stripper, and compressor units.

## Quick Start

```python
from tep import TEPSimulator

# Create and initialize
sim = TEPSimulator()
sim.initialize()

# Run simulation
result = sim.simulate(duration_hours=2.0)

# Access results
print(f"Time points: {len(result.time)}")
print(f"Final reactor temp: {result.measurements[-1, 8]:.1f} °C")
print(f"Final reactor pressure: {result.measurements[-1, 6]:.1f} kPa")
```

## With Fault Injection

```python
from tep import TEPSimulator

sim = TEPSimulator()
sim.initialize()

# Apply IDV(4) - Cooling water inlet temperature step at t=1 hour
result = sim.simulate(
    duration_hours=4.0,
    disturbances={4: (1.0, 1)}  # {idv_number: (time_hours, value)}
)
```

## Key Imports

```python
from tep import (
    TEPSimulator,           # Main simulator class
    SimulationResult,       # Result dataclass
    ControlMode,            # OPEN_LOOP, CLOSED_LOOP, MANUAL
    BaseFaultDetector,      # Base class for custom detectors
    DetectionResult,        # Detection output
    FaultDetectorRegistry,  # Registry for built-in detectors
    register_detector,      # Decorator for custom detectors
)
```

## Reference Files

- [simulator-reference.md](simulator-reference.md) - TEPSimulator API details
- [fault-detection.md](fault-detection.md) - Fault detection framework
- [process-variables.md](process-variables.md) - XMEAS, XMV, IDV variable tables
- [cli-reference.md](cli-reference.md) - CLI commands (tep-sim, tep-web)

## Common Tasks

### Run Normal Operation
```python
sim = TEPSimulator()
sim.initialize()
result = sim.simulate(duration_hours=8.0)
```

### Apply Multiple Disturbances
```python
result = sim.simulate(
    duration_hours=8.0,
    disturbances={
        1: (1.0, 1),  # IDV(1) at t=1h
        4: (2.0, 1),  # IDV(4) at t=2h
    }
)
```

### Use Fault Detection
```python
from tep import FaultDetectorRegistry

detector = FaultDetectorRegistry.create("pca", window_size=200)
for i, xmeas in enumerate(result.measurements):
    detection = detector.process(xmeas, i)
    if detection.is_fault:
        print(f"Fault at step {i}: class={detection.fault_class}")
```

### Manual Valve Control
```python
sim = TEPSimulator()
sim.initialize()

# Set manual valve position (XMV index 0-11, value 0-100%)
sim.set_manual_mv(mv_index=9, value=50.0)  # Reactor cooling water

result = sim.simulate(duration_hours=1.0)
```

### Batch CLI Simulation
```bash
# Normal 8-hour run
tep-sim --duration 8 --output normal.dat

# With fault injection
tep-sim --duration 8 --faults 4 --fault-times 1.0 --output fault4.dat

# Multiple faults
tep-sim --duration 8 --faults 1,4,7 --fault-times 1.0,2.0,3.0 --output multi.dat
```

### Launch Dashboard
```bash
tep-web                          # Opens browser automatically
tep-web --no-browser --port 8080 # Headless mode
```

## Process Overview

The TEP simulates a chemical plant producing products G and H from reactants A, C, D, E:
- **Reactor**: Exothermic reactions with cooling water control
- **Separator**: Vapor-liquid separation with condenser
- **Stripper**: Steam-heated purification
- **Compressor**: Recycle gas compression

Key measurements:
- XMEAS(1-6): Feed and recycle flows
- XMEAS(7-9): Reactor pressure, level, temperature
- XMEAS(23-41): Composition analyzers (sampled, with delay)

Key manipulated variables:
- XMV(1-4): Feed flow valves
- XMV(10): Reactor cooling water
- XMV(11): Condenser cooling water

## Installation

```bash
# Default (Python backend, no compiler needed)
pip install -e .

# With Fortran acceleration (~5-10x faster, requires gfortran)
pip install -e . --config-settings=setup-args=-Dfortran=enabled

# With dev tools (pytest, matplotlib, dash, streamlit)
pip install -e ".[dev]"
```
