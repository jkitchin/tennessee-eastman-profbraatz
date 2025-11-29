# Pure Python Backend for Tennessee Eastman Process

This document describes the pure Python implementation of the TEP simulator, which provides a Fortran-free alternative while maintaining statistical similarity to the original.

## Overview

The Python backend (`tep/python_backend.py`) is a complete reimplementation of the original Fortran code (`teprob.f`) in pure Python. It enables running TEP simulations in environments where Fortran compilation is unavailable (e.g., cloud notebooks, restricted systems).

## Backend Selection

```python
from tep import TEPSimulator

# Use Python backend (no Fortran required)
sim_py = TEPSimulator(backend='python')

# Use Fortran backend (highest accuracy, requires gfortran)
sim_f = TEPSimulator(backend='fortran')

# Check available backends
from tep import get_available_backends, get_default_backend
print(get_available_backends())  # ['fortran', 'python'] or ['python']
print(get_default_backend())     # 'fortran' if available, else 'python'
```

## Architecture

### Fortran Common Blocks → Python Dataclasses

The Fortran code uses common blocks for shared state. These are replicated as Python dataclasses:

| Fortran Block | Python Class | Contents |
|---------------|--------------|----------|
| `/CONST/` | `ConstBlock` | Thermodynamic properties (Antoine coefficients, enthalpies, densities) |
| `/TEPROC/` | `TEProcBlock` | Process state (temperatures, pressures, compositions, flows) |
| `/WLK/` | `WalkBlock` | Disturbance random walk parameters |
| `/RANDSD/` | Instance variable `_g` | Random number generator seed |
| `/PV/` | `_xmeas`, `_xmv` arrays | Measurements and manipulated variables |
| `/DVEC/` | `_idv` array | Disturbance vector |

### Key Components

```
PythonTEProcess
├── State Arrays
│   ├── yy[50]     - State variables
│   ├── yp[50]     - State derivatives
│   ├── _xmeas[41] - Measurements
│   ├── _xmv[12]   - Manipulated variables
│   └── _idv[20]   - Disturbances
├── Common Blocks
│   ├── _const     - Physical constants (ConstBlock)
│   ├── _teproc    - Process state (TEProcBlock)
│   └── _wlk       - Random walk state (WalkBlock)
├── Initialization
│   └── teinit()   - Initialize process to steady state
└── Simulation
    └── tefunc()   - Compute derivatives (called by integrator)
```

### Helper Functions (TESUB1-8)

| Function | Purpose |
|----------|---------|
| `_tesub1` | Reaction rate kinetics |
| `_tesub2` | Reaction rate calculations |
| `_tesub3` | Vapor-liquid equilibrium (VLE) |
| `_tesub4` | Flash calculations for separator/stripper |
| `_tesub5` | Liquid enthalpy calculation |
| `_tesub6` | Vapor enthalpy calculation |
| `_tesub7` | Linear congruential random number generator |
| `_tesub8` | Disturbance random walk updates |

## Random Number Generator

The Python backend replicates the Fortran linear congruential generator exactly:

```python
def _tesub7(self, i: int) -> float:
    """Linear congruential RNG matching Fortran."""
    self._g = (self._g * 9228907.0) % 4294967296.0
    if i >= 0:
        return self._g / 4294967296.0      # [0, 1)
    else:
        return 2.0 * self._g / 4294967296.0 - 1.0  # (-1, 1)
```

Setting the same seed produces identical random sequences:

```python
# Reproducible simulation
sim = TEPSimulator(backend='python', random_seed=1234)
```

## Statistical Validation

The Python backend was validated against reference data from the original Fortran simulator. All 22 continuous process measurements show <5% relative difference:

### Validation Results (Base Case, No Faults)

| XMEAS | Description | Python Mean | Reference Mean | Rel. Diff (%) |
|-------|-------------|-------------|----------------|---------------|
| 1 | A Feed (kscmh) | 0.257 | 0.251 | 2.37 |
| 2 | D Feed (kg/hr) | 3664.26 | 3663.54 | 0.02 |
| 3 | E Feed (kg/hr) | 4508.74 | 4509.29 | 0.01 |
| 4 | A+C Feed (kscmh) | 9.351 | 9.350 | 0.01 |
| 5 | Recycle Flow | 26.921 | 26.901 | 0.07 |
| 6 | Reactor Feed | 42.374 | 42.339 | 0.08 |
| 7 | Reactor Pressure | 2705.0 | 2705.0 | <0.01 |
| 8 | Reactor Level | 75.00 | 75.00 | <0.01 |
| 9 | Reactor Temp | 120.40 | 120.40 | <0.01 |
| 10 | Purge Rate | 0.337 | 0.337 | 0.02 |
| 11 | Sep Temp | 80.11 | 80.11 | <0.01 |
| 12 | Sep Level | 50.00 | 50.00 | <0.01 |
| 13 | Sep Pressure | 2633.7 | 2633.7 | <0.01 |
| 14 | Sep Underflow | 25.17 | 25.16 | 0.02 |
| 15 | Stripper Level | 50.00 | 50.00 | <0.01 |
| 16 | Stripper Pressure | 3102.2 | 3102.2 | <0.01 |
| 17 | Stripper Underflow | 22.95 | 22.95 | <0.01 |
| 18 | Stripper Temp | 65.73 | 65.73 | <0.01 |
| 19 | Stripper Steam | 230.3 | 230.3 | <0.01 |
| 20 | Compr Work | 341.4 | 341.4 | <0.01 |
| 21 | Reactor CW Out | 94.60 | 94.59 | 0.01 |
| 22 | Sep CW Out | 77.32 | 77.29 | 0.04 |

**All 22 continuous measurements within 5% of Fortran reference.**

## Performance Considerations

| Aspect | Fortran Backend | Python Backend |
|--------|-----------------|----------------|
| Speed | ~5-10x faster | Baseline |
| Accuracy | Original code | Statistical match |
| Dependencies | gfortran required | numpy only |
| Portability | Platform-specific build | Any Python environment |

For production use with performance requirements, prefer the Fortran backend when available. The Python backend is ideal for:
- Environments without Fortran compilers
- Educational and prototyping purposes
- Debugging and understanding process dynamics
- Cross-platform deployment

## Usage Examples

### Basic Simulation

```python
from tep import TEPSimulator

sim = TEPSimulator(backend='python')
result = sim.simulate(duration_hours=1.0)

print(f"Final reactor temp: {result.measurements[-1, 8]:.2f} C")
print(f"Final reactor pressure: {result.measurements[-1, 6]:.2f} kPa")
```

### With Fault Injection

```python
sim = TEPSimulator(backend='python')
sim.set_disturbance(1, True)  # IDV(1): A/C feed ratio step
result = sim.simulate(duration_hours=2.0)
```

### Direct Backend Access

```python
from tep import PythonTEProcess

# Low-level access
process = PythonTEProcess(random_seed=42)
process.teinit()

# Get state and derivatives
time = 0.0
process.tefunc(time)
print(f"State derivatives: {process.yp[:5]}")
```

## Testing

Run the Python backend tests:

```bash
pytest tests/test_python_backend.py -v
```

Tests include:
- Initialization correctness
- Random seed reproducibility
- Disturbance activation
- Thermodynamic calculations (Antoine, enthalpy, density)
- Comparison against Fortran backend (when available)
- Validation against reference trajectories
