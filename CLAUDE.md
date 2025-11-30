# Tennessee Eastman Process Simulator

## Project Overview
Python interface to the Tennessee Eastman Process (TEP) simulator. The TEP is an industrial chemical process benchmark for control systems and fault detection research. Includes a pure Python implementation, optional Fortran acceleration, and a JAX backend for autodiff/GPU support.

## Build & Install
```bash
# Default install (Python backend only, no compiler needed)
pip install -e .

# With Fortran acceleration (requires gfortran)
pip install -e . --config-settings=setup-args=-Dfortran=enabled

# With JAX support (for autodiff, JIT, vmap, GPU)
pip install -e ".[jax]"

# With optional dependencies
pip install -e ".[dev]"        # Dev tools (pytest, matplotlib, dash)
pip install -e ".[web]"        # Dash web dashboard
```

## Backend Selection
```python
from tep import TEPSimulator, get_available_backends, is_jax_available

# Check available backends
print(get_available_backends())  # ['python'], ['fortran', 'python'], or ['jax', 'python']

# Use specific backend
sim = TEPSimulator(backend='python')   # Pure Python (always available)
sim = TEPSimulator(backend='fortran')  # Fortran (if installed)
sim = TEPSimulator(backend='jax')      # JAX (if installed)
```

## JAX Backend Usage
The JAX backend enables JIT compilation, automatic differentiation, and batch simulations:

```python
from tep.jax_backend import JaxTEProcess
import jax

# Initialize
process = JaxTEProcess()
key = jax.random.PRNGKey(1234)
state, key = process.initialize(key)

# JIT-compiled simulation
step_jit = jax.jit(process.step)
for _ in range(3600):
    state, key = step_jit(state, key)

# Batch simulation with vmap
def simulate(key):
    state, key = process.initialize(key)
    for _ in range(100):
        state, key = process.step(state, key)
    return state.yy

keys = jax.random.split(key, 16)
batch_results = jax.vmap(simulate)(keys)  # Run 16 simulations in parallel

# Automatic differentiation
grad_fn = jax.grad(lambda z: process._tesub1(z, 100.0, 0))
grad_h = grad_fn(state.reactor.xlr)  # Gradient of enthalpy w.r.t. composition
```

## Performance Benchmarks (CPU)
| Backend | Single Sim | Batch=16 | Notes |
|---------|------------|----------|-------|
| Python | ~1500 steps/sec | N/A | Baseline |
| JAX (JIT) | ~1100 steps/sec | ~10700 steps/sec | 7x throughput with batching |
| Fortran | ~8000 steps/sec | N/A | 5x single-sim speedup |

JAX excels at batch simulations via `vmap`. For single simulations on CPU, Python or Fortran may be faster.

## Test Commands
```bash
pytest                           # Run all tests
pytest tests/test_simulator.py   # Run specific test file
pytest tests/test_jax_backend.py # Run JAX backend tests
pytest -xvs tests/test_simulator.py::test_name  # Run single test with output
```

## CLI Commands
```bash
tep-sim --duration 2 --faults 1 --output data.dat  # Batch simulation
tep-web                                             # Launch Dash dashboard
```

## Architecture
- `tep/simulator.py` - High-level TEPSimulator interface (backend-agnostic)
- `tep/python_backend.py` - Pure Python implementation of TEP process
- `tep/fortran_backend.py` - f2py wrapper for Fortran TEINIT/TEFUNC (optional)
- `tep/jax_backend.py` - JAX implementation with JIT/autodiff/vmap support (optional)
- `tep/constants.py` - Physical constants, initial states, variable names
- `tep/controllers.py` - PI controllers, decentralized control
- `tep/controller_base.py`, `controller_plugins.py` - Controller plugin system
- `tep/detector_base.py`, `detector_plugins.py` - Fault detection framework
- `tep/cli.py` - Batch simulation CLI (tep-sim)
- `tep/dashboard_dash.py` - Dash web dashboard (tep-web)
- `tep/_fortran/` - Compiled Fortran extension (optional)

## Key Patterns
- Python backend is default; Fortran is optional for ~5-10x speedup
- JAX backend enables autodiff, JIT compilation, and vmap for batch simulations
- Fault detectors use plugin system with `@register_detector` decorator
- Controllers use plugin system with `@register_controller` decorator
- SimulationResult dataclass holds time, states, measurements, mvs arrays
- All backends produce statistically similar results (<1.5% difference)

## Testing Notes
- Tests compare Python outputs against reference .dat files
- Random seed control enables reproducibility within each backend
- JAX backend uses JAX PRNGKey for reproducible random numbers
- Use `pytest -xvs` for verbose output when debugging
