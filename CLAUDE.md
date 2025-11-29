# Tennessee Eastman Process Simulator

## Project Overview
Python interface to the Tennessee Eastman Process (TEP) simulator. The TEP is an industrial chemical process benchmark for control systems and fault detection research. Includes both a pure Python implementation and optional Fortran acceleration.

## Build & Install
```bash
# Default install (Python backend only, no compiler needed)
pip install -e .

# With Fortran acceleration (requires gfortran)
pip install -e . --config-settings=setup-args=-Dfortran=enabled

# With optional dependencies
pip install -e ".[dev]"        # Dev tools (pytest, matplotlib, dash, streamlit)
pip install -e ".[web]"        # Dash web dashboard
pip install -e ".[streamlit]"  # Streamlit web dashboard
```

## Backend Selection
```python
from tep import TEPSimulator, get_available_backends, is_fortran_available

# Check available backends
print(get_available_backends())  # ['python'] or ['fortran', 'python']

# Use specific backend
sim = TEPSimulator(backend='python')   # Pure Python (always available)
sim = TEPSimulator(backend='fortran')  # Fortran (if installed)
```

## Test Commands
```bash
pytest                           # Run all tests
pytest tests/test_simulator.py   # Run specific test file
pytest -xvs tests/test_simulator.py::test_name  # Run single test with output
```

## CLI Commands
```bash
tep-sim --duration 2 --faults 1 --output data.dat  # Batch simulation
tep-web                                             # Launch Dash dashboard
tep-streamlit                                       # Launch Streamlit dashboard
```

## Architecture
- `tep/simulator.py` - High-level TEPSimulator interface (backend-agnostic)
- `tep/python_backend.py` - Pure Python implementation of TEP process
- `tep/fortran_backend.py` - f2py wrapper for Fortran TEINIT/TEFUNC (optional)
- `tep/constants.py` - Physical constants, initial states, variable names
- `tep/controllers.py` - PI controllers, decentralized control
- `tep/controller_base.py`, `controller_plugins.py` - Controller plugin system
- `tep/detector_base.py`, `detector_plugins.py` - Fault detection framework
- `tep/cli.py` - Batch simulation CLI (tep-sim)
- `tep/dashboard_dash.py` - Dash web dashboard (tep-web)
- `tep/dashboard_streamlit.py` - Streamlit web dashboard (tep-streamlit)
- `tep/_fortran/` - Compiled Fortran extension (optional)

## Key Patterns
- Python backend is default; Fortran is optional for ~5-10x speedup
- Fault detectors use plugin system with `@register_detector` decorator
- Controllers use plugin system with `@register_controller` decorator
- SimulationResult dataclass holds time, states, measurements, mvs arrays
- Both backends produce statistically similar results (<1.5% difference)

## Testing Notes
- Tests compare Python outputs against reference .dat files
- Random seed control enables reproducibility within each backend
- Use `pytest -xvs` for verbose output when debugging
