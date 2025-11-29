# Tennessee Eastman Process Simulator

## Project Overview
Python interface to the Tennessee Eastman Process (TEP) simulator using original Fortran code via f2py. The TEP is an industrial chemical process benchmark for control systems and fault detection research.

## Build & Install
```bash
# Requires gfortran (brew install gcc on macOS)
pip install -e .           # Basic install
pip install -e ".[dev]"    # With dev dependencies (pytest, matplotlib, dash)
pip install -e ".[web]"    # With web dashboard support
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
tep-web                                             # Launch web dashboard
tep-web --no-browser --port 8080                   # Dashboard with options
```

## Architecture
- `tep/simulator.py` - High-level TEPSimulator interface
- `tep/fortran_backend.py` - f2py wrapper for Fortran TEINIT/TEFUNC
- `tep/constants.py` - Physical constants, initial states, variable names
- `tep/controllers.py` - PI controllers, decentralized control
- `tep/controller_base.py`, `controller_plugins.py` - Controller plugin system
- `tep/detector_base.py`, `detector_plugins.py` - Fault detection framework
- `tep/cli.py` - Batch simulation CLI (tep-sim)
- `tep/dashboard_dash.py` - Web dashboard (tep-web)
- `tep/_fortran/` - Compiled Fortran extension

## Key Patterns
- Fortran backend is required; pure Python fallback was removed
- Fault detectors use plugin system with `@register_detector` decorator
- Controllers use plugin system with `@register_controller` decorator
- SimulationResult dataclass holds time, states, measurements, mvs arrays

## Testing Notes
- Tests compare Python outputs against reference .dat files from original Fortran
- Fortran random seed behavior means exact reproducibility requires seed control
- Use `pytest -xvs` for verbose output when debugging
