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
pip install -e ".[dev]"        # Dev tools (pytest, matplotlib, dash)
pip install -e ".[web]"        # Dash web dashboard
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
```

## Architecture
- `tep/simulator.py` - High-level TEPSimulator interface (backend-agnostic)
- `tep/python_backend.py` - Pure Python implementation of TEP process
- `tep/fortran_backend.py` - f2py wrapper for Fortran TEINIT/TEFUNC (optional)
- `tep/constants.py` - Physical constants, initial states, variable names
- `tep/controllers.py` - PI controllers, decentralized control
- `tep/controller_base.py`, `controller_plugins.py` - Controller plugin system
- `tep/detector_base.py`, `detector_plugins.py` - Fault detection framework
- `tep/fault_base.py`, `fault_plugins.py` - Custom fault injection plugin system
- `tep/cli.py` - Batch simulation CLI (tep-sim)
- `tep/dashboard_dash.py` - Dash web dashboard (tep-web)
- `tep/_fortran/` - Compiled Fortran extension (optional)

## Key Patterns
- Python backend is default; Fortran is optional for ~5-10x speedup
- Fault detectors use plugin system with `@register_detector` decorator
- Controllers use plugin system with `@register_controller` decorator
- Fault plugins use plugin system with `@register_fault` decorator
- SimulationResult dataclass holds time, states, measurements, mvs arrays
- Both backends produce statistically similar results (<1.5% difference)

## Fault Plugin System
Custom faults can be defined as individual plugin classes:

```python
from tep import TEPSimulator, BaseFaultPlugin, FaultEffect, register_fault

# Use pre-defined IDV fault plugins
sim = TEPSimulator()
sim.initialize()
sim.add_fault('idv4_reactor_cw', activate_at=1.0, magnitude=0.5)

# Create custom fault plugin
@register_fault(name='my_fault', description='Custom temperature fault')
class MyFault(BaseFaultPlugin):
    name = 'my_fault'

    def apply(self, time, process_state):
        return [FaultEffect('reactor_cw_inlet_temp', 'additive', 5.0 * self.magnitude)]

    def reset(self):
        pass

sim.add_fault('my_fault', activate_at=2.0, magnitude=0.5)
result = sim.simulate(duration_hours=4.0)
```

Available fault effect variables:
- Feed compositions: `feed_comp_a`, `feed_comp_b`, `feed_comp_c`
- Temperatures: `feed_temp_d`, `feed_temp_c`, `reactor_cw_inlet_temp`, `condenser_cw_inlet_temp`
- Flow multipliers: `flow_a`, `flow_c`
- Valve positions: `valve_reactor_cw`, `valve_condenser_cw`

## Testing Notes
- Tests compare Python outputs against reference .dat files
- Random seed control enables reproducibility within each backend
- Use `pytest -xvs` for verbose output when debugging
