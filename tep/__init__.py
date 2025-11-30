"""
Tennessee Eastman Process (TEP) Simulator

This package provides a Python interface to the Tennessee Eastman Process
simulator with two backend options:

Backends:
    - 'python': Pure Python implementation (default, no compilation needed)
    - 'fortran': Original Fortran code via f2py (optional, ~5-10x faster)

Installation:
    Default (Python only):
        pip install tep

    With Fortran acceleration (requires gfortran):
        pip install tep --config-settings=setup-args=-Dfortran=enabled

Based on the original Fortran code by James J. Downs and Ernest F. Vogel,
with modifications by Evan L. Russell, Leo H. Chiang, and Richard D. Braatz.

This implementation is designed for:
- Process simulation and control research
- Fault detection and diagnosis studies
- Real-time dashboard integration
- Educational purposes

Example:
    >>> from tep import TEPSimulator
    >>> sim = TEPSimulator()  # Uses default backend (python unless fortran installed)
    >>> result = sim.simulate(duration_hours=1.0)

References:
    J.J. Downs and E.F. Vogel, "A plant-wide industrial process control problem,"
    Computers and Chemical Engineering, 17:245-255 (1993).

    E.L. Russell, L.H. Chiang, and R.D. Braatz. Data-driven Techniques for Fault
    Detection and Diagnosis in Chemical Processes, Springer-Verlag, London, 2000.
"""

from .simulator import TEPSimulator, ControlMode
from .python_backend import PythonTEProcess
from .controllers import PIController, DecentralizedController, ManualController

# Try to import Fortran backend (optional)
# The import of FortranTEProcess may succeed even if the extension isn't built,
# so we need to actually try using it to confirm availability.
try:
    from .fortran_backend import FortranTEProcess
    # Actually test that the extension is available
    from tep._fortran import teprob as _teprob
    _FORTRAN_AVAILABLE = True
except ImportError:
    _FORTRAN_AVAILABLE = False
    FortranTEProcess = None
from .controller_base import (
    BaseController,
    ControllerRegistry,
    CompositeController,
    register_controller,
)
from .detector_base import (
    BaseFaultDetector,
    FaultDetectorRegistry,
    DetectionResult,
    DetectionMetrics,
    register_detector,
)
from .constants import (
    NUM_STATES, NUM_MEASUREMENTS, NUM_MANIPULATED_VARS, NUM_DISTURBANCES,
    COMPONENT_NAMES, MEASUREMENT_NAMES, MANIPULATED_VAR_NAMES, DISTURBANCE_NAMES
)

# Import plugins to register them
from . import controller_plugins
from . import detector_plugins

__version__ = "2.3.0"
__author__ = "Fortran wrapper of Downs & Vogel (1993)"


def get_available_backends():
    """Get list of available simulation backends.

    Returns
    -------
    list
        List of available backend names. Always includes 'python'.
        Includes 'fortran' if the Fortran extension was compiled.
    """
    backends = ["python"]
    if _FORTRAN_AVAILABLE:
        backends.insert(0, "fortran")
    return backends


def get_default_backend():
    """Get the default backend name.

    Returns
    -------
    str
        Returns backend from TEP_BACKEND env var if set,
        otherwise 'fortran' if available, otherwise 'python'.
    """
    import os
    env_backend = os.environ.get('TEP_BACKEND', '').lower()
    if env_backend in ('python', 'fortran'):
        if env_backend == 'fortran' and not _FORTRAN_AVAILABLE:
            return 'python'
        return env_backend
    return "fortran" if _FORTRAN_AVAILABLE else "python"


def is_fortran_available():
    """Check if Fortran backend is available.

    Returns
    -------
    bool
        True if Fortran extension was compiled and can be imported.
    """
    return _FORTRAN_AVAILABLE


__all__ = [
    # Simulator
    "TEPSimulator",
    "ControlMode",
    # Backends
    "PythonTEProcess",
    "FortranTEProcess",
    # Controllers
    "PIController",
    "DecentralizedController",
    "ManualController",
    # Controller Plugin System
    "BaseController",
    "ControllerRegistry",
    "CompositeController",
    "register_controller",
    "controller_plugins",
    # Fault Detection System
    "BaseFaultDetector",
    "FaultDetectorRegistry",
    "DetectionResult",
    "DetectionMetrics",
    "register_detector",
    "detector_plugins",
    # Constants
    "NUM_STATES",
    "NUM_MEASUREMENTS",
    "NUM_MANIPULATED_VARS",
    "NUM_DISTURBANCES",
    "COMPONENT_NAMES",
    "MEASUREMENT_NAMES",
    "MANIPULATED_VAR_NAMES",
    "DISTURBANCE_NAMES",
    # Utilities
    "get_available_backends",
    "get_default_backend",
    "is_fortran_available",
    # Dashboard launchers
    "run_dashboard",
]


def run_dashboard():
    """Launch the interactive Dash web dashboard.

    Requires: pip install tep[web]
    """
    from .dashboard_dash import run_dashboard as _run_dashboard
    _run_dashboard()
