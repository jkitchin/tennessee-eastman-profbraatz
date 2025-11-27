"""
Tennessee Eastman Process (TEP) Simulator

This package provides a Python interface to the Tennessee Eastman Process
simulator using the original Fortran code via f2py for exact reproduction
of simulation results.

Based on the original Fortran code by James J. Downs and Ernest F. Vogel,
with modifications by Evan L. Russell, Leo H. Chiang, and Richard D. Braatz.

This implementation is designed for:
- Process simulation and control research
- Fault detection and diagnosis studies
- Real-time dashboard integration
- Educational purposes

Requirements:
    - Fortran compiler (gfortran) during installation

References:
    J.J. Downs and E.F. Vogel, "A plant-wide industrial process control problem,"
    Computers and Chemical Engineering, 17:245-255 (1993).

    E.L. Russell, L.H. Chiang, and R.D. Braatz. Data-driven Techniques for Fault
    Detection and Diagnosis in Chemical Processes, Springer-Verlag, London, 2000.
"""

from .simulator import TEPSimulator
from .fortran_backend import FortranTEProcess
from .controllers import PIController, DecentralizedController
from .constants import (
    NUM_STATES, NUM_MEASUREMENTS, NUM_MANIPULATED_VARS, NUM_DISTURBANCES,
    COMPONENT_NAMES, MEASUREMENT_NAMES, MANIPULATED_VAR_NAMES, DISTURBANCE_NAMES
)

__version__ = "2.0.0"
__author__ = "Fortran wrapper of Downs & Vogel (1993)"


def get_available_backends():
    """Get list of available simulation backends.

    Returns
    -------
    list
        List containing 'fortran' (only backend available).
    """
    return ["fortran"]


def get_default_backend():
    """Get the default backend name.

    Returns
    -------
    str
        Always returns 'fortran'.
    """
    return "fortran"


__all__ = [
    "TEPSimulator",
    "FortranTEProcess",
    "PIController",
    "DecentralizedController",
    "NUM_STATES",
    "NUM_MEASUREMENTS",
    "NUM_MANIPULATED_VARS",
    "NUM_DISTURBANCES",
    "COMPONENT_NAMES",
    "MEASUREMENT_NAMES",
    "MANIPULATED_VAR_NAMES",
    "DISTURBANCE_NAMES",
    "get_available_backends",
    "get_default_backend",
]


def run_dashboard():
    """Launch the interactive GUI dashboard."""
    from .dashboard import TEPDashboard
    app = TEPDashboard()
    app.run()
