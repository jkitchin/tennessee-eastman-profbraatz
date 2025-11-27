"""
Tennessee Eastman Process (TEP) Simulator - Pure Python Implementation

A faithful translation of the original Fortran code by James J. Downs and Ernest F. Vogel,
with modifications by Evan L. Russell, Leo H. Chiang, and Richard D. Braatz.

This modular implementation is designed for:
- Process simulation and control research
- Fault detection and diagnosis studies
- Real-time dashboard integration
- Educational purposes

References:
    J.J. Downs and E.F. Vogel, "A plant-wide industrial process control problem,"
    Computers and Chemical Engineering, 17:245-255 (1993).

    E.L. Russell, L.H. Chiang, and R.D. Braatz. Data-driven Techniques for Fault
    Detection and Diagnosis in Chemical Processes, Springer-Verlag, London, 2000.
"""

from .simulator import TEPSimulator
from .process import TEProcess
from .controllers import PIController, DecentralizedController
from .constants import (
    NUM_STATES, NUM_MEASUREMENTS, NUM_MANIPULATED_VARS, NUM_DISTURBANCES,
    COMPONENT_NAMES, MEASUREMENT_NAMES, MANIPULATED_VAR_NAMES, DISTURBANCE_NAMES
)

__version__ = "1.1.0"
__author__ = "Python translation of Downs & Vogel (1993)"

__all__ = [
    "TEPSimulator",
    "TEProcess",
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
]


def run_dashboard():
    """Launch the interactive GUI dashboard."""
    from .dashboard import TEPDashboard
    app = TEPDashboard()
    app.run()
