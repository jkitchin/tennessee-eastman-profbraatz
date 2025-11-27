"""
Constants and physical properties for the Tennessee Eastman Process.

This module contains all physical constants, component properties, equipment
parameters, and naming conventions used throughout the TEP simulation.

All values are taken directly from the original Fortran implementation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

# =============================================================================
# Process Dimensions
# =============================================================================
NUM_STATES = 50  # Number of differential equations
NUM_COMPONENTS = 8  # Chemical components A-H
NUM_STREAMS = 13  # Process streams
NUM_MEASUREMENTS = 41  # Total measurements (22 continuous + 19 sampled)
NUM_CONTINUOUS_MEAS = 22  # Continuous measurements
NUM_SAMPLED_MEAS = 19  # Sampled measurements
NUM_MANIPULATED_VARS = 12  # Manipulated variables
NUM_DISTURBANCES = 20  # Process disturbances

# =============================================================================
# Component Names and Properties
# =============================================================================
COMPONENT_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H"]

# Molecular weights (kg/kmol)
XMW = np.array([2.0, 25.4, 28.0, 32.0, 46.0, 48.0, 62.0, 76.0])

# Antoine coefficients for vapor pressure: ln(P) = AVP + BVP/(T + CVP)
# Components 1-3 (A, B, C) are non-condensable gases
AVP = np.array([0.0, 0.0, 0.0, 15.92, 16.35, 16.35, 16.43, 17.21])
BVP = np.array([0.0, 0.0, 0.0, -1444.0, -2114.0, -2114.0, -2748.0, -3318.0])
CVP = np.array([0.0, 0.0, 0.0, 259.0, 265.5, 265.5, 232.9, 249.6])

# Liquid density coefficients: rho = AD + BD*T + CD*T^2
AD = np.array([1.0, 1.0, 1.0, 23.3, 33.9, 32.8, 49.9, 50.5])
BD = np.array([0.0, 0.0, 0.0, -0.0700, -0.0957, -0.0995, -0.0191, -0.0541])
CD = np.array([0.0, 0.0, 0.0, -0.0002, -0.000152, -0.000233, -0.000425, -0.000150])

# Liquid enthalpy coefficients: H = AH*T + BH*T^2/2 + CH*T^3/3
AH = np.array([1.0e-6, 1.0e-6, 1.0e-6, 0.960e-6, 0.573e-6, 0.652e-6, 0.515e-6, 0.471e-6])
BH = np.array([0.0, 0.0, 0.0, 8.70e-9, 2.41e-9, 2.18e-9, 5.65e-10, 8.70e-10])
CH = np.array([0.0, 0.0, 0.0, 4.81e-11, 1.82e-11, 1.94e-11, 3.82e-12, 2.62e-12])

# Heat of vaporization coefficients
AV = np.array([1.0e-6, 1.0e-6, 1.0e-6, 86.7e-6, 160.0e-6, 160.0e-6, 225.0e-6, 209.0e-6])

# Gas heat capacity coefficients: Cp = AG + BG*T + CG*T^2
AG = np.array([3.411e-6, 0.3799e-6, 0.2491e-6, 0.3567e-6, 0.3463e-6, 0.3930e-6, 0.170e-6, 0.150e-6])
BG = np.array([7.18e-10, 1.08e-9, 1.36e-11, 8.51e-10, 8.96e-10, 1.02e-9, 0.0, 0.0])
CG = np.array([6.0e-13, -3.98e-13, -3.93e-14, -3.12e-13, -3.27e-13, -3.12e-13, 0.0, 0.0])

# =============================================================================
# Equipment Parameters
# =============================================================================
# Equipment volumes (m^3)
VTR = 1300.0  # Reactor total volume
VTS = 3500.0  # Separator total volume
VTC = 156.5  # Stripper (condenser) total volume
VTV = 5000.0  # Compressor volume

# Cooling water properties
HWR = 7060.0  # Reactor cooling water heat transfer coefficient
HWS = 11138.0  # Separator cooling water heat transfer coefficient

# Heat of reaction
HTR = np.array([0.06899381054, 0.05, 0.0])

# Gas constant
RG = 998.9

# Compressor parameters
CPFLMX = 280275.0  # Maximum compressor flow
CPPRMX = 1.3  # Maximum pressure ratio

# =============================================================================
# Valve Parameters
# =============================================================================
# Valve ranges (maximum flows)
VRNG = np.array([
    400.0,   # XMV(1) - D Feed Flow
    400.0,   # XMV(2) - E Feed Flow
    100.0,   # XMV(3) - A Feed Flow
    1500.0,  # XMV(4) - A and C Feed Flow
    0.0,     # XMV(5) - Compressor Recycle (not used in VRNG)
    0.0,     # XMV(6) - Purge Valve (not used in VRNG)
    1500.0,  # XMV(7) - Separator Pot Liquid Flow
    1000.0,  # XMV(8) - Stripper Liquid Product Flow
    0.03,    # XMV(9) - Stripper Steam Valve
    1000.0,  # XMV(10) - Reactor Cooling Water Flow
    1200.0,  # XMV(11) - Condenser Cooling Water Flow
    0.0,     # XMV(12) - Agitator Speed (not used in VRNG)
])

# Valve time constants (seconds, will be converted to hours)
VTAU_SECONDS = np.array([8.0, 8.0, 6.0, 9.0, 7.0, 5.0, 5.0, 5.0, 120.0, 5.0, 5.0, 5.0])
VTAU = VTAU_SECONDS / 3600.0  # Convert to hours

# Valve sticking threshold
VST_DEFAULT = 2.0

# =============================================================================
# Initial Steady-State Values
# =============================================================================
# These are the initial state values for steady-state operation
INITIAL_STATES = np.array([
    10.40491389,    # YY(1)
    4.363996017,    # YY(2)
    7.570059737,    # YY(3)
    0.4230042431,   # YY(4)
    24.15513437,    # YY(5)
    2.942597645,    # YY(6)
    154.3770655,    # YY(7)
    159.1865960,    # YY(8)
    2.808522723,    # YY(9)
    63.75581199,    # YY(10)
    26.74026066,    # YY(11)
    46.38532432,    # YY(12)
    0.2464521543,   # YY(13)
    15.20484404,    # YY(14)
    1.852266172,    # YY(15)
    52.44639459,    # YY(16)
    41.20394008,    # YY(17)
    0.5699317760,   # YY(18)
    0.4306056376,   # YY(19)
    7.9906200783e-03,  # YY(20)
    0.9056036089,   # YY(21)
    1.6054258216e-02,  # YY(22)
    0.7509759687,   # YY(23)
    8.8582855955e-02,  # YY(24)
    48.27726193,    # YY(25)
    39.38459028,    # YY(26)
    0.3755297257,   # YY(27)
    107.7562698,    # YY(28)
    29.77250546,    # YY(29)
    88.32481135,    # YY(30)
    23.03929507,    # YY(31)
    62.85848794,    # YY(32)
    5.546318688,    # YY(33)
    11.92244772,    # YY(34)
    5.555448243,    # YY(35)
    0.9218489762,   # YY(36)
    94.59927549,    # YY(37)
    77.29698353,    # YY(38)
    63.05263039,    # YY(39) - XMV(1)
    53.97970677,    # YY(40) - XMV(2)
    24.64355755,    # YY(41) - XMV(3)
    61.30192144,    # YY(42) - XMV(4)
    22.21000000,    # YY(43) - XMV(5)
    40.06374673,    # YY(44) - XMV(6)
    38.10034370,    # YY(45) - XMV(7)
    46.53415582,    # YY(46) - XMV(8)
    47.44573456,    # YY(47) - XMV(9)
    41.10581288,    # YY(48) - XMV(10)
    18.11349055,    # YY(49) - XMV(11)
    50.00000000,    # YY(50) - XMV(12)
])

# Initial feed stream compositions
XST_INITIAL = {
    1: np.array([0.0, 0.0001, 0.0, 0.9999, 0.0, 0.0, 0.0, 0.0]),  # Stream 1: A Feed (mostly D)
    2: np.array([0.0, 0.0, 0.0, 0.0, 0.9999, 0.0001, 0.0, 0.0]),  # Stream 2: D Feed (mostly E)
    3: np.array([0.9999, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Stream 3: E Feed (mostly A)
    4: np.array([0.4850, 0.0050, 0.5100, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Stream 4: A and C Feed
}

# Initial feed stream temperatures (deg C)
TST_INITIAL = {1: 45.0, 2: 45.0, 3: 45.0, 4: 45.0}

# Stripper separation factors (initial)
SFR_INITIAL = np.array([0.99500, 0.99100, 0.99000, 0.91600, 0.93600, 0.93800, 0.05800, 0.03010])

# =============================================================================
# Measurement Noise Standard Deviations
# =============================================================================
XNS = np.array([
    0.0012,   # XMEAS(1)
    18.000,   # XMEAS(2)
    22.000,   # XMEAS(3)
    0.0500,   # XMEAS(4)
    0.2000,   # XMEAS(5)
    0.2100,   # XMEAS(6)
    0.3000,   # XMEAS(7)
    0.5000,   # XMEAS(8)
    0.0100,   # XMEAS(9)
    0.0017,   # XMEAS(10)
    0.0100,   # XMEAS(11)
    1.0000,   # XMEAS(12)
    0.3000,   # XMEAS(13)
    0.1250,   # XMEAS(14)
    1.0000,   # XMEAS(15)
    0.3000,   # XMEAS(16)
    0.1150,   # XMEAS(17)
    0.0100,   # XMEAS(18)
    1.1500,   # XMEAS(19)
    0.2000,   # XMEAS(20)
    0.0100,   # XMEAS(21)
    0.0100,   # XMEAS(22)
    0.250,    # XMEAS(23)
    0.100,    # XMEAS(24)
    0.250,    # XMEAS(25)
    0.100,    # XMEAS(26)
    0.250,    # XMEAS(27)
    0.025,    # XMEAS(28)
    0.250,    # XMEAS(29)
    0.100,    # XMEAS(30)
    0.250,    # XMEAS(31)
    0.100,    # XMEAS(32)
    0.250,    # XMEAS(33)
    0.025,    # XMEAS(34)
    0.050,    # XMEAS(35)
    0.050,    # XMEAS(36)
    0.010,    # XMEAS(37)
    0.010,    # XMEAS(38)
    0.010,    # XMEAS(39)
    0.500,    # XMEAS(40)
    0.500,    # XMEAS(41)
])

# =============================================================================
# Disturbance Random Walk Parameters
# =============================================================================
@dataclass
class WalkParameters:
    """Parameters for disturbance random walk generation."""
    hspan: float  # Time span variation
    hzero: float  # Time span base value
    sspan: float  # Signal span variation
    szero: float  # Signal base value
    spspan: float = 0.0  # Derivative span variation


WALK_PARAMS = [
    WalkParameters(0.2, 0.5, 0.03, 0.485, 0.0),   # IDV(8) - A composition
    WalkParameters(0.7, 1.0, 0.003, 0.005, 0.0),  # IDV(8) - B composition
    WalkParameters(0.25, 0.5, 10.0, 45.0, 0.0),   # IDV(9) - D Feed temp
    WalkParameters(0.7, 1.0, 10.0, 45.0, 0.0),    # IDV(10) - C Feed temp
    WalkParameters(0.15, 0.25, 10.0, 35.0, 0.0),  # IDV(11) - Reactor CW temp
    WalkParameters(0.15, 0.25, 10.0, 40.0, 0.0),  # IDV(12) - Condenser CW temp
    WalkParameters(1.0, 2.0, 0.25, 1.0, 0.0),     # IDV(13) - Reaction kinetics 1
    WalkParameters(1.0, 2.0, 0.25, 1.0, 0.0),     # IDV(13) - Reaction kinetics 2
    WalkParameters(0.4, 0.5, 0.25, 0.0, 0.0),     # IDV(16)
    WalkParameters(1.5, 2.0, 0.0, 0.0, 0.0),      # IDV(17)
    WalkParameters(2.0, 3.0, 0.0, 0.0, 0.0),      # IDV(18)
    WalkParameters(1.5, 2.0, 0.0, 0.0, 0.0),      # IDV(20)
]

# =============================================================================
# Measurement Names and Units
# =============================================================================
MEASUREMENT_NAMES = [
    "A Feed (stream 1)",                    # XMEAS(1) - kscmh
    "D Feed (stream 2)",                    # XMEAS(2) - kg/hr
    "E Feed (stream 3)",                    # XMEAS(3) - kg/hr
    "A and C Feed (stream 4)",              # XMEAS(4) - kscmh
    "Recycle Flow (stream 8)",              # XMEAS(5) - kscmh
    "Reactor Feed Rate (stream 6)",         # XMEAS(6) - kscmh
    "Reactor Pressure",                     # XMEAS(7) - kPa gauge
    "Reactor Level",                        # XMEAS(8) - %
    "Reactor Temperature",                  # XMEAS(9) - deg C
    "Purge Rate (stream 9)",                # XMEAS(10) - kscmh
    "Product Sep Temp",                     # XMEAS(11) - deg C
    "Product Sep Level",                    # XMEAS(12) - %
    "Prod Sep Pressure",                    # XMEAS(13) - kPa gauge
    "Prod Sep Underflow (stream 10)",       # XMEAS(14) - m3/hr
    "Stripper Level",                       # XMEAS(15) - %
    "Stripper Pressure",                    # XMEAS(16) - kPa gauge
    "Stripper Underflow (stream 11)",       # XMEAS(17) - m3/hr
    "Stripper Temperature",                 # XMEAS(18) - deg C
    "Stripper Steam Flow",                  # XMEAS(19) - kg/hr
    "Compressor Work",                      # XMEAS(20) - kW
    "Reactor Cooling Water Outlet Temp",    # XMEAS(21) - deg C
    "Separator Cooling Water Outlet Temp",  # XMEAS(22) - deg C
    "Reactor Feed Component A",             # XMEAS(23) - mol%
    "Reactor Feed Component B",             # XMEAS(24) - mol%
    "Reactor Feed Component C",             # XMEAS(25) - mol%
    "Reactor Feed Component D",             # XMEAS(26) - mol%
    "Reactor Feed Component E",             # XMEAS(27) - mol%
    "Reactor Feed Component F",             # XMEAS(28) - mol%
    "Purge Gas Component A",                # XMEAS(29) - mol%
    "Purge Gas Component B",                # XMEAS(30) - mol%
    "Purge Gas Component C",                # XMEAS(31) - mol%
    "Purge Gas Component D",                # XMEAS(32) - mol%
    "Purge Gas Component E",                # XMEAS(33) - mol%
    "Purge Gas Component F",                # XMEAS(34) - mol%
    "Purge Gas Component G",                # XMEAS(35) - mol%
    "Purge Gas Component H",                # XMEAS(36) - mol%
    "Product Component D",                  # XMEAS(37) - mol%
    "Product Component E",                  # XMEAS(38) - mol%
    "Product Component F",                  # XMEAS(39) - mol%
    "Product Component G",                  # XMEAS(40) - mol%
    "Product Component H",                  # XMEAS(41) - mol%
]

MEASUREMENT_UNITS = [
    "kscmh", "kg/hr", "kg/hr", "kscmh", "kscmh", "kscmh", "kPa", "%",
    "deg C", "kscmh", "deg C", "%", "kPa", "m3/hr", "%", "kPa",
    "m3/hr", "deg C", "kg/hr", "kW", "deg C", "deg C",
    "mol%", "mol%", "mol%", "mol%", "mol%", "mol%",
    "mol%", "mol%", "mol%", "mol%", "mol%", "mol%", "mol%", "mol%",
    "mol%", "mol%", "mol%", "mol%", "mol%",
]

# =============================================================================
# Manipulated Variable Names and Descriptions
# =============================================================================
MANIPULATED_VAR_NAMES = [
    "D Feed Flow (stream 2)",           # XMV(1)
    "E Feed Flow (stream 3)",           # XMV(2)
    "A Feed Flow (stream 1)",           # XMV(3)
    "A and C Feed Flow (stream 4)",     # XMV(4)
    "Compressor Recycle Valve",         # XMV(5)
    "Purge Valve (stream 9)",           # XMV(6)
    "Separator Pot Liquid Flow",        # XMV(7)
    "Stripper Liquid Product Flow",     # XMV(8)
    "Stripper Steam Valve",             # XMV(9)
    "Reactor Cooling Water Flow",       # XMV(10)
    "Condenser Cooling Water Flow",     # XMV(11)
    "Agitator Speed",                   # XMV(12)
]

# =============================================================================
# Disturbance Names and Descriptions
# =============================================================================
DISTURBANCE_NAMES = [
    "A/C Feed Ratio, B Composition Constant (Step)",       # IDV(1)
    "B Composition, A/C Ratio Constant (Step)",            # IDV(2)
    "D Feed Temperature (Step)",                           # IDV(3)
    "Reactor Cooling Water Inlet Temperature (Step)",      # IDV(4)
    "Condenser Cooling Water Inlet Temperature (Step)",    # IDV(5)
    "A Feed Loss (Step)",                                  # IDV(6)
    "C Header Pressure Loss (Step)",                       # IDV(7)
    "A, B, C Feed Composition (Random Variation)",         # IDV(8)
    "D Feed Temperature (Random Variation)",               # IDV(9)
    "C Feed Temperature (Random Variation)",               # IDV(10)
    "Reactor Cooling Water Inlet Temp (Random Variation)", # IDV(11)
    "Condenser Cooling Water Inlet Temp (Random Var)",     # IDV(12)
    "Reaction Kinetics (Slow Drift)",                      # IDV(13)
    "Reactor Cooling Water Valve (Sticking)",              # IDV(14)
    "Condenser Cooling Water Valve (Sticking)",            # IDV(15)
    "Unknown",                                             # IDV(16)
    "Unknown",                                             # IDV(17)
    "Unknown",                                             # IDV(18)
    "Unknown",                                             # IDV(19)
    "Unknown",                                             # IDV(20)
]

# =============================================================================
# Safety Limits
# =============================================================================
@dataclass
class SafetyLimits:
    """Process safety limits that trigger shutdown."""
    reactor_pressure_max: float = 3000.0    # kPa
    reactor_level_max: float = 24.0         # %
    reactor_level_min: float = 2.0          # %
    reactor_temp_max: float = 175.0         # deg C
    separator_level_max: float = 12.0       # %
    separator_level_min: float = 1.0        # %
    stripper_level_max: float = 8.0         # %
    stripper_level_min: float = 1.0         # %


SAFETY_LIMITS = SafetyLimits()

# =============================================================================
# Default Random Seed
# =============================================================================
DEFAULT_RANDOM_SEED = 4651207995
