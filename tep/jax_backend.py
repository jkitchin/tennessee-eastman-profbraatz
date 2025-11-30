"""JAX backend for Tennessee Eastman Process simulation.

This module provides a JAX-based implementation of the TEP simulator,
enabling:
- JIT compilation for faster execution
- Automatic differentiation for gradient-based optimization
- Vectorization via vmap for batch simulations
- GPU/TPU acceleration

The implementation mirrors the Python backend interface but uses
functional programming patterns required by JAX.

Example:
    >>> from tep.jax_backend import JaxTEProcess
    >>> import jax
    >>>
    >>> key = jax.random.PRNGKey(1234)
    >>> process = JaxTEProcess(random_key=key)
    >>> state = process.initialize()
    >>>
    >>> # Run simulation
    >>> for _ in range(3600):
    ...     state, key = process.step(state, key)
    >>>
    >>> xmeas = process.get_xmeas(state)
"""

from __future__ import annotations

import os

# Force JAX to use CPU backend - Apple Metal GPU support is experimental and incomplete
# Users can override with JAX_PLATFORMS environment variable if needed
if 'JAX_PLATFORMS' not in os.environ:
    os.environ['JAX_PLATFORMS'] = 'cpu'

import jax

# Note: 64-bit floats are not supported on all platforms (e.g., Apple Metal GPU)
# Using 32-bit floats for broader compatibility. Set JAX_ENABLE_X64=true for 64-bit on CPU.
# jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax
from typing import NamedTuple, Optional, Tuple
import numpy as np

# Try to import constants from the tep package
try:
    from . import constants
except ImportError:
    from tep import constants


# =============================================================================
# JAX-Compatible State Structures (NamedTuples are automatic pytrees)
# =============================================================================

class ConstBlock(NamedTuple):
    """Physical constants (immutable).

    Contains thermodynamic properties for the 8 chemical components.
    This is created once and never modified.
    """
    xmw: jnp.ndarray    # Molecular weights (kg/kmol)
    avp: jnp.ndarray    # Antoine vapor pressure coefficients
    bvp: jnp.ndarray
    cvp: jnp.ndarray
    ad: jnp.ndarray     # Liquid density coefficients
    bd: jnp.ndarray
    cd: jnp.ndarray
    ah: jnp.ndarray     # Liquid enthalpy coefficients
    bh: jnp.ndarray
    ch: jnp.ndarray
    av: jnp.ndarray     # Heat of vaporization
    ag: jnp.ndarray     # Gas heat capacity coefficients
    bg: jnp.ndarray
    cg: jnp.ndarray


class ReactorState(NamedTuple):
    """Reactor state variables."""
    uclr: jnp.ndarray   # Liquid component holdup (8,)
    ucvr: jnp.ndarray   # Vapor component holdup (8,)
    utlr: float         # Total liquid holdup
    utvr: float         # Total vapor holdup
    xlr: jnp.ndarray    # Liquid mole fractions (8,)
    xvr: jnp.ndarray    # Vapor mole fractions (8,)
    etr: float          # Total energy
    esr: float          # Specific energy
    tcr: float          # Temperature (C)
    tkr: float          # Temperature (K)
    dlr: float          # Liquid density
    vlr: float          # Liquid volume
    vvr: float          # Vapor volume
    vtr: float          # Total volume
    ptr: float          # Total pressure
    ppr: jnp.ndarray    # Partial pressures (8,)
    crxr: jnp.ndarray   # Reaction rates by component (8,)
    rr: jnp.ndarray     # Reaction rates (4,)
    rh: float           # Heat of reaction
    fwr: float          # Cooling water flow
    twr: float          # Cooling water temperature
    qur: float          # Heat transfer
    hwr: float          # Heat transfer coefficient
    uar: float          # Overall heat transfer


class SeparatorState(NamedTuple):
    """Separator state variables."""
    ucls: jnp.ndarray   # Liquid component holdup (8,)
    ucvs: jnp.ndarray   # Vapor component holdup (8,)
    utls: float         # Total liquid holdup
    utvs: float         # Total vapor holdup
    xls: jnp.ndarray    # Liquid mole fractions (8,)
    xvs: jnp.ndarray    # Vapor mole fractions (8,)
    ets: float          # Total energy
    ess: float          # Specific energy
    tcs: float          # Temperature (C)
    tks: float          # Temperature (K)
    dls: float          # Liquid density
    vls: float          # Liquid volume
    vvs: float          # Vapor volume
    vts: float          # Total volume
    pts: float          # Total pressure
    pps: jnp.ndarray    # Partial pressures (8,)
    fws: float          # Cooling water flow
    tws: float          # Cooling water temperature
    qus: float          # Heat transfer
    hws: float          # Heat transfer coefficient


class StripperState(NamedTuple):
    """Stripper/Condenser state variables."""
    uclc: jnp.ndarray   # Liquid component holdup (8,)
    utlc: float         # Total liquid holdup
    xlc: jnp.ndarray    # Liquid mole fractions (8,)
    etc: float          # Total energy
    esc: float          # Specific energy
    tcc: float          # Temperature (C)
    dlc: float          # Liquid density
    vlc: float          # Liquid volume
    vtc: float          # Total volume
    quc: float          # Heat transfer


class CompressorState(NamedTuple):
    """Compressor state variables."""
    ucvv: jnp.ndarray   # Vapor component holdup (8,)
    utvv: float         # Total vapor holdup
    xvv: jnp.ndarray    # Vapor mole fractions (8,)
    etv: float          # Total energy
    esv: float          # Specific energy
    tcv: float          # Temperature (C)
    tkv: float          # Temperature (K)
    vtv: float          # Volume
    ptv: float          # Pressure
    cpflmx: float       # Max compressor flow
    cpprmx: float       # Max pressure ratio
    cpdh: float         # Compressor delta enthalpy


class ValveState(NamedTuple):
    """Valve state variables."""
    vcv: jnp.ndarray    # Valve command values (12,)
    vrng: jnp.ndarray   # Valve ranges (12,)
    vtau: jnp.ndarray   # Time constants (12,)
    vst: jnp.ndarray    # Sticking threshold (12,)
    ivst: jnp.ndarray   # Sticking flags (12,)


class StreamState(NamedTuple):
    """Stream flow and composition state."""
    ftm: jnp.ndarray    # Total molar flows (13,)
    fcm: jnp.ndarray    # Component molar flows (8, 13)
    xst: jnp.ndarray    # Stream compositions (8, 13)
    xmws: jnp.ndarray   # Stream molecular weights (13,)
    hst: jnp.ndarray    # Stream enthalpies (13,)
    tst: jnp.ndarray    # Stream temperatures (13,)
    sfr: jnp.ndarray    # Separation factors (8,)


class WalkState(NamedTuple):
    """Disturbance random walk state."""
    adist: jnp.ndarray  # Spline coefficients (12,)
    bdist: jnp.ndarray
    cdist: jnp.ndarray
    ddist: jnp.ndarray
    tlast: jnp.ndarray  # Last update times (12,)
    tnext: jnp.ndarray  # Next update times (12,)
    hspan: jnp.ndarray  # Time span variation (12,)
    hzero: jnp.ndarray  # Time span base (12,)
    sspan: jnp.ndarray  # Signal span variation (12,)
    szero: jnp.ndarray  # Signal base (12,)
    spspan: jnp.ndarray # Derivative span (12,)
    idvwlk: jnp.ndarray # Disturbance walk flags (12,)


class MeasurementState(NamedTuple):
    """Measurement and sampling state."""
    xmeas: jnp.ndarray  # Measurements (41,)
    xdel: jnp.ndarray   # Delayed measurements (41,)
    xns: jnp.ndarray    # Noise std devs (41,)
    tgas: float         # Gas sampling time
    tprod: float        # Product sampling time


class TEPState(NamedTuple):
    """Complete TEP process state.

    This is the main state object passed through all JAX functions.
    It contains all sub-states needed for the simulation.
    """
    # Integration state
    yy: jnp.ndarray         # State vector (50,)
    yp: jnp.ndarray         # Derivative vector (50,)
    time: float             # Current time (hours)

    # Process sub-states
    reactor: ReactorState
    separator: SeparatorState
    stripper: StripperState
    compressor: CompressorState
    valves: ValveState
    streams: StreamState
    walks: WalkState
    measurements: MeasurementState

    # Control inputs
    xmv: jnp.ndarray        # Manipulated variables (12,)
    idv: jnp.ndarray        # Disturbance vector (20,)

    # Cooling water inlet temps
    tcwr: float
    tcws: float

    # Heat of reaction coefficients
    htr: jnp.ndarray        # (3,)
    agsp: float             # Agitator speed


# =============================================================================
# Helper Functions for Creating Default States
# =============================================================================

def create_const_block() -> ConstBlock:
    """Create the physical constants block."""
    return ConstBlock(
        xmw=jnp.array([2.0, 25.4, 28.0, 32.0, 46.0, 48.0, 62.0, 76.0]),
        avp=jnp.array([0.0, 0.0, 0.0, 15.92, 16.35, 16.35, 16.43, 17.21]),
        bvp=jnp.array([0.0, 0.0, 0.0, -1444.0, -2114.0, -2114.0, -2748.0, -3318.0]),
        cvp=jnp.array([0.0, 0.0, 0.0, 259.0, 265.5, 265.5, 232.9, 249.6]),
        ad=jnp.array([1.0, 1.0, 1.0, 23.3, 33.9, 32.8, 49.9, 50.5]),
        bd=jnp.array([0.0, 0.0, 0.0, -0.0700, -0.0957, -0.0995, -0.0191, -0.0541]),
        cd=jnp.array([0.0, 0.0, 0.0, -0.0002, -0.000152, -0.000233, -0.000425, -0.000150]),
        ah=jnp.array([1.0e-6, 1.0e-6, 1.0e-6, 0.960e-6, 0.573e-6, 0.652e-6, 0.515e-6, 0.471e-6]),
        bh=jnp.array([0.0, 0.0, 0.0, 8.70e-9, 2.41e-9, 2.18e-9, 5.65e-10, 8.70e-10]),
        ch=jnp.array([0.0, 0.0, 0.0, 4.81e-11, 1.82e-11, 1.94e-11, 3.82e-12, 2.62e-12]),
        av=jnp.array([1.0e-6, 1.0e-6, 1.0e-6, 86.7e-6, 160.0e-6, 160.0e-6, 225.0e-6, 209.0e-6]),
        ag=jnp.array([3.411e-6, 0.3799e-6, 0.2491e-6, 0.3567e-6, 0.3463e-6, 0.3930e-6, 0.170e-6, 0.150e-6]),
        bg=jnp.array([7.18e-10, 1.08e-9, 1.36e-11, 8.51e-10, 8.96e-10, 1.02e-9, 0.0, 0.0]),
        cg=jnp.array([6.0e-13, -3.98e-13, -3.93e-14, -3.12e-13, -3.27e-13, -3.12e-13, 0.0, 0.0]),
    )


def create_initial_reactor_state() -> ReactorState:
    """Create initial reactor state."""
    return ReactorState(
        uclr=jnp.zeros(8),
        ucvr=jnp.zeros(8),
        utlr=0.0,
        utvr=0.0,
        xlr=jnp.zeros(8),
        xvr=jnp.zeros(8),
        etr=0.0,
        esr=0.0,
        tcr=120.0,  # Initial guess
        tkr=393.15,
        dlr=0.0,
        vlr=0.0,
        vvr=0.0,
        vtr=1300.0,
        ptr=0.0,
        ppr=jnp.zeros(8),
        crxr=jnp.zeros(8),
        rr=jnp.zeros(4),
        rh=0.0,
        fwr=0.0,
        twr=0.0,
        qur=0.0,
        hwr=7060.0,
        uar=0.0,
    )


def create_initial_separator_state() -> SeparatorState:
    """Create initial separator state."""
    return SeparatorState(
        ucls=jnp.zeros(8),
        ucvs=jnp.zeros(8),
        utls=0.0,
        utvs=0.0,
        xls=jnp.zeros(8),
        xvs=jnp.zeros(8),
        ets=0.0,
        ess=0.0,
        tcs=80.0,  # Initial guess
        tks=353.15,
        dls=0.0,
        vls=0.0,
        vvs=0.0,
        vts=3500.0,
        pts=0.0,
        pps=jnp.zeros(8),
        fws=0.0,
        tws=0.0,
        qus=0.0,
        hws=11138.0,
    )


def create_initial_stripper_state() -> StripperState:
    """Create initial stripper state."""
    return StripperState(
        uclc=jnp.zeros(8),
        utlc=0.0,
        xlc=jnp.zeros(8),
        etc=0.0,
        esc=0.0,
        tcc=65.0,  # Initial guess
        dlc=0.0,
        vlc=0.0,
        vtc=156.5,
        quc=0.0,
    )


def create_initial_compressor_state() -> CompressorState:
    """Create initial compressor state."""
    return CompressorState(
        ucvv=jnp.zeros(8),
        utvv=0.0,
        xvv=jnp.zeros(8),
        etv=0.0,
        esv=0.0,
        tcv=100.0,  # Initial guess
        tkv=373.15,
        vtv=5000.0,
        ptv=0.0,
        cpflmx=280275.0,
        cpprmx=1.3,
        cpdh=0.0,
    )


def create_initial_valve_state() -> ValveState:
    """Create initial valve state."""
    return ValveState(
        vcv=jnp.zeros(12),
        vrng=jnp.array([
            400.0, 400.0, 100.0, 1500.0, 0.0, 0.0,
            1500.0, 1000.0, 0.03, 1000.0, 1200.0, 0.0
        ]),
        vtau=jnp.array([8.0, 8.0, 6.0, 9.0, 7.0, 5.0, 5.0, 5.0, 120.0, 5.0, 5.0, 5.0]) / 3600.0,
        vst=jnp.full(12, 2.0),
        ivst=jnp.zeros(12, dtype=jnp.int32),
    )


def create_initial_stream_state() -> StreamState:
    """Create initial stream state."""
    # Initialize feed stream compositions
    xst = jnp.zeros((8, 13))
    # Stream 1: A Feed (mostly D)
    xst = xst.at[0, 0].set(0.0)
    xst = xst.at[1, 0].set(0.0001)
    xst = xst.at[2, 0].set(0.0)
    xst = xst.at[3, 0].set(0.9999)
    # Stream 2: D Feed (mostly E)
    xst = xst.at[4, 1].set(0.9999)
    xst = xst.at[5, 1].set(0.0001)
    # Stream 3: E Feed (mostly A)
    xst = xst.at[0, 2].set(0.9999)
    xst = xst.at[1, 2].set(0.0001)
    # Stream 4: A and C Feed
    xst = xst.at[0, 3].set(0.4850)
    xst = xst.at[1, 3].set(0.0050)
    xst = xst.at[2, 3].set(0.5100)

    tst = jnp.zeros(13)
    tst = tst.at[0].set(45.0)
    tst = tst.at[1].set(45.0)
    tst = tst.at[2].set(45.0)
    tst = tst.at[3].set(45.0)

    return StreamState(
        ftm=jnp.zeros(13),
        fcm=jnp.zeros((8, 13)),
        xst=xst,
        xmws=jnp.zeros(13),
        hst=jnp.zeros(13),
        tst=tst,
        sfr=jnp.array([0.99500, 0.99100, 0.99000, 0.91600, 0.93600, 0.93800, 0.05800, 0.03010]),
    )


def create_initial_walk_state() -> WalkState:
    """Create initial walk state for disturbances."""
    return WalkState(
        adist=jnp.array([0.485, 0.005, 45.0, 45.0, 35.0, 40.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        bdist=jnp.zeros(12),
        cdist=jnp.zeros(12),
        ddist=jnp.zeros(12),
        tlast=jnp.zeros(12),
        tnext=jnp.full(12, 0.1),
        hspan=jnp.array([0.2, 0.7, 0.25, 0.7, 0.15, 0.15, 1.0, 1.0, 0.4, 1.5, 2.0, 1.5]),
        hzero=jnp.array([0.5, 1.0, 0.5, 1.0, 0.25, 0.25, 2.0, 2.0, 0.5, 2.0, 3.0, 2.0]),
        sspan=jnp.array([0.03, 0.003, 10.0, 10.0, 10.0, 10.0, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0]),
        szero=jnp.array([0.485, 0.005, 45.0, 45.0, 35.0, 40.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        spspan=jnp.zeros(12),
        idvwlk=jnp.zeros(12, dtype=jnp.int32),
    )


def create_initial_measurement_state() -> MeasurementState:
    """Create initial measurement state."""
    xns = jnp.array([
        0.0012, 18.000, 22.000, 0.0500, 0.2000, 0.2100, 0.3000, 0.5000,
        0.0100, 0.0017, 0.0100, 1.0000, 0.3000, 0.1250, 1.0000, 0.3000,
        0.1150, 0.0100, 1.1500, 0.2000, 0.0100, 0.0100,
        0.250, 0.100, 0.250, 0.100, 0.250, 0.025,
        0.250, 0.100, 0.250, 0.100, 0.250, 0.025, 0.050, 0.050,
        0.010, 0.010, 0.010, 0.500, 0.500
    ])
    return MeasurementState(
        xmeas=jnp.zeros(41),
        xdel=jnp.zeros(41),
        xns=xns,
        tgas=0.1,
        tprod=0.25,
    )


# Initial state vector (from Python backend) - stored as Python list, converted lazily
_INITIAL_YY_LIST = [
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
]

def _get_initial_yy():
    """Get initial state as JAX array (created on each call for device compatibility)."""
    return jnp.array(_INITIAL_YY_LIST)

# For backwards compatibility
INITIAL_YY = None  # Will be set lazily or by _get_initial_yy()


# =============================================================================
# JAX TEP Process Class
# =============================================================================

class JaxTEProcess:
    """JAX-based implementation of the Tennessee Eastman Process.

    This class provides a functional interface to the TEP simulator,
    compatible with JAX transformations (jit, grad, vmap).

    Key differences from PythonTEProcess:
    - State is passed explicitly (functional style)
    - Random keys are threaded through functions
    - All operations use jax.numpy instead of numpy
    - Methods return new state instead of modifying in-place

    Example:
        >>> import jax
        >>> from tep.jax_backend import JaxTEProcess
        >>>
        >>> key = jax.random.PRNGKey(1234)
        >>> process = JaxTEProcess()
        >>> state, key = process.initialize(key)
        >>>
        >>> # JIT-compiled step function
        >>> step_fn = jax.jit(process.step)
        >>>
        >>> # Run simulation
        >>> for _ in range(3600):
        ...     state, key = step_fn(state, key)
        >>>
        >>> # Get measurements
        >>> xmeas = process.get_xmeas(state)
        >>>
        >>> # Batch simulation with vmap
        >>> keys = jax.random.split(key, 100)
        >>> batch_init = jax.vmap(process.initialize)
        >>> states, keys = batch_init(keys)
    """

    def __init__(self):
        """Initialize the JAX TEP process.

        Note: Unlike PythonTEProcess, this doesn't store state internally.
        State is passed explicitly to all methods.
        """
        self._nn = 50  # Number of state variables
        self._const = create_const_block()

    @property
    def const(self) -> ConstBlock:
        """Get the physical constants block."""
        return self._const

    def initialize(self, key: jax.Array) -> Tuple[TEPState, jax.Array]:
        """Initialize the process to steady-state conditions.

        Parameters
        ----------
        key : jax.Array
            JAX random key for reproducibility.

        Returns
        -------
        state : TEPState
            Initial process state.
        key : jax.Array
            Updated random key.
        """
        # Split key for any randomness needed during init
        key, subkey = jax.random.split(key)

        # Create initial sub-states
        reactor = create_initial_reactor_state()
        separator = create_initial_separator_state()
        stripper = create_initial_stripper_state()
        compressor = create_initial_compressor_state()
        valves = create_initial_valve_state()
        streams = create_initial_stream_state()
        walks = create_initial_walk_state()
        measurements = create_initial_measurement_state()

        # Get initial state vector (lazily initialized)
        initial_yy = _get_initial_yy()

        # Initial manipulated variables from state vector
        xmv = initial_yy[38:50]

        # Update valve command values to match initial MVs
        valves = valves._replace(vcv=xmv)

        # Create initial state
        state = TEPState(
            yy=initial_yy,
            yp=jnp.zeros(self._nn),
            time=0.0,
            reactor=reactor,
            separator=separator,
            stripper=stripper,
            compressor=compressor,
            valves=valves,
            streams=streams,
            walks=walks,
            measurements=measurements,
            xmv=xmv,
            idv=jnp.zeros(20, dtype=jnp.int32),
            tcwr=35.0,
            tcws=40.0,
            htr=jnp.array([0.06899381054, 0.05, 0.0]),
            agsp=0.0,
        )

        # Compute initial derivatives
        state, key = self._tefunc(state, key)

        return state, key

    def step(
        self,
        state: TEPState,
        key: jax.Array,
        dt: float = 1.0/3600.0
    ) -> Tuple[TEPState, jax.Array]:
        """Single Euler integration step.

        Parameters
        ----------
        state : TEPState
            Current process state.
        key : jax.Array
            JAX random key.
        dt : float
            Time step in hours. Default is 1 second (1/3600 hours).

        Returns
        -------
        state : TEPState
            Updated process state.
        key : jax.Array
            Updated random key.
        """
        # Compute derivatives
        state, key = self._tefunc(state, key)

        # Euler integration: yy = yy + yp * dt
        new_yy = state.yy + state.yp * dt
        new_time = state.time + dt

        # Apply valve constraints
        new_xmv = jnp.clip(state.xmv, 0.0, 100.0)

        # Update state
        state = state._replace(
            yy=new_yy,
            time=new_time,
            xmv=new_xmv,
        )

        return state, key

    def _tefunc(
        self,
        state: TEPState,
        key: jax.Array
    ) -> Tuple[TEPState, jax.Array]:
        """Compute state derivatives.

        This is the core simulation function that computes all process
        dynamics using JAX-compatible operations.

        Parameters
        ----------
        state : TEPState
            Current process state.
        key : jax.Array
            JAX random key.

        Returns
        -------
        state : TEPState
            State with updated derivatives (yp) and updated sub-states.
        key : jax.Array
            Updated random key.
        """
        const = self._const
        time = state.time
        yy = state.yy
        idv = state.idv
        walks = state.walks
        streams = state.streams

        # Normalize IDV values to 0 or 1
        idv = jnp.where(idv > 0, 1, 0)

        # Map IDV to walk indices
        idvwlk = jnp.array([
            idv[7],   # IDV(8)
            idv[7],   # IDV(8)
            idv[8],   # IDV(9)
            idv[9],   # IDV(10)
            idv[10],  # IDV(11)
            idv[11],  # IDV(12)
            idv[12],  # IDV(13)
            idv[12],  # IDV(13)
            idv[15],  # IDV(16)
            idv[16],  # IDV(17)
            idv[17],  # IDV(18)
            idv[19],  # IDV(20)
        ], dtype=jnp.int32)
        walks = walks._replace(idvwlk=idvwlk)

        # Update random walks for disturbances 0-8
        # Using lax.fori_loop for JIT compatibility
        def update_walk_0_8(i, carry):
            walks, key = carry
            hwlk = walks.tnext[i] - walks.tlast[i]
            swlk = walks.adist[i] + hwlk * (walks.bdist[i] + hwlk * (walks.cdist[i] + hwlk * walks.ddist[i]))
            spwlk = walks.bdist[i] + hwlk * (2.0 * walks.cdist[i] + 3.0 * hwlk * walks.ddist[i])

            # Conditionally update if time >= tnext[i]
            should_update = time >= walks.tnext[i]
            new_tlast = jnp.where(should_update, walks.tnext[i], walks.tlast[i])
            walks = walks._replace(tlast=walks.tlast.at[i].set(new_tlast))

            # Call tesub5 conditionally
            walks_updated, key_new = self._tesub5(walks, i, swlk, spwlk, walks.idvwlk[i], key)

            # Select updated or original walks based on condition
            walks = jax.tree.map(
                lambda old, new: jnp.where(should_update, new, old),
                walks,
                walks_updated
            )
            key = jnp.where(should_update, key_new, key)
            return walks, key

        walks, key = lax.fori_loop(0, 9, update_walk_0_8, (walks, key))

        # Update random walks for disturbances 9-11 (special handling)
        def update_walk_9_11(i, carry):
            walks, key = carry
            hwlk = walks.tnext[i] - walks.tlast[i]
            swlk = walks.adist[i] + hwlk * (walks.bdist[i] + hwlk * (walks.cdist[i] + hwlk * walks.ddist[i]))
            spwlk = walks.bdist[i] + hwlk * (2.0 * walks.cdist[i] + 3.0 * hwlk * walks.ddist[i])

            should_update = time >= walks.tnext[i]
            new_tlast = jnp.where(should_update, walks.tnext[i], walks.tlast[i])

            # Special case: if swlk > 0.1
            key, k1 = jax.random.split(key)
            r1 = 2.0 * jax.random.uniform(k1) - 1.0
            hwlk_new = walks.hspan[i] * r1 + walks.hzero[i]

            # Case 1: swlk > 0.1
            adist_case1 = swlk
            bdist_case1 = spwlk
            cdist_case1 = -(3.0 * swlk + 0.2 * spwlk) / 0.01
            ddist_case1 = (2.0 * swlk + 0.1 * spwlk) / 0.001
            tnext_case1 = new_tlast + 0.1

            # Case 2: swlk <= 0.1
            adist_case2 = 0.0
            bdist_case2 = 0.0
            cdist_case2 = walks.idvwlk[i].astype(float) / (hwlk_new ** 2)
            ddist_case2 = 0.0
            tnext_case2 = new_tlast + hwlk_new

            # Select based on swlk
            use_case1 = swlk > 0.1
            new_adist = jnp.where(use_case1, adist_case1, adist_case2)
            new_bdist = jnp.where(use_case1, bdist_case1, bdist_case2)
            new_cdist = jnp.where(use_case1, cdist_case1, cdist_case2)
            new_ddist = jnp.where(use_case1, ddist_case1, ddist_case2)
            new_tnext = jnp.where(use_case1, tnext_case1, tnext_case2)

            # Only update if should_update
            walks = walks._replace(
                tlast=walks.tlast.at[i].set(jnp.where(should_update, new_tlast, walks.tlast[i])),
                adist=walks.adist.at[i].set(jnp.where(should_update, new_adist, walks.adist[i])),
                bdist=walks.bdist.at[i].set(jnp.where(should_update, new_bdist, walks.bdist[i])),
                cdist=walks.cdist.at[i].set(jnp.where(should_update, new_cdist, walks.cdist[i])),
                ddist=walks.ddist.at[i].set(jnp.where(should_update, new_ddist, walks.ddist[i])),
                tnext=walks.tnext.at[i].set(jnp.where(should_update, new_tnext, walks.tnext[i])),
            )
            return walks, key

        walks, key = lax.fori_loop(9, 12, update_walk_9_11, (walks, key))

        # Reset walks at time = 0
        def reset_walks(walks):
            return walks._replace(
                adist=walks.szero,
                bdist=jnp.zeros(12),
                cdist=jnp.zeros(12),
                ddist=jnp.zeros(12),
                tlast=jnp.zeros(12),
                tnext=jnp.full(12, 0.1),
            )

        walks = lax.cond(time == 0.0, reset_walks, lambda w: w, walks)

        # Apply disturbances to feed compositions
        xst = streams.xst
        tst = streams.tst

        xst_0_3 = self._tesub8(walks, 1, time) - idv[0] * 0.03 - idv[1] * 2.43719e-3
        xst_1_3 = self._tesub8(walks, 2, time) + idv[1] * 0.005
        xst_2_3 = 1.0 - xst_0_3 - xst_1_3

        xst = xst.at[0, 3].set(xst_0_3)
        xst = xst.at[1, 3].set(xst_1_3)
        xst = xst.at[2, 3].set(xst_2_3)

        tst = tst.at[0].set(self._tesub8(walks, 3, time) + idv[2] * 5.0)
        tst = tst.at[3].set(self._tesub8(walks, 4, time))

        tcwr = self._tesub8(walks, 5, time) + idv[3] * 5.0
        tcws = self._tesub8(walks, 6, time) + idv[4] * 5.0
        r1f = self._tesub8(walks, 7, time)
        r2f = self._tesub8(walks, 8, time)

        streams = streams._replace(xst=xst, tst=tst)

        # Extract state variables from yy
        # Reactor: non-condensables in vapor (0-2), condensables in liquid (3-7)
        ucvr = jnp.array([yy[0], yy[1], yy[2], 0.0, 0.0, 0.0, 0.0, 0.0])
        uclr = jnp.array([0.0, 0.0, 0.0, yy[3], yy[4], yy[5], yy[6], yy[7]])

        # Separator
        ucvs = jnp.array([yy[9], yy[10], yy[11], 0.0, 0.0, 0.0, 0.0, 0.0])
        ucls = jnp.array([0.0, 0.0, 0.0, yy[12], yy[13], yy[14], yy[15], yy[16]])

        # Condenser
        uclc = yy[18:26]

        # Compressor
        ucvv = yy[27:35]

        # Energy states
        etr = yy[8]
        ets = yy[17]
        etc = yy[26]
        etv = yy[35]

        # Cooling water temperatures
        twr = yy[36]
        tws = yy[37]

        # Valve positions
        vpos = yy[38:50]

        # Calculate total holdups
        utlr = jnp.sum(uclr)
        utls = jnp.sum(ucls)
        utlc = jnp.sum(uclc)
        utvv = jnp.sum(ucvv)

        # Calculate mole fractions (with guards against division by zero)
        xlr = jnp.where(utlr > 0, uclr / utlr, jnp.zeros(8))
        xls = jnp.where(utls > 0, ucls / utls, jnp.zeros(8))
        xlc = jnp.where(utlc > 0, uclc / utlc, jnp.zeros(8))
        xvv = jnp.where(utvv > 0, ucvv / utvv, jnp.zeros(8))

        # Calculate specific energies
        esr = jnp.where(utlr > 0, etr / utlr, 0.0)
        ess = jnp.where(utls > 0, ets / utls, 0.0)
        esc = jnp.where(utlc > 0, etc / utlc, 0.0)
        esv = jnp.where(utvv > 0, etv / utvv, 0.0)

        # Calculate temperatures from enthalpies
        tcr_init = jnp.where(state.reactor.tcr > 0, state.reactor.tcr, 120.0)
        tcs_init = jnp.where(state.separator.tcs > 0, state.separator.tcs, 80.0)
        tcc_init = jnp.where(state.stripper.tcc > 0, state.stripper.tcc, 65.0)
        tcv_init = jnp.where(state.compressor.tcv > 0, state.compressor.tcv, 100.0)

        tcr = self._tesub2(xlr, tcr_init, esr, 0)
        tkr = tcr + 273.15
        tcs = self._tesub2(xls, tcs_init, ess, 0)
        tks = tcs + 273.15
        tcc = self._tesub2(xlc, tcc_init, esc, 0)
        tcv = self._tesub2(xvv, tcv_init, esv, 2)
        tkv = tcv + 273.15

        # Calculate densities
        dlr = self._tesub4(xlr, tcr)
        dls = self._tesub4(xls, tcs)
        dlc = self._tesub4(xlc, tcc)

        # Calculate volumes
        vlr = jnp.where(dlr > 0, utlr / dlr, 0.0)
        vls = jnp.where(dls > 0, utls / dls, 0.0)
        vlc = jnp.where(dlc > 0, utlc / dlc, 0.0)
        vtr = 1300.0  # Reactor total volume
        vts = 3500.0  # Separator total volume
        vtc = 156.5   # Stripper total volume
        vtv = 5000.0  # Compressor volume
        vvr = vtr - vlr
        vvs = vts - vls

        # Gas constant
        rg = 998.9

        # Calculate pressures (reactor and separator)
        # Non-condensable components (ideal gas)
        ppr_nc = jnp.where(vvr > 0, ucvr[:3] * rg * tkr / vvr, jnp.zeros(3))
        pps_nc = jnp.where(vvs > 0, ucvs[:3] * rg * tks / vvs, jnp.zeros(3))

        # Condensable components (vapor pressure)
        vpr_r = jnp.exp(const.avp[3:] + const.bvp[3:] / (tcr + const.cvp[3:]))
        ppr_c = vpr_r * xlr[3:]
        vpr_s = jnp.exp(const.avp[3:] + const.bvp[3:] / (tcs + const.cvp[3:]))
        pps_c = vpr_s * xls[3:]

        # Combine partial pressures
        ppr = jnp.concatenate([ppr_nc, ppr_c])
        pps = jnp.concatenate([pps_nc, pps_c])
        ptr = jnp.sum(ppr)
        pts = jnp.sum(pps)

        # Compressor pressure
        ptv = jnp.where(vtv > 0, utvv * rg * tkv / vtv, 0.0)

        # Calculate vapor compositions
        xvr = jnp.where(ptr > 0, ppr / ptr, jnp.zeros(8))
        xvs = jnp.where(pts > 0, pps / pts, jnp.zeros(8))

        # Calculate total vapor holdups
        utvr = jnp.where((rg * tkr) > 0, ptr * vvr / rg / tkr, 0.0)
        utvs = jnp.where((rg * tks) > 0, pts * vvs / rg / tks, 0.0)

        # Update condensable vapor holdups
        ucvr = ucvr.at[3:].set(utvr * xvr[3:])
        ucvs = ucvs.at[3:].set(utvs * xvs[3:])

        # Reaction kinetics
        rr = jnp.zeros(4)
        rr = rr.at[0].set(jnp.exp(31.5859536 - 40000.0 / 1.987 / tkr) * r1f)
        rr = rr.at[1].set(jnp.exp(3.00094014 - 20000.0 / 1.987 / tkr) * r2f)
        rr = rr.at[2].set(jnp.exp(53.4060443 - 60000.0 / 1.987 / tkr))
        rr = rr.at[3].set(rr[2] * 0.767488334)

        # Partial pressure terms
        pp_valid = (ppr[0] > 0.0) & (ppr[2] > 0.0)
        r1f_pp = jnp.where(pp_valid, ppr[0] ** 1.1544, 0.0)
        r2f_pp = jnp.where(pp_valid, ppr[2] ** 0.3735, 0.0)

        rr = rr.at[0].set(jnp.where(pp_valid, rr[0] * r1f_pp * r2f_pp * ppr[3], 0.0))
        rr = rr.at[1].set(jnp.where(pp_valid, rr[1] * r1f_pp * r2f_pp * ppr[4], 0.0))
        rr = rr.at[2].set(rr[2] * ppr[0] * ppr[4])
        rr = rr.at[3].set(rr[3] * ppr[0] * ppr[3])

        # Scale by vapor volume
        rr = rr * vvr

        # Component reaction rates
        crxr = jnp.array([
            -rr[0] - rr[1] - rr[2],      # Component 0 (A)
            0.0,                          # Component 1 (B)
            -rr[0] - rr[1],               # Component 2 (C)
            -rr[0] - 1.5 * rr[3],         # Component 3 (D)
            -rr[1] - rr[2],               # Component 4 (E)
            rr[2] + rr[3],                # Component 5 (F)
            rr[0],                        # Component 6 (G)
            rr[1],                        # Component 7 (H)
        ])

        # Heat of reaction
        htr = state.htr
        rh = rr[0] * htr[0] + rr[1] * htr[1]

        # Stream compositions and molecular weights
        xst = xst.at[:, 5].set(xvv)
        xst = xst.at[:, 7].set(xvr)
        xst = xst.at[:, 8].set(xvs)
        xst = xst.at[:, 9].set(xvs)
        xst = xst.at[:, 10].set(xls)
        xst = xst.at[:, 12].set(xlc)

        xmws = jnp.zeros(13)
        xmws = xmws.at[0].set(jnp.sum(xst[:, 0] * const.xmw))
        xmws = xmws.at[1].set(jnp.sum(xst[:, 1] * const.xmw))
        xmws = xmws.at[5].set(jnp.sum(xst[:, 5] * const.xmw))
        xmws = xmws.at[7].set(jnp.sum(xst[:, 7] * const.xmw))
        xmws = xmws.at[8].set(jnp.sum(xst[:, 8] * const.xmw))
        xmws = xmws.at[9].set(jnp.sum(xst[:, 9] * const.xmw))

        # Stream temperatures
        tst = tst.at[5].set(tcv)
        tst = tst.at[7].set(tcr)
        tst = tst.at[8].set(tcs)
        tst = tst.at[9].set(tcs)
        tst = tst.at[10].set(tcs)
        tst = tst.at[12].set(tcc)

        # Calculate stream enthalpies
        hst = jnp.zeros(13)
        hst = hst.at[0].set(self._tesub1(xst[:, 0], tst[0], 1))
        hst = hst.at[1].set(self._tesub1(xst[:, 1], tst[1], 1))
        hst = hst.at[2].set(self._tesub1(xst[:, 2], tst[2], 1))
        hst = hst.at[3].set(self._tesub1(xst[:, 3], tst[3], 1))
        hst = hst.at[5].set(self._tesub1(xst[:, 5], tst[5], 1))
        hst = hst.at[7].set(self._tesub1(xst[:, 7], tst[7], 1))
        hst = hst.at[8].set(self._tesub1(xst[:, 8], tst[8], 1))
        hst = hst.at[9].set(hst[8])
        hst = hst.at[10].set(self._tesub1(xst[:, 10], tst[10], 0))
        hst = hst.at[12].set(self._tesub1(xst[:, 12], tst[12], 0))

        # Calculate flows
        vrng = state.valves.vrng
        ftm = jnp.zeros(13)
        ftm = ftm.at[0].set(vpos[0] * vrng[0] / 100.0)
        ftm = ftm.at[1].set(vpos[1] * vrng[1] / 100.0)
        ftm = ftm.at[2].set(vpos[2] * (1.0 - idv[5]) * vrng[2] / 100.0)
        ftm = ftm.at[3].set(vpos[3] * (1.0 - idv[6] * 0.2) * vrng[3] / 100.0 + 1.0e-10)
        ftm = ftm.at[10].set(vpos[6] * vrng[6] / 100.0)
        ftm = ftm.at[12].set(vpos[7] * vrng[7] / 100.0)

        uac = vpos[8] * vrng[8] * (1.0 + self._tesub8(walks, 9, time)) / 100.0
        fwr = vpos[9] * vrng[9] / 100.0
        fws = vpos[10] * vrng[10] / 100.0
        agsp = (vpos[11] + 150.0) / 100.0

        # Pressure-driven flows
        dlp = jnp.maximum(ptv - ptr, 0.0)
        flms = 1937.6 * jnp.sqrt(dlp)
        ftm = ftm.at[5].set(jnp.where(xmws[5] > 0, flms / xmws[5], 0.0))

        dlp = jnp.maximum(ptr - pts, 0.0)
        flms = 4574.21 * jnp.sqrt(dlp) * (1.0 - 0.25 * self._tesub8(walks, 12, time))
        ftm = ftm.at[7].set(jnp.where(xmws[7] > 0, flms / xmws[7], 0.0))

        dlp = jnp.maximum(pts - 760.0, 0.0)
        flms = vpos[5] * 0.151169 * jnp.sqrt(dlp)
        ftm = ftm.at[9].set(jnp.where(xmws[9] > 0, flms / xmws[9], 0.0))

        # Compressor
        cpflmx = 280275.0
        cpprmx = 1.3
        pr = jnp.where(pts > 0, ptv / pts, 1.0)
        pr = jnp.clip(pr, 1.0, cpprmx)
        flcoef = cpflmx / 1.197
        flms = cpflmx + flcoef * (1.0 - pr ** 3)
        cpdh = jnp.where(
            (xmws[8] * pts) > 0,
            flms * (tcs + 273.15) * 1.8e-6 * 1.9872 * (ptv - pts) / (xmws[8] * pts),
            0.0
        )

        dlp = jnp.maximum(ptv - pts, 0.0)
        flms = flms - vpos[4] * 53.349 * jnp.sqrt(dlp)
        flms = jnp.maximum(flms, 1.0e-3)
        ftm = ftm.at[8].set(jnp.where(xmws[8] > 0, flms / xmws[8], 0.0))
        hst = hst.at[8].set(jnp.where(ftm[8] > 0, hst[8] + cpdh / ftm[8], hst[8]))

        # Component flows
        fcm = jnp.zeros((8, 13))
        for j in [0, 1, 2, 3, 5, 7, 8, 9, 10, 12]:
            fcm = fcm.at[:, j].set(xst[:, j] * ftm[j])

        # Stripper separation
        tmpfac = jnp.where(
            tcc > 170.0,
            tcc - 120.262,
            jnp.where(
                tcc < 5.292,
                0.1,
                363.744 / (177.0 - tcc) - 2.22579488
            )
        )
        vovrl = jnp.where(ftm[10] > 0.1, ftm[3] / ftm[10] * tmpfac, 0.0)

        sfr = streams.sfr
        sfr_updated = jnp.array([
            sfr[0], sfr[1], sfr[2],
            jnp.where(ftm[10] > 0.1, 8.5010 * vovrl / (1.0 + 8.5010 * vovrl), 0.9999),
            jnp.where(ftm[10] > 0.1, 11.402 * vovrl / (1.0 + 11.402 * vovrl), 0.999),
            jnp.where(ftm[10] > 0.1, 11.795 * vovrl / (1.0 + 11.795 * vovrl), 0.999),
            jnp.where(ftm[10] > 0.1, 0.0480 * vovrl / (1.0 + 0.0480 * vovrl), 0.99),
            jnp.where(ftm[10] > 0.1, 0.0242 * vovrl / (1.0 + 0.0242 * vovrl), 0.98),
        ])

        # Stripper inlet flows
        fin = fcm[:, 3] + fcm[:, 10]

        # Stripper separation
        fcm = fcm.at[:, 4].set(sfr_updated * fin)
        fcm = fcm.at[:, 11].set(fin - fcm[:, 4])
        ftm = ftm.at[4].set(jnp.sum(fcm[:, 4]))
        ftm = ftm.at[11].set(jnp.sum(fcm[:, 11]))

        # Stream compositions
        xst = xst.at[:, 4].set(jnp.where(ftm[4] > 0, fcm[:, 4] / ftm[4], jnp.zeros(8)))
        xst = xst.at[:, 11].set(jnp.where(ftm[11] > 0, fcm[:, 11] / ftm[11], jnp.zeros(8)))

        tst = tst.at[4].set(tcc)
        tst = tst.at[11].set(tcc)
        hst = hst.at[4].set(self._tesub1(xst[:, 4], tst[4], 1))
        hst = hst.at[11].set(self._tesub1(xst[:, 11], tst[11], 0))

        # Stream 7 = Stream 6
        ftm = ftm.at[6].set(ftm[5])
        hst = hst.at[6].set(hst[5])
        tst = tst.at[6].set(tst[5])
        xst = xst.at[:, 6].set(xst[:, 5])
        fcm = fcm.at[:, 6].set(fcm[:, 5])

        # Heat transfer calculations
        uarlev = jnp.where(
            vlr / 7.8 > 50.0,
            1.0,
            jnp.where(vlr / 7.8 < 10.0, 0.0, 0.025 * vlr / 7.8 - 0.25)
        )
        hwr = 7060.0
        hws = 11138.0

        uar = uarlev * (-0.5 * agsp ** 2 + 2.75 * agsp - 2.5) * 855490.0e-6
        qur = uar * (twr - tcr) * (1.0 - 0.35 * self._tesub8(walks, 10, time))

        uas = 0.404655 * (1.0 - 1.0 / (1.0 + (ftm[7] / 3528.73) ** 4))
        qus = uas * (tws - tst[7]) * (1.0 - 0.25 * self._tesub8(walks, 11, time))

        quc = jnp.where(tcc < 100.0, uac * (100.0 - tcc), 0.0)

        # Calculate measurements (without noise first)
        xmeas = jnp.zeros(41)
        xmeas = xmeas.at[0].set(ftm[2] * 0.359 / 35.3145)
        xmeas = xmeas.at[1].set(ftm[0] * xmws[0] * 0.454)
        xmeas = xmeas.at[2].set(ftm[1] * xmws[1] * 0.454)
        xmeas = xmeas.at[3].set(ftm[3] * 0.359 / 35.3145)
        xmeas = xmeas.at[4].set(ftm[8] * 0.359 / 35.3145)
        xmeas = xmeas.at[5].set(ftm[5] * 0.359 / 35.3145)
        xmeas = xmeas.at[6].set((ptr - 760.0) / 760.0 * 101.325)
        xmeas = xmeas.at[7].set((vlr - 84.6) / 666.7 * 100.0)
        xmeas = xmeas.at[8].set(tcr)
        xmeas = xmeas.at[9].set(ftm[9] * 0.359 / 35.3145)
        xmeas = xmeas.at[10].set(tcs)
        xmeas = xmeas.at[11].set((vls - 27.5) / 290.0 * 100.0)
        xmeas = xmeas.at[12].set((pts - 760.0) / 760.0 * 101.325)
        xmeas = xmeas.at[13].set(jnp.where(dls > 0, ftm[10] / dls / 35.3145, 0.0))
        xmeas = xmeas.at[14].set((vlc - 78.25) / vtc * 100.0)
        xmeas = xmeas.at[15].set((ptv - 760.0) / 760.0 * 101.325)
        xmeas = xmeas.at[16].set(jnp.where(dlc > 0, ftm[12] / dlc / 35.3145, 0.0))
        xmeas = xmeas.at[17].set(tcc)
        xmeas = xmeas.at[18].set(quc * 1.04e3 * 0.454)
        xmeas = xmeas.at[19].set(cpdh * 0.29307e3)
        xmeas = xmeas.at[20].set(twr)
        xmeas = xmeas.at[21].set(tws)

        # Safety shutdown check
        isd = (
            (xmeas[6] > 3000.0) |
            (vlr / 35.3145 > 24.0) |
            (vlr / 35.3145 < 2.0) |
            (xmeas[8] > 175.0) |
            (vls / 35.3145 > 12.0) |
            (vls / 35.3145 < 1.0) |
            (vlc / 35.3145 > 8.0) |
            (vlc / 35.3145 < 1.0)
        )

        # Add measurement noise (only after time 0 and not shutdown)
        def add_noise(carry):
            xmeas, key = carry
            def add_noise_single(i, xmeas_key):
                xmeas, key = xmeas_key
                noise, key = self._tesub6(key, state.measurements.xns[i])
                xmeas = xmeas.at[i].set(xmeas[i] + noise)
                return xmeas, key
            xmeas, key = lax.fori_loop(0, 22, add_noise_single, (xmeas, key))
            return xmeas, key

        def no_noise(carry):
            return carry

        should_add_noise = (time > 0.0) & (~isd)
        xmeas, key = lax.cond(should_add_noise, add_noise, no_noise, (xmeas, key))

        # Sampled composition measurements
        xcmp = jnp.zeros(41)
        xcmp = xcmp.at[22].set(xst[0, 6] * 100.0)
        xcmp = xcmp.at[23].set(xst[1, 6] * 100.0)
        xcmp = xcmp.at[24].set(xst[2, 6] * 100.0)
        xcmp = xcmp.at[25].set(xst[3, 6] * 100.0)
        xcmp = xcmp.at[26].set(xst[4, 6] * 100.0)
        xcmp = xcmp.at[27].set(xst[5, 6] * 100.0)
        xcmp = xcmp.at[28].set(xst[0, 9] * 100.0)
        xcmp = xcmp.at[29].set(xst[1, 9] * 100.0)
        xcmp = xcmp.at[30].set(xst[2, 9] * 100.0)
        xcmp = xcmp.at[31].set(xst[3, 9] * 100.0)
        xcmp = xcmp.at[32].set(xst[4, 9] * 100.0)
        xcmp = xcmp.at[33].set(xst[5, 9] * 100.0)
        xcmp = xcmp.at[34].set(xst[6, 9] * 100.0)
        xcmp = xcmp.at[35].set(xst[7, 9] * 100.0)
        xcmp = xcmp.at[36].set(xst[3, 12] * 100.0)
        xcmp = xcmp.at[37].set(xst[4, 12] * 100.0)
        xcmp = xcmp.at[38].set(xst[5, 12] * 100.0)
        xcmp = xcmp.at[39].set(xst[6, 12] * 100.0)
        xcmp = xcmp.at[40].set(xst[7, 12] * 100.0)

        # Handle delayed measurements
        meas = state.measurements
        xdel = meas.xdel
        tgas = meas.tgas
        tprod = meas.tprod

        # Initialize at time 0
        def init_delays(carry):
            xmeas, xdel, tgas, tprod = carry
            xdel = xdel.at[22:41].set(xcmp[22:41])
            xmeas = xmeas.at[22:41].set(xcmp[22:41])
            return xmeas, xdel, 0.1, 0.25

        def keep_delays(carry):
            return carry

        xmeas, xdel, tgas, tprod = lax.cond(
            time == 0.0,
            init_delays,
            keep_delays,
            (xmeas, xdel, tgas, tprod)
        )

        # Gas sampling update
        def update_gas(carry):
            xmeas, xdel, tgas, key = carry
            def add_gas_noise(i, xmeas_xdel_key):
                xmeas, xdel, key = xmeas_xdel_key
                noise, key = self._tesub6(key, meas.xns[i])
                xmeas = xmeas.at[i].set(xdel[i] + noise)
                xdel = xdel.at[i].set(xcmp[i])
                return xmeas, xdel, key
            xmeas, xdel, key = lax.fori_loop(22, 36, add_gas_noise, (xmeas, xdel, key))
            return xmeas, xdel, tgas + 0.1, key

        def no_gas_update(carry):
            return carry

        xmeas, xdel, tgas, key = lax.cond(
            time >= tgas,
            update_gas,
            no_gas_update,
            (xmeas, xdel, tgas, key)
        )

        # Product sampling update
        def update_prod(carry):
            xmeas, xdel, tprod, key = carry
            def add_prod_noise(i, xmeas_xdel_key):
                xmeas, xdel, key = xmeas_xdel_key
                noise, key = self._tesub6(key, meas.xns[i])
                xmeas = xmeas.at[i].set(xdel[i] + noise)
                xdel = xdel.at[i].set(xcmp[i])
                return xmeas, xdel, key
            xmeas, xdel, key = lax.fori_loop(36, 41, add_prod_noise, (xmeas, xdel, key))
            return xmeas, xdel, tprod + 0.25, key

        def no_prod_update(carry):
            return carry

        xmeas, xdel, tprod, key = lax.cond(
            time >= tprod,
            update_prod,
            no_prod_update,
            (xmeas, xdel, tprod, key)
        )

        # State derivatives
        yp = jnp.zeros(50)

        # Reactor component balances (0-7)
        yp = yp.at[0:8].set(fcm[:, 6] - fcm[:, 7] + crxr)

        # Separator component balances (9-16)
        yp = yp.at[9:17].set(fcm[:, 7] - fcm[:, 8] - fcm[:, 9] - fcm[:, 10])

        # Condenser component balances (18-25)
        yp = yp.at[18:26].set(fcm[:, 11] - fcm[:, 12])

        # Compressor component balances (27-34)
        yp = yp.at[27:35].set(
            fcm[:, 0] + fcm[:, 1] + fcm[:, 2] + fcm[:, 4] + fcm[:, 8] - fcm[:, 5]
        )

        # Energy balances
        yp = yp.at[8].set(hst[6] * ftm[6] - hst[7] * ftm[7] + rh + qur)
        yp = yp.at[17].set(
            hst[7] * ftm[7] - hst[8] * ftm[8] - hst[9] * ftm[9] - hst[10] * ftm[10] + qus
        )
        yp = yp.at[26].set(
            hst[3] * ftm[3] + hst[10] * ftm[10] - hst[4] * ftm[4] - hst[12] * ftm[12] + quc
        )
        yp = yp.at[35].set(
            hst[0] * ftm[0] + hst[1] * ftm[1] + hst[2] * ftm[2] +
            hst[4] * ftm[4] + hst[8] * ftm[8] - hst[5] * ftm[5]
        )

        # Cooling water temperatures
        yp = yp.at[36].set((fwr * 500.53 * (tcwr - twr) - qur * 1.0e6 / 1.8) / hwr)
        yp = yp.at[37].set((fws * 500.53 * (tcws - tws) - qus * 1.0e6 / 1.8) / hws)

        # Valve sticking
        ivst = state.valves.ivst
        ivst = ivst.at[9].set(idv[13])   # IDV(14)
        ivst = ivst.at[10].set(idv[14])  # IDV(15)
        ivst = ivst.at[4].set(idv[18])   # IDV(19)
        ivst = ivst.at[6].set(idv[18])   # IDV(19)
        ivst = ivst.at[7].set(idv[18])   # IDV(19)
        ivst = ivst.at[8].set(idv[18])   # IDV(19)

        # Valve dynamics
        vcv = state.valves.vcv
        vst = state.valves.vst
        vtau = state.valves.vtau
        xmv = state.xmv

        def valve_update(i, carry):
            vcv, yp = carry
            should_update = (time == 0.0) | (jnp.abs(vcv[i] - xmv[i]) > vst[i] * ivst[i])
            new_vcv = jnp.where(should_update, xmv[i], vcv[i])
            new_vcv = jnp.clip(new_vcv, 0.0, 100.0)
            vcv = vcv.at[i].set(new_vcv)
            yp = yp.at[38 + i].set((new_vcv - vpos[i]) / vtau[i])
            return vcv, yp

        vcv, yp = lax.fori_loop(0, 12, valve_update, (vcv, yp))

        # Shutdown: zero all derivatives
        yp = jnp.where(isd, jnp.zeros(50), yp)

        # Update all sub-states
        reactor = state.reactor._replace(
            uclr=uclr, ucvr=ucvr, utlr=utlr, utvr=utvr,
            xlr=xlr, xvr=xvr, etr=etr, esr=esr,
            tcr=tcr, tkr=tkr, dlr=dlr, vlr=vlr, vvr=vvr, vtr=vtr,
            ptr=ptr, ppr=ppr, crxr=crxr, rr=rr, rh=rh,
            fwr=fwr, twr=twr, qur=qur, uar=uar,
        )

        separator = state.separator._replace(
            ucls=ucls, ucvs=ucvs, utls=utls, utvs=utvs,
            xls=xls, xvs=xvs, ets=ets, ess=ess,
            tcs=tcs, tks=tks, dls=dls, vls=vls, vvs=vvs, vts=vts,
            pts=pts, pps=pps, fws=fws, tws=tws, qus=qus,
        )

        stripper = state.stripper._replace(
            uclc=uclc, utlc=utlc, xlc=xlc,
            etc=etc, esc=esc, tcc=tcc, dlc=dlc, vlc=vlc, vtc=vtc, quc=quc,
        )

        compressor = state.compressor._replace(
            ucvv=ucvv, utvv=utvv, xvv=xvv,
            etv=etv, esv=esv, tcv=tcv, tkv=tkv, vtv=vtv, ptv=ptv,
            cpdh=cpdh,
        )

        valves = state.valves._replace(vcv=vcv, ivst=ivst)

        streams = streams._replace(
            ftm=ftm, fcm=fcm, xst=xst, xmws=xmws, hst=hst, tst=tst, sfr=sfr_updated,
        )

        measurements = meas._replace(xmeas=xmeas, xdel=xdel, tgas=tgas, tprod=tprod)

        # Create updated state
        state = state._replace(
            yy=yy,
            yp=yp,
            reactor=reactor,
            separator=separator,
            stripper=stripper,
            compressor=compressor,
            valves=valves,
            streams=streams,
            walks=walks,
            measurements=measurements,
            idv=idv,
            tcwr=tcwr,
            tcws=tcws,
            agsp=agsp,
        )

        return state, key

    # =========================================================================
    # Thermodynamic Functions (to be implemented)
    # =========================================================================

    def _tesub1(
        self,
        z: jnp.ndarray,
        t: float,
        ity: int
    ) -> float:
        """Calculate enthalpy from composition and temperature.

        Parameters
        ----------
        z : jnp.ndarray
            Mole fractions (8 elements).
        t : float
            Temperature (deg C).
        ity : int
            0 = liquid, 1 = gas, 2 = gas with pressure correction.

        Returns
        -------
        float
            Specific enthalpy.
        """
        const = self._const

        # Liquid enthalpy calculation (vectorized)
        hi_liquid = t * (const.ah + const.bh * t / 2.0 + const.ch * t**2 / 3.0)
        hi_liquid = 1.8 * hi_liquid
        h_liquid = jnp.sum(z * const.xmw * hi_liquid)

        # Gas enthalpy calculation (vectorized)
        hi_gas = t * (const.ag + const.bg * t / 2.0 + const.cg * t**2 / 3.0)
        hi_gas = 1.8 * hi_gas + const.av
        h_gas = jnp.sum(z * const.xmw * hi_gas)

        # Select liquid or gas based on ity
        # ity == 0 -> liquid, ity >= 1 -> gas
        h = jnp.where(ity == 0, h_liquid, h_gas)

        # Apply pressure correction for ity == 2 (compressor)
        r = 3.57696e-6
        correction = r * (t + 273.15)
        h = jnp.where(ity == 2, h - correction, h)

        return h

    def _tesub2(
        self,
        z: jnp.ndarray,
        t_init: float,
        h: float,
        ity: int
    ) -> float:
        """Calculate temperature from composition and enthalpy.

        Uses Newton iteration with JAX's lax.while_loop.

        Parameters
        ----------
        z : jnp.ndarray
            Mole fractions (8 elements).
        t_init : float
            Initial temperature guess (deg C).
        h : float
            Target specific enthalpy.
        ity : int
            0 = liquid, 1 = gas, 2 = gas with pressure correction.

        Returns
        -------
        float
            Temperature (deg C).
        """
        # Newton iteration using lax.while_loop for JIT compatibility

        def newton_cond(carry):
            """Condition: continue while |dt| >= tolerance and iteration < max."""
            t, dt, iteration = carry
            return (jnp.abs(dt) >= 1.0e-12) & (iteration < 100)

        def newton_body(carry):
            """One Newton iteration step."""
            t, _, iteration = carry
            htest = self._tesub1(z, t, ity)
            err = htest - h
            dh = self._tesub3(z, t, ity)
            # Guard against zero derivative
            dt = jnp.where(jnp.abs(dh) > 1e-20, -err / dh, 0.0)
            new_t = t + dt
            return (new_t, dt, iteration + 1)

        # Initial carry: (temperature, dt, iteration_count)
        # Start with dt=1.0 to ensure loop runs at least once
        initial_carry = (t_init, 1.0, 0)

        # Run Newton iteration
        final_t, final_dt, final_iter = lax.while_loop(
            newton_cond,
            newton_body,
            initial_carry
        )

        # Return final temperature (or initial if didn't converge)
        return jnp.where(final_iter >= 100, t_init, final_t)

    def _tesub3(
        self,
        z: jnp.ndarray,
        t: float,
        ity: int
    ) -> float:
        """Calculate enthalpy derivative dH/dT.

        Parameters
        ----------
        z : jnp.ndarray
            Mole fractions (8 elements).
        t : float
            Temperature (deg C).
        ity : int
            0 = liquid, 1 = gas, 2 = gas with pressure correction.

        Returns
        -------
        float
            dH/dT.
        """
        const = self._const

        # Liquid dH/dT (vectorized)
        dhi_liquid = const.ah + const.bh * t + const.ch * t**2
        dhi_liquid = 1.8 * dhi_liquid
        dh_liquid = jnp.sum(z * const.xmw * dhi_liquid)

        # Gas dH/dT (vectorized)
        dhi_gas = const.ag + const.bg * t + const.cg * t**2
        dhi_gas = 1.8 * dhi_gas
        dh_gas = jnp.sum(z * const.xmw * dhi_gas)

        # Select liquid or gas based on ity
        dh = jnp.where(ity == 0, dh_liquid, dh_gas)

        # Apply pressure correction for ity == 2
        r = 3.57696e-6
        dh = jnp.where(ity == 2, dh - r, dh)

        return dh

    def _tesub4(
        self,
        x: jnp.ndarray,
        t: float
    ) -> float:
        """Calculate liquid density from composition and temperature.

        Parameters
        ----------
        x : jnp.ndarray
            Mole fractions (8 elements).
        t : float
            Temperature (deg C).

        Returns
        -------
        float
            Liquid density (kmol/m^3).
        """
        const = self._const

        # Calculate molar volume (vectorized)
        # v = sum(x_i * M_i / rho_i) where rho_i = ad + bd*t + cd*t^2
        rho_components = const.ad + (const.bd + const.cd * t) * t
        v = jnp.sum(x * const.xmw / rho_components)

        # Return density = 1/v, with guard against division by zero
        return jnp.where(v > 0, 1.0 / v, 1.0)

    # =========================================================================
    # Random Number Functions (JAX-based)
    # =========================================================================

    def _tesub6(
        self,
        key: jax.Array,
        std: float
    ) -> Tuple[float, jax.Array]:
        """Generate random noise with given standard deviation.

        Uses JAX's native random number generation instead of LCG.
        Sum of 12 uniforms approximates normal distribution.

        Parameters
        ----------
        key : jax.Array
            JAX random key.
        std : float
            Standard deviation.

        Returns
        -------
        noise : float
            Random noise value.
        key : jax.Array
            Updated random key.
        """
        key, subkey = jax.random.split(key)
        # Sum of 12 uniform randoms approximates normal (matches Python backend)
        uniforms = jax.random.uniform(subkey, shape=(12,))
        noise = (jnp.sum(uniforms) - 6.0) * std
        return noise, key

    def _tesub5(
        self,
        walks: WalkState,
        idx: int,
        s: float,
        sp: float,
        idvflag: int,
        key: jax.Array
    ) -> Tuple[WalkState, jax.Array]:
        """Generate next random walk interval (cubic spline parameters).

        Parameters
        ----------
        walks : WalkState
            Current walk state.
        idx : int
            Disturbance index (0-based).
        s : float
            Current signal value.
        sp : float
            Current signal derivative.
        idvflag : int
            Disturbance active flag (0 or 1).
        key : jax.Array
            JAX random key.

        Returns
        -------
        walks : WalkState
            Updated walk state.
        key : jax.Array
            Updated random key.
        """
        # Generate random values
        key, k1, k2, k3 = jax.random.split(key, 4)
        r1 = 2.0 * jax.random.uniform(k1) - 1.0  # Range [-1, 1]
        r2 = 2.0 * jax.random.uniform(k2) - 1.0
        r3 = 2.0 * jax.random.uniform(k3) - 1.0

        # Calculate interval and target
        h = walks.hspan[idx] * r1 + walks.hzero[idx]
        s1 = walks.sspan[idx] * r2 * idvflag + walks.szero[idx]
        s1p = walks.spspan[idx] * r3 * idvflag

        # Calculate cubic spline coefficients
        h2 = h * h
        h3 = h2 * h
        new_adist = walks.adist.at[idx].set(s)
        new_bdist = walks.bdist.at[idx].set(sp)
        new_cdist = walks.cdist.at[idx].set((3.0 * (s1 - s) - h * (s1p + 2.0 * sp)) / h2)
        new_ddist = walks.ddist.at[idx].set((2.0 * (s - s1) + h * (s1p + sp)) / h3)
        new_tnext = walks.tnext.at[idx].set(walks.tlast[idx] + h)

        walks = walks._replace(
            adist=new_adist,
            bdist=new_bdist,
            cdist=new_cdist,
            ddist=new_ddist,
            tnext=new_tnext,
        )
        return walks, key

    def _tesub7(
        self,
        key: jax.Array,
        signed: bool = True
    ) -> Tuple[float, jax.Array]:
        """Generate random number using JAX.

        Replaces the LCG-based random number generator with JAX random.

        Parameters
        ----------
        key : jax.Array
            JAX random key.
        signed : bool
            If True, returns value in [-1, 1). If False, returns [0, 1).

        Returns
        -------
        value : float
            Random number.
        key : jax.Array
            Updated random key.
        """
        key, subkey = jax.random.split(key)
        r = jax.random.uniform(subkey)
        value = jnp.where(signed, 2.0 * r - 1.0, r)
        return value, key

    def _tesub8(
        self,
        walks: WalkState,
        i: int,
        t: float
    ) -> float:
        """Evaluate cubic spline for disturbance walk.

        Parameters
        ----------
        walks : WalkState
            Walk state containing spline coefficients.
        i : int
            Disturbance index (1-based, as in Fortran).
        t : float
            Current time (hours).

        Returns
        -------
        float
            Disturbance value.
        """
        idx = i - 1  # Convert to 0-based
        h = t - walks.tlast[idx]
        return (walks.adist[idx] +
                h * (walks.bdist[idx] +
                     h * (walks.cdist[idx] +
                          h * walks.ddist[idx])))

    # =========================================================================
    # Public Interface Methods
    # =========================================================================

    def get_xmeas(self, state: TEPState) -> jnp.ndarray:
        """Get current measurement values.

        Parameters
        ----------
        state : TEPState
            Current process state.

        Returns
        -------
        jnp.ndarray
            Measurement values (41,).
        """
        return state.measurements.xmeas

    def get_xmv(self, state: TEPState) -> jnp.ndarray:
        """Get current manipulated variables.

        Parameters
        ----------
        state : TEPState
            Current process state.

        Returns
        -------
        jnp.ndarray
            Manipulated variable values (12,).
        """
        return state.xmv

    def set_xmv(
        self,
        state: TEPState,
        index: int,
        value: float
    ) -> TEPState:
        """Set a manipulated variable.

        Parameters
        ----------
        state : TEPState
            Current process state.
        index : int
            1-based index of the manipulated variable (1-12).
        value : float
            Value to set (will be clipped to 0-100).

        Returns
        -------
        TEPState
            Updated process state.
        """
        new_xmv = state.xmv.at[index - 1].set(jnp.clip(value, 0.0, 100.0))
        return state._replace(xmv=new_xmv)

    def set_idv(
        self,
        state: TEPState,
        index: int,
        value: int
    ) -> TEPState:
        """Set a disturbance variable.

        Parameters
        ----------
        state : TEPState
            Current process state.
        index : int
            1-based index of the disturbance (1-20).
        value : int
            0 to disable, 1 to enable the disturbance.

        Returns
        -------
        TEPState
            Updated process state.
        """
        new_idv = state.idv.at[index - 1].set(value)
        return state._replace(idv=new_idv)

    def clear_disturbances(self, state: TEPState) -> TEPState:
        """Clear all disturbances (set IDV to 0).

        Parameters
        ----------
        state : TEPState
            Current process state.

        Returns
        -------
        TEPState
            Updated process state with all disturbances cleared.
        """
        return state._replace(idv=jnp.zeros(20, dtype=jnp.int32))

    def get_state(self, state: TEPState) -> jnp.ndarray:
        """Get current state vector.

        Parameters
        ----------
        state : TEPState
            Current process state.

        Returns
        -------
        jnp.ndarray
            State vector (50,).
        """
        return state.yy

    def is_shutdown(self, state: TEPState) -> bool:
        """Check if process is in shutdown state.

        Parameters
        ----------
        state : TEPState
            Current process state.

        Returns
        -------
        bool
            True if process has triggered a safety shutdown.
        """
        xmeas = state.measurements.xmeas
        reactor = state.reactor
        separator = state.separator
        stripper = state.stripper

        # Use JAX-compatible checks (returns bool array, need to convert)
        conditions = jnp.array([
            xmeas[6] > 3000.0,
            reactor.vlr / 35.3145 > 24.0,
            reactor.vlr / 35.3145 < 2.0,
            xmeas[8] > 175.0,
            separator.vls / 35.3145 > 12.0,
            separator.vls / 35.3145 < 1.0,
            stripper.vlc / 35.3145 > 8.0,
            stripper.vlc / 35.3145 < 1.0,
        ])
        return bool(jnp.any(conditions))


# =============================================================================
# Wrapper Class for API Compatibility with TEPSimulator
# =============================================================================

class JaxTEProcessWrapper:
    """Wrapper that provides a stateful interface matching PythonTEProcess.

    This wrapper stores state internally to provide the same API as the
    Python backend, making it a drop-in replacement for TEPSimulator.

    For pure JAX usage (JIT, vmap, grad), use JaxTEProcess directly.
    """

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize the JAX TEP process wrapper.

        Parameters
        ----------
        random_seed : int, optional
            Random seed for reproducibility.
        """
        self._process = JaxTEProcess()
        self._nn = 50
        self._initialized = False
        self.time = 0.0

        # Initialize random key
        if random_seed is None:
            random_seed = 1234
        self._key = jax.random.PRNGKey(random_seed)

        # State will be set during initialize()
        self._state: Optional[TEPState] = None

    def initialize(self):
        """Initialize the process to steady-state conditions."""
        self._state, self._key = self._process.initialize(self._key)
        self.time = 0.0
        self._initialized = True

    def _initialize(self):
        """Internal initialization method for TEPSimulator compatibility."""
        self.initialize()

    def step(self, dt: float = 1.0/3600.0):
        """Single Euler integration step."""
        if not self._initialized:
            raise RuntimeError("Process not initialized. Call initialize() first.")

        self._state, self._key = self._process.step(self._state, self._key, dt)
        self.time = float(self._state.time)

    @property
    def yy(self) -> np.ndarray:
        """Get current state vector as numpy array."""
        if self._state is None:
            return np.zeros(self._nn)
        return np.array(self._state.yy)

    @yy.setter
    def yy(self, value: np.ndarray):
        """Set state vector."""
        if self._state is not None:
            self._state = self._state._replace(yy=jnp.array(value))

    @property
    def yp(self) -> np.ndarray:
        """Get current derivative vector as numpy array."""
        if self._state is None:
            return np.zeros(self._nn)
        return np.array(self._state.yp)

    @property
    def xmeas(self) -> np.ndarray:
        """Get current measurement values."""
        if self._state is None:
            return np.zeros(41)
        return np.array(self._process.get_xmeas(self._state))

    @property
    def xmv(self) -> np.ndarray:
        """Get current manipulated variables."""
        if self._state is None:
            return np.zeros(12)
        return np.array(self._process.get_xmv(self._state))

    @property
    def idv(self) -> np.ndarray:
        """Get current disturbance vector."""
        if self._state is None:
            return np.zeros(20, dtype=np.int32)
        return np.array(self._state.idv)

    def get_xmeas(self) -> np.ndarray:
        """Get current measurement values (for TEPSimulator compatibility)."""
        return self.xmeas.copy()

    def get_xmv(self) -> np.ndarray:
        """Get current manipulated variables (for TEPSimulator compatibility)."""
        return self.xmv.copy()

    def set_xmv(self, index: int, value: float):
        """Set a manipulated variable."""
        if self._state is not None:
            self._state = self._process.set_xmv(self._state, index, value)

    def set_idv(self, index: int, value: int):
        """Set a disturbance variable."""
        if self._state is not None:
            self._state = self._process.set_idv(self._state, index, value)

    def clear_disturbances(self):
        """Clear all disturbances."""
        if self._state is not None:
            self._state = self._process.clear_disturbances(self._state)

    def evaluate(self, time: float, yy: np.ndarray) -> np.ndarray:
        """Evaluate derivatives using TEFUNC."""
        if self._state is None:
            return np.zeros(self._nn)

        temp_state = self._state._replace(
            yy=jnp.array(yy),
            time=time
        )
        temp_state, self._key = self._process._tefunc(temp_state, self._key)
        return np.array(temp_state.yp)

    def is_shutdown(self) -> bool:
        """Check if process is in shutdown state."""
        if self._state is None:
            return False
        return self._process.is_shutdown(self._state)

    def get_state(self) -> np.ndarray:
        """Get current state vector."""
        return self.yy.copy()

    def set_state(self, state: np.ndarray):
        """Set state vector."""
        if len(state) != self._nn:
            raise ValueError(f"State must have {self._nn} elements")
        self.yy = state

    @property
    def state(self) -> 'JaxTEProcessState':
        """Get state wrapper for TEPSimulator compatibility."""
        return JaxTEProcessState(self)

    @property
    def disturbances(self) -> 'JaxDisturbanceManager':
        """Get disturbance manager for API compatibility."""
        return JaxDisturbanceManager(self)


class JaxTEProcessState:
    """State wrapper that mimics PythonTEProcessState interface.

    This provides the `state.yy`, `state.xmeas`, `state.xmv` interface
    expected by TEPSimulator.
    """

    def __init__(self, wrapper: 'JaxTEProcessWrapper'):
        self._wrapper = wrapper

    @property
    def yy(self) -> np.ndarray:
        return self._wrapper.yy

    @yy.setter
    def yy(self, value: np.ndarray):
        self._wrapper.yy = value

    @property
    def xmeas(self) -> np.ndarray:
        return self._wrapper.xmeas

    @property
    def xmv(self) -> np.ndarray:
        return self._wrapper.xmv


class JaxDisturbanceManager:
    """Disturbance manager wrapper for API compatibility."""

    def __init__(self, wrapper: 'JaxTEProcessWrapper'):
        self._wrapper = wrapper

    def clear_all_disturbances(self):
        """Clear all disturbances (set IDV to 0)."""
        self._wrapper.clear_disturbances()

    def set_disturbance(self, index: int, value: int):
        """Set a disturbance variable."""
        self._wrapper.set_idv(index, value)


# =============================================================================
# Utility Functions
# =============================================================================

def is_jax_available() -> bool:
    """Check if JAX is available."""
    try:
        import jax
        return True
    except ImportError:
        return False


def get_jax_backend_info() -> dict:
    """Get information about the JAX backend configuration."""
    try:
        import jax
        return {
            "available": True,
            "version": jax.__version__,
            "devices": [str(d) for d in jax.devices()],
            "default_backend": jax.default_backend(),
        }
    except ImportError:
        return {"available": False}
