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

import jax
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


# Initial state vector (from Python backend)
INITIAL_YY = jnp.array([
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
], dtype=jnp.float64)


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

        # Initial manipulated variables from state vector
        xmv = INITIAL_YY[38:50]

        # Update valve command values to match initial MVs
        valves = valves._replace(vcv=xmv)

        # Create initial state
        state = TEPState(
            yy=INITIAL_YY,
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
        """Compute state derivatives (placeholder).

        This is the core simulation function that will be implemented
        to compute all process dynamics using JAX-compatible operations.

        Parameters
        ----------
        state : TEPState
            Current process state.
        key : jax.Array
            JAX random key.

        Returns
        -------
        state : TEPState
            State with updated derivatives (yp).
        key : jax.Array
            Updated random key.
        """
        # TODO: Implement full TEFUNC logic with JAX operations
        # For now, return zero derivatives (placeholder)
        state = state._replace(yp=jnp.zeros(self._nn))
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
        # TODO: Implement with jnp operations
        # Will use jnp.where instead of if/else for JIT compatibility
        raise NotImplementedError("_tesub1 not yet implemented")

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
        # TODO: Implement with lax.while_loop for JIT compatibility
        raise NotImplementedError("_tesub2 not yet implemented")

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
        # TODO: Implement with jnp operations
        raise NotImplementedError("_tesub3 not yet implemented")

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
        # TODO: Implement with jnp operations
        raise NotImplementedError("_tesub4 not yet implemented")

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
