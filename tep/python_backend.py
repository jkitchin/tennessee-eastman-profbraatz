"""Pure Python backend for Tennessee Eastman Process simulation.

This module provides a complete Python implementation of the TEP simulator,
replicating the original Fortran code (teprob.f) without any Fortran dependency.

The implementation faithfully reproduces:
- All common blocks as Python data structures
- The TEINIT initialization routine
- The TEFUNC derivative function
- All helper subroutines (TESUB1-8)
- The linear congruential random number generator
- Disturbance random walks and measurement noise

This enables running TEP simulations in environments where Fortran compilation
is not available, while maintaining statistical similarity to the original.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
import copy


# =============================================================================
# Common Block Data Structures
# =============================================================================

@dataclass
class ConstBlock:
    """Physical constants common block (/CONST/).

    Contains thermodynamic properties for the 8 chemical components:
    - Antoine coefficients for vapor pressure
    - Enthalpy coefficients (liquid and gas)
    - Density coefficients
    - Heat of vaporization
    - Molecular weights
    """
    # Molecular weights (kg/kmol)
    xmw: np.ndarray = field(default_factory=lambda: np.array([
        2.0, 25.4, 28.0, 32.0, 46.0, 48.0, 62.0, 76.0
    ]))

    # Antoine vapor pressure: ln(P) = AVP + BVP/(T + CVP)
    # Components 1-3 (A,B,C) are non-condensable gases (coefficients = 0)
    avp: np.ndarray = field(default_factory=lambda: np.array([
        0.0, 0.0, 0.0, 15.92, 16.35, 16.35, 16.43, 17.21
    ]))
    bvp: np.ndarray = field(default_factory=lambda: np.array([
        0.0, 0.0, 0.0, -1444.0, -2114.0, -2114.0, -2748.0, -3318.0
    ]))
    cvp: np.ndarray = field(default_factory=lambda: np.array([
        0.0, 0.0, 0.0, 259.0, 265.5, 265.5, 232.9, 249.6
    ]))

    # Liquid density: rho = AD + BD*T + CD*T^2
    ad: np.ndarray = field(default_factory=lambda: np.array([
        1.0, 1.0, 1.0, 23.3, 33.9, 32.8, 49.9, 50.5
    ]))
    bd: np.ndarray = field(default_factory=lambda: np.array([
        0.0, 0.0, 0.0, -0.0700, -0.0957, -0.0995, -0.0191, -0.0541
    ]))
    cd: np.ndarray = field(default_factory=lambda: np.array([
        0.0, 0.0, 0.0, -0.0002, -0.000152, -0.000233, -0.000425, -0.000150
    ]))

    # Liquid enthalpy: H = AH*T + BH*T^2/2 + CH*T^3/3
    ah: np.ndarray = field(default_factory=lambda: np.array([
        1.0e-6, 1.0e-6, 1.0e-6, 0.960e-6, 0.573e-6, 0.652e-6, 0.515e-6, 0.471e-6
    ]))
    bh: np.ndarray = field(default_factory=lambda: np.array([
        0.0, 0.0, 0.0, 8.70e-9, 2.41e-9, 2.18e-9, 5.65e-10, 8.70e-10
    ]))
    ch: np.ndarray = field(default_factory=lambda: np.array([
        0.0, 0.0, 0.0, 4.81e-11, 1.82e-11, 1.94e-11, 3.82e-12, 2.62e-12
    ]))

    # Heat of vaporization
    av: np.ndarray = field(default_factory=lambda: np.array([
        1.0e-6, 1.0e-6, 1.0e-6, 86.7e-6, 160.0e-6, 160.0e-6, 225.0e-6, 209.0e-6
    ]))

    # Gas heat capacity: Cp = AG + BG*T + CG*T^2
    ag: np.ndarray = field(default_factory=lambda: np.array([
        3.411e-6, 0.3799e-6, 0.2491e-6, 0.3567e-6, 0.3463e-6, 0.3930e-6, 0.170e-6, 0.150e-6
    ]))
    bg: np.ndarray = field(default_factory=lambda: np.array([
        7.18e-10, 1.08e-9, 1.36e-11, 8.51e-10, 8.96e-10, 1.02e-9, 0.0, 0.0
    ]))
    cg: np.ndarray = field(default_factory=lambda: np.array([
        6.0e-13, -3.98e-13, -3.93e-14, -3.12e-13, -3.27e-13, -3.12e-13, 0.0, 0.0
    ]))


@dataclass
class TEProcBlock:
    """Process state common block (/TEPROC/).

    Contains all internal process variables for reactor, separator,
    stripper (condenser), and compressor.
    """
    # Reactor state
    uclr: np.ndarray = field(default_factory=lambda: np.zeros(8))  # Liquid component holdup
    ucvr: np.ndarray = field(default_factory=lambda: np.zeros(8))  # Vapor component holdup
    utlr: float = 0.0  # Total liquid holdup
    utvr: float = 0.0  # Total vapor holdup
    xlr: np.ndarray = field(default_factory=lambda: np.zeros(8))   # Liquid mole fractions
    xvr: np.ndarray = field(default_factory=lambda: np.zeros(8))   # Vapor mole fractions
    etr: float = 0.0   # Total energy
    esr: float = 0.0   # Specific energy
    tcr: float = 0.0   # Temperature (C)
    tkr: float = 0.0   # Temperature (K)
    dlr: float = 0.0   # Liquid density
    vlr: float = 0.0   # Liquid volume
    vvr: float = 0.0   # Vapor volume
    vtr: float = 1300.0  # Total volume
    ptr: float = 0.0   # Total pressure
    ppr: np.ndarray = field(default_factory=lambda: np.zeros(8))   # Partial pressures
    crxr: np.ndarray = field(default_factory=lambda: np.zeros(8))  # Reaction rates by component
    rr: np.ndarray = field(default_factory=lambda: np.zeros(4))    # Reaction rates
    rh: float = 0.0    # Heat of reaction
    fwr: float = 0.0   # Cooling water flow
    twr: float = 0.0   # Cooling water temperature
    qur: float = 0.0   # Heat transfer
    hwr: float = 7060.0  # Heat transfer coefficient
    uar: float = 0.0   # Overall heat transfer

    # Separator state
    ucls: np.ndarray = field(default_factory=lambda: np.zeros(8))
    ucvs: np.ndarray = field(default_factory=lambda: np.zeros(8))
    utls: float = 0.0
    utvs: float = 0.0
    xls: np.ndarray = field(default_factory=lambda: np.zeros(8))
    xvs: np.ndarray = field(default_factory=lambda: np.zeros(8))
    ets: float = 0.0
    ess: float = 0.0
    tcs: float = 0.0
    tks: float = 0.0
    dls: float = 0.0
    vls: float = 0.0
    vvs: float = 0.0
    vts: float = 3500.0
    pts: float = 0.0
    pps: np.ndarray = field(default_factory=lambda: np.zeros(8))
    fws: float = 0.0
    tws: float = 0.0
    qus: float = 0.0
    hws: float = 11138.0

    # Stripper/Condenser state
    uclc: np.ndarray = field(default_factory=lambda: np.zeros(8))
    utlc: float = 0.0
    xlc: np.ndarray = field(default_factory=lambda: np.zeros(8))
    etc: float = 0.0
    esc: float = 0.0
    tcc: float = 0.0
    dlc: float = 0.0
    vlc: float = 0.0
    vtc: float = 156.5
    quc: float = 0.0

    # Compressor state
    ucvv: np.ndarray = field(default_factory=lambda: np.zeros(8))
    utvv: float = 0.0
    xvv: np.ndarray = field(default_factory=lambda: np.zeros(8))
    etv: float = 0.0
    esv: float = 0.0
    tcv: float = 0.0
    tkv: float = 0.0
    vtv: float = 5000.0
    ptv: float = 0.0

    # Valve states
    vcv: np.ndarray = field(default_factory=lambda: np.zeros(12))  # Valve command values
    vrng: np.ndarray = field(default_factory=lambda: np.array([
        400.0, 400.0, 100.0, 1500.0, 0.0, 0.0,
        1500.0, 1000.0, 0.03, 1000.0, 1200.0, 0.0
    ]))  # Valve ranges
    vtau: np.ndarray = field(default_factory=lambda: np.array([
        8.0, 8.0, 6.0, 9.0, 7.0, 5.0, 5.0, 5.0, 120.0, 5.0, 5.0, 5.0
    ]) / 3600.0)  # Time constants (hours)

    # Stream flows and compositions
    ftm: np.ndarray = field(default_factory=lambda: np.zeros(13))  # Total molar flows
    fcm: np.ndarray = field(default_factory=lambda: np.zeros((8, 13)))  # Component molar flows
    xst: np.ndarray = field(default_factory=lambda: np.zeros((8, 13)))  # Stream compositions
    xmws: np.ndarray = field(default_factory=lambda: np.zeros(13))  # Stream molecular weights
    hst: np.ndarray = field(default_factory=lambda: np.zeros(13))  # Stream enthalpies
    tst: np.ndarray = field(default_factory=lambda: np.zeros(13))  # Stream temperatures
    sfr: np.ndarray = field(default_factory=lambda: np.zeros(8))   # Separation factors

    # Compressor parameters
    cpflmx: float = 280275.0
    cpprmx: float = 1.3
    cpdh: float = 0.0

    # Cooling water inlet temperatures
    tcwr: float = 0.0
    tcws: float = 0.0

    # Heat of reaction and agitator
    htr: np.ndarray = field(default_factory=lambda: np.array([0.06899381054, 0.05, 0.0]))
    agsp: float = 0.0

    # Measurement delays and noise
    xdel: np.ndarray = field(default_factory=lambda: np.zeros(41))  # Delayed measurements
    xns: np.ndarray = field(default_factory=lambda: np.zeros(41))   # Noise std devs
    tgas: float = 0.1    # Gas sampling time
    tprod: float = 0.25  # Product sampling time

    # Valve sticking
    vst: np.ndarray = field(default_factory=lambda: np.full(12, 2.0))  # Sticking threshold
    ivst: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.int32))  # Sticking flags


@dataclass
class WalkBlock:
    """Disturbance random walk common block (/WLK/).

    Contains parameters for cubic spline interpolation of random disturbances.
    """
    adist: np.ndarray = field(default_factory=lambda: np.zeros(12))
    bdist: np.ndarray = field(default_factory=lambda: np.zeros(12))
    cdist: np.ndarray = field(default_factory=lambda: np.zeros(12))
    ddist: np.ndarray = field(default_factory=lambda: np.zeros(12))
    tlast: np.ndarray = field(default_factory=lambda: np.zeros(12))
    tnext: np.ndarray = field(default_factory=lambda: np.full(12, 0.1))
    hspan: np.ndarray = field(default_factory=lambda: np.array([
        0.2, 0.7, 0.25, 0.7, 0.15, 0.15, 1.0, 1.0, 0.4, 1.5, 2.0, 1.5
    ]))
    hzero: np.ndarray = field(default_factory=lambda: np.array([
        0.5, 1.0, 0.5, 1.0, 0.25, 0.25, 2.0, 2.0, 0.5, 2.0, 3.0, 2.0
    ]))
    sspan: np.ndarray = field(default_factory=lambda: np.array([
        0.03, 0.003, 10.0, 10.0, 10.0, 10.0, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0
    ]))
    szero: np.ndarray = field(default_factory=lambda: np.array([
        0.485, 0.005, 45.0, 45.0, 35.0, 40.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0
    ]))
    spspan: np.ndarray = field(default_factory=lambda: np.zeros(12))
    idvwlk: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.int32))


# =============================================================================
# Pure Python TEP Process Implementation
# =============================================================================

class PythonTEProcess:
    """Pure Python implementation of the Tennessee Eastman Process.

    This class replicates the original Fortran implementation (teprob.f)
    without any Fortran dependency. It provides the same interface as
    FortranTEProcess for drop-in replacement.

    The implementation includes:
    - Linear congruential random number generator (matching Fortran)
    - Full process dynamics for reactor, separator, stripper, compressor
    - Reaction kinetics and heat transfer
    - Valve dynamics with sticking
    - Measurement noise and sampling delays
    - 20 process disturbances

    Example:
        >>> process = PythonTEProcess(random_seed=1234)
        >>> process.initialize()
        >>> for _ in range(3600):  # 1 hour
        ...     process.step()
        >>> print(process.xmeas[:5])
    """

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize the Python TEP process.

        Parameters
        ----------
        random_seed : int, optional
            Random seed for reproducibility. If None, uses the default
            Fortran seed (4651207995).
        """
        self._nn = 50  # Number of state variables
        self._initialized = False
        self.time = 0.0
        self._random_seed = random_seed

        # Random number generator state
        self._g = 4651207995.0  # Default Fortran seed

        # State vectors
        self.yy = np.zeros(self._nn, dtype=np.float64)
        self.yp = np.zeros(self._nn, dtype=np.float64)

        # Common blocks
        self._const = ConstBlock()
        self._teproc = TEProcBlock()
        self._wlk = WalkBlock()

        # Process variables (PV common block)
        self._xmeas = np.zeros(41, dtype=np.float64)
        self._xmv = np.zeros(12, dtype=np.float64)

        # Disturbance vector (DVEC common block)
        self._idv = np.zeros(20, dtype=np.int32)

        # Create state wrapper for API compatibility
        self.state = PythonTEProcessState(self)

        # Create disturbances wrapper for API compatibility
        self.disturbances = PythonDisturbanceManager(self)

    # =========================================================================
    # Random Number Generator (TESUB7)
    # =========================================================================

    def _tesub7(self, i: int) -> float:
        """Linear congruential random number generator.

        Replicates Fortran TESUB7:
            G = MOD(G * 9228907, 2^32)
            if i >= 0: return G / 2^32        (range 0 to 1)
            if i < 0:  return 2*G/2^32 - 1    (range -1 to 1)

        Parameters
        ----------
        i : int
            If >= 0, returns value in [0, 1)
            If < 0, returns value in [-1, 1)

        Returns
        -------
        float
            Random number
        """
        self._g = (self._g * 9228907.0) % 4294967296.0
        if i >= 0:
            return self._g / 4294967296.0
        else:
            return 2.0 * self._g / 4294967296.0 - 1.0

    # =========================================================================
    # Thermodynamic Functions (TESUB1-4)
    # =========================================================================

    def _tesub1(self, z: np.ndarray, t: float, ity: int) -> float:
        """Calculate enthalpy from composition and temperature.

        Parameters
        ----------
        z : np.ndarray
            Mole fractions (8 elements)
        t : float
            Temperature (deg C)
        ity : int
            0 = liquid, 1 = gas, 2 = gas with pressure correction

        Returns
        -------
        float
            Specific enthalpy
        """
        const = self._const
        h = 0.0

        if ity == 0:
            # Liquid enthalpy
            for i in range(8):
                hi = t * (const.ah[i] + const.bh[i] * t / 2.0 + const.ch[i] * t**2 / 3.0)
                hi = 1.8 * hi
                h += z[i] * const.xmw[i] * hi
        else:
            # Gas enthalpy
            for i in range(8):
                hi = t * (const.ag[i] + const.bg[i] * t / 2.0 + const.cg[i] * t**2 / 3.0)
                hi = 1.8 * hi
                hi += const.av[i]
                h += z[i] * const.xmw[i] * hi

        if ity == 2:
            # Pressure correction for compressor
            r = 3.57696e-6
            h -= r * (t + 273.15)

        return h

    def _tesub2(self, z: np.ndarray, t_init: float, h: float, ity: int) -> float:
        """Calculate temperature from composition and enthalpy (Newton iteration).

        Parameters
        ----------
        z : np.ndarray
            Mole fractions (8 elements)
        t_init : float
            Initial temperature guess (deg C)
        h : float
            Target specific enthalpy
        ity : int
            0 = liquid, 1 = gas, 2 = gas with pressure correction

        Returns
        -------
        float
            Temperature (deg C)
        """
        t = t_init
        for _ in range(100):
            htest = self._tesub1(z, t, ity)
            err = htest - h
            dh = self._tesub3(z, t, ity)
            dt = -err / dh
            t += dt
            if abs(dt) < 1.0e-12:
                return t
        return t_init  # Return initial guess if not converged

    def _tesub3(self, z: np.ndarray, t: float, ity: int) -> float:
        """Calculate enthalpy derivative dH/dT.

        Parameters
        ----------
        z : np.ndarray
            Mole fractions (8 elements)
        t : float
            Temperature (deg C)
        ity : int
            0 = liquid, 1 = gas, 2 = gas with pressure correction

        Returns
        -------
        float
            dH/dT
        """
        const = self._const
        dh = 0.0

        if ity == 0:
            # Liquid
            for i in range(8):
                dhi = const.ah[i] + const.bh[i] * t + const.ch[i] * t**2
                dhi = 1.8 * dhi
                dh += z[i] * const.xmw[i] * dhi
        else:
            # Gas
            for i in range(8):
                dhi = const.ag[i] + const.bg[i] * t + const.cg[i] * t**2
                dhi = 1.8 * dhi
                dh += z[i] * const.xmw[i] * dhi

        if ity == 2:
            r = 3.57696e-6
            dh -= r

        return dh

    def _tesub4(self, x: np.ndarray, t: float) -> float:
        """Calculate liquid density from composition and temperature.

        Parameters
        ----------
        x : np.ndarray
            Mole fractions (8 elements)
        t : float
            Temperature (deg C)

        Returns
        -------
        float
            Liquid density (kmol/m^3)
        """
        const = self._const
        v = 0.0
        for i in range(8):
            v += x[i] * const.xmw[i] / (const.ad[i] + (const.bd[i] + const.cd[i] * t) * t)
        return 1.0 / v if v > 0 else 1.0

    # =========================================================================
    # Disturbance Walk Functions (TESUB5, TESUB6, TESUB8)
    # =========================================================================

    def _tesub5(self, s: float, sp: float, idx: int, idvflag: int):
        """Generate next random walk interval (cubic spline parameters).

        Updates the walk block parameters for disturbance index idx.

        Parameters
        ----------
        s : float
            Current signal value
        sp : float
            Current signal derivative
        idx : int
            Disturbance index (0-based)
        idvflag : int
            Disturbance active flag (0 or 1)
        """
        wlk = self._wlk

        # Generate random interval
        h = wlk.hspan[idx] * self._tesub7(-1) + wlk.hzero[idx]
        s1 = wlk.sspan[idx] * self._tesub7(-1) * idvflag + wlk.szero[idx]
        s1p = wlk.spspan[idx] * self._tesub7(-1) * idvflag

        # Calculate cubic spline coefficients
        wlk.adist[idx] = s
        wlk.bdist[idx] = sp
        wlk.cdist[idx] = (3.0 * (s1 - s) - h * (s1p + 2.0 * sp)) / h**2
        wlk.ddist[idx] = (2.0 * (s - s1) + h * (s1p + sp)) / h**3
        wlk.tnext[idx] = wlk.tlast[idx] + h

    def _tesub6(self, std: float) -> float:
        """Generate random noise with given standard deviation.

        Uses sum of 12 uniform random numbers (approximates normal distribution).

        Parameters
        ----------
        std : float
            Standard deviation

        Returns
        -------
        float
            Random noise value
        """
        x = 0.0
        for _ in range(12):
            x += self._tesub7(1)
        return (x - 6.0) * std

    def _tesub8(self, i: int, t: float) -> float:
        """Evaluate cubic spline for disturbance walk.

        Parameters
        ----------
        i : int
            Disturbance index (1-based, as in Fortran)
        t : float
            Current time (hours)

        Returns
        -------
        float
            Disturbance value
        """
        wlk = self._wlk
        idx = i - 1  # Convert to 0-based
        h = t - wlk.tlast[idx]
        return wlk.adist[idx] + h * (wlk.bdist[idx] + h * (wlk.cdist[idx] + h * wlk.ddist[idx]))

    # =========================================================================
    # Initialization (TEINIT)
    # =========================================================================

    def _initialize(self):
        """Internal initialization method for TEPSimulator compatibility."""
        self.initialize()

    def initialize(self):
        """Initialize the process to steady-state conditions.

        Replicates Fortran TEINIT subroutine. Sets initial state values,
        physical constants, and calls TEFUNC to compute initial derivatives.
        """
        # Reset random seed
        self._g = 4651207995.0

        # Initialize physical constants (already done in ConstBlock)
        # The constants are initialized in the dataclass defaults

        # Set initial state values (YY)
        self.yy = np.array([
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
        ], dtype=np.float64)

        # Initialize XMV from state and valve parameters
        tp = self._teproc
        for i in range(12):
            self._xmv[i] = self.yy[38 + i]
            tp.vcv[i] = self._xmv[i]
            tp.vst[i] = 2.0
            tp.ivst[i] = 0

        # Initialize separation factors
        tp.sfr = np.array([0.99500, 0.99100, 0.99000, 0.91600, 0.93600, 0.93800, 0.05800, 0.03010])

        # Initialize feed stream compositions
        # Stream 1: A Feed (mostly D)
        tp.xst[0, 0] = 0.0
        tp.xst[1, 0] = 0.0001
        tp.xst[2, 0] = 0.0
        tp.xst[3, 0] = 0.9999
        tp.xst[4, 0] = 0.0
        tp.xst[5, 0] = 0.0
        tp.xst[6, 0] = 0.0
        tp.xst[7, 0] = 0.0
        tp.tst[0] = 45.0

        # Stream 2: D Feed (mostly E)
        tp.xst[0, 1] = 0.0
        tp.xst[1, 1] = 0.0
        tp.xst[2, 1] = 0.0
        tp.xst[3, 1] = 0.0
        tp.xst[4, 1] = 0.9999
        tp.xst[5, 1] = 0.0001
        tp.xst[6, 1] = 0.0
        tp.xst[7, 1] = 0.0
        tp.tst[1] = 45.0

        # Stream 3: E Feed (mostly A)
        tp.xst[0, 2] = 0.9999
        tp.xst[1, 2] = 0.0001
        tp.xst[2, 2] = 0.0
        tp.xst[3, 2] = 0.0
        tp.xst[4, 2] = 0.0
        tp.xst[5, 2] = 0.0
        tp.xst[6, 2] = 0.0
        tp.xst[7, 2] = 0.0
        tp.tst[2] = 45.0

        # Stream 4: A and C Feed
        tp.xst[0, 3] = 0.4850
        tp.xst[1, 3] = 0.0050
        tp.xst[2, 3] = 0.5100
        tp.xst[3, 3] = 0.0
        tp.xst[4, 3] = 0.0
        tp.xst[5, 3] = 0.0
        tp.xst[6, 3] = 0.0
        tp.xst[7, 3] = 0.0
        tp.tst[3] = 45.0

        # Initialize measurement noise standard deviations
        tp.xns = np.array([
            0.0012, 18.000, 22.000, 0.0500, 0.2000, 0.2100, 0.3000, 0.5000,
            0.0100, 0.0017, 0.0100, 1.0000, 0.3000, 0.1250, 1.0000, 0.3000,
            0.1150, 0.0100, 1.1500, 0.2000, 0.0100, 0.0100,
            0.250, 0.100, 0.250, 0.100, 0.250, 0.025,
            0.250, 0.100, 0.250, 0.100, 0.250, 0.025, 0.050, 0.050,
            0.010, 0.010, 0.010, 0.500, 0.500
        ])

        # Clear disturbances
        self._idv[:] = 0

        # Initialize walk block
        wlk = self._wlk
        for i in range(12):
            wlk.tlast[i] = 0.0
            wlk.tnext[i] = 0.1
            wlk.adist[i] = wlk.szero[i]
            wlk.bdist[i] = 0.0
            wlk.cdist[i] = 0.0
            wlk.ddist[i] = 0.0

        # Set time to zero
        self.time = 0.0

        # Override random seed if provided
        if self._random_seed is not None:
            self._g = float(self._random_seed)

        # Call TEFUNC to compute initial derivatives
        self._tefunc()

        self._initialized = True

    # =========================================================================
    # Main Simulation Function (TEFUNC)
    # =========================================================================

    def _tefunc(self):
        """Compute state derivatives.

        Replicates Fortran TEFUNC subroutine. This is the core simulation
        function that computes all process dynamics.
        """
        tp = self._teproc
        wlk = self._wlk
        const = self._const
        idv = self._idv
        time = self.time
        yy = self.yy
        yp = self.yp

        # Normalize IDV values to 0 or 1
        for i in range(20):
            idv[i] = 1 if idv[i] > 0 else 0

        # Map IDV to walk indices
        wlk.idvwlk[0] = idv[7]   # IDV(8)
        wlk.idvwlk[1] = idv[7]   # IDV(8)
        wlk.idvwlk[2] = idv[8]   # IDV(9)
        wlk.idvwlk[3] = idv[9]   # IDV(10)
        wlk.idvwlk[4] = idv[10]  # IDV(11)
        wlk.idvwlk[5] = idv[11]  # IDV(12)
        wlk.idvwlk[6] = idv[12]  # IDV(13)
        wlk.idvwlk[7] = idv[12]  # IDV(13)
        wlk.idvwlk[8] = idv[15]  # IDV(16)
        wlk.idvwlk[9] = idv[16]  # IDV(17)
        wlk.idvwlk[10] = idv[17] # IDV(18)
        wlk.idvwlk[11] = idv[19] # IDV(20)

        # Update random walks for disturbances 1-9
        for i in range(9):
            if time >= wlk.tnext[i]:
                hwlk = wlk.tnext[i] - wlk.tlast[i]
                swlk = wlk.adist[i] + hwlk * (wlk.bdist[i] + hwlk * (wlk.cdist[i] + hwlk * wlk.ddist[i]))
                spwlk = wlk.bdist[i] + hwlk * (2.0 * wlk.cdist[i] + 3.0 * hwlk * wlk.ddist[i])
                wlk.tlast[i] = wlk.tnext[i]
                self._tesub5(swlk, spwlk, i, wlk.idvwlk[i])

        # Update random walks for disturbances 10-12 (special handling)
        for i in range(9, 12):
            if time >= wlk.tnext[i]:
                hwlk = wlk.tnext[i] - wlk.tlast[i]
                swlk = wlk.adist[i] + hwlk * (wlk.bdist[i] + hwlk * (wlk.cdist[i] + hwlk * wlk.ddist[i]))
                spwlk = wlk.bdist[i] + hwlk * (2.0 * wlk.cdist[i] + 3.0 * hwlk * wlk.ddist[i])
                wlk.tlast[i] = wlk.tnext[i]

                if swlk > 0.1:
                    wlk.adist[i] = swlk
                    wlk.bdist[i] = spwlk
                    wlk.cdist[i] = -(3.0 * swlk + 0.2 * spwlk) / 0.01
                    wlk.ddist[i] = (2.0 * swlk + 0.1 * spwlk) / 0.001
                    wlk.tnext[i] = wlk.tlast[i] + 0.1
                else:
                    hwlk = wlk.hspan[i] * self._tesub7(-1) + wlk.hzero[i]
                    wlk.adist[i] = 0.0
                    wlk.bdist[i] = 0.0
                    wlk.cdist[i] = float(wlk.idvwlk[i]) / hwlk**2
                    wlk.ddist[i] = 0.0
                    wlk.tnext[i] = wlk.tlast[i] + hwlk

        # Reset walks at time = 0
        if time == 0.0:
            for i in range(12):
                wlk.adist[i] = wlk.szero[i]
                wlk.bdist[i] = 0.0
                wlk.cdist[i] = 0.0
                wlk.ddist[i] = 0.0
                wlk.tlast[i] = 0.0
                wlk.tnext[i] = 0.1

        # Apply disturbances to feed compositions
        tp.xst[0, 3] = self._tesub8(1, time) - idv[0] * 0.03 - idv[1] * 2.43719e-3
        tp.xst[1, 3] = self._tesub8(2, time) + idv[1] * 0.005
        tp.xst[2, 3] = 1.0 - tp.xst[0, 3] - tp.xst[1, 3]
        tp.tst[0] = self._tesub8(3, time) + idv[2] * 5.0
        tp.tst[3] = self._tesub8(4, time)
        tp.tcwr = self._tesub8(5, time) + idv[3] * 5.0
        tp.tcws = self._tesub8(6, time) + idv[4] * 5.0
        r1f = self._tesub8(7, time)
        r2f = self._tesub8(8, time)

        # Extract state variables
        # Reactor vapor holdup (components 1-3)
        for i in range(3):
            tp.ucvr[i] = yy[i]
            tp.ucvs[i] = yy[i + 9]
            tp.uclr[i] = 0.0
            tp.ucls[i] = 0.0

        # Reactor liquid holdup (components 4-8)
        for i in range(3, 8):
            tp.uclr[i] = yy[i]
            tp.ucls[i] = yy[i + 9]

        # Condenser and compressor holdup
        for i in range(8):
            tp.uclc[i] = yy[i + 18]
            tp.ucvv[i] = yy[i + 27]

        # Energy states
        tp.etr = yy[8]
        tp.ets = yy[17]
        tp.etc = yy[26]
        tp.etv = yy[35]

        # Cooling water temperatures
        tp.twr = yy[36]
        tp.tws = yy[37]

        # Valve positions
        vpos = yy[38:50].copy()

        # Calculate total holdups
        tp.utlr = np.sum(tp.uclr)
        tp.utls = np.sum(tp.ucls)
        tp.utlc = np.sum(tp.uclc)
        tp.utvv = np.sum(tp.ucvv)

        # Calculate mole fractions
        for i in range(8):
            tp.xlr[i] = tp.uclr[i] / tp.utlr if tp.utlr > 0 else 0.0
            tp.xls[i] = tp.ucls[i] / tp.utls if tp.utls > 0 else 0.0
            tp.xlc[i] = tp.uclc[i] / tp.utlc if tp.utlc > 0 else 0.0
            tp.xvv[i] = tp.ucvv[i] / tp.utvv if tp.utvv > 0 else 0.0

        # Calculate specific energies
        tp.esr = tp.etr / tp.utlr if tp.utlr > 0 else 0.0
        tp.ess = tp.ets / tp.utls if tp.utls > 0 else 0.0
        tp.esc = tp.etc / tp.utlc if tp.utlc > 0 else 0.0
        tp.esv = tp.etv / tp.utvv if tp.utvv > 0 else 0.0

        # Calculate temperatures from enthalpies
        tp.tcr = self._tesub2(tp.xlr, tp.tcr if tp.tcr > 0 else 120.0, tp.esr, 0)
        tp.tkr = tp.tcr + 273.15
        tp.tcs = self._tesub2(tp.xls, tp.tcs if tp.tcs > 0 else 80.0, tp.ess, 0)
        tp.tks = tp.tcs + 273.15
        tp.tcc = self._tesub2(tp.xlc, tp.tcc if tp.tcc > 0 else 65.0, tp.esc, 0)
        tp.tcv = self._tesub2(tp.xvv, tp.tcv if tp.tcv > 0 else 100.0, tp.esv, 2)
        tp.tkv = tp.tcv + 273.15

        # Calculate densities
        tp.dlr = self._tesub4(tp.xlr, tp.tcr)
        tp.dls = self._tesub4(tp.xls, tp.tcs)
        tp.dlc = self._tesub4(tp.xlc, tp.tcc)

        # Calculate volumes
        tp.vlr = tp.utlr / tp.dlr if tp.dlr > 0 else 0.0
        tp.vls = tp.utls / tp.dls if tp.dls > 0 else 0.0
        tp.vlc = tp.utlc / tp.dlc if tp.dlc > 0 else 0.0
        tp.vvr = tp.vtr - tp.vlr
        tp.vvs = tp.vts - tp.vls

        # Gas constant
        rg = 998.9

        # Calculate pressures (reactor and separator)
        tp.ptr = 0.0
        tp.pts = 0.0

        # Non-condensable components (ideal gas)
        for i in range(3):
            tp.ppr[i] = tp.ucvr[i] * rg * tp.tkr / tp.vvr if tp.vvr > 0 else 0.0
            tp.ptr += tp.ppr[i]
            tp.pps[i] = tp.ucvs[i] * rg * tp.tks / tp.vvs if tp.vvs > 0 else 0.0
            tp.pts += tp.pps[i]

        # Condensable components (vapor pressure)
        for i in range(3, 8):
            vpr = np.exp(const.avp[i] + const.bvp[i] / (tp.tcr + const.cvp[i]))
            tp.ppr[i] = vpr * tp.xlr[i]
            tp.ptr += tp.ppr[i]
            vpr = np.exp(const.avp[i] + const.bvp[i] / (tp.tcs + const.cvp[i]))
            tp.pps[i] = vpr * tp.xls[i]
            tp.pts += tp.pps[i]

        # Compressor pressure
        tp.ptv = tp.utvv * rg * tp.tkv / tp.vtv if tp.vtv > 0 else 0.0

        # Calculate vapor compositions
        for i in range(8):
            tp.xvr[i] = tp.ppr[i] / tp.ptr if tp.ptr > 0 else 0.0
            tp.xvs[i] = tp.pps[i] / tp.pts if tp.pts > 0 else 0.0

        # Calculate total vapor holdups
        tp.utvr = tp.ptr * tp.vvr / rg / tp.tkr if (rg * tp.tkr) > 0 else 0.0
        tp.utvs = tp.pts * tp.vvs / rg / tp.tks if (rg * tp.tks) > 0 else 0.0

        # Update condensable vapor holdups
        for i in range(3, 8):
            tp.ucvr[i] = tp.utvr * tp.xvr[i]
            tp.ucvs[i] = tp.utvs * tp.xvs[i]

        # Reaction kinetics
        tp.rr[0] = np.exp(31.5859536 - 40000.0 / 1.987 / tp.tkr) * r1f
        tp.rr[1] = np.exp(3.00094014 - 20000.0 / 1.987 / tp.tkr) * r2f
        tp.rr[2] = np.exp(53.4060443 - 60000.0 / 1.987 / tp.tkr)
        tp.rr[3] = tp.rr[2] * 0.767488334

        if tp.ppr[0] > 0.0 and tp.ppr[2] > 0.0:
            r1f_pp = tp.ppr[0] ** 1.1544
            r2f_pp = tp.ppr[2] ** 0.3735
            tp.rr[0] = tp.rr[0] * r1f_pp * r2f_pp * tp.ppr[3]
            tp.rr[1] = tp.rr[1] * r1f_pp * r2f_pp * tp.ppr[4]
        else:
            tp.rr[0] = 0.0
            tp.rr[1] = 0.0

        tp.rr[2] = tp.rr[2] * tp.ppr[0] * tp.ppr[4]
        tp.rr[3] = tp.rr[3] * tp.ppr[0] * tp.ppr[3]

        # Scale by vapor volume
        for i in range(4):
            tp.rr[i] = tp.rr[i] * tp.vvr

        # Component reaction rates
        tp.crxr[0] = -tp.rr[0] - tp.rr[1] - tp.rr[2]
        tp.crxr[1] = 0.0
        tp.crxr[2] = -tp.rr[0] - tp.rr[1]
        tp.crxr[3] = -tp.rr[0] - 1.5 * tp.rr[3]
        tp.crxr[4] = -tp.rr[1] - tp.rr[2]
        tp.crxr[5] = tp.rr[2] + tp.rr[3]
        tp.crxr[6] = tp.rr[0]
        tp.crxr[7] = tp.rr[1]

        # Heat of reaction
        tp.rh = tp.rr[0] * tp.htr[0] + tp.rr[1] * tp.htr[1]

        # Stream compositions and molecular weights
        tp.xmws[0] = 0.0
        tp.xmws[1] = 0.0
        tp.xmws[5] = 0.0
        tp.xmws[7] = 0.0
        tp.xmws[8] = 0.0
        tp.xmws[9] = 0.0

        for i in range(8):
            tp.xst[i, 5] = tp.xvv[i]
            tp.xst[i, 7] = tp.xvr[i]
            tp.xst[i, 8] = tp.xvs[i]
            tp.xst[i, 9] = tp.xvs[i]
            tp.xst[i, 10] = tp.xls[i]
            tp.xst[i, 12] = tp.xlc[i]
            tp.xmws[0] += tp.xst[i, 0] * const.xmw[i]
            tp.xmws[1] += tp.xst[i, 1] * const.xmw[i]
            tp.xmws[5] += tp.xst[i, 5] * const.xmw[i]
            tp.xmws[7] += tp.xst[i, 7] * const.xmw[i]
            tp.xmws[8] += tp.xst[i, 8] * const.xmw[i]
            tp.xmws[9] += tp.xst[i, 9] * const.xmw[i]

        # Stream temperatures
        tp.tst[5] = tp.tcv
        tp.tst[7] = tp.tcr
        tp.tst[8] = tp.tcs
        tp.tst[9] = tp.tcs
        tp.tst[10] = tp.tcs
        tp.tst[12] = tp.tcc

        # Calculate stream enthalpies
        tp.hst[0] = self._tesub1(tp.xst[:, 0], tp.tst[0], 1)
        tp.hst[1] = self._tesub1(tp.xst[:, 1], tp.tst[1], 1)
        tp.hst[2] = self._tesub1(tp.xst[:, 2], tp.tst[2], 1)
        tp.hst[3] = self._tesub1(tp.xst[:, 3], tp.tst[3], 1)
        tp.hst[5] = self._tesub1(tp.xst[:, 5], tp.tst[5], 1)
        tp.hst[7] = self._tesub1(tp.xst[:, 7], tp.tst[7], 1)
        tp.hst[8] = self._tesub1(tp.xst[:, 8], tp.tst[8], 1)
        tp.hst[9] = tp.hst[8]
        tp.hst[10] = self._tesub1(tp.xst[:, 10], tp.tst[10], 0)
        tp.hst[12] = self._tesub1(tp.xst[:, 12], tp.tst[12], 0)

        # Calculate flows
        tp.ftm[0] = vpos[0] * tp.vrng[0] / 100.0
        tp.ftm[1] = vpos[1] * tp.vrng[1] / 100.0
        tp.ftm[2] = vpos[2] * (1.0 - idv[5]) * tp.vrng[2] / 100.0
        tp.ftm[3] = vpos[3] * (1.0 - idv[6] * 0.2) * tp.vrng[3] / 100.0 + 1.0e-10
        tp.ftm[10] = vpos[6] * tp.vrng[6] / 100.0
        tp.ftm[12] = vpos[7] * tp.vrng[7] / 100.0

        uac = vpos[8] * tp.vrng[8] * (1.0 + self._tesub8(9, time)) / 100.0
        tp.fwr = vpos[9] * tp.vrng[9] / 100.0
        tp.fws = vpos[10] * tp.vrng[10] / 100.0
        tp.agsp = (vpos[11] + 150.0) / 100.0

        # Pressure-driven flows
        dlp = tp.ptv - tp.ptr
        if dlp < 0.0:
            dlp = 0.0
        flms = 1937.6 * np.sqrt(dlp)
        tp.ftm[5] = flms / tp.xmws[5] if tp.xmws[5] > 0 else 0.0

        dlp = tp.ptr - tp.pts
        if dlp < 0.0:
            dlp = 0.0
        flms = 4574.21 * np.sqrt(dlp) * (1.0 - 0.25 * self._tesub8(12, time))
        tp.ftm[7] = flms / tp.xmws[7] if tp.xmws[7] > 0 else 0.0

        dlp = tp.pts - 760.0
        if dlp < 0.0:
            dlp = 0.0
        flms = vpos[5] * 0.151169 * np.sqrt(dlp)
        tp.ftm[9] = flms / tp.xmws[9] if tp.xmws[9] > 0 else 0.0

        # Compressor
        pr = tp.ptv / tp.pts if tp.pts > 0 else 1.0
        if pr < 1.0:
            pr = 1.0
        if pr > tp.cpprmx:
            pr = tp.cpprmx
        flcoef = tp.cpflmx / 1.197
        flms = tp.cpflmx + flcoef * (1.0 - pr**3)
        tp.cpdh = flms * (tp.tcs + 273.15) * 1.8e-6 * 1.9872 * (tp.ptv - tp.pts) / (tp.xmws[8] * tp.pts) if (tp.xmws[8] * tp.pts) > 0 else 0.0

        dlp = tp.ptv - tp.pts
        if dlp < 0.0:
            dlp = 0.0
        flms = flms - vpos[4] * 53.349 * np.sqrt(dlp)
        if flms < 1.0e-3:
            flms = 1.0e-3
        tp.ftm[8] = flms / tp.xmws[8] if tp.xmws[8] > 0 else 0.0
        tp.hst[8] = tp.hst[8] + tp.cpdh / tp.ftm[8] if tp.ftm[8] > 0 else tp.hst[8]

        # Component flows
        for i in range(8):
            tp.fcm[i, 0] = tp.xst[i, 0] * tp.ftm[0]
            tp.fcm[i, 1] = tp.xst[i, 1] * tp.ftm[1]
            tp.fcm[i, 2] = tp.xst[i, 2] * tp.ftm[2]
            tp.fcm[i, 3] = tp.xst[i, 3] * tp.ftm[3]
            tp.fcm[i, 5] = tp.xst[i, 5] * tp.ftm[5]
            tp.fcm[i, 7] = tp.xst[i, 7] * tp.ftm[7]
            tp.fcm[i, 8] = tp.xst[i, 8] * tp.ftm[8]
            tp.fcm[i, 9] = tp.xst[i, 9] * tp.ftm[9]
            tp.fcm[i, 10] = tp.xst[i, 10] * tp.ftm[10]
            tp.fcm[i, 12] = tp.xst[i, 12] * tp.ftm[12]

        # Stripper separation
        if tp.ftm[10] > 0.1:
            if tp.tcc > 170.0:
                tmpfac = tp.tcc - 120.262
            elif tp.tcc < 5.292:
                tmpfac = 0.1
            else:
                tmpfac = 363.744 / (177.0 - tp.tcc) - 2.22579488
            vovrl = tp.ftm[3] / tp.ftm[10] * tmpfac
            tp.sfr[3] = 8.5010 * vovrl / (1.0 + 8.5010 * vovrl)
            tp.sfr[4] = 11.402 * vovrl / (1.0 + 11.402 * vovrl)
            tp.sfr[5] = 11.795 * vovrl / (1.0 + 11.795 * vovrl)
            tp.sfr[6] = 0.0480 * vovrl / (1.0 + 0.0480 * vovrl)
            tp.sfr[7] = 0.0242 * vovrl / (1.0 + 0.0242 * vovrl)
        else:
            tp.sfr[3] = 0.9999
            tp.sfr[4] = 0.999
            tp.sfr[5] = 0.999
            tp.sfr[6] = 0.99
            tp.sfr[7] = 0.98

        # Stripper inlet flows
        fin = np.zeros(8)
        for i in range(8):
            fin[i] = tp.fcm[i, 3] + tp.fcm[i, 10]

        # Stripper separation
        tp.ftm[4] = 0.0
        tp.ftm[11] = 0.0
        for i in range(8):
            tp.fcm[i, 4] = tp.sfr[i] * fin[i]
            tp.fcm[i, 11] = fin[i] - tp.fcm[i, 4]
            tp.ftm[4] += tp.fcm[i, 4]
            tp.ftm[11] += tp.fcm[i, 11]

        # Stream compositions
        for i in range(8):
            tp.xst[i, 4] = tp.fcm[i, 4] / tp.ftm[4] if tp.ftm[4] > 0 else 0.0
            tp.xst[i, 11] = tp.fcm[i, 11] / tp.ftm[11] if tp.ftm[11] > 0 else 0.0

        tp.tst[4] = tp.tcc
        tp.tst[11] = tp.tcc
        tp.hst[4] = self._tesub1(tp.xst[:, 4], tp.tst[4], 1)
        tp.hst[11] = self._tesub1(tp.xst[:, 11], tp.tst[11], 0)

        # Stream 7 = Stream 6
        tp.ftm[6] = tp.ftm[5]
        tp.hst[6] = tp.hst[5]
        tp.tst[6] = tp.tst[5]
        for i in range(8):
            tp.xst[i, 6] = tp.xst[i, 5]
            tp.fcm[i, 6] = tp.fcm[i, 5]

        # Heat transfer calculations
        if tp.vlr / 7.8 > 50.0:
            uarlev = 1.0
        elif tp.vlr / 7.8 < 10.0:
            uarlev = 0.0
        else:
            uarlev = 0.025 * tp.vlr / 7.8 - 0.25

        tp.uar = uarlev * (-0.5 * tp.agsp**2 + 2.75 * tp.agsp - 2.5) * 855490.0e-6
        tp.qur = tp.uar * (tp.twr - tp.tcr) * (1.0 - 0.35 * self._tesub8(10, time))

        uas = 0.404655 * (1.0 - 1.0 / (1.0 + (tp.ftm[7] / 3528.73)**4))
        tp.qus = uas * (tp.tws - tp.tst[7]) * (1.0 - 0.25 * self._tesub8(11, time))

        tp.quc = 0.0
        if tp.tcc < 100.0:
            tp.quc = uac * (100.0 - tp.tcc)

        # Calculate measurements (without noise first)
        self._xmeas[0] = tp.ftm[2] * 0.359 / 35.3145
        self._xmeas[1] = tp.ftm[0] * tp.xmws[0] * 0.454
        self._xmeas[2] = tp.ftm[1] * tp.xmws[1] * 0.454
        self._xmeas[3] = tp.ftm[3] * 0.359 / 35.3145
        self._xmeas[4] = tp.ftm[8] * 0.359 / 35.3145
        self._xmeas[5] = tp.ftm[5] * 0.359 / 35.3145
        self._xmeas[6] = (tp.ptr - 760.0) / 760.0 * 101.325
        self._xmeas[7] = (tp.vlr - 84.6) / 666.7 * 100.0
        self._xmeas[8] = tp.tcr
        self._xmeas[9] = tp.ftm[9] * 0.359 / 35.3145
        self._xmeas[10] = tp.tcs
        self._xmeas[11] = (tp.vls - 27.5) / 290.0 * 100.0
        self._xmeas[12] = (tp.pts - 760.0) / 760.0 * 101.325
        self._xmeas[13] = tp.ftm[10] / tp.dls / 35.3145 if tp.dls > 0 else 0.0
        self._xmeas[14] = (tp.vlc - 78.25) / tp.vtc * 100.0
        self._xmeas[15] = (tp.ptv - 760.0) / 760.0 * 101.325
        self._xmeas[16] = tp.ftm[12] / tp.dlc / 35.3145 if tp.dlc > 0 else 0.0
        self._xmeas[17] = tp.tcc
        self._xmeas[18] = tp.quc * 1.04e3 * 0.454
        self._xmeas[19] = tp.cpdh * 0.29307e3
        self._xmeas[20] = tp.twr
        self._xmeas[21] = tp.tws

        # Safety shutdown check
        isd = 0
        if self._xmeas[6] > 3000.0:
            isd = 1
        if tp.vlr / 35.3145 > 24.0:
            isd = 1
        if tp.vlr / 35.3145 < 2.0:
            isd = 1
        if self._xmeas[8] > 175.0:
            isd = 1
        if tp.vls / 35.3145 > 12.0:
            isd = 1
        if tp.vls / 35.3145 < 1.0:
            isd = 1
        if tp.vlc / 35.3145 > 8.0:
            isd = 1
        if tp.vlc / 35.3145 < 1.0:
            isd = 1

        # Add measurement noise (only after time 0)
        if time > 0.0 and isd == 0:
            for i in range(22):
                self._xmeas[i] += self._tesub6(tp.xns[i])

        # Sampled composition measurements (with delay)
        xcmp = np.zeros(41)
        xcmp[22] = tp.xst[0, 6] * 100.0
        xcmp[23] = tp.xst[1, 6] * 100.0
        xcmp[24] = tp.xst[2, 6] * 100.0
        xcmp[25] = tp.xst[3, 6] * 100.0
        xcmp[26] = tp.xst[4, 6] * 100.0
        xcmp[27] = tp.xst[5, 6] * 100.0
        xcmp[28] = tp.xst[0, 9] * 100.0
        xcmp[29] = tp.xst[1, 9] * 100.0
        xcmp[30] = tp.xst[2, 9] * 100.0
        xcmp[31] = tp.xst[3, 9] * 100.0
        xcmp[32] = tp.xst[4, 9] * 100.0
        xcmp[33] = tp.xst[5, 9] * 100.0
        xcmp[34] = tp.xst[6, 9] * 100.0
        xcmp[35] = tp.xst[7, 9] * 100.0
        xcmp[36] = tp.xst[3, 12] * 100.0
        xcmp[37] = tp.xst[4, 12] * 100.0
        xcmp[38] = tp.xst[5, 12] * 100.0
        xcmp[39] = tp.xst[6, 12] * 100.0
        xcmp[40] = tp.xst[7, 12] * 100.0

        if time == 0.0:
            for i in range(22, 41):
                tp.xdel[i] = xcmp[i]
                self._xmeas[i] = xcmp[i]
            tp.tgas = 0.1
            tp.tprod = 0.25

        if time >= tp.tgas:
            for i in range(22, 36):
                self._xmeas[i] = tp.xdel[i] + self._tesub6(tp.xns[i])
                tp.xdel[i] = xcmp[i]
            tp.tgas += 0.1

        if time >= tp.tprod:
            for i in range(36, 41):
                self._xmeas[i] = tp.xdel[i] + self._tesub6(tp.xns[i])
                tp.xdel[i] = xcmp[i]
            tp.tprod += 0.25

        # State derivatives
        # Reactor component balances
        for i in range(8):
            yp[i] = tp.fcm[i, 6] - tp.fcm[i, 7] + tp.crxr[i]
            yp[i + 9] = tp.fcm[i, 7] - tp.fcm[i, 8] - tp.fcm[i, 9] - tp.fcm[i, 10]
            yp[i + 18] = tp.fcm[i, 11] - tp.fcm[i, 12]
            yp[i + 27] = tp.fcm[i, 0] + tp.fcm[i, 1] + tp.fcm[i, 2] + tp.fcm[i, 4] + tp.fcm[i, 8] - tp.fcm[i, 5]

        # Energy balances
        yp[8] = tp.hst[6] * tp.ftm[6] - tp.hst[7] * tp.ftm[7] + tp.rh + tp.qur
        yp[17] = tp.hst[7] * tp.ftm[7] - tp.hst[8] * tp.ftm[8] - tp.hst[9] * tp.ftm[9] - tp.hst[10] * tp.ftm[10] + tp.qus
        yp[26] = tp.hst[3] * tp.ftm[3] + tp.hst[10] * tp.ftm[10] - tp.hst[4] * tp.ftm[4] - tp.hst[12] * tp.ftm[12] + tp.quc
        yp[35] = tp.hst[0] * tp.ftm[0] + tp.hst[1] * tp.ftm[1] + tp.hst[2] * tp.ftm[2] + tp.hst[4] * tp.ftm[4] + tp.hst[8] * tp.ftm[8] - tp.hst[5] * tp.ftm[5]

        # Cooling water temperatures
        yp[36] = (tp.fwr * 500.53 * (tp.tcwr - tp.twr) - tp.qur * 1.0e6 / 1.8) / tp.hwr
        yp[37] = (tp.fws * 500.53 * (tp.tcws - tp.tws) - tp.qus * 1.0e6 / 1.8) / tp.hws

        # Valve sticking
        tp.ivst[9] = idv[13]   # IDV(14)
        tp.ivst[10] = idv[14]  # IDV(15)
        tp.ivst[4] = idv[18]   # IDV(19)
        tp.ivst[6] = idv[18]   # IDV(19)
        tp.ivst[7] = idv[18]   # IDV(19)
        tp.ivst[8] = idv[18]   # IDV(19)

        # Valve dynamics
        for i in range(12):
            if time == 0.0 or abs(tp.vcv[i] - self._xmv[i]) > tp.vst[i] * tp.ivst[i]:
                tp.vcv[i] = self._xmv[i]
            tp.vcv[i] = np.clip(tp.vcv[i], 0.0, 100.0)
            yp[38 + i] = (tp.vcv[i] - vpos[i]) / tp.vtau[i]

        # Shutdown: zero all derivatives
        if isd != 0:
            yp[:] = 0.0

    # =========================================================================
    # Public Interface Methods
    # =========================================================================

    def step(self, dt: float = 1.0/3600.0):
        """Single Euler integration step.

        Parameters
        ----------
        dt : float
            Time step in hours. Default is 1 second (1/3600 hours).
        """
        if not self._initialized:
            raise RuntimeError("Process not initialized. Call initialize() first.")

        # Call tefunc to get derivatives
        self._tefunc()

        # Euler integration: yy = yy + yp * dt
        self.yy[:] = self.yy + self.yp * dt
        self.time += dt

        # Apply valve constraints
        self._apply_constraints()

    def _apply_constraints(self):
        """Apply valve constraints (0-100%)."""
        for i in range(11):
            self._xmv[i] = np.clip(self._xmv[i], 0.0, 100.0)

    @property
    def xmeas(self) -> np.ndarray:
        """Get current measurement values."""
        return self._xmeas.copy()

    @property
    def xmv(self) -> np.ndarray:
        """Get current manipulated variables."""
        return self._xmv.copy()

    @property
    def idv(self) -> np.ndarray:
        """Get current disturbance vector."""
        return self._idv.copy()

    def set_xmv(self, index: int, value: float):
        """Set a manipulated variable.

        Parameters
        ----------
        index : int
            1-based index of the manipulated variable (1-12).
        value : float
            Value to set (will be clipped to 0-100).
        """
        if not 1 <= index <= 12:
            raise ValueError(f"XMV index must be 1-12, got {index}")
        self._xmv[index - 1] = np.clip(value, 0.0, 100.0)

    def set_idv(self, index: int, value: int):
        """Set a disturbance variable.

        Parameters
        ----------
        index : int
            1-based index of the disturbance (1-20).
        value : int
            0 to disable, 1 to enable the disturbance.
        """
        if not 1 <= index <= 20:
            raise ValueError(f"IDV index must be 1-20, got {index}")
        self._idv[index - 1] = value

    def clear_disturbances(self):
        """Clear all disturbances (set IDV to 0)."""
        self._idv[:] = 0

    def get_xmeas(self) -> np.ndarray:
        """Get current measurement values (for TEPSimulator compatibility)."""
        return self._xmeas.copy()

    def get_xmv(self) -> np.ndarray:
        """Get current manipulated variables (for TEPSimulator compatibility)."""
        return self._xmv.copy()

    def evaluate(self, time: float, yy: np.ndarray) -> np.ndarray:
        """Evaluate derivatives using TEFUNC.

        Parameters
        ----------
        time : float
            Current time in hours.
        yy : np.ndarray
            Current state vector.

        Returns
        -------
        np.ndarray
            Derivative vector.
        """
        self.yy[:] = yy
        self.time = time
        self._tefunc()
        return self.yp.copy()

    def is_shutdown(self) -> bool:
        """Check if process is in shutdown state."""
        # Check shutdown conditions
        tp = self._teproc
        if self._xmeas[6] > 3000.0:
            return True
        if tp.vlr / 35.3145 > 24.0 or tp.vlr / 35.3145 < 2.0:
            return True
        if self._xmeas[8] > 175.0:
            return True
        if tp.vls / 35.3145 > 12.0 or tp.vls / 35.3145 < 1.0:
            return True
        if tp.vlc / 35.3145 > 8.0 or tp.vlc / 35.3145 < 1.0:
            return True
        return False

    def get_state(self) -> np.ndarray:
        """Get current state vector."""
        return self.yy.copy()

    def set_state(self, state: np.ndarray):
        """Set state vector."""
        if len(state) != self._nn:
            raise ValueError(f"State must have {self._nn} elements")
        self.yy[:] = state


class PythonTEProcessState:
    """State wrapper that mimics TEProcess state interface."""

    def __init__(self, process: 'PythonTEProcess'):
        self._process = process

    @property
    def yy(self) -> np.ndarray:
        return self._process.yy

    @yy.setter
    def yy(self, value: np.ndarray):
        self._process.yy[:] = value

    @property
    def xmeas(self) -> np.ndarray:
        return self._process.xmeas

    @property
    def xmv(self) -> np.ndarray:
        return self._process.xmv


class PythonDisturbanceManager:
    """Disturbance manager wrapper for API compatibility."""

    def __init__(self, process: 'PythonTEProcess'):
        self._process = process

    def clear_all_disturbances(self):
        """Clear all disturbances (set IDV to 0)."""
        self._process.clear_disturbances()

    def set_disturbance(self, index: int, value: int):
        """Set a disturbance variable."""
        self._process.set_idv(index, value)
