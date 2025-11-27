"""
Core Tennessee Eastman Process model.

This module implements the main process dynamics from TEFUNC in the
original Fortran code. It calculates the derivatives of all 50 state
variables for numerical integration.

The process consists of:
- Reactor (exothermic reactions, gas-liquid equilibrium)
- Product Separator (liquid-vapor separation)
- Stripper (steam stripping column)
- Compressor (vapor recycle)
- Condenser (vapor condensation)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict
from .constants import (
    NUM_STATES, NUM_COMPONENTS, NUM_STREAMS, NUM_MEASUREMENTS,
    NUM_MANIPULATED_VARS, XMW, AVP, BVP, CVP, XNS,
    VTR, VTS, VTC, VTV, HWR, HWS, HTR, RG, CPFLMX, CPPRMX,
    VRNG, VTAU, VST_DEFAULT, INITIAL_STATES, XST_INITIAL, TST_INITIAL,
    SFR_INITIAL, SAFETY_LIMITS
)
from .thermodynamics import (
    calculate_enthalpy, calculate_temperature, calculate_liquid_density,
    calculate_vapor_pressure, calculate_mixture_molecular_weight
)
from .disturbances import DisturbanceManager, RandomGenerator


@dataclass
class ProcessState:
    """
    Container for all process state variables and intermediate calculations.

    This provides a structured way to access process variables by name
    rather than array indices.
    """
    # State vector (50 elements)
    yy: np.ndarray = field(default_factory=lambda: INITIAL_STATES.copy())

    # Measurements (41 elements)
    xmeas: np.ndarray = field(default_factory=lambda: np.zeros(NUM_MEASUREMENTS))

    # Manipulated variables (12 elements)
    xmv: np.ndarray = field(default_factory=lambda: INITIAL_STATES[38:50].copy())

    # Valve positions (actual, after dynamics)
    vpos: np.ndarray = field(default_factory=lambda: np.zeros(12))

    # Valve command values
    vcv: np.ndarray = field(default_factory=lambda: np.zeros(12))

    # Stream compositions (8 components x 13 streams)
    xst: np.ndarray = field(default_factory=lambda: np.zeros((NUM_COMPONENTS, NUM_STREAMS)))

    # Stream temperatures
    tst: np.ndarray = field(default_factory=lambda: np.zeros(NUM_STREAMS))

    # Stream enthalpies
    hst: np.ndarray = field(default_factory=lambda: np.zeros(NUM_STREAMS))

    # Stream total flows
    ftm: np.ndarray = field(default_factory=lambda: np.zeros(NUM_STREAMS))

    # Stream component flows
    fcm: np.ndarray = field(default_factory=lambda: np.zeros((NUM_COMPONENTS, NUM_STREAMS)))

    # Stream molecular weights
    xmws: np.ndarray = field(default_factory=lambda: np.zeros(NUM_STREAMS))

    # Separation factors
    sfr: np.ndarray = field(default_factory=lambda: SFR_INITIAL.copy())

    # Delayed compositions for sampled measurements
    xdel: np.ndarray = field(default_factory=lambda: np.zeros(NUM_MEASUREMENTS))

    # Valve sticking status
    ivst: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=int))

    # Valve sticking threshold
    vst: np.ndarray = field(default_factory=lambda: np.full(12, VST_DEFAULT))

    # Sampling times
    tgas: float = 0.1
    tprod: float = 0.25

    # Shutdown flag
    isd: int = 0


class TEProcess:
    """
    Tennessee Eastman Process model.

    This class implements the core process dynamics including:
    - Mass and energy balances for all units
    - Reaction kinetics
    - Thermodynamic equilibrium calculations
    - Measurement generation with noise
    - Safety shutdown logic
    """

    def __init__(self, random_seed: int = None):
        """
        Initialize the TEP process model.

        Args:
            random_seed: Optional seed for random number generation
        """
        from .constants import DEFAULT_RANDOM_SEED

        if random_seed is None:
            random_seed = DEFAULT_RANDOM_SEED

        self.disturbances = DisturbanceManager(random_seed)
        self.rng = RandomGenerator(random_seed)
        self.state = ProcessState()

        # Initialize state
        self._initialize()

    def _initialize(self):
        """Initialize process to steady-state conditions."""
        # Set initial states
        self.state.yy = INITIAL_STATES.copy()

        # Initialize manipulated variables from state vector
        self.state.xmv = self.state.yy[38:50].copy()
        self.state.vcv = self.state.xmv.copy()

        # Initialize feed stream compositions
        for stream_idx, comp in XST_INITIAL.items():
            self.state.xst[:, stream_idx - 1] = comp

        # Initialize feed temperatures
        for stream_idx, temp in TST_INITIAL.items():
            self.state.tst[stream_idx - 1] = temp

        # Initialize separation factors
        self.state.sfr = SFR_INITIAL.copy()

        # Reset disturbances
        self.disturbances.clear_all_disturbances()

    def set_idv(self, index: int, value: int = 1):
        """Set a disturbance flag."""
        self.disturbances.set_idv(index, value)

    def set_xmv(self, index: int, value: float):
        """
        Set a manipulated variable.

        Args:
            index: MV index (1-12 in Fortran convention)
            value: Valve position (0-100%)
        """
        idx = index - 1 if index >= 1 else index
        if 0 <= idx < NUM_MANIPULATED_VARS:
            self.state.xmv[idx] = np.clip(value, 0.0, 100.0)

    def get_xmeas(self) -> np.ndarray:
        """Get current measurements."""
        return self.state.xmeas.copy()

    def get_xmv(self) -> np.ndarray:
        """Get current manipulated variable values."""
        return self.state.xmv.copy()

    def evaluate(self, time: float, yy: np.ndarray) -> np.ndarray:
        """
        Evaluate process derivatives (TEFUNC equivalent).

        This is the main function that computes dy/dt for all 50 state variables.

        Args:
            time: Current simulation time (hours)
            yy: Current state vector (50 elements)

        Returns:
            Derivative vector yp (50 elements)
        """
        # Update disturbance random walks
        self.disturbances.update_walks(time)

        # Initialize derivative vector
        yp = np.zeros(NUM_STATES)

        # Extract state variables
        # States 1-3: Reactor vapor component holdups (A, B, C)
        ucvr = np.zeros(NUM_COMPONENTS)
        ucvr[0:3] = yy[0:3]

        # States 4-8: Reactor liquid component holdups (D, E, F, G, H)
        uclr = np.zeros(NUM_COMPONENTS)
        uclr[3:8] = yy[3:8]

        # State 9: Reactor total energy
        etr = yy[8]

        # States 10-17: Separator liquid holdups (8 components)
        ucls = np.zeros(NUM_COMPONENTS)
        ucvs = np.zeros(NUM_COMPONENTS)
        ucls[0:3] = 0.0
        ucvs[0:3] = yy[9:12]
        ucls[3:8] = yy[12:17]

        # State 18: Separator total energy
        ets = yy[17]

        # States 19-26: Stripper/Condenser liquid holdups
        uclc = yy[18:26]

        # State 27: Stripper total energy
        etc = yy[26]

        # States 28-35: Vapor header holdups
        ucvv = yy[27:35]

        # State 36: Vapor header energy
        etv = yy[35]

        # States 37-38: Cooling water temperatures
        twr = yy[36]  # Reactor cooling water
        tws = yy[37]  # Separator cooling water

        # States 39-50: Valve positions
        vpos = yy[38:50]
        self.state.vpos = vpos.copy()

        # Get disturbance-affected parameters
        xst4_abc = self.disturbances.get_xst_composition(time)
        self.state.xst[0, 3] = xst4_abc[0]  # A in stream 4
        self.state.xst[1, 3] = xst4_abc[1]  # B in stream 4
        self.state.xst[2, 3] = xst4_abc[2]  # C in stream 4

        tst1 = self.disturbances.get_feed_temperature(1, time)
        tst4 = self.disturbances.get_feed_temperature(4, time)
        self.state.tst[0] = tst1
        self.state.tst[3] = tst4

        tcwr = self.disturbances.get_cooling_water_temp('reactor', time)
        tcws = self.disturbances.get_cooling_water_temp('condenser', time)

        r1f = self.disturbances.get_reaction_factor(1, time)
        r2f = self.disturbances.get_reaction_factor(2, time)

        # Calculate total holdups
        utlr = np.sum(uclr)
        utls = np.sum(ucls)
        utlc = np.sum(uclc)
        utvv = np.sum(ucvv)

        # Calculate mole fractions
        xlr = uclr / utlr if utlr > 0 else np.zeros(NUM_COMPONENTS)
        xls = ucls / utls if utls > 0 else np.zeros(NUM_COMPONENTS)
        xlc = uclc / utlc if utlc > 0 else np.zeros(NUM_COMPONENTS)
        xvv = ucvv / utvv if utvv > 0 else np.zeros(NUM_COMPONENTS)

        # Calculate specific energies
        esr = etr / utlr if utlr > 0 else 0.0
        ess = ets / utls if utls > 0 else 0.0
        esc = etc / utlc if utlc > 0 else 0.0
        esv = etv / utvv if utvv > 0 else 0.0

        # Calculate temperatures from energy
        tcr = calculate_temperature(xlr, esr, phase=0, initial_guess=120.0)
        tkr = tcr + 273.15

        tcs = calculate_temperature(xls, ess, phase=0, initial_guess=80.0)
        tks = tcs + 273.15

        tcc = calculate_temperature(xlc, esc, phase=0, initial_guess=65.0)

        tcv = calculate_temperature(xvv, esv, phase=2, initial_guess=100.0)
        tkv = tcv + 273.15

        # Calculate liquid densities
        dlr = calculate_liquid_density(xlr, tcr)
        dls = calculate_liquid_density(xls, tcs)
        dlc = calculate_liquid_density(xlc, tcc)

        # Calculate volumes
        vlr = utlr / dlr if dlr > 0 else 0.0
        vls = utls / dls if dls > 0 else 0.0
        vlc = utlc / dlc if dlc > 0 else 0.0

        vvr = VTR - vlr
        vvs = VTS - vls

        # Calculate pressures
        # Non-condensable components (A, B, C) - ideal gas
        ppr = np.zeros(NUM_COMPONENTS)
        pps = np.zeros(NUM_COMPONENTS)

        for i in range(3):
            ppr[i] = ucvr[i] * RG * tkr / vvr if vvr > 0 else 0.0
            pps[i] = ucvs[i] * RG * tks / vvs if vvs > 0 else 0.0

        # Condensable components (D-H) - vapor pressure
        for i in range(3, NUM_COMPONENTS):
            vpr = np.exp(AVP[i] + BVP[i] / (tcr + CVP[i])) if tcr + CVP[i] != 0 else 0.0
            ppr[i] = vpr * xlr[i]

            vpr = np.exp(AVP[i] + BVP[i] / (tcs + CVP[i])) if tcs + CVP[i] != 0 else 0.0
            pps[i] = vpr * xls[i]

        ptr = np.sum(ppr)  # Reactor total pressure
        pts = np.sum(pps)  # Separator total pressure
        ptv = utvv * RG * tkv / VTV if utvv > 0 else 0.0  # Vapor header pressure

        # Calculate vapor compositions
        xvr = ppr / ptr if ptr > 0 else np.zeros(NUM_COMPONENTS)
        xvs = pps / pts if pts > 0 else np.zeros(NUM_COMPONENTS)

        # Calculate total vapor holdups
        utvr = ptr * vvr / RG / tkr if tkr > 0 else 0.0
        utvs = pts * vvs / RG / tks if tks > 0 else 0.0

        # Update vapor component holdups for condensables
        for i in range(3, NUM_COMPONENTS):
            ucvr[i] = utvr * xvr[i]
            ucvs[i] = utvs * xvs[i]

        # Calculate reaction rates
        rr = np.zeros(4)
        rr[0] = np.exp(31.5859536 - 40000.0 / 1.987 / tkr) * r1f
        rr[1] = np.exp(3.00094014 - 20000.0 / 1.987 / tkr) * r2f
        rr[2] = np.exp(53.4060443 - 60000.0 / 1.987 / tkr)
        rr[3] = rr[2] * 0.767488334

        if ppr[0] > 0 and ppr[2] > 0:
            r1f_pp = ppr[0] ** 1.1544
            r2f_pp = ppr[2] ** 0.3735
            rr[0] = rr[0] * r1f_pp * r2f_pp * ppr[3]
            rr[1] = rr[1] * r1f_pp * r2f_pp * ppr[4]
        else:
            rr[0] = 0.0
            rr[1] = 0.0

        rr[2] = rr[2] * ppr[0] * ppr[4]
        rr[3] = rr[3] * ppr[0] * ppr[3]

        # Scale by reactor vapor volume
        rr = rr * vvr

        # Component reaction rates
        crxr = np.zeros(NUM_COMPONENTS)
        crxr[0] = -rr[0] - rr[1] - rr[2]  # A
        crxr[2] = -rr[0] - rr[1]  # C
        crxr[3] = -rr[0] - 1.5 * rr[3]  # D
        crxr[4] = -rr[1] - rr[2]  # E
        crxr[5] = rr[2] + rr[3]  # F
        crxr[6] = rr[0]  # G
        crxr[7] = rr[1]  # H

        # Heat of reaction
        rh = rr[0] * HTR[0] + rr[1] * HTR[1]

        # Update stream compositions
        self.state.xst[:, 5] = xvv  # Stream 6
        self.state.xst[:, 7] = xvr  # Stream 8
        self.state.xst[:, 8] = xvs  # Stream 9
        self.state.xst[:, 9] = xvs  # Stream 10
        self.state.xst[:, 10] = xls  # Stream 11
        self.state.xst[:, 12] = xlc  # Stream 13

        # Calculate stream molecular weights
        xmws = np.zeros(NUM_STREAMS)
        for s in [0, 1, 5, 7, 8, 9]:
            xmws[s] = np.sum(self.state.xst[:, s] * XMW)

        self.state.xmws = xmws

        # Update stream temperatures
        self.state.tst[5] = tcv
        self.state.tst[7] = tcr
        self.state.tst[8] = tcs
        self.state.tst[9] = tcs
        self.state.tst[10] = tcs
        self.state.tst[12] = tcc

        # Calculate stream enthalpies
        for s in [0, 1, 2, 3]:
            self.state.hst[s] = calculate_enthalpy(self.state.xst[:, s], self.state.tst[s], phase=1)
        self.state.hst[5] = calculate_enthalpy(self.state.xst[:, 5], self.state.tst[5], phase=1)
        self.state.hst[7] = calculate_enthalpy(self.state.xst[:, 7], self.state.tst[7], phase=1)
        self.state.hst[8] = calculate_enthalpy(self.state.xst[:, 8], self.state.tst[8], phase=1)
        self.state.hst[9] = self.state.hst[8]
        self.state.hst[10] = calculate_enthalpy(self.state.xst[:, 10], self.state.tst[10], phase=0)
        self.state.hst[12] = calculate_enthalpy(self.state.xst[:, 12], self.state.tst[12], phase=0)

        # Calculate flows
        # Feed flows (valve-controlled)
        ftm = np.zeros(NUM_STREAMS)
        ftm[0] = vpos[0] * VRNG[0] / 100.0  # D Feed
        ftm[1] = vpos[1] * VRNG[1] / 100.0  # E Feed
        ftm[2] = vpos[2] * self.disturbances.get_feed_loss_factor() * VRNG[2] / 100.0  # A Feed
        ftm[3] = vpos[3] * self.disturbances.get_header_pressure_factor() * VRNG[3] / 100.0 + 1e-10  # A+C Feed
        ftm[10] = vpos[6] * VRNG[6] / 100.0  # Separator underflow
        ftm[12] = vpos[7] * VRNG[7] / 100.0  # Stripper product

        # Stripper steam heat transfer
        uac = vpos[8] * VRNG[8] * self.disturbances.get_uac_factor(time) / 100.0

        # Cooling water flows
        fwr = vpos[9] * VRNG[9] / 100.0
        fws = vpos[10] * VRNG[10] / 100.0

        # Agitator speed factor
        agsp = (vpos[11] + 150.0) / 100.0

        # Reactor-to-separator flow (pressure-driven)
        dlp = ptv - ptr
        if dlp < 0:
            dlp = 0.0
        flms = 1937.6 * np.sqrt(dlp)
        ftm[5] = flms / xmws[5] if xmws[5] > 0 else 0.0  # Stream 6

        # Reactor outlet flow (pressure-driven)
        dlp = ptr - pts
        if dlp < 0:
            dlp = 0.0
        flms = 4574.21 * np.sqrt(dlp) * self.disturbances.get_reactor_flow_factor(time)
        ftm[7] = flms / xmws[7] if xmws[7] > 0 else 0.0  # Stream 8

        # Purge flow (pressure-driven)
        dlp = pts - 760.0
        if dlp < 0:
            dlp = 0.0
        flms = vpos[5] * 0.151169 * np.sqrt(dlp)
        ftm[9] = flms / xmws[9] if xmws[9] > 0 else 0.0  # Stream 10

        # Compressor flow
        pr = ptv / pts if pts > 0 else 1.0
        if pr < 1.0:
            pr = 1.0
        if pr > CPPRMX:
            pr = CPPRMX

        flcoef = CPFLMX / 1.197
        flms = CPFLMX + flcoef * (1.0 - pr**3)
        cpdh = flms * (tcs + 273.15) * 1.8e-6 * 1.9872 * (ptv - pts) / (xmws[8] * pts) if xmws[8] * pts > 0 else 0.0

        dlp = ptv - pts
        if dlp < 0:
            dlp = 0.0
        flms = flms - vpos[4] * 53.349 * np.sqrt(dlp)
        if flms < 1e-3:
            flms = 1e-3
        ftm[8] = flms / xmws[8] if xmws[8] > 0 else 0.0  # Stream 9

        # Add compressor work to stream 9 enthalpy
        self.state.hst[8] = self.state.hst[8] + cpdh / ftm[8] if ftm[8] > 0 else self.state.hst[8]

        self.state.ftm = ftm

        # Calculate component flows
        fcm = np.zeros((NUM_COMPONENTS, NUM_STREAMS))
        for i in range(NUM_COMPONENTS):
            fcm[i, 0] = self.state.xst[i, 0] * ftm[0]
            fcm[i, 1] = self.state.xst[i, 1] * ftm[1]
            fcm[i, 2] = self.state.xst[i, 2] * ftm[2]
            fcm[i, 3] = self.state.xst[i, 3] * ftm[3]
            fcm[i, 5] = self.state.xst[i, 5] * ftm[5]
            fcm[i, 7] = self.state.xst[i, 7] * ftm[7]
            fcm[i, 8] = self.state.xst[i, 8] * ftm[8]
            fcm[i, 9] = self.state.xst[i, 9] * ftm[9]
            fcm[i, 10] = self.state.xst[i, 10] * ftm[10]
            fcm[i, 12] = self.state.xst[i, 12] * ftm[12]

        # Stripper separation
        if ftm[10] > 0.1:
            if tcc > 170.0:
                tmpfac = tcc - 120.262
            elif tcc < 5.292:
                tmpfac = 0.1
            else:
                tmpfac = 363.744 / (177.0 - tcc) - 2.22579488

            vovrl = ftm[3] / ftm[10] * tmpfac

            sfr = self.state.sfr.copy()
            sfr[3] = 8.5010 * vovrl / (1.0 + 8.5010 * vovrl)
            sfr[4] = 11.402 * vovrl / (1.0 + 11.402 * vovrl)
            sfr[5] = 11.795 * vovrl / (1.0 + 11.795 * vovrl)
            sfr[6] = 0.0480 * vovrl / (1.0 + 0.0480 * vovrl)
            sfr[7] = 0.0242 * vovrl / (1.0 + 0.0242 * vovrl)
            self.state.sfr = sfr
        else:
            sfr = np.array([0.9999, 0.999, 0.999, 0.9999, 0.999, 0.999, 0.99, 0.98])
            self.state.sfr = sfr

        # Stripper inlet
        fin = fcm[:, 3] + fcm[:, 10]

        # Stripper vapor/liquid split
        ftm[4] = 0.0  # Stream 5 (vapor to recycle)
        ftm[11] = 0.0  # Stream 12 (liquid to stripper bottom)
        for i in range(NUM_COMPONENTS):
            fcm[i, 4] = self.state.sfr[i] * fin[i]
            fcm[i, 11] = fin[i] - fcm[i, 4]
            ftm[4] += fcm[i, 4]
            ftm[11] += fcm[i, 11]

        # Update stream compositions for stripper
        self.state.xst[:, 4] = fcm[:, 4] / ftm[4] if ftm[4] > 0 else np.zeros(NUM_COMPONENTS)
        self.state.xst[:, 11] = fcm[:, 11] / ftm[11] if ftm[11] > 0 else np.zeros(NUM_COMPONENTS)

        self.state.tst[4] = tcc
        self.state.tst[11] = tcc
        self.state.hst[4] = calculate_enthalpy(self.state.xst[:, 4], self.state.tst[4], phase=1)
        self.state.hst[11] = calculate_enthalpy(self.state.xst[:, 11], self.state.tst[11], phase=0)

        # Stream 7 = Stream 6 (feed to reactor)
        ftm[6] = ftm[5]
        self.state.hst[6] = self.state.hst[5]
        self.state.tst[6] = self.state.tst[5]
        self.state.xst[:, 6] = self.state.xst[:, 5].copy()
        fcm[:, 6] = fcm[:, 5].copy()

        self.state.fcm = fcm
        self.state.ftm = ftm

        # Heat transfer calculations
        # Reactor heat transfer
        if vlr / 7.8 > 50.0:
            uarlev = 1.0
        elif vlr / 7.8 < 10.0:
            uarlev = 0.0
        else:
            uarlev = 0.025 * vlr / 7.8 - 0.25

        uar = uarlev * (-0.5 * agsp**2 + 2.75 * agsp - 2.5) * 855490e-6
        qur = uar * (twr - tcr) * self.disturbances.get_reactor_ht_factor(time)

        # Separator heat transfer
        uas = 0.404655 * (1.0 - 1.0 / (1.0 + (ftm[7] / 3528.73)**4))
        qus = uas * (tws - self.state.tst[7]) * self.disturbances.get_separator_ht_factor(time)

        # Condenser heat transfer
        quc = 0.0
        if tcc < 100.0:
            quc = uac * (100.0 - tcc)

        # Calculate measurements
        self._calculate_measurements(
            ftm, vlr, vls, vlc, ptr, pts, ptv, tcr, tcs, tcc, tcv,
            twr, tws, dlr, dls, dlc, cpdh, quc, time
        )

        # Check safety shutdown
        isd = self._check_safety(vlr, vls, vlc)

        # Calculate derivatives
        # Reactor component balances (states 1-8)
        for i in range(NUM_COMPONENTS):
            yp[i] = fcm[i, 6] - fcm[i, 7] + crxr[i]

        # Reactor energy balance (state 9)
        yp[8] = self.state.hst[6] * ftm[6] - self.state.hst[7] * ftm[7] + rh + qur

        # Separator component balances (states 10-17)
        for i in range(NUM_COMPONENTS):
            yp[i + 9] = fcm[i, 7] - fcm[i, 8] - fcm[i, 9] - fcm[i, 10]

        # Separator energy balance (state 18)
        yp[17] = (self.state.hst[7] * ftm[7] - self.state.hst[8] * ftm[8] -
                  self.state.hst[9] * ftm[9] - self.state.hst[10] * ftm[10] + qus)

        # Stripper/Condenser component balances (states 19-26)
        for i in range(NUM_COMPONENTS):
            yp[i + 18] = fcm[i, 11] - fcm[i, 12]

        # Stripper energy balance (state 27)
        yp[26] = (self.state.hst[3] * ftm[3] + self.state.hst[10] * ftm[10] -
                  self.state.hst[4] * ftm[4] - self.state.hst[12] * ftm[12] + quc)

        # Vapor header component balances (states 28-35)
        for i in range(NUM_COMPONENTS):
            yp[i + 27] = (fcm[i, 0] + fcm[i, 1] + fcm[i, 2] + fcm[i, 4] +
                         fcm[i, 8] - fcm[i, 5])

        # Vapor header energy balance (state 36)
        yp[35] = (self.state.hst[0] * ftm[0] + self.state.hst[1] * ftm[1] +
                  self.state.hst[2] * ftm[2] + self.state.hst[4] * ftm[4] +
                  self.state.hst[8] * ftm[8] - self.state.hst[5] * ftm[5])

        # Cooling water dynamics (states 37-38)
        yp[36] = (fwr * 500.53 * (tcwr - twr) - qur * 1e6 / 1.8) / HWR
        yp[37] = (fws * 500.53 * (tcws - tws) - qus * 1e6 / 1.8) / HWS

        # Valve dynamics (states 39-50)
        # Update sticking flags
        self.state.ivst[9] = int(self.disturbances.is_valve_stuck(10))
        self.state.ivst[10] = int(self.disturbances.is_valve_stuck(11))
        self.state.ivst[4] = int(self.disturbances.idv[18])  # IDV(19)
        self.state.ivst[6] = int(self.disturbances.idv[18])
        self.state.ivst[7] = int(self.disturbances.idv[18])
        self.state.ivst[8] = int(self.disturbances.idv[18])

        for i in range(12):
            # Check if valve should respond to command
            if time == 0.0 or abs(self.state.vcv[i] - self.state.xmv[i]) > self.state.vst[i] * self.state.ivst[i]:
                self.state.vcv[i] = self.state.xmv[i]

            self.state.vcv[i] = np.clip(self.state.vcv[i], 0.0, 100.0)
            yp[i + 38] = (self.state.vcv[i] - vpos[i]) / VTAU[i]

        # Shutdown: freeze all derivatives
        if isd != 0:
            yp[:] = 0.0

        self.state.isd = isd

        return yp

    def _calculate_measurements(
        self, ftm, vlr, vls, vlc, ptr, pts, ptv, tcr, tcs, tcc, tcv,
        twr, tws, dlr, dls, dlc, cpdh, quc, time
    ):
        """Calculate process measurements with noise."""
        xmeas = np.zeros(NUM_MEASUREMENTS)

        # Continuous measurements (1-22)
        xmeas[0] = ftm[2] * 0.359 / 35.3145  # A Feed
        xmeas[1] = ftm[0] * self.state.xmws[0] * 0.454  # D Feed
        xmeas[2] = ftm[1] * self.state.xmws[1] * 0.454  # E Feed
        xmeas[3] = ftm[3] * 0.359 / 35.3145  # A+C Feed
        xmeas[4] = ftm[8] * 0.359 / 35.3145  # Recycle
        xmeas[5] = ftm[5] * 0.359 / 35.3145  # Reactor feed
        xmeas[6] = (ptr - 760.0) / 760.0 * 101.325  # Reactor pressure
        xmeas[7] = (vlr - 84.6) / 666.7 * 100.0  # Reactor level
        xmeas[8] = tcr  # Reactor temperature
        xmeas[9] = ftm[9] * 0.359 / 35.3145  # Purge rate
        xmeas[10] = tcs  # Separator temperature
        xmeas[11] = (vls - 27.5) / 290.0 * 100.0  # Separator level
        xmeas[12] = (pts - 760.0) / 760.0 * 101.325  # Separator pressure
        xmeas[13] = ftm[10] / dls / 35.3145 if dls > 0 else 0.0  # Separator underflow
        xmeas[14] = (vlc - 78.25) / VTC * 100.0  # Stripper level
        xmeas[15] = (ptv - 760.0) / 760.0 * 101.325  # Stripper pressure
        xmeas[16] = ftm[12] / dlc / 35.3145 if dlc > 0 else 0.0  # Stripper underflow
        xmeas[17] = tcc  # Stripper temperature
        xmeas[18] = quc * 1.04e3 * 0.454  # Stripper steam flow
        xmeas[19] = cpdh * 0.29307e3  # Compressor work
        xmeas[20] = twr  # Reactor CW outlet temp
        xmeas[21] = tws  # Separator CW outlet temp

        # Check if safe to add noise
        if time > 0.0 and self.state.isd == 0:
            for i in range(22):
                noise = self.rng.gaussian(XNS[i])
                xmeas[i] = xmeas[i] + noise

        # Sampled measurements - composition analyzers
        # Calculate current compositions
        xcmp = np.zeros(NUM_MEASUREMENTS)
        xcmp[22:28] = self.state.xst[:6, 6] * 100.0  # Reactor feed (stream 7)
        xcmp[28:36] = self.state.xst[:, 9] * 100.0  # Purge (stream 10)
        xcmp[36:41] = self.state.xst[3:8, 12] * 100.0  # Product (stream 13)

        # Initialize delayed values at t=0
        if time == 0.0:
            self.state.xdel[22:41] = xcmp[22:41]
            xmeas[22:41] = xcmp[22:41]
            self.state.tgas = 0.1
            self.state.tprod = 0.25
        else:
            # Retain previous sampled measurements between analyzer updates
            # (like Fortran's COMMON block persistence)
            xmeas[22:41] = self.state.xmeas[22:41]

        # Gas analyzer (0.1 hr sampling)
        if time >= self.state.tgas:
            for i in range(22, 36):
                xmeas[i] = self.state.xdel[i]
                if self.state.isd == 0:
                    xmeas[i] += self.rng.gaussian(XNS[i])
                self.state.xdel[i] = xcmp[i]
            self.state.tgas += 0.1

        # Product analyzer (0.25 hr sampling)
        if time >= self.state.tprod:
            for i in range(36, 41):
                xmeas[i] = self.state.xdel[i]
                if self.state.isd == 0:
                    xmeas[i] += self.rng.gaussian(XNS[i])
                self.state.xdel[i] = xcmp[i]
            self.state.tprod += 0.25

        self.state.xmeas = xmeas

    def _check_safety(self, vlr, vls, vlc) -> int:
        """Check safety limits and return shutdown flag."""
        limits = SAFETY_LIMITS

        isd = 0
        if self.state.xmeas[6] > limits.reactor_pressure_max:
            isd = 1
        if vlr / 35.3145 > limits.reactor_level_max:
            isd = 1
        if vlr / 35.3145 < limits.reactor_level_min:
            isd = 1
        if self.state.xmeas[8] > limits.reactor_temp_max:
            isd = 1
        if vls / 35.3145 > limits.separator_level_max:
            isd = 1
        if vls / 35.3145 < limits.separator_level_min:
            isd = 1
        if vlc / 35.3145 > limits.stripper_level_max:
            isd = 1
        if vlc / 35.3145 < limits.stripper_level_min:
            isd = 1

        return isd

    def is_shutdown(self) -> bool:
        """Check if process is in shutdown state."""
        return self.state.isd != 0
