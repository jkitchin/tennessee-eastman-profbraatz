"""
Disturbance generation for the Tennessee Eastman Process.

This module implements the disturbance mechanisms from the original Fortran code,
including:
- 20 predefined process disturbances (IDV flags)
- Random walk generation for time-varying disturbances
- Step changes
- Slow drifts
- Valve sticking effects
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from .constants import WALK_PARAMS, DEFAULT_RANDOM_SEED, NUM_DISTURBANCES


class RandomGenerator:
    """
    Linear congruential random number generator matching Fortran implementation.

    Uses the same constants as the original TESUB7 function to ensure
    reproducibility when using the same seed.
    """

    def __init__(self, seed: int = DEFAULT_RANDOM_SEED):
        """
        Initialize the random generator.

        Args:
            seed: Initial seed value (default matches Fortran code)
        """
        self.g = float(seed)

    def random(self, signed: bool = False) -> float:
        """
        Generate a random number.

        Equivalent to TESUB7 in the original Fortran code.

        Args:
            signed: If True, return value in [-1, 1]; if False, return [0, 1]

        Returns:
            Random number
        """
        # Linear congruential generator
        self.g = (self.g * 9228907.0) % 4294967296.0

        if signed:
            return 2.0 * self.g / 4294967296.0 - 1.0
        else:
            return self.g / 4294967296.0

    def gaussian(self, std: float = 1.0) -> float:
        """
        Generate Gaussian random number using sum of 12 uniforms.

        Equivalent to TESUB6 in the original Fortran code.

        Args:
            std: Standard deviation

        Returns:
            Gaussian random number with zero mean and given std
        """
        x = 0.0
        for _ in range(12):
            x += self.random(signed=False)
        x = (x - 6.0) * std
        return x

    def set_seed(self, seed: int):
        """Set a new seed value."""
        self.g = float(seed)


@dataclass
class WalkState:
    """State variables for cubic spline random walk disturbance."""
    adist: float = 0.0  # Spline coefficient a
    bdist: float = 0.0  # Spline coefficient b
    cdist: float = 0.0  # Spline coefficient c
    ddist: float = 0.0  # Spline coefficient d
    tlast: float = 0.0  # Time of last knot point
    tnext: float = 0.1  # Time of next knot point


class DisturbanceManager:
    """
    Manages all process disturbances for the TEP simulation.

    This class handles:
    - IDV flag states (20 disturbances)
    - Random walk disturbances for time-varying inputs
    - Step change implementation
    - Slow drift generation
    - Valve sticking effects
    """

    def __init__(self, random_seed: int = DEFAULT_RANDOM_SEED):
        """
        Initialize the disturbance manager.

        Args:
            random_seed: Seed for random number generation
        """
        self.rng = RandomGenerator(random_seed)

        # IDV flags (1-indexed in Fortran, 0-indexed here)
        self.idv = np.zeros(NUM_DISTURBANCES, dtype=int)

        # Mapping from walk index to IDV flag
        # This matches IDVWLK assignment in TEFUNC
        self._idvwlk_map = {
            0: 7,   # IDV(8) -> walk 1 (A composition)
            1: 7,   # IDV(8) -> walk 2 (B composition)
            2: 8,   # IDV(9) -> walk 3 (D feed temp)
            3: 9,   # IDV(10) -> walk 4 (C feed temp)
            4: 10,  # IDV(11) -> walk 5 (Reactor CW temp)
            5: 11,  # IDV(12) -> walk 6 (Condenser CW temp)
            6: 12,  # IDV(13) -> walk 7 (Reaction kinetics 1)
            7: 12,  # IDV(13) -> walk 8 (Reaction kinetics 2)
            8: 15,  # IDV(16) -> walk 9
            9: 16,  # IDV(17) -> walk 10
            10: 17, # IDV(18) -> walk 11
            11: 19, # IDV(20) -> walk 12
        }

        # Initialize walk states for 12 random walk disturbances
        self.walk_states: List[WalkState] = []
        for i, params in enumerate(WALK_PARAMS):
            state = WalkState()
            state.adist = params.szero
            state.bdist = 0.0
            state.cdist = 0.0
            state.ddist = 0.0
            state.tlast = 0.0
            state.tnext = 0.1
            self.walk_states.append(state)

    def set_idv(self, index: int, value: int = 1):
        """
        Set a disturbance flag.

        Args:
            index: Disturbance index (1-20 in Fortran convention, converted internally)
            value: Flag value (0 = off, nonzero = on)
        """
        # Convert from 1-indexed to 0-indexed
        idx = index - 1 if index >= 1 else index
        if 0 <= idx < NUM_DISTURBANCES:
            self.idv[idx] = 1 if value > 0 else 0

    def clear_all_disturbances(self):
        """Turn off all disturbances."""
        self.idv[:] = 0

    def get_idvwlk(self, walk_index: int) -> int:
        """Get the IDV flag value for a specific walk index."""
        idv_idx = self._idvwlk_map.get(walk_index, 0)
        return self.idv[idv_idx] if idv_idx < NUM_DISTURBANCES else 0

    def update_walks(self, time: float):
        """
        Update all random walk disturbances.

        This should be called at each integration step.

        Args:
            time: Current simulation time (hours)
        """
        # Update walks 0-8 (standard random walks)
        for i in range(9):
            if time >= self.walk_states[i].tnext:
                self._update_standard_walk(i, time)

        # Update walks 9-11 (special walks with decay)
        for i in range(9, 12):
            if time >= self.walk_states[i].tnext:
                self._update_decay_walk(i, time)

        # Reset at time = 0
        if time == 0.0:
            for i, params in enumerate(WALK_PARAMS):
                self.walk_states[i].adist = params.szero
                self.walk_states[i].bdist = 0.0
                self.walk_states[i].cdist = 0.0
                self.walk_states[i].ddist = 0.0
                self.walk_states[i].tlast = 0.0
                self.walk_states[i].tnext = 0.1

    def _update_standard_walk(self, walk_index: int, time: float):
        """Update a standard cubic spline random walk."""
        state = self.walk_states[walk_index]
        params = WALK_PARAMS[walk_index]
        idvwlk = self.get_idvwlk(walk_index)

        hwlk = state.tnext - state.tlast
        swlk = state.adist + hwlk * (state.bdist + hwlk * (state.cdist + hwlk * state.ddist))
        spwlk = state.bdist + hwlk * (2.0 * state.cdist + 3.0 * hwlk * state.ddist)

        state.tlast = state.tnext

        # Generate new knot point
        h = params.hspan * self.rng.random(signed=True) + params.hzero
        s1 = params.sspan * self.rng.random(signed=True) * idvwlk + params.szero
        s1p = params.spspan * self.rng.random(signed=True) * idvwlk

        # Update spline coefficients
        state.adist = swlk
        state.bdist = spwlk
        state.cdist = (3.0 * (s1 - swlk) - h * (s1p + 2.0 * spwlk)) / h**2
        state.ddist = (2.0 * (swlk - s1) + h * (s1p + spwlk)) / h**3
        state.tnext = state.tlast + h

    def _update_decay_walk(self, walk_index: int, time: float):
        """Update a decay-type random walk (for walks 9-11)."""
        state = self.walk_states[walk_index]
        params = WALK_PARAMS[walk_index]
        idvwlk = self.get_idvwlk(walk_index)

        hwlk = state.tnext - state.tlast
        swlk = state.adist + hwlk * (state.bdist + hwlk * (state.cdist + hwlk * state.ddist))
        spwlk = state.bdist + hwlk * (2.0 * state.cdist + 3.0 * hwlk * state.ddist)

        state.tlast = state.tnext

        if swlk > 0.1:
            # Decay to zero
            state.adist = swlk
            state.bdist = spwlk
            state.cdist = -(3.0 * swlk + 0.2 * spwlk) / 0.01
            state.ddist = (2.0 * swlk + 0.1 * spwlk) / 0.001
            state.tnext = state.tlast + 0.1
        else:
            # Generate new pulse
            hwlk = params.hspan * self.rng.random(signed=True) + params.hzero
            state.adist = 0.0
            state.bdist = 0.0
            state.cdist = float(idvwlk) / hwlk**2 if hwlk > 0 else 0.0
            state.ddist = 0.0
            state.tnext = state.tlast + hwlk

    def get_walk_value(self, walk_index: int, time: float) -> float:
        """
        Get the current value of a random walk disturbance.

        Equivalent to TESUB8 in the original Fortran code.

        Args:
            walk_index: Walk index (0-11)
            time: Current time (hours)

        Returns:
            Current disturbance value
        """
        if 0 <= walk_index < len(self.walk_states):
            state = self.walk_states[walk_index]
            h = time - state.tlast
            return state.adist + h * (state.bdist + h * (state.cdist + h * state.ddist))
        return 0.0

    def get_xst_composition(self, time: float) -> np.ndarray:
        """
        Get stream 4 composition including IDV(1) and IDV(2) effects.

        Returns:
            Array of [A, B, C] mole fractions for stream 4
        """
        xst1 = self.get_walk_value(0, time) - self.idv[0] * 0.03 - self.idv[1] * 2.43719e-3
        xst2 = self.get_walk_value(1, time) + self.idv[1] * 0.005
        xst3 = 1.0 - xst1 - xst2
        return np.array([xst1, xst2, xst3])

    def get_feed_temperature(self, stream: int, time: float) -> float:
        """
        Get feed stream temperature including disturbance effects.

        Args:
            stream: Stream number (1 or 4)
            time: Current time (hours)

        Returns:
            Temperature (deg C)
        """
        if stream == 1:
            # D Feed temperature (stream 2 in Fortran)
            return self.get_walk_value(2, time) + self.idv[2] * 5.0
        elif stream == 4:
            # A and C Feed temperature
            return self.get_walk_value(3, time)
        return 45.0  # Default

    def get_cooling_water_temp(self, unit: str, time: float) -> float:
        """
        Get cooling water inlet temperature.

        Args:
            unit: 'reactor' or 'condenser'
            time: Current time (hours)

        Returns:
            Cooling water inlet temperature (deg C)
        """
        if unit == 'reactor':
            return self.get_walk_value(4, time) + self.idv[3] * 5.0
        elif unit == 'condenser':
            return self.get_walk_value(5, time) + self.idv[4] * 5.0
        return 35.0  # Default

    def get_reaction_factor(self, reaction: int, time: float) -> float:
        """
        Get reaction rate modification factor for reaction kinetics drift.

        Args:
            reaction: Reaction number (1 or 2)
            time: Current time (hours)

        Returns:
            Reaction rate factor (1.0 = normal)
        """
        if reaction == 1:
            return self.get_walk_value(6, time)
        elif reaction == 2:
            return self.get_walk_value(7, time)
        return 1.0

    def is_valve_stuck(self, valve: int) -> bool:
        """
        Check if a valve is in sticking mode.

        Args:
            valve: Valve number (10 for reactor CW, 11 for condenser CW)

        Returns:
            True if valve is stuck
        """
        if valve == 10:
            return bool(self.idv[13])  # IDV(14)
        elif valve == 11:
            return bool(self.idv[14])  # IDV(15)
        return False

    def get_feed_loss_factor(self) -> float:
        """Get A feed loss factor for IDV(6)."""
        return 1.0 - self.idv[5]

    def get_header_pressure_factor(self) -> float:
        """Get C header pressure loss factor for IDV(7)."""
        return 1.0 - self.idv[6] * 0.2

    def get_uac_factor(self, time: float) -> float:
        """Get stripper steam valve heat transfer factor."""
        return 1.0 + self.get_walk_value(8, time)

    def get_reactor_ht_factor(self, time: float) -> float:
        """Get reactor heat transfer degradation factor."""
        return 1.0 - 0.35 * self.get_walk_value(9, time)

    def get_separator_ht_factor(self, time: float) -> float:
        """Get separator heat transfer degradation factor."""
        return 1.0 - 0.25 * self.get_walk_value(10, time)

    def get_reactor_flow_factor(self, time: float) -> float:
        """Get reactor flow restriction factor."""
        return 1.0 - 0.25 * self.get_walk_value(11, time)
