"""
Tests for the constants module.
"""

import pytest
import numpy as np
from tep.constants import (
    NUM_STATES, NUM_COMPONENTS, NUM_MEASUREMENTS, NUM_MANIPULATED_VARS,
    NUM_DISTURBANCES, XMW, INITIAL_STATES, MEASUREMENT_NAMES,
    MANIPULATED_VAR_NAMES, DISTURBANCE_NAMES, XNS, VRNG, VTAU
)


class TestDimensions:
    """Test that dimensions match expected values."""

    def test_num_states(self):
        assert NUM_STATES == 50

    def test_num_components(self):
        assert NUM_COMPONENTS == 8

    def test_num_measurements(self):
        assert NUM_MEASUREMENTS == 41

    def test_num_mvs(self):
        assert NUM_MANIPULATED_VARS == 12

    def test_num_disturbances(self):
        assert NUM_DISTURBANCES == 20


class TestMolecularWeights:
    """Test molecular weight constants."""

    def test_xmw_shape(self):
        assert len(XMW) == NUM_COMPONENTS

    def test_xmw_positive(self):
        assert all(XMW > 0)

    def test_component_a_mw(self):
        """Component A (hydrogen-like) should have MW = 2."""
        assert XMW[0] == 2.0

    def test_component_h_mw(self):
        """Component H should have MW = 76."""
        assert XMW[7] == 76.0


class TestInitialStates:
    """Test initial state values."""

    def test_initial_states_shape(self):
        assert len(INITIAL_STATES) == NUM_STATES

    def test_initial_states_finite(self):
        assert all(np.isfinite(INITIAL_STATES))

    def test_valve_positions_in_range(self):
        """Valve positions (states 39-50) should be 0-100%."""
        valve_positions = INITIAL_STATES[38:50]
        assert all(valve_positions >= 0)
        assert all(valve_positions <= 100)


class TestNamingConventions:
    """Test that names arrays have correct lengths."""

    def test_measurement_names_length(self):
        assert len(MEASUREMENT_NAMES) == NUM_MEASUREMENTS

    def test_mv_names_length(self):
        assert len(MANIPULATED_VAR_NAMES) == NUM_MANIPULATED_VARS

    def test_disturbance_names_length(self):
        assert len(DISTURBANCE_NAMES) == NUM_DISTURBANCES


class TestNoiseStdDev:
    """Test noise standard deviation values."""

    def test_xns_shape(self):
        assert len(XNS) == NUM_MEASUREMENTS

    def test_xns_non_negative(self):
        assert all(XNS >= 0)


class TestValveParameters:
    """Test valve range and time constant values."""

    def test_vrng_shape(self):
        assert len(VRNG) == NUM_MANIPULATED_VARS

    def test_vrng_non_negative(self):
        assert all(VRNG >= 0)

    def test_vtau_shape(self):
        assert len(VTAU) == NUM_MANIPULATED_VARS

    def test_vtau_positive(self):
        """Valve time constants should be positive."""
        assert all(VTAU > 0)

    def test_vtau_reasonable(self):
        """Valve time constants should be reasonable (< 1 hour)."""
        assert all(VTAU < 1.0)
