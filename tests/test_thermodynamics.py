"""
Tests for the thermodynamics module.
"""

import pytest
import numpy as np
from tep.thermodynamics import (
    calculate_enthalpy, calculate_temperature, calculate_liquid_density,
    calculate_vapor_pressure, calculate_mixture_molecular_weight,
    calculate_enthalpy_derivative
)
from tep.constants import NUM_COMPONENTS, XMW


class TestEnthalpy:
    """Test enthalpy calculations."""

    @pytest.fixture
    def pure_component_d(self):
        """Pure component D composition."""
        comp = np.zeros(NUM_COMPONENTS)
        comp[3] = 1.0  # Pure D
        return comp

    @pytest.fixture
    def mixture(self):
        """Typical mixture composition."""
        comp = np.array([0.1, 0.05, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05])
        return comp / comp.sum()

    def test_liquid_enthalpy_positive_temp(self, pure_component_d):
        """Liquid enthalpy should be finite at normal temperatures."""
        H = calculate_enthalpy(pure_component_d, 50.0, phase=0)
        assert np.isfinite(H)

    def test_vapor_enthalpy_positive_temp(self, mixture):
        """Vapor enthalpy should be finite at normal temperatures."""
        H = calculate_enthalpy(mixture, 100.0, phase=1)
        assert np.isfinite(H)

    def test_enthalpy_increases_with_temp(self, mixture):
        """Enthalpy should increase with temperature."""
        H1 = calculate_enthalpy(mixture, 50.0, phase=0)
        H2 = calculate_enthalpy(mixture, 100.0, phase=0)
        assert H2 > H1

    def test_vapor_enthalpy_higher_than_liquid(self, mixture):
        """Vapor enthalpy should be higher than liquid at same T."""
        H_liq = calculate_enthalpy(mixture, 80.0, phase=0)
        H_vap = calculate_enthalpy(mixture, 80.0, phase=1)
        assert H_vap > H_liq


class TestTemperature:
    """Test temperature calculations (inverse of enthalpy)."""

    @pytest.fixture
    def mixture(self):
        comp = np.array([0.1, 0.05, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05])
        return comp / comp.sum()

    def test_temperature_roundtrip(self, mixture):
        """Temperature calculation should be inverse of enthalpy."""
        T_original = 80.0
        H = calculate_enthalpy(mixture, T_original, phase=0)
        T_calculated = calculate_temperature(mixture, H, phase=0, initial_guess=100.0)
        assert abs(T_calculated - T_original) < 0.1

    def test_temperature_convergence(self, mixture):
        """Temperature iteration should converge."""
        H = calculate_enthalpy(mixture, 120.0, phase=0)
        T = calculate_temperature(mixture, H, phase=0, initial_guess=50.0)
        assert np.isfinite(T)
        assert abs(T - 120.0) < 1.0


class TestLiquidDensity:
    """Test liquid density calculations."""

    @pytest.fixture
    def pure_component_d(self):
        comp = np.zeros(NUM_COMPONENTS)
        comp[3] = 1.0
        return comp

    def test_density_positive(self, pure_component_d):
        """Density should be positive."""
        rho = calculate_liquid_density(pure_component_d, 50.0)
        assert rho > 0

    def test_density_finite(self, pure_component_d):
        """Density should be finite."""
        rho = calculate_liquid_density(pure_component_d, 50.0)
        assert np.isfinite(rho)

    def test_density_decreases_with_temp(self, pure_component_d):
        """Liquid density typically decreases with temperature."""
        rho1 = calculate_liquid_density(pure_component_d, 30.0)
        rho2 = calculate_liquid_density(pure_component_d, 80.0)
        assert rho1 > rho2


class TestVaporPressure:
    """Test vapor pressure calculations."""

    def test_non_condensable_zero_pressure(self):
        """Non-condensable components (A, B, C) should have zero vapor pressure."""
        for i in range(3):
            P = calculate_vapor_pressure(i, 50.0)
            assert P == 0.0

    def test_condensable_positive_pressure(self):
        """Condensable components should have positive vapor pressure."""
        for i in range(3, 8):
            P = calculate_vapor_pressure(i, 50.0)
            assert P > 0

    def test_vapor_pressure_increases_with_temp(self):
        """Vapor pressure should increase with temperature."""
        P1 = calculate_vapor_pressure(4, 50.0)
        P2 = calculate_vapor_pressure(4, 100.0)
        assert P2 > P1


class TestMolecularWeight:
    """Test mixture molecular weight calculation."""

    def test_pure_component(self):
        """Pure component MW should match tabulated value."""
        comp = np.zeros(NUM_COMPONENTS)
        comp[3] = 1.0  # Pure D
        mw = calculate_mixture_molecular_weight(comp)
        assert abs(mw - XMW[3]) < 0.001

    def test_mixture_intermediate(self):
        """Mixture MW should be between min and max component MW."""
        comp = np.ones(NUM_COMPONENTS) / NUM_COMPONENTS
        mw = calculate_mixture_molecular_weight(comp)
        assert mw > XMW.min()
        assert mw < XMW.max()


class TestEnthalpyDerivative:
    """Test enthalpy derivative (heat capacity) calculations."""

    @pytest.fixture
    def mixture(self):
        comp = np.array([0.1, 0.05, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05])
        return comp / comp.sum()

    def test_derivative_positive(self, mixture):
        """Heat capacity should be positive."""
        dH = calculate_enthalpy_derivative(mixture, 50.0, phase=0)
        assert dH > 0

    def test_derivative_finite(self, mixture):
        """Heat capacity should be finite."""
        dH = calculate_enthalpy_derivative(mixture, 100.0, phase=1)
        assert np.isfinite(dH)
