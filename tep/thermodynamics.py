"""
Thermodynamic utility functions for the Tennessee Eastman Process.

This module implements the thermodynamic calculations from TESUB1-4 in the
original Fortran code, including:
- Enthalpy calculations (liquid and vapor)
- Temperature from enthalpy (inverse calculation)
- Heat capacity calculations
- Liquid density calculations
- Vapor pressure calculations
"""

import numpy as np
from typing import Tuple, Optional
from .constants import (
    AH, BH, CH, AG, BG, CG, AV, AD, BD, CD, AVP, BVP, CVP, XMW,
    NUM_COMPONENTS
)


def calculate_enthalpy(
    composition: np.ndarray,
    temperature: float,
    phase: int = 0
) -> float:
    """
    Calculate mixture enthalpy from composition and temperature.

    Equivalent to TESUB1 in the original Fortran code.

    Args:
        composition: Mole fractions of 8 components (should sum to 1)
        temperature: Temperature in degrees Celsius
        phase: 0 for liquid, 1 for vapor, 2 for vapor with compressibility correction

    Returns:
        Enthalpy value (arbitrary units consistent with original code)
    """
    H = 0.0
    T = temperature

    if phase == 0:
        # Liquid enthalpy
        for i in range(NUM_COMPONENTS):
            Hi = T * (AH[i] + BH[i] * T / 2.0 + CH[i] * T**2 / 3.0)
            Hi = 1.8 * Hi  # Unit conversion
            H += composition[i] * XMW[i] * Hi
    else:
        # Vapor enthalpy
        for i in range(NUM_COMPONENTS):
            Hi = T * (AG[i] + BG[i] * T / 2.0 + CG[i] * T**2 / 3.0)
            Hi = 1.8 * Hi  # Unit conversion
            Hi = Hi + AV[i]  # Add heat of vaporization
            H += composition[i] * XMW[i] * Hi

    # Compressibility correction for phase 2
    if phase == 2:
        R = 3.57696e-6
        H = H - R * (T + 273.15)

    return H


def calculate_enthalpy_derivative(
    composition: np.ndarray,
    temperature: float,
    phase: int = 0
) -> float:
    """
    Calculate derivative of enthalpy with respect to temperature.

    Equivalent to TESUB3 in the original Fortran code.
    Used for Newton-Raphson iteration in temperature calculation.

    Args:
        composition: Mole fractions of 8 components
        temperature: Temperature in degrees Celsius
        phase: 0 for liquid, 1 for vapor, 2 for vapor with compressibility correction

    Returns:
        dH/dT value
    """
    dH = 0.0
    T = temperature

    if phase == 0:
        # Liquid heat capacity
        for i in range(NUM_COMPONENTS):
            dHi = AH[i] + BH[i] * T + CH[i] * T**2
            dHi = 1.8 * dHi
            dH += composition[i] * XMW[i] * dHi
    else:
        # Vapor heat capacity
        for i in range(NUM_COMPONENTS):
            dHi = AG[i] + BG[i] * T + CG[i] * T**2
            dHi = 1.8 * dHi
            dH += composition[i] * XMW[i] * dHi

    # Compressibility correction for phase 2
    if phase == 2:
        R = 3.57696e-6
        dH = dH - R

    return dH


def calculate_temperature(
    composition: np.ndarray,
    enthalpy: float,
    phase: int = 0,
    initial_guess: float = 100.0,
    max_iterations: int = 100,
    tolerance: float = 1.0e-12
) -> float:
    """
    Calculate temperature from composition and enthalpy using Newton-Raphson.

    Equivalent to TESUB2 in the original Fortran code.

    Args:
        composition: Mole fractions of 8 components
        enthalpy: Target enthalpy value
        phase: 0 for liquid, 1 for vapor, 2 for vapor with compressibility
        initial_guess: Initial temperature estimate (deg C)
        max_iterations: Maximum Newton-Raphson iterations
        tolerance: Convergence tolerance for temperature change

    Returns:
        Temperature in degrees Celsius
    """
    T = initial_guess
    T_initial = initial_guess

    for _ in range(max_iterations):
        H_test = calculate_enthalpy(composition, T, phase)
        error = H_test - enthalpy
        dH = calculate_enthalpy_derivative(composition, T, phase)

        if abs(dH) < 1e-20:
            # Avoid division by zero
            break

        dT = -error / dH
        T = T + dT

        if abs(dT) < tolerance:
            return T

    # If no convergence, return initial guess (as in Fortran)
    return T_initial


def calculate_liquid_density(
    composition: np.ndarray,
    temperature: float
) -> float:
    """
    Calculate liquid mixture density.

    Equivalent to TESUB4 in the original Fortran code.
    Uses a volume-weighted mixing rule.

    Args:
        composition: Mole fractions of 8 components
        temperature: Temperature in degrees Celsius

    Returns:
        Liquid density (kmol/m^3)
    """
    T = temperature
    V = 0.0

    for i in range(NUM_COMPONENTS):
        # Molar volume of component i
        density_i = AD[i] + (BD[i] + CD[i] * T) * T
        if density_i > 0:
            V += composition[i] * XMW[i] / density_i

    if V > 0:
        return 1.0 / V
    else:
        return 1.0  # Fallback


def calculate_vapor_pressure(
    component_index: int,
    temperature: float
) -> float:
    """
    Calculate vapor pressure of a pure component using Antoine equation.

    Args:
        component_index: 0-based component index (0-7 for A-H)
        temperature: Temperature in degrees Celsius

    Returns:
        Vapor pressure (kPa)
    """
    i = component_index

    # Components A, B, C (indices 0-2) are non-condensable
    if i < 3:
        return 0.0

    # Antoine equation: ln(P) = A + B/(T + C)
    if temperature + CVP[i] != 0:
        return np.exp(AVP[i] + BVP[i] / (temperature + CVP[i]))
    else:
        return 0.0


def calculate_mixture_molecular_weight(composition: np.ndarray) -> float:
    """
    Calculate mixture average molecular weight.

    Args:
        composition: Mole fractions of 8 components

    Returns:
        Average molecular weight (kg/kmol)
    """
    return np.sum(composition * XMW)


def calculate_bubble_point_pressure(
    liquid_composition: np.ndarray,
    temperature: float
) -> Tuple[float, np.ndarray]:
    """
    Calculate bubble point pressure and vapor composition.

    Uses Raoult's law for condensable components and assumes
    non-condensable components follow ideal gas behavior.

    Args:
        liquid_composition: Liquid mole fractions
        temperature: Temperature in degrees Celsius

    Returns:
        Tuple of (total_pressure, vapor_composition)
    """
    partial_pressures = np.zeros(NUM_COMPONENTS)

    # Calculate partial pressures using Raoult's law
    for i in range(NUM_COMPONENTS):
        if i < 3:
            # Non-condensable - handled separately
            partial_pressures[i] = 0.0
        else:
            Psat = calculate_vapor_pressure(i, temperature)
            partial_pressures[i] = Psat * liquid_composition[i]

    total_pressure = np.sum(partial_pressures)

    # Vapor composition
    if total_pressure > 0:
        vapor_composition = partial_pressures / total_pressure
    else:
        vapor_composition = np.zeros(NUM_COMPONENTS)

    return total_pressure, vapor_composition
