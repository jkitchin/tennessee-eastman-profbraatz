"""
Numerical integration methods for the Tennessee Eastman Process.

This module provides various integration methods for solving the
system of ordinary differential equations.
"""

import numpy as np
from typing import Callable, Tuple
from enum import Enum


class IntegratorType(Enum):
    """Available integration methods."""
    EULER = "euler"
    RK4 = "rk4"
    RK45 = "rk45"


def euler_step(
    func: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray,
    dt: float
) -> Tuple[float, np.ndarray]:
    """
    Perform one Euler integration step.

    This is the same method used in the original Fortran code.

    Args:
        func: Derivative function f(t, y) -> dy/dt
        t: Current time
        y: Current state vector
        dt: Time step

    Returns:
        Tuple of (new_time, new_state)
    """
    yp = func(t, y)
    y_new = y + yp * dt
    t_new = t + dt

    return t_new, y_new


def rk4_step(
    func: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray,
    dt: float
) -> Tuple[float, np.ndarray]:
    """
    Perform one 4th-order Runge-Kutta integration step.

    More accurate than Euler for smooth systems.

    Args:
        func: Derivative function f(t, y) -> dy/dt
        t: Current time
        y: Current state vector
        dt: Time step

    Returns:
        Tuple of (new_time, new_state)
    """
    k1 = func(t, y)
    k2 = func(t + dt/2, y + dt/2 * k1)
    k3 = func(t + dt/2, y + dt/2 * k2)
    k4 = func(t + dt, y + dt * k3)

    y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    t_new = t + dt

    return t_new, y_new


class Integrator:
    """
    Base class for ODE integrators.

    Provides a common interface for different integration methods.
    """

    def __init__(self, method: IntegratorType = IntegratorType.EULER):
        """
        Initialize the integrator.

        Args:
            method: Integration method to use
        """
        self.method = method

    def step(
        self,
        func: Callable[[float, np.ndarray], np.ndarray],
        t: float,
        y: np.ndarray,
        dt: float
    ) -> Tuple[float, np.ndarray]:
        """
        Perform one integration step.

        Args:
            func: Derivative function f(t, y) -> dy/dt
            t: Current time
            y: Current state vector
            dt: Time step

        Returns:
            Tuple of (new_time, new_state)
        """
        if self.method == IntegratorType.EULER:
            return euler_step(func, t, y, dt)
        elif self.method == IntegratorType.RK4:
            return rk4_step(func, t, y, dt)
        else:
            # Default to Euler
            return euler_step(func, t, y, dt)

    def integrate(
        self,
        func: Callable[[float, np.ndarray], np.ndarray],
        t0: float,
        y0: np.ndarray,
        t_end: float,
        dt: float,
        callback: Callable[[float, np.ndarray], bool] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate from t0 to t_end.

        Args:
            func: Derivative function f(t, y) -> dy/dt
            t0: Initial time
            y0: Initial state vector
            t_end: Final time
            dt: Time step
            callback: Optional callback called at each step.
                      If callback returns True, integration stops.

        Returns:
            Tuple of (time_array, state_array)
        """
        # Estimate number of steps
        n_steps = int(np.ceil((t_end - t0) / dt)) + 1

        # Initialize output arrays
        times = [t0]
        states = [y0.copy()]

        t = t0
        y = y0.copy()

        while t < t_end:
            # Adjust final step if needed
            if t + dt > t_end:
                dt_step = t_end - t
            else:
                dt_step = dt

            # Perform integration step
            t, y = self.step(func, t, y, dt_step)

            # Store results
            times.append(t)
            states.append(y.copy())

            # Call callback if provided
            if callback is not None:
                if callback(t, y):
                    break

        return np.array(times), np.array(states)
