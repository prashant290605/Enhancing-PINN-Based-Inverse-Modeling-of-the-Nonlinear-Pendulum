"""Nonlinear pendulum solver using numerical integration."""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Dict, Any, Callable


def pendulum_ode(
    t: float, state: np.ndarray, g: float, L: float, damping: float
) -> np.ndarray:
    """
    Pendulum ODE: [θ', θ''] = [θ_dot, -(g/L)sin(θ) - damping*θ_dot]

    Args:
        t: time
        state: [theta, theta_dot]
        g: gravitational acceleration
        L: pendulum length
        damping: damping coefficient

    Returns:
        derivatives [theta_dot, theta_ddot]
    """
    theta, theta_dot = state
    theta_ddot = -(g / L) * np.sin(theta) - damping * theta_dot
    return np.array([theta_dot, theta_ddot])


def solve_nonlinear_pendulum(
    t: np.ndarray,
    theta0: float,
    theta_dot0: float,
    g: float = 9.81,
    L: float = 1.0,
    damping: float = 0.0,
    method: str = "RK45",
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve nonlinear pendulum using scipy's solve_ivp.

    Args:
        t: time array
        theta0: initial angle (rad)
        theta_dot0: initial angular velocity (rad/s)
        g: gravitational acceleration
        L: pendulum length
        damping: damping coefficient
        method: integration method
        rtol: relative tolerance
        atol: absolute tolerance

    Returns:
        theta: angle trajectory
        theta_dot: angular velocity trajectory
    """
    state0 = np.array([theta0, theta_dot0])

    sol = solve_ivp(
        lambda t, y: pendulum_ode(t, y, g, L, damping),
        (t[0], t[-1]),
        state0,
        method=method,
        t_eval=t,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    return sol.y[0], sol.y[1]


def solve_rk4(
    t: np.ndarray,
    theta0: float,
    theta_dot0: float,
    g: float = 9.81,
    L: float = 1.0,
    damping: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve pendulum using explicit 4th-order Runge-Kutta.

    Args:
        t: time array (must be uniformly spaced)
        theta0: initial angle
        theta_dot0: initial angular velocity
        g: gravitational acceleration
        L: pendulum length
        damping: damping coefficient

    Returns:
        theta: angle trajectory
        theta_dot: angular velocity trajectory
    """
    n = len(t)
    dt = t[1] - t[0]

    theta = np.zeros(n)
    theta_dot = np.zeros(n)

    theta[0] = theta0
    theta_dot[0] = theta_dot0

    for i in range(n - 1):
        state = np.array([theta[i], theta_dot[i]])

        k1 = pendulum_ode(t[i], state, g, L, damping)
        k2 = pendulum_ode(t[i] + dt / 2, state + dt * k1 / 2, g, L, damping)
        k3 = pendulum_ode(t[i] + dt / 2, state + dt * k2 / 2, g, L, damping)
        k4 = pendulum_ode(t[i] + dt, state + dt * k3, g, L, damping)

        state_next = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        theta[i + 1] = state_next[0]
        theta_dot[i + 1] = state_next[1]

    return theta, theta_dot


def compute_energy_nonlinear(
    theta: np.ndarray, theta_dot: np.ndarray, g: float = 9.81, L: float = 1.0, m: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Compute energy components for nonlinear pendulum.

    Args:
        theta: angle trajectory
        theta_dot: angular velocity trajectory
        g: gravitational acceleration
        L: pendulum length
        m: pendulum mass

    Returns:
        dictionary with 'kinetic', 'potential', 'total' energy arrays
    """
    # Kinetic energy: (1/2) m L² θ_dot²
    kinetic = 0.5 * m * L**2 * theta_dot**2

    # Potential energy: m g L (1 - cos(θ))
    potential = m * g * L * (1 - np.cos(theta))

    total = kinetic + potential

    return {"kinetic": kinetic, "potential": potential, "total": total}


def compute_phase_portrait(
    theta_range: Tuple[float, float],
    theta_dot_range: Tuple[float, float],
    n_grid: int = 50,
    g: float = 9.81,
    L: float = 1.0,
    damping: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute phase portrait (vector field) for pendulum.

    Args:
        theta_range: (theta_min, theta_max)
        theta_dot_range: (theta_dot_min, theta_dot_max)
        n_grid: grid resolution
        g: gravitational acceleration
        L: pendulum length
        damping: damping coefficient

    Returns:
        theta_grid: 2D grid of theta values
        theta_dot_grid: 2D grid of theta_dot values
        dtheta: theta derivatives at grid points
        dtheta_dot: theta_dot derivatives at grid points
    """
    theta_vals = np.linspace(theta_range[0], theta_range[1], n_grid)
    theta_dot_vals = np.linspace(theta_dot_range[0], theta_dot_range[1], n_grid)

    theta_grid, theta_dot_grid = np.meshgrid(theta_vals, theta_dot_vals)

    dtheta = theta_dot_grid
    dtheta_dot = -(g / L) * np.sin(theta_grid) - damping * theta_dot_grid

    return theta_grid, theta_dot_grid, dtheta, dtheta_dot


def estimate_period_nonlinear(
    theta0: float, g: float = 9.81, L: float = 1.0, n_terms: int = 5
) -> float:
    """
    Estimate period of nonlinear pendulum using series expansion.

    Uses the formula: T = T₀ * [1 + (1/16)θ₀² + (11/3072)θ₀⁴ + ...]
    where T₀ = 2π√(L/g)

    Args:
        theta0: initial amplitude (rad)
        g: gravitational acceleration
        L: pendulum length
        n_terms: number of terms in series

    Returns:
        estimated period
    """
    T0 = 2 * np.pi * np.sqrt(L / g)

    # Series coefficients for period correction
    # These come from elliptic integral expansion
    correction = 1.0
    theta0_sq = theta0**2

    if n_terms >= 2:
        correction += (1 / 16) * theta0_sq

    if n_terms >= 3:
        correction += (11 / 3072) * theta0_sq**2

    if n_terms >= 4:
        correction += (173 / 737280) * theta0_sq**3

    return T0 * correction

