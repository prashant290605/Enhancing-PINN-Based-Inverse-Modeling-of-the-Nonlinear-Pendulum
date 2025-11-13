"""Analytic solution for small-angle pendulum approximation."""

import numpy as np
from typing import Tuple, Dict, Any


def analytic_small_angle(theta0: float, g: float, L: float, t_grid: np.ndarray) -> np.ndarray:
    """
    Analytical small-angle pendulum solution.
    
    Equation: θ(t) = θ₀ cos(√(g/L) t)
    
    Args:
        theta0: initial angle (radians)
        g: gravitational acceleration (m/s²)
        L: pendulum length (m)
        t_grid: time array
        
    Returns:
        theta: angle trajectory
    """
    omega0 = np.sqrt(g / L)
    return theta0 * np.cos(omega0 * t_grid)


def solve_small_angle_undamped(
    t: np.ndarray, theta0: float, theta_dot0: float, g: float = 9.81, L: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analytic solution for undamped small-angle pendulum.

    Equation: θ''(t) + (g/L)θ(t) = 0
    Solution: θ(t) = A*cos(ω₀*t) + B*sin(ω₀*t)
              where ω₀ = sqrt(g/L)

    Args:
        t: time array
        theta0: initial angle (rad)
        theta_dot0: initial angular velocity (rad/s)
        g: gravitational acceleration (m/s^2)
        L: pendulum length (m)

    Returns:
        theta: angle trajectory
        theta_dot: angular velocity trajectory
    """
    omega0 = np.sqrt(g / L)

    A = theta0
    B = theta_dot0 / omega0

    theta = A * np.cos(omega0 * t) + B * np.sin(omega0 * t)
    theta_dot = -A * omega0 * np.sin(omega0 * t) + B * omega0 * np.cos(omega0 * t)

    return theta, theta_dot


def solve_small_angle_damped(
    t: np.ndarray,
    theta0: float,
    theta_dot0: float,
    g: float = 9.81,
    L: float = 1.0,
    damping: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analytic solution for damped small-angle pendulum.

    Equation: θ''(t) + 2ζω₀θ'(t) + ω₀²θ(t) = 0
              where ω₀ = sqrt(g/L), ζ = damping/(2ω₀)

    For underdamped case (ζ < 1):
    Solution: θ(t) = e^(-ζω₀t)[A*cos(ωₐt) + B*sin(ωₐt)]
              where ωₐ = ω₀*sqrt(1-ζ²)

    Args:
        t: time array
        theta0: initial angle (rad)
        theta_dot0: initial angular velocity (rad/s)
        g: gravitational acceleration (m/s^2)
        L: pendulum length (m)
        damping: damping coefficient (1/s)

    Returns:
        theta: angle trajectory
        theta_dot: angular velocity trajectory
    """
    omega0 = np.sqrt(g / L)

    if damping == 0:
        return solve_small_angle_undamped(t, theta0, theta_dot0, g, L)

    # Underdamped case
    omega_d = np.sqrt(omega0**2 - damping**2)

    if omega_d.imag != 0:
        # Overdamped or critically damped - use exponential solution
        # For simplicity, we'll handle underdamped case only
        raise ValueError("Overdamped case not implemented. Use damping < sqrt(g/L)")

    exp_term = np.exp(-damping * t)

    A = theta0
    B = (theta_dot0 + damping * theta0) / omega_d

    cos_term = np.cos(omega_d * t)
    sin_term = np.sin(omega_d * t)

    theta = exp_term * (A * cos_term + B * sin_term)

    # Derivative using product rule
    theta_dot = exp_term * (
        -damping * (A * cos_term + B * sin_term) + omega_d * (-A * sin_term + B * cos_term)
    )

    return theta, theta_dot


def compute_energy_small_angle(
    theta: np.ndarray, theta_dot: np.ndarray, g: float = 9.81, L: float = 1.0, m: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Compute energy components for small-angle approximation.

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

    # Potential energy (small angle): (1/2) m g L θ²
    potential = 0.5 * m * g * L * theta**2

    total = kinetic + potential

    return {"kinetic": kinetic, "potential": potential, "total": total}


def get_period_small_angle(g: float = 9.81, L: float = 1.0) -> float:
    """
    Compute period of small-angle pendulum.

    Args:
        g: gravitational acceleration
        L: pendulum length

    Returns:
        period (s)
    """
    omega0 = np.sqrt(g / L)
    return 2 * np.pi / omega0


def validate_small_angle_approximation(theta_max: float, threshold: float = 0.2) -> bool:
    """
    Check if small-angle approximation is valid.

    Rule of thumb: |θ| < 0.2 rad (~11.5°) for <1% error

    Args:
        theta_max: maximum angle (rad)
        threshold: threshold for validity (rad)

    Returns:
        True if approximation is valid
    """
    return abs(theta_max) < threshold

