"""Simulators for pendulum dynamics: analytic small-angle and nonlinear numerical."""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Optional, Literal


class SmallAnglePendulum:
    """Analytic solution for small-angle pendulum approximation."""

    def __init__(self, g: float = 9.81, L: float = 1.0, damping: float = 0.0):
        """
        Initialize small-angle pendulum.

        Args:
            g: gravitational acceleration (m/s^2)
            L: pendulum length (m)
            damping: damping coefficient (1/s)
        """
        self.g = g
        self.L = L
        self.damping = damping
        self.omega0 = np.sqrt(g / L)

    def solve(
        self, t: np.ndarray, theta0: float, theta_dot0: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute analytic solution for small angles.

        Args:
            t: time array
            theta0: initial angle (rad)
            theta_dot0: initial angular velocity (rad/s)

        Returns:
            theta: angle trajectory
            theta_dot: angular velocity trajectory
        """
        if self.damping == 0:
            # Undamped case
            A = theta0
            B = theta_dot0 / self.omega0
            theta = A * np.cos(self.omega0 * t) + B * np.sin(self.omega0 * t)
            theta_dot = -A * self.omega0 * np.sin(self.omega0 * t) + B * self.omega0 * np.cos(
                self.omega0 * t
            )
        else:
            # Underdamped case (assuming damping < omega0)
            omega_d = np.sqrt(self.omega0**2 - self.damping**2)
            exp_term = np.exp(-self.damping * t)
            A = theta0
            B = (theta_dot0 + self.damping * theta0) / omega_d
            theta = exp_term * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
            theta_dot = exp_term * (
                -self.damping * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
                + omega_d * (-A * np.sin(omega_d * t) + B * np.cos(omega_d * t))
            )

        return theta, theta_dot


class NonlinearPendulum:
    """Nonlinear pendulum solver using numerical integration."""

    def __init__(self, g: float = 9.81, L: float = 1.0, damping: float = 0.0):
        """
        Initialize nonlinear pendulum.

        Args:
            g: gravitational acceleration (m/s^2)
            L: pendulum length (m)
            damping: damping coefficient (1/s)
        """
        self.g = g
        self.L = L
        self.damping = damping

    def _dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Compute derivatives for pendulum ODE.

        Args:
            t: time
            state: [theta, theta_dot]

        Returns:
            derivatives: [theta_dot, theta_ddot]
        """
        theta, theta_dot = state
        theta_ddot = -(self.g / self.L) * np.sin(theta) - self.damping * theta_dot
        return np.array([theta_dot, theta_ddot])

    def solve(
        self,
        t: np.ndarray,
        theta0: float,
        theta_dot0: float,
        method: str = "RK45",
        rtol: float = 1e-9,
        atol: float = 1e-12,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve nonlinear pendulum using numerical integration.

        Args:
            t: time array
            theta0: initial angle (rad)
            theta_dot0: initial angular velocity (rad/s)
            method: integration method ('RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA')
            rtol: relative tolerance
            atol: absolute tolerance

        Returns:
            theta: angle trajectory
            theta_dot: angular velocity trajectory
        """
        state0 = np.array([theta0, theta_dot0])
        sol = solve_ivp(
            self._dynamics,
            (t[0], t[-1]),
            state0,
            method=method,
            t_eval=t,
            rtol=rtol,
            atol=atol,
        )

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        theta = sol.y[0]
        theta_dot = sol.y[1]

        return theta, theta_dot

    def solve_rk4(
        self, t: np.ndarray, theta0: float, theta_dot0: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve using explicit RK4 method.

        Args:
            t: time array (must be uniformly spaced)
            theta0: initial angle (rad)
            theta_dot0: initial angular velocity (rad/s)

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

            k1 = self._dynamics(t[i], state)
            k2 = self._dynamics(t[i] + dt / 2, state + dt * k1 / 2)
            k3 = self._dynamics(t[i] + dt / 2, state + dt * k2 / 2)
            k4 = self._dynamics(t[i] + dt, state + dt * k3)

            state_next = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

            theta[i + 1] = state_next[0]
            theta_dot[i + 1] = state_next[1]

        return theta, theta_dot


def simulate_pendulum(
    theta0: float,
    omega0: float,
    g: float,
    L: float,
    c: float,
    t_grid: np.ndarray,
    method: Literal["ivp", "rk4"] = "ivp",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate nonlinear damped pendulum.
    
    ODE: θ̈ + (g/L)sin(θ) + c·θ̇ = 0
    
    Args:
        theta0: initial angle (radians)
        omega0: initial angular velocity (rad/s)
        g: gravitational acceleration (m/s²)
        L: pendulum length (m)
        c: damping coefficient (1/s)
        t_grid: time array
        method: "ivp" for solve_ivp or "rk4" for RK4
        
    Returns:
        t: time array
        theta: angle trajectory
        omega: angular velocity trajectory
    """
    pendulum = NonlinearPendulum(g=g, L=L, damping=c)
    
    if method == "ivp":
        theta, omega = pendulum.solve(t_grid, theta0, omega0)
    elif method == "rk4":
        theta, omega = pendulum.solve_rk4(t_grid, theta0, omega0)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return t_grid, theta, omega


def add_noise(theta: np.ndarray, sigma: float, seed: Optional[int] = None) -> np.ndarray:
    """
    Add Gaussian noise to angle trajectory.
    
    Args:
        theta: angle array
        sigma: noise standard deviation
        seed: random seed
        
    Returns:
        theta_noisy: noisy angle array
    """
    if seed is not None:
        np.random.seed(seed)
    
    noise = np.random.normal(0, sigma, theta.shape)
    return theta + noise


def subsample(
    t: np.ndarray,
    theta: np.ndarray,
    k: int,
    irregular: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subsample trajectory to k points.
    
    Args:
        t: time array
        theta: angle array
        k: number of points to keep
        irregular: if True, use random sampling; if False, use uniform
        seed: random seed for irregular sampling
        
    Returns:
        t_s: subsampled time array
        theta_s: subsampled angle array
    """
    n = len(t)
    
    if k >= n:
        return t, theta
    
    if irregular:
        if seed is not None:
            np.random.seed(seed)
        # Random sampling without replacement, then sort
        indices = np.sort(np.random.choice(n, k, replace=False))
    else:
        # Uniform sampling
        indices = np.linspace(0, n - 1, k, dtype=int)
    
    return t[indices], theta[indices]


def make_sparse_measurements(
    t: np.ndarray,
    theta: np.ndarray,
    omega: np.ndarray,
    n_sparse: int,
    noise_std: float = 0.0,
    irregular: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sparse measurements with noise from dense trajectory.
    
    Args:
        t: dense time array
        theta: dense angle trajectory
        omega: dense angular velocity trajectory
        n_sparse: number of sparse observations
        noise_std: noise standard deviation
        irregular: if True, use random sampling; if False, use uniform
        seed: random seed
        
    Returns:
        t_obs: sparse observation times
        theta_obs_noisy: sparse noisy angle observations
        omega_obs_noisy: sparse noisy velocity observations
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(t)
    
    if n_sparse >= n:
        indices = np.arange(n)
    elif irregular:
        # Random sampling without replacement, then sort
        indices = np.sort(np.random.choice(n, n_sparse, replace=False))
    else:
        # Uniform sampling
        indices = np.linspace(0, n - 1, n_sparse, dtype=int)
    
    t_obs = t[indices]
    theta_obs = theta[indices]
    omega_obs = omega[indices]
    
    # Add noise if specified
    if noise_std > 0:
        theta_obs_noisy = theta_obs + np.random.normal(0, noise_std, theta_obs.shape)
        omega_obs_noisy = omega_obs + np.random.normal(0, noise_std, omega_obs.shape)
    else:
        theta_obs_noisy = theta_obs.copy()
        omega_obs_noisy = omega_obs.copy()
    
    return t_obs, theta_obs_noisy, omega_obs_noisy


def generate_pendulum_data(
    g: float = 9.81,
    L: float = 1.0,
    damping: float = 0.1,
    theta0: float = np.pi / 6,
    theta_dot0: float = 0.0,
    t_span: Tuple[float, float] = (0.0, 10.0),
    n_points: int = 100,
    noise_std: float = 0.0,
    use_small_angle: bool = False,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate pendulum trajectory data.

    Args:
        g: gravitational acceleration
        L: pendulum length
        damping: damping coefficient
        theta0: initial angle
        theta_dot0: initial angular velocity
        t_span: (t_start, t_end)
        n_points: number of time points
        noise_std: standard deviation of Gaussian noise
        use_small_angle: use small-angle approximation
        seed: random seed for noise

    Returns:
        t: time array
        theta: angle trajectory (with noise if noise_std > 0)
        theta_dot: angular velocity trajectory (with noise if noise_std > 0)
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.linspace(t_span[0], t_span[1], n_points)

    if use_small_angle:
        pendulum = SmallAnglePendulum(g=g, L=L, damping=damping)
    else:
        pendulum = NonlinearPendulum(g=g, L=L, damping=damping)

    theta, theta_dot = pendulum.solve(t, theta0, theta_dot0)

    if noise_std > 0:
        theta += np.random.normal(0, noise_std, theta.shape)
        theta_dot += np.random.normal(0, noise_std, theta_dot.shape)

    return t, theta, theta_dot

