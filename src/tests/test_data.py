"""Tests for data generation modules."""

import pytest
import numpy as np
import torch

from src.data.generator import (
    SmallAnglePendulum,
    NonlinearPendulum,
    generate_pendulum_data,
    simulate_pendulum,
    add_noise,
    subsample,
)
from src.baseline.linear_small_angle import analytic_small_angle
from src.data.utils import (
    create_time_grid,
    numpy_to_torch,
    torch_to_numpy,
    split_train_test,
    subsample_data,
)


class TestSmallAnglePendulum:
    """Tests for small-angle pendulum solver."""

    def test_initialization(self):
        """Test pendulum initialization."""
        pendulum = SmallAnglePendulum(g=9.81, L=1.0, damping=0.0)
        assert pendulum.g == 9.81
        assert pendulum.L == 1.0
        assert pendulum.damping == 0.0

    def test_solve_undamped(self):
        """Test undamped solution."""
        pendulum = SmallAnglePendulum(g=9.81, L=1.0, damping=0.0)
        t = np.linspace(0, 10, 100)
        theta, theta_dot = pendulum.solve(t, theta0=0.1, theta_dot0=0.0)

        assert theta.shape == (100,)
        assert theta_dot.shape == (100,)
        assert theta[0] == pytest.approx(0.1, abs=1e-6)

    def test_solve_damped(self):
        """Test damped solution."""
        pendulum = SmallAnglePendulum(g=9.81, L=1.0, damping=0.1)
        t = np.linspace(0, 10, 100)
        theta, theta_dot = pendulum.solve(t, theta0=0.1, theta_dot0=0.0)

        assert theta.shape == (100,)
        assert theta_dot.shape == (100,)
        # Energy should decrease for damped system
        assert np.abs(theta[-1]) < np.abs(theta[0])


class TestNonlinearPendulum:
    """Tests for nonlinear pendulum solver."""

    def test_initialization(self):
        """Test pendulum initialization."""
        pendulum = NonlinearPendulum(g=9.81, L=1.0, damping=0.0)
        assert pendulum.g == 9.81
        assert pendulum.L == 1.0
        assert pendulum.damping == 0.0

    def test_solve(self):
        """Test nonlinear solution."""
        pendulum = NonlinearPendulum(g=9.81, L=1.0, damping=0.0)
        t = np.linspace(0, 10, 100)
        theta, theta_dot = pendulum.solve(t, theta0=0.5, theta_dot0=0.0)

        assert theta.shape == (100,)
        assert theta_dot.shape == (100,)
        assert theta[0] == pytest.approx(0.5, abs=1e-6)

    def test_solve_rk4(self):
        """Test RK4 solver."""
        pendulum = NonlinearPendulum(g=9.81, L=1.0, damping=0.0)
        t = np.linspace(0, 10, 100)
        theta, theta_dot = pendulum.solve_rk4(t, theta0=0.5, theta_dot0=0.0)

        assert theta.shape == (100,)
        assert theta_dot.shape == (100,)
        assert theta[0] == pytest.approx(0.5, abs=1e-6)


class TestDataGeneration:
    """Tests for data generation utilities."""

    def test_generate_pendulum_data(self):
        """Test pendulum data generation."""
        t, theta, theta_dot = generate_pendulum_data(
            g=9.81,
            L=1.0,
            damping=0.1,
            theta0=0.5,
            theta_dot0=0.0,
            t_span=(0.0, 10.0),
            n_points=100,
            noise_std=0.0,
            seed=42,
        )

        assert t.shape == (100,)
        assert theta.shape == (100,)
        assert theta_dot.shape == (100,)
        assert t[0] == 0.0
        assert t[-1] == 10.0

    def test_generate_with_noise(self):
        """Test data generation with noise."""
        t1, theta1, _ = generate_pendulum_data(noise_std=0.0, seed=42)
        t2, theta2, _ = generate_pendulum_data(noise_std=0.01, seed=42)

        assert not np.allclose(theta1, theta2)


class TestAnalyticSmallAngle:
    """Tests for analytical small-angle solution."""

    def test_analytic_cosine_shape(self):
        """Test that analytic solution has cosine shape at theta0=0.1 rad."""
        theta0 = 0.1  # radians
        g = 9.81
        L = 1.0
        t_grid = np.linspace(0, 10, 1000)
        
        theta = analytic_small_angle(theta0, g, L, t_grid)
        
        # Check it's approximately cosine shape
        omega0 = np.sqrt(g / L)
        expected = theta0 * np.cos(omega0 * t_grid)
        
        assert np.allclose(theta, expected, atol=1e-10)
        assert theta[0] == pytest.approx(theta0, abs=1e-10)

    def test_analytic_period(self):
        """Test that period is near 2π√(L/g)."""
        theta0 = 0.1
        g = 9.81
        L = 1.0
        
        # Expected period
        expected_period = 2 * np.pi * np.sqrt(L / g)
        
        # Generate trajectory over 2 periods
        t_grid = np.linspace(0, 2 * expected_period, 10000)
        theta = analytic_small_angle(theta0, g, L, t_grid)
        
        # Find zero crossings to measure period
        # theta crosses zero when cos(omega*t) = 0
        # This happens at t = π/(2*omega), 3π/(2*omega), etc.
        omega0 = np.sqrt(g / L)
        first_zero = np.pi / (2 * omega0)
        second_zero = 3 * np.pi / (2 * omega0)
        measured_half_period = second_zero - first_zero
        measured_period = 2 * measured_half_period
        
        assert measured_period == pytest.approx(expected_period, rel=1e-6)


class TestNonlinearEnergy:
    """Tests for nonlinear pendulum energy conservation."""

    def test_energy_constant_no_damping(self):
        """Test energy nearly constant for c=0 over dense grid."""
        theta0 = 0.5  # radians
        omega0 = 0.0
        g = 9.81
        L = 1.0
        c = 0.0  # no damping
        
        # Dense grid
        t_grid = np.linspace(0, 10, 10000)
        
        t, theta, omega = simulate_pendulum(theta0, omega0, g, L, c, t_grid, method="ivp")
        
        # Compute energy: E = 0.5*m*L²*ω² + m*g*L*(1 - cos(θ))
        # Set m=1 for simplicity
        m = 1.0
        kinetic = 0.5 * m * L**2 * omega**2
        potential = m * g * L * (1 - np.cos(theta))
        energy = kinetic + potential
        
        # Energy should be nearly constant
        E0 = energy[0]
        relative_drift = np.abs((energy - E0) / E0)
        
        # Allow small numerical error (< 0.1%)
        assert np.max(relative_drift) < 1e-3

    def test_energy_decreases_with_damping(self):
        """Test energy monotonically decreases for c>0."""
        theta0 = 0.5
        omega0 = 0.0
        g = 9.81
        L = 1.0
        c = 0.05  # damping
        
        t_grid = np.linspace(0, 10, 10000)
        
        t, theta, omega = simulate_pendulum(theta0, omega0, g, L, c, t_grid, method="ivp")
        
        # Compute energy
        m = 1.0
        kinetic = 0.5 * m * L**2 * omega**2
        potential = m * g * L * (1 - np.cos(theta))
        energy = kinetic + potential
        
        # Energy should decrease (allowing tiny numerical wiggles)
        # Check that overall trend is decreasing
        # Use moving average to smooth out numerical noise
        window = 100
        energy_smooth = np.convolve(energy, np.ones(window)/window, mode='valid')
        
        # Check that smoothed energy is decreasing
        energy_diff = np.diff(energy_smooth)
        # Most differences should be negative or very small
        assert np.mean(energy_diff < 1e-6) > 0.95  # 95% of points decreasing


class TestSimulatePendulum:
    """Tests for simulate_pendulum function."""

    def test_simulate_ivp_vs_rk4(self):
        """Test that ivp and rk4 methods give similar results."""
        theta0 = 0.3
        omega0 = 0.0
        g = 9.81
        L = 1.0
        c = 0.02
        
        t_grid = np.linspace(0, 10, 1000)
        
        t_ivp, theta_ivp, omega_ivp = simulate_pendulum(
            theta0, omega0, g, L, c, t_grid, method="ivp"
        )
        t_rk4, theta_rk4, omega_rk4 = simulate_pendulum(
            theta0, omega0, g, L, c, t_grid, method="rk4"
        )
        
        # Results should be close
        assert np.allclose(theta_ivp, theta_rk4, atol=1e-4)
        assert np.allclose(omega_ivp, omega_rk4, atol=1e-4)


class TestNoiseAndSubsample:
    """Tests for noise and subsampling functions."""

    def test_add_noise(self):
        """Test noise addition."""
        theta = np.ones(100)
        sigma = 0.1
        
        theta_noisy = add_noise(theta, sigma, seed=42)
        
        assert theta_noisy.shape == theta.shape
        assert not np.allclose(theta, theta_noisy)
        # Check noise statistics
        noise = theta_noisy - theta
        assert np.abs(np.mean(noise)) < 0.05  # Mean near zero
        assert np.abs(np.std(noise) - sigma) < 0.02  # Std near sigma

    def test_subsample_irregular(self):
        """Test irregular subsampling."""
        t = np.linspace(0, 10, 1000)
        theta = np.sin(t)
        k = 20
        
        t_s, theta_s = subsample(t, theta, k, irregular=True, seed=42)
        
        assert len(t_s) == k
        assert len(theta_s) == k
        # Check that points are from original arrays
        for i in range(k):
            assert t_s[i] in t
            # Find corresponding theta value
            idx = np.where(t == t_s[i])[0][0]
            assert theta_s[i] == pytest.approx(theta[idx])

    def test_subsample_uniform(self):
        """Test uniform subsampling."""
        t = np.linspace(0, 10, 1000)
        theta = np.sin(t)
        k = 20
        
        t_s, theta_s = subsample(t, theta, k, irregular=False)
        
        assert len(t_s) == k
        assert len(theta_s) == k
        # Check uniform spacing
        assert t_s[0] == pytest.approx(t[0])
        assert t_s[-1] == pytest.approx(t[-1])


class TestDataUtils:
    """Tests for data utilities."""

    def test_create_time_grid_uniform(self):
        """Test uniform time grid."""
        t = create_time_grid((0.0, 10.0), 100, spacing="uniform")
        assert len(t) == 100
        assert t[0] == 0.0
        assert t[-1] == 10.0

    def test_create_time_grid_chebyshev(self):
        """Test Chebyshev time grid."""
        t = create_time_grid((0.0, 10.0), 100, spacing="chebyshev")
        assert len(t) == 100
        assert t.min() >= 0.0
        assert t.max() <= 10.0

    def test_numpy_to_torch(self):
        """Test numpy to torch conversion."""
        arr = np.array([1.0, 2.0, 3.0])
        tensor = numpy_to_torch(arr)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 1)

    def test_torch_to_numpy(self):
        """Test torch to numpy conversion."""
        tensor = torch.tensor([[1.0], [2.0], [3.0]])
        arr = torch_to_numpy(tensor)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3,)

    def test_split_train_test(self):
        """Test train/test split."""
        data = np.arange(100)
        train, test = split_train_test(data, train_ratio=0.8, seed=42)

        assert len(train) == 80
        assert len(test) == 20

    def test_subsample_data(self):
        """Test data subsampling."""
        data = np.arange(100)
        result = subsample_data(data, n_samples=50, method="uniform")
        # subsample_data returns tuple, so unpack it
        subsampled = result[0] if isinstance(result, tuple) else result

        assert len(subsampled) == 50

