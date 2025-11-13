"""Tests for loss functions."""

import pytest
import torch
import numpy as np

from src.models.losses import (
    grad,
    grad2,
    compute_derivatives,
    loss_data,
    loss_ic,
    loss_phys,
    loss_passivity,
    physics_loss,
    initial_condition_loss,
    data_loss,
    passivity_loss,
    combined_loss,
    total_loss,
)


class TestDerivatives:
    """Tests for automatic differentiation."""

    def test_compute_derivatives(self):
        """Test derivative computation."""
        # Create simple function: theta = t^2
        t = torch.linspace(0, 1, 10, requires_grad=True).unsqueeze(-1)
        theta = t**2

        theta_dot, theta_ddot = compute_derivatives(theta, t)

        # Expected: theta_dot = 2t, theta_ddot = 2
        assert theta_dot.shape == theta.shape
        assert theta_ddot.shape == theta.shape

        # Check approximate values
        assert theta_dot[5].item() == pytest.approx(2 * t[5].item(), abs=1e-3)
        assert theta_ddot[5].item() == pytest.approx(2.0, abs=1e-2)


class TestPhysicsLoss:
    """Tests for physics loss."""

    def test_physics_loss_shape(self):
        """Test physics loss computation."""
        t = torch.linspace(0, 1, 10, requires_grad=True).unsqueeze(-1)
        theta = torch.sin(t)
        g = torch.tensor([9.81])
        L = torch.tensor([1.0])

        loss = physics_loss(theta, t, g, L)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar

    def test_physics_loss_with_damping(self):
        """Test physics loss with damping."""
        t = torch.linspace(0, 1, 10, requires_grad=True).unsqueeze(-1)
        theta = torch.sin(t)
        g = torch.tensor([9.81])
        L = torch.tensor([1.0])
        damping = torch.tensor([0.1])

        loss = physics_loss(theta, t, g, L, damping)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0


class TestInitialConditionLoss:
    """Tests for initial condition loss."""

    def test_ic_loss(self):
        """Test initial condition loss."""
        theta_pred = torch.tensor([[0.5]])
        theta_dot_pred = torch.tensor([[0.0]])
        theta0 = 0.5
        theta_dot0 = 0.0

        loss = initial_condition_loss(theta_pred, theta_dot_pred, theta0, theta_dot0)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_ic_loss_nonzero(self):
        """Test IC loss with error."""
        theta_pred = torch.tensor([[0.6]])
        theta_dot_pred = torch.tensor([[0.1]])
        theta0 = 0.5
        theta_dot0 = 0.0

        loss = initial_condition_loss(theta_pred, theta_dot_pred, theta0, theta_dot0)

        assert loss.item() > 0.0


class TestDataLoss:
    """Tests for data loss."""

    def test_data_loss(self):
        """Test data loss computation."""
        theta_pred = torch.tensor([[0.5], [0.6], [0.7]])
        theta_data = torch.tensor([[0.5], [0.6], [0.7]])

        loss = data_loss(theta_pred, theta_data)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_data_loss_with_weights(self):
        """Test weighted data loss."""
        theta_pred = torch.tensor([[0.5], [0.6], [0.7]])
        theta_data = torch.tensor([[0.5], [0.6], [0.7]])
        weights = torch.tensor([[1.0], [2.0], [3.0]])

        loss = data_loss(theta_pred, theta_data, weights)

        assert isinstance(loss, torch.Tensor)


class TestPassivityLoss:
    """Tests for passivity loss."""

    def test_passivity_loss_shape(self):
        """Test passivity loss computation."""
        t = torch.linspace(0, 1, 10, requires_grad=True).unsqueeze(-1)
        theta = torch.sin(t)
        g = torch.tensor([9.81])
        L = torch.tensor([1.0])

        loss = passivity_loss(theta, t, g, L)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
    
    def test_passivity_conservative_solution(self):
        """
        Test passivity loss for a perfect analytic conservative solution with c=0.
        
        For a conservative pendulum (no damping), the energy should be constant,
        so dH/dt ≈ 0, and loss_passivity should be ≈0.
        """
        # Use analytic small-angle solution: θ(t) = θ₀ cos(ω₀ t) with ω₀ = √(g/L)
        g = 9.81
        L = 1.0
        theta0 = 0.1  # small angle (radians)
        omega0 = np.sqrt(g / L)
        
        # Time grid
        t_np = np.linspace(0, 5, 100)
        t = torch.tensor(t_np, dtype=torch.float32, requires_grad=True).unsqueeze(-1)
        
        # Analytic solution
        theta_np = theta0 * np.cos(omega0 * t_np)
        theta = torch.tensor(theta_np, dtype=torch.float32).unsqueeze(-1)
        
        # Compute derivatives analytically (for perfect solution)
        theta_dot_np = -theta0 * omega0 * np.sin(omega0 * t_np)
        theta_ddot_np = -theta0 * omega0**2 * np.cos(omega0 * t_np)
        
        theta_dot = torch.tensor(theta_dot_np, dtype=torch.float32).unsqueeze(-1)
        theta_ddot = torch.tensor(theta_ddot_np, dtype=torch.float32).unsqueeze(-1)
        
        # Compute passivity loss
        loss = loss_passivity(theta, theta_dot, theta_ddot, g, L, m=1.0, eps=1e-5)
        
        # For conservative system, energy should be constant, so loss should be ≈0
        # Allow small tolerance for numerical precision
        assert loss.item() < 1e-3, f"Passivity loss for conservative solution should be ≈0, got {loss.item()}"
    
    def test_passivity_ascending_energy(self):
        """
        Test passivity loss for injected ascending-energy fake signal.
        
        For a signal with increasing energy, loss_passivity should be > 0.
        """
        # Create fake signal with increasing energy
        # θ(t) = A·t (linearly increasing angle)
        # θ̇(t) = A (constant velocity)
        # θ̈(t) = 0
        g = 9.81
        L = 1.0
        A = 0.5
        
        t_np = np.linspace(0, 5, 100)
        t = torch.tensor(t_np, dtype=torch.float32, requires_grad=True).unsqueeze(-1)
        
        # Fake signal with increasing energy
        theta = torch.tensor(A * t_np, dtype=torch.float32).unsqueeze(-1)
        theta_dot = torch.ones_like(theta) * A
        theta_ddot = torch.zeros_like(theta)
        
        # Compute passivity loss
        loss = loss_passivity(theta, theta_dot, theta_ddot, g, L, m=1.0, eps=1e-5)
        
        # Energy is increasing, so loss should be > 0
        assert loss.item() > 0, f"Passivity loss for ascending energy should be > 0, got {loss.item()}"


class TestCombinedLoss:
    """Tests for combined loss."""

    def test_combined_loss(self):
        """Test combined loss computation."""
        t = torch.linspace(0, 1, 10, requires_grad=True).unsqueeze(-1)
        theta = torch.sin(t)
        theta_data = torch.sin(t)
        g = torch.tensor([9.81])
        L = torch.tensor([1.0])
        damping = torch.tensor([0.1])

        weights = {"physics": 1.0, "data": 1.0, "ic": 1.0}

        losses = combined_loss(
            theta,
            t,
            theta_data,
            g,
            L,
            damping,
            theta0=0.0,
            theta_dot0=1.0,
            weights=weights,
            use_passivity=False,
        )

        assert "physics" in losses
        assert "data" in losses
        assert "ic" in losses
        assert "total" in losses

    def test_combined_loss_with_passivity(self):
        """Test combined loss with passivity."""
        t = torch.linspace(0, 1, 10, requires_grad=True).unsqueeze(-1)
        theta = torch.sin(t)
        theta_data = torch.sin(t)
        g = torch.tensor([9.81])
        L = torch.tensor([1.0])
        damping = torch.tensor([0.1])

        weights = {"physics": 1.0, "data": 1.0, "ic": 1.0, "passivity": 1.0}

        losses = combined_loss(
            theta,
            t,
            theta_data,
            g,
            L,
            damping,
            theta0=0.0,
            theta_dot0=1.0,
            weights=weights,
            use_passivity=True,
        )

        assert "passivity" in losses

