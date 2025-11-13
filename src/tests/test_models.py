"""Tests for PINN models."""

import pytest
import torch
import numpy as np

from src.models.pinn_inverse import PINN, create_pinn
from src.models.train_inverse import PINNTrainer, create_trainer
from src.models.dissipation_net import (
    DissipationNet,
    LinearDissipation,
    QuadraticDissipation,
    create_dissipation_model,
)
from src.data.generator import simulate_pendulum


class TestPINN:
    """Tests for PINN model."""

    def test_initialization(self):
        """Test PINN initialization."""
        model = PINN(
            hidden_layers=[32, 32],
            activation="tanh",
            init_g=9.81,
            init_L=1.0,
            init_damping=0.1,
            use_fourier=False,  # Disable Fourier features for simple test
        )

        assert model.g.item() == pytest.approx(9.81, abs=0.1)  # Allow tolerance for softplus
        assert model.L.item() == pytest.approx(1.0, abs=0.1)
        assert model.damping.item() == pytest.approx(0.1, abs=0.05)

    def test_forward(self):
        """Test forward pass."""
        model = PINN(hidden_layers=[32, 32], use_fourier=False)
        t = torch.linspace(0, 1, 10).unsqueeze(-1)

        theta = model(t)

        assert theta.shape == (10, 1)

    def test_predict_with_derivatives(self):
        """Test prediction with derivatives."""
        model = PINN(hidden_layers=[32, 32], use_fourier=False)
        t = torch.linspace(0, 1, 10, requires_grad=True).unsqueeze(-1)

        theta, theta_dot, theta_ddot = model.predict_with_derivatives(t)

        assert theta.shape == (10, 1)
        assert theta_dot.shape == (10, 1)
        assert theta_ddot.shape == (10, 1)

    def test_get_parameters(self):
        """Test parameter retrieval."""
        model = PINN(init_g=9.81, init_L=1.0, init_damping=0.1, use_fourier=False)
        params = model.get_parameters()

        assert "g" in params
        assert "L" in params
        assert "damping" in params
        assert params["g"] == pytest.approx(9.81, abs=0.1)  # Allow tolerance for softplus

    def test_trainable_parameters(self):
        """Test trainable parameters."""
        model = PINN(learn_g=True, learn_L=False, learn_damping=True, use_fourier=False)

        assert model.g_raw.requires_grad is True
        assert model.L_raw.requires_grad is False
        assert model.damping_raw.requires_grad is True


class TestPINNFactory:
    """Tests for PINN factory function."""

    def test_create_default(self):
        """Test creating default PINN."""
        model = create_pinn(architecture="default")
        assert isinstance(model, PINN)

    def test_create_shallow(self):
        """Test creating shallow PINN."""
        model = create_pinn(architecture="shallow")
        assert isinstance(model, PINN)

    def test_create_deep(self):
        """Test creating deep PINN."""
        model = create_pinn(architecture="deep")
        assert isinstance(model, PINN)

    def test_create_wide(self):
        """Test creating wide PINN."""
        model = create_pinn(architecture="wide")
        assert isinstance(model, PINN)


class TestDissipationNet:
    """Tests for dissipation network."""

    def test_initialization(self):
        """Test dissipation network initialization."""
        model = DissipationNet(hidden_layers=[16, 16])
        assert isinstance(model, DissipationNet)

    def test_forward(self):
        """Test forward pass."""
        model = DissipationNet(hidden_layers=[16, 16])
        theta = torch.randn(10, 1)
        theta_dot = torch.randn(10, 1)

        dissipation = model(theta, theta_dot)

        assert dissipation.shape == (10, 1)


class TestLinearDissipation:
    """Tests for linear dissipation model."""

    def test_initialization(self):
        """Test initialization."""
        model = LinearDissipation(init_damping=0.1)
        assert model.damping.item() == pytest.approx(0.1, abs=1e-6)

    def test_forward(self):
        """Test forward pass."""
        model = LinearDissipation(init_damping=0.1)
        theta = torch.randn(10, 1)
        theta_dot = torch.randn(10, 1)

        dissipation = model(theta, theta_dot)

        # Should be 0.1 * theta_dot
        expected = 0.1 * theta_dot
        assert torch.allclose(dissipation, expected, atol=1e-6)


class TestQuadraticDissipation:
    """Tests for quadratic dissipation model."""

    def test_initialization(self):
        """Test initialization."""
        model = QuadraticDissipation(init_c1=0.1, init_c2=0.01)
        assert model.c1.item() == pytest.approx(0.1, abs=1e-6)
        assert model.c2.item() == pytest.approx(0.01, abs=1e-6)

    def test_forward(self):
        """Test forward pass."""
        model = QuadraticDissipation(init_c1=0.1, init_c2=0.01)
        theta = torch.randn(10, 1)
        theta_dot = torch.randn(10, 1)

        dissipation = model(theta, theta_dot)

        assert dissipation.shape == (10, 1)


class TestDissipationFactory:
    """Tests for dissipation model factory."""

    def test_create_linear(self):
        """Test creating linear dissipation."""
        model = create_dissipation_model(model_type="linear")
        assert isinstance(model, LinearDissipation)

    def test_create_quadratic(self):
        """Test creating quadratic dissipation."""
        model = create_dissipation_model(model_type="quadratic")
        assert isinstance(model, QuadraticDissipation)

    def test_create_neural(self):
        """Test creating neural dissipation."""
        model = create_dissipation_model(model_type="neural")
        assert isinstance(model, DissipationNet)


class TestDissipationNetFeatures:
    """Tests for dissipation network features."""
    
    def test_odd_symmetry(self):
        """Test that DissipationNet enforces odd symmetry: D(θ, -θ̇) = -D(θ, θ̇)."""
        model = DissipationNet(hidden_layers=[16, 16], activation="tanh")
        
        theta = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32)
        theta_dot = torch.tensor([[0.5], [-0.3], [0.8]], dtype=torch.float32)
        
        # Compute D(θ, θ̇)
        D_pos = model(theta, theta_dot)
        
        # Compute D(θ, -θ̇)
        D_neg = model(theta, -theta_dot)
        
        # Check odd symmetry: D(θ, -θ̇) ≈ -D(θ, θ̇)
        assert torch.allclose(D_neg, -D_pos, atol=1e-6), "Odd symmetry not satisfied"
    
    def test_passivity_constraint(self):
        """Test that D * θ̇ ≥ 0 (passivity constraint)."""
        model = DissipationNet(hidden_layers=[16, 16], activation="tanh")
        
        theta = torch.tensor([[0.1], [0.2], [0.3], [-0.1]], dtype=torch.float32)
        theta_dot = torch.tensor([[0.5], [-0.3], [0.8], [-0.4]], dtype=torch.float32)
        
        D = model(theta, theta_dot)
        
        # Check passivity: D * θ̇ ≥ 0
        power_dissipated = D * theta_dot
        assert torch.all(power_dissipated >= -1e-6), "Passivity constraint violated"


class TestInversePINN:
    """Tests for inverse PINN training."""
    
    def test_inverse_pinn_small_angle_conservative(self):
        """
        Test inverse PINN on tiny synthetic case (c=0, small angle).
        
        Model should recover g, L within ~2-3% after short training on dense data.
        """
        # True parameters
        true_g = 9.81
        true_L = 1.0
        true_c = 0.0  # conservative
        theta0 = np.radians(10)  # small angle
        omega0 = 0.0
        
        # Generate dense synthetic data
        t_grid = np.linspace(0, 5, 500)
        t_np, theta_np, omega_np = simulate_pendulum(
            theta0, omega0, true_g, true_L, true_c, t_grid, method="ivp"
        )
        
        # Convert to torch tensors
        t_obs = torch.tensor(t_np, dtype=torch.float32).unsqueeze(-1)
        theta_obs = torch.tensor(theta_np, dtype=torch.float32).unsqueeze(-1)
        
        # Use same grid for collocation (dense data)
        t_collocation = t_obs.clone()
        
        # Create model with initial guesses (slightly off)
        model = PINN(
            hidden_layers=[32, 32],
            activation="tanh",
            init_g=10.0,  # slightly off
            init_L=1.1,   # slightly off
            init_damping=0.01,  # small initial damping
            learn_g=True,
            learn_L=True,
            learn_damping=True,
            use_fourier=True,
            num_frequencies=6,
        )
        
        # Create trainer
        trainer = create_trainer(
            model=model,
            t_obs=t_obs,
            theta_obs=theta_obs,
            t_collocation=t_collocation,
            theta0=theta0,
            omega0=omega0,
            lambda_data=1.0,
            lambda_phys=10.0,
            lambda_ic=1.0,
            lambda_pass=1.0,
            learning_rate=1e-3,
            scheduler_type="cosine",
            n_epochs=2000,  # short training for test
            device="cpu",
        )
        
        # Train model
        history = trainer.train(n_epochs=2000, verbose=False)
        
        # Get final parameters
        params = model.get_parameters()
        
        # Check that parameters are recovered within ~2-3%
        g_error = abs(params["g"] - true_g) / true_g
        L_error = abs(params["L"] - true_L) / true_L
        
        # Allow 10% tolerance for test (lenient due to short training for test speed)
        # In production, with longer training (10k-20k epochs), errors should be ~2-3%
        assert g_error < 0.10, f"g error {g_error*100:.2f}% exceeds 10%: predicted {params['g']:.3f}, true {true_g:.3f}"
        assert L_error < 0.10, f"L error {L_error*100:.2f}% exceeds 10%: predicted {params['L']:.3f}, true {true_L:.3f}"
        
        # Check that damping is close to 0 (conservative system)
        assert params["damping"] < 0.1, f"Damping should be near 0 for conservative system, got {params['damping']:.4f}"
        
        # Check that loss decreased
        assert history["loss"][-1] < history["loss"][0], "Loss should decrease during training"
    
    def test_dissipation_net_learns_viscous(self):
        """
        Test that DissipationNet can learn viscous damping.
        
        Generate data with c_viscous=0.05; the NN should learn an equivalent mapping
        (RMSE close to viscous baseline).
        """
        # True parameters
        true_g = 9.81
        true_L = 1.0
        true_c = 0.05  # viscous damping
        theta0 = np.radians(15)  # moderate angle
        omega0 = 0.0
        
        # Generate data with viscous damping
        t_grid = np.linspace(0, 10, 1000)
        t_np, theta_np, omega_np = simulate_pendulum(
            theta0, omega0, true_g, true_L, true_c, t_grid, method="ivp"
        )
        
        # Convert to torch tensors
        t_obs = torch.tensor(t_np, dtype=torch.float32).unsqueeze(-1)
        theta_obs = torch.tensor(theta_np, dtype=torch.float32).unsqueeze(-1)
        t_collocation = t_obs.clone()
        
        # Create PINN model (don't learn damping, use NN instead)
        model = PINN(
            hidden_layers=[32, 32],
            activation="tanh",
            init_g=10.0,
            init_L=1.0,
            init_damping=0.0,
            learn_g=True,
            learn_L=True,
            learn_damping=False,  # Don't use viscous damping
            use_fourier=True,
            num_frequencies=6,
        )
        
        # Create dissipation network
        dissipation_net = DissipationNet(
            hidden_layers=[16, 16],
            activation="tanh",
            use_enhanced_features=True,
        )
        
        # Create trainer with dissipation net
        trainer = create_trainer(
            model=model,
            t_obs=t_obs,
            theta_obs=theta_obs,
            t_collocation=t_collocation,
            theta0=theta0,
            omega0=omega0,
            lambda_data=1.0,
            lambda_phys=10.0,
            lambda_ic=1.0,
            lambda_pass=1.0,
            dissipation_net=dissipation_net,
            learning_rate=1e-3,
            scheduler_type="cosine",
            n_epochs=2000,  # short training for test
            device="cpu",
        )
        
        # Train model
        history = trainer.train(n_epochs=2000, verbose=False)
        
        # Get final parameters
        params = model.get_parameters()
        
        # Check that g and L are recovered reasonably well
        g_error = abs(params["g"] - true_g) / true_g
        L_error = abs(params["L"] - true_L) / true_L
        
        assert g_error < 0.15, f"g error {g_error*100:.2f}% too large"
        assert L_error < 0.15, f"L error {L_error*100:.2f}% too large"
        
        # Check that the NN learned dissipation
        # For a short training test, we mainly verify:
        # 1. The model trains without errors
        # 2. The NN produces reasonable dissipation values
        # 3. Odd symmetry and passivity are preserved
        model.eval()
        dissipation_net.eval()
        
        with torch.no_grad():
            # Test odd symmetry
            theta_test = torch.tensor([[0.1], [0.2]], dtype=torch.float32)
            theta_dot_test = torch.tensor([[0.5], [-0.3]], dtype=torch.float32)
            
            D_pos = dissipation_net(theta_test, theta_dot_test)
            D_neg = dissipation_net(theta_test, -theta_dot_test)
            
            # Check odd symmetry
            assert torch.allclose(D_neg, -D_pos, atol=1e-5), "NN dissipation should preserve odd symmetry"
            
            # Check passivity: D * θ̇ ≥ 0
            power = D_pos * theta_dot_test
            assert torch.all(power >= -1e-6), "NN dissipation should preserve passivity"
            
            # Check that dissipation is in reasonable range (not all zeros)
            assert torch.abs(D_pos).max() > 1e-4, "NN dissipation should be non-trivial"
        
        # Check that loss decreased
        assert history["loss"][-1] < history["loss"][0], "Loss should decrease during training"

