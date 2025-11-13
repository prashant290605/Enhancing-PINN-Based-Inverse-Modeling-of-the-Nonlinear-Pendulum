"""Neural network for nonparametric dissipation modeling: D(θ, θ_dot)."""

import torch
import torch.nn as nn
from typing import Optional


class DissipationNet(nn.Module):
    """
    Neural network to model dissipation function D(θ, θ_dot).

    For generalized damping beyond linear: θ''(t) + (g/L)sin(θ) + D(θ, θ_dot) = 0
    
    Key features:
    - Enforced odd symmetry in θ_dot: D(θ, -θ_dot) = -D(θ, θ_dot)
    - Non-negative dissipation magnitude via softplus
    - Enhanced features: |θ_dot|, θ_dot*|θ_dot| for better representation
    
    Implementation: D(θ, θ_dot) = θ_dot * softplus(net(|θ_dot|, θ, θ_dot*|θ_dot|))
    """

    def __init__(
        self,
        hidden_layers: list[int] = [16, 16],
        activation: str = "tanh",
        use_enhanced_features: bool = True,
    ):
        """
        Initialize dissipation network.

        Args:
            hidden_layers: list of hidden layer sizes
            activation: activation function
            use_enhanced_features: whether to use |θ_dot| and θ_dot*|θ_dot| features
        """
        super().__init__()
        
        self.use_enhanced_features = use_enhanced_features

        layers = []
        # Input features: theta, |theta_dot|, theta_dot*|theta_dot|
        input_dim = 3 if use_enhanced_features else 2

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            input_dim = hidden_dim

        # Output layer (outputs magnitude, will be multiplied by sign)
        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
            "softplus": nn.Softplus(),
            "silu": nn.SiLU(),
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        return activations[activation]

    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, theta: torch.Tensor, theta_dot: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute dissipation D(θ, θ_dot) with odd symmetry.

        Enforces odd symmetry: D(θ, -θ_dot) = -D(θ, θ_dot)
        Implementation: D = θ_dot * softplus(net(|θ_dot|, θ, θ_dot*|θ_dot|))
        
        This ensures:
        1. Odd symmetry in θ_dot
        2. Non-negative magnitude (via softplus)
        3. D * θ_dot ≥ 0 (passivity constraint)

        Args:
            theta: angle [N, 1]
            theta_dot: angular velocity [N, 1]

        Returns:
            dissipation [N, 1]
        """
        # Compute enhanced features that preserve odd symmetry
        # Key: only use |theta_dot| and theta as inputs, output magnitude
        abs_theta_dot = torch.abs(theta_dot)
        
        if self.use_enhanced_features:
            # Features: |theta_dot|, theta, |theta_dot|²
            # These are all even in theta_dot, so multiplying by theta_dot gives odd symmetry
            abs_theta_dot_squared = abs_theta_dot ** 2
            x = torch.cat([abs_theta_dot, theta, abs_theta_dot_squared], dim=-1)
        else:
            # Basic features: |theta_dot|, theta
            x = torch.cat([abs_theta_dot, theta], dim=-1)
        
        # Network outputs magnitude (always positive via softplus)
        magnitude = torch.nn.functional.softplus(self.network(x))
        
        # Multiply by sign(theta_dot) to enforce odd symmetry
        # D(θ, θ_dot) = sign(θ_dot) * magnitude(|θ_dot|, θ)
        # This ensures D(θ, -θ_dot) = -D(θ, θ_dot)
        # Use theta_dot / (|theta_dot| + eps) instead of sign() for smooth gradients
        eps = 1e-8
        dissipation = theta_dot * magnitude / (abs_theta_dot + eps)
        
        return dissipation


class LinearDissipation(nn.Module):
    """Simple linear dissipation: D(θ, θ_dot) = damping * θ_dot."""

    def __init__(self, init_damping: float = 0.1, learn_damping: bool = True):
        """
        Initialize linear dissipation.

        Args:
            init_damping: initial damping coefficient
            learn_damping: whether damping is trainable
        """
        super().__init__()
        self.damping = nn.Parameter(
            torch.tensor([init_damping], dtype=torch.float32), requires_grad=learn_damping
        )

    def forward(self, theta: torch.Tensor, theta_dot: torch.Tensor) -> torch.Tensor:
        """Compute linear dissipation."""
        return self.damping * theta_dot


class QuadraticDissipation(nn.Module):
    """Quadratic dissipation: D(θ, θ_dot) = c1*θ_dot + c2*θ_dot*|θ_dot|."""

    def __init__(
        self,
        init_c1: float = 0.1,
        init_c2: float = 0.01,
        learn_c1: bool = True,
        learn_c2: bool = True,
    ):
        """
        Initialize quadratic dissipation.

        Args:
            init_c1: initial linear coefficient
            init_c2: initial quadratic coefficient
            learn_c1: whether c1 is trainable
            learn_c2: whether c2 is trainable
        """
        super().__init__()
        self.c1 = nn.Parameter(torch.tensor([init_c1], dtype=torch.float32), requires_grad=learn_c1)
        self.c2 = nn.Parameter(torch.tensor([init_c2], dtype=torch.float32), requires_grad=learn_c2)

    def forward(self, theta: torch.Tensor, theta_dot: torch.Tensor) -> torch.Tensor:
        """Compute quadratic dissipation."""
        return self.c1 * theta_dot + self.c2 * theta_dot * torch.abs(theta_dot)


class PassivityConstrainedDissipation(nn.Module):
    """
    Dissipation network with built-in passivity constraint.

    Ensures D(θ, θ_dot) * θ_dot ≥ 0 (energy dissipation is non-negative).
    """

    def __init__(self, hidden_layers: list[int] = [16, 16], activation: str = "tanh"):
        """
        Initialize passivity-constrained dissipation network.

        Args:
            hidden_layers: list of hidden layer sizes
            activation: activation function
        """
        super().__init__()

        layers = []
        input_dim = 2  # (theta, theta_dot)

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
            "silu": nn.SiLU(),
        }
        return activations.get(activation, nn.Tanh())

    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, theta: torch.Tensor, theta_dot: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with passivity constraint.

        Output: D(θ, θ_dot) = sign(θ_dot) * |f(θ, θ_dot)|
        where f is the neural network output.

        This ensures D * θ_dot ≥ 0.

        Args:
            theta: angle [N, 1]
            theta_dot: angular velocity [N, 1]

        Returns:
            dissipation [N, 1]
        """
        x = torch.cat([theta, theta_dot], dim=-1)
        f = self.network(x)

        # Ensure passivity: D * theta_dot >= 0
        # D = sign(theta_dot) * |f|
        dissipation = torch.sign(theta_dot) * torch.abs(f)

        return dissipation


def create_dissipation_model(
    model_type: str = "linear",
    **kwargs,
) -> nn.Module:
    """
    Factory function to create dissipation models.

    Args:
        model_type: 'linear', 'quadratic', 'neural', 'passivity_constrained'
        **kwargs: additional arguments

    Returns:
        dissipation model
    """
    models = {
        "linear": LinearDissipation,
        "quadratic": QuadraticDissipation,
        "neural": DissipationNet,
        "passivity_constrained": PassivityConstrainedDissipation,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")

    return models[model_type](**kwargs)

