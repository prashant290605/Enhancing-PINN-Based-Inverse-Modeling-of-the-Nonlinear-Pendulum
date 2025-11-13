"""Inverse PINN model: learn θ(t) and infer system parameters (g, L, damping)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple


class FourierFeatures(nn.Module):
    """Fourier feature encoding for time input."""
    
    def __init__(self, num_frequencies: int = 8):
        """
        Initialize Fourier features.
        
        Args:
            num_frequencies: number of frequency components (default: 8)
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        # Create frequency scales: [1, 2, 3, ..., num_frequencies]
        self.register_buffer('frequencies', torch.arange(1, num_frequencies + 1, dtype=torch.float32))
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier feature encoding.
        
        Args:
            t: time input [N, 1]
            
        Returns:
            features: [N, 1 + 2*num_frequencies] (original t + sin/cos features)
        """
        # Compute sin and cos features for each frequency
        # t: [N, 1], frequencies: [num_frequencies]
        # t * frequencies: [N, num_frequencies]
        angles = 2 * np.pi * t * self.frequencies.unsqueeze(0)
        
        sin_features = torch.sin(angles)  # [N, num_frequencies]
        cos_features = torch.cos(angles)  # [N, num_frequencies]
        
        # Concatenate: [t, sin(2πt), cos(2πt), sin(4πt), cos(4πt), ...]
        features = torch.cat([t, sin_features, cos_features], dim=1)
        
        return features
    
    @property
    def output_dim(self) -> int:
        """Output dimension: 1 (original) + 2*num_frequencies (sin/cos)."""
        return 1 + 2 * self.num_frequencies


class PINN(nn.Module):
    """Physics-Informed Neural Network for pendulum inverse problem."""

    def __init__(
        self,
        hidden_layers: list[int] = [32, 32, 32],
        activation: str = "tanh",
        init_g: float = 9.81,
        init_L: float = 1.0,
        init_damping: float = 0.1,
        learn_g: bool = True,
        learn_L: bool = True,
        learn_damping: bool = True,
        use_fourier: bool = True,
        num_frequencies: int = 8,
    ):
        """
        Initialize PINN.

        Args:
            hidden_layers: list of hidden layer sizes
            activation: activation function ('tanh', 'relu', 'sigmoid', 'gelu')
            init_g: initial value for g
            init_L: initial value for L
            init_damping: initial value for damping
            learn_g: whether g is trainable
            learn_L: whether L is trainable
            learn_damping: whether damping is trainable
            use_fourier: whether to use Fourier feature encoding
            num_frequencies: number of Fourier frequencies (6-8 recommended)
        """
        super().__init__()
        
        self.use_fourier = use_fourier
        
        # Fourier feature encoding
        if use_fourier:
            self.fourier = FourierFeatures(num_frequencies=num_frequencies)
            input_dim = self.fourier.output_dim
        else:
            self.fourier = None
            input_dim = 1

        # Build neural network
        layers = []

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            input_dim = hidden_dim

        # Output layer (theta)
        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights (Xavier/Glorot)
        self._initialize_weights()

        # Physical parameters (trainable, stored in unconstrained space)
        # Use softplus to enforce positivity: param = softplus(param_raw)
        # Inverse: param_raw = log(exp(param) - 1) ≈ log(param) for large param
        self.g_raw = nn.Parameter(
            torch.tensor([self._inverse_softplus(init_g)], dtype=torch.float32), 
            requires_grad=learn_g
        )
        self.L_raw = nn.Parameter(
            torch.tensor([self._inverse_softplus(init_L)], dtype=torch.float32), 
            requires_grad=learn_L
        )
        self.damping_raw = nn.Parameter(
            torch.tensor([self._inverse_softplus(init_damping)], dtype=torch.float32), 
            requires_grad=learn_damping
        )

        self.learn_g = learn_g
        self.learn_L = learn_L
        self.learn_damping = learn_damping
    
    @staticmethod
    def _inverse_softplus(x: float, beta: float = 1.0) -> float:
        """Inverse of softplus for initialization."""
        return np.log(np.exp(beta * x) - 1) / beta
    
    @property
    def g(self) -> torch.Tensor:
        """Get g with positivity constraint via softplus."""
        return F.softplus(self.g_raw)
    
    @property
    def L(self) -> torch.Tensor:
        """Get L with positivity constraint via softplus."""
        return F.softplus(self.L_raw)
    
    @property
    def damping(self) -> torch.Tensor:
        """Get damping with positivity constraint via softplus."""
        return F.softplus(self.damping_raw)

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
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        return activations[activation]

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict theta(t).

        Args:
            t: time inputs [N, 1]

        Returns:
            theta: angle predictions [N, 1]
        """
        # Apply Fourier features if enabled
        if self.use_fourier:
            features = self.fourier(t)
        else:
            features = t
        
        return self.network(features)

    def get_parameters(self) -> Dict[str, float]:
        """Get current physical parameters (with positivity constraints applied)."""
        return {
            "g": self.g.item(),
            "L": self.L.item(),
            "damping": self.damping.item(),
        }

    def predict_with_derivatives(
        self, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict theta and its derivatives.

        Args:
            t: time inputs [N, 1] (must have requires_grad=True)

        Returns:
            theta: angle [N, 1]
            theta_dot: angular velocity [N, 1]
            theta_ddot: angular acceleration [N, 1]
        """
        theta = self.forward(t)

        theta_dot = torch.autograd.grad(
            theta,
            t,
            grad_outputs=torch.ones_like(theta),
            create_graph=True,
            retain_graph=True,
        )[0]

        theta_ddot = torch.autograd.grad(
            theta_dot,
            t,
            grad_outputs=torch.ones_like(theta_dot),
            create_graph=True,
            retain_graph=True,
        )[0]

        return theta, theta_dot, theta_ddot


class AdaptivePINN(PINN):
    """PINN with adaptive activation functions (learnable parameters)."""

    def __init__(
        self,
        hidden_layers: list[int] = [32, 32, 32],
        activation: str = "adaptive_tanh",
        init_g: float = 9.81,
        init_L: float = 1.0,
        init_damping: float = 0.1,
        learn_g: bool = True,
        learn_L: bool = True,
        learn_damping: bool = True,
    ):
        """Initialize adaptive PINN with learnable activation scaling."""
        # Build network manually for adaptive activations
        self.hidden_layers = hidden_layers
        self.activation_type = activation

        # Call parent init but we'll rebuild the network
        super().__init__(
            hidden_layers=hidden_layers,
            activation="tanh",  # temporary
            init_g=init_g,
            init_L=init_L,
            init_damping=init_damping,
            learn_g=learn_g,
            learn_L=learn_L,
            learn_damping=learn_damping,
        )

        # Rebuild with adaptive activations
        if "adaptive" in activation:
            self._build_adaptive_network()

    def _build_adaptive_network(self):
        """Build network with adaptive activations."""
        layers = []
        input_dim = 1

        for i, hidden_dim in enumerate(self.hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))

            # Adaptive activation with learnable parameter
            if self.activation_type == "adaptive_tanh":
                layers.append(AdaptiveTanh())
            else:
                layers.append(nn.Tanh())

            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
        self._initialize_weights()


class AdaptiveTanh(nn.Module):
    """Adaptive tanh activation: a * tanh(b * x)."""

    def __init__(self, init_a: float = 1.0, init_b: float = 1.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([init_a]))
        self.b = nn.Parameter(torch.tensor([init_b]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * torch.tanh(self.b * x)


def create_pinn(
    architecture: str = "default",
    **kwargs,
) -> PINN:
    """
    Factory function to create PINN models.

    Args:
        architecture: 'default', 'shallow', 'deep', 'wide', 'adaptive'
        **kwargs: additional arguments for PINN

    Returns:
        PINN model
    """
    architectures = {
        "default": {"hidden_layers": [32, 32, 32], "activation": "tanh"},
        "shallow": {"hidden_layers": [64], "activation": "tanh"},
        "deep": {"hidden_layers": [32, 32, 32, 32, 32], "activation": "tanh"},
        "wide": {"hidden_layers": [128, 128], "activation": "tanh"},
        "adaptive": {"hidden_layers": [32, 32, 32], "activation": "adaptive_tanh"},
    }

    if architecture not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}")

    config = architectures[architecture]
    config.update(kwargs)

    if "adaptive" in architecture:
        return AdaptivePINN(**config)
    else:
        return PINN(**config)

