"""Loss functions for PINN training: physics, IC/BC, passivity, data."""

import torch
from typing import Dict, Optional, Union


def grad(y: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """
    Compute first derivative dy/dx using automatic differentiation.
    
    Args:
        y: output tensor [N, 1]
        x: input tensor [N, 1] (must have requires_grad=True)
        create_graph: whether to create computation graph for higher-order derivatives
        
    Returns:
        dy/dx: first derivative [N, 1]
    """
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=create_graph,
        retain_graph=True,
    )[0]


def grad2(y: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """
    Compute second derivative d²y/dx² using automatic differentiation.
    
    Args:
        y: output tensor [N, 1]
        x: input tensor [N, 1] (must have requires_grad=True)
        create_graph: whether to create computation graph
        
    Returns:
        d²y/dx²: second derivative [N, 1]
    """
    dy_dx = grad(y, x, create_graph=True)
    d2y_dx2 = grad(dy_dx, x, create_graph=create_graph)
    return d2y_dx2


def compute_derivatives(
    theta: torch.Tensor, t: torch.Tensor, create_graph: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute first and second derivatives θ̇, θ̈ from network output θ̂(t).

    Args:
        theta: angle predictions [N, 1]
        t: time inputs [N, 1] (must have requires_grad=True)
        create_graph: whether to create computation graph for higher-order derivatives

    Returns:
        theta_dot: first derivative θ̇ [N, 1]
        theta_ddot: second derivative θ̈ [N, 1]
    """
    theta_dot = grad(theta, t, create_graph=True)
    theta_ddot = grad(theta_dot, t, create_graph=create_graph)
    return theta_dot, theta_ddot


def loss_data(
    theta_hat_at_obs: torch.Tensor,
    theta_obs: torch.Tensor,
) -> torch.Tensor:
    """
    Data fitting loss.
    
    Args:
        theta_hat_at_obs: predicted angles at observation points [N, 1]
        theta_obs: observed angles [N, 1]
        
    Returns:
        mean squared error
    """
    return torch.mean((theta_hat_at_obs - theta_obs) ** 2)


def loss_velocity(
    omega_hat_at_obs: torch.Tensor,
    omega_obs: torch.Tensor,
) -> torch.Tensor:
    """
    Velocity data fitting loss.
    
    Args:
        omega_hat_at_obs: predicted angular velocities at observation points [N, 1]
        omega_obs: observed angular velocities [N, 1]
        
    Returns:
        mean squared error
    """
    return torch.mean((omega_hat_at_obs - omega_obs) ** 2)


def loss_ic(
    theta_hat0: torch.Tensor,
    omega_hat0: torch.Tensor,
    theta0: float,
    omega0: float,
) -> torch.Tensor:
    """
    Initial condition loss.
    
    Args:
        theta_hat0: predicted angle at t=0 [1, 1]
        omega_hat0: predicted angular velocity at t=0 [1, 1]
        theta0: true initial angle
        omega0: true initial angular velocity
        
    Returns:
        mean squared error
    """
    loss_theta = (theta_hat0 - theta0) ** 2
    loss_omega = (omega_hat0 - omega0) ** 2
    return torch.mean(loss_theta + loss_omega)


def loss_phys(
    theta_ddot: torch.Tensor,
    theta: torch.Tensor,
    theta_dot: torch.Tensor,
    g: Union[torch.Tensor, float],
    L: Union[torch.Tensor, float],
    D: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Physics loss: residual of pendulum ODE.
    
    Equation: θ̈ + (g/L)sin(θ) + D = 0
    
    where D is either:
    - c·θ̇ (linear damping)
    - D_φ(θ, θ̇) (neural network dissipation)
    
    Args:
        theta_ddot: second time derivative θ̈ [N, 1]
        theta: angle [N, 1]
        theta_dot: first time derivative θ̇ [N, 1]
        g: gravitational acceleration
        L: pendulum length
        D: dissipation term [N, 1] (optional)
        
    Returns:
        mean squared residual
    """
    # Convert to tensors if needed
    if not isinstance(g, torch.Tensor):
        g = torch.tensor(g, dtype=theta.dtype, device=theta.device)
    if not isinstance(L, torch.Tensor):
        L = torch.tensor(L, dtype=theta.dtype, device=theta.device)
    
    # Physics residual: θ̈ + (g/L)sin(θ) + D = 0
    residual = theta_ddot + (g / L) * torch.sin(theta)
    
    if D is not None:
        residual = residual + D
    
    return torch.mean(residual ** 2)


def physics_loss(
    theta: torch.Tensor,
    t: torch.Tensor,
    g: torch.Tensor,
    L: torch.Tensor,
    damping: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Physics-informed loss: residual of pendulum ODE.

    Equation: θ''(t) + (g/L)sin(θ) + damping*θ'(t) = 0

    Args:
        theta: angle predictions [N, 1]
        t: time inputs [N, 1] (requires_grad=True)
        g: gravitational acceleration (trainable parameter)
        L: pendulum length (trainable parameter)
        damping: optional damping coefficient (trainable parameter)

    Returns:
        mean squared residual
    """
    theta_dot, theta_ddot = compute_derivatives(theta, t)

    # Compute dissipation term
    D = damping * theta_dot if damping is not None else None
    
    return loss_phys(theta_ddot, theta, theta_dot, g, L, D)


def initial_condition_loss(
    theta_pred: torch.Tensor,
    theta_dot_pred: torch.Tensor,
    theta0: float,
    theta_dot0: float,
) -> torch.Tensor:
    """
    Initial condition loss.

    Args:
        theta_pred: predicted angle at t=0 [1, 1]
        theta_dot_pred: predicted angular velocity at t=0 [1, 1]
        theta0: true initial angle
        theta_dot0: true initial angular velocity

    Returns:
        mean squared error
    """
    loss_theta = (theta_pred - theta0) ** 2
    loss_theta_dot = (theta_dot_pred - theta_dot0) ** 2

    return torch.mean(loss_theta + loss_theta_dot)


def data_loss(
    theta_pred: torch.Tensor, theta_data: torch.Tensor, weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Data fitting loss.

    Args:
        theta_pred: predicted angles [N, 1]
        theta_data: observed angles [N, 1]
        weights: optional sample weights [N, 1]

    Returns:
        weighted mean squared error
    """
    residual = (theta_pred - theta_data) ** 2

    if weights is not None:
        residual = residual * weights

    return torch.mean(residual)


def loss_passivity(
    theta: torch.Tensor,
    theta_dot: torch.Tensor,
    theta_ddot: torch.Tensor,
    g: Union[torch.Tensor, float],
    L: Union[torch.Tensor, float],
    m: float = 1.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Passivity constraint loss (hinge loss).
    
    Energy: H = (1/2)L²θ̇² + gL(1-cos(θ))  (with m=1)
    
    Energy derivative: Ḣ = ∂H/∂θ · θ̇ + ∂H/∂θ̇ · θ̈
    
    For passive systems: Ḣ ≤ 0 (energy should not increase)
    
    Loss: mean(relu(Ḣ + eps)²) where eps ≈ 1e-5
    
    Args:
        theta: angle [N, 1]
        theta_dot: angular velocity θ̇ [N, 1]
        theta_ddot: angular acceleration θ̈ [N, 1]
        g: gravitational acceleration
        L: pendulum length
        m: pendulum mass (default: 1.0)
        eps: small tolerance for numerical stability
        
    Returns:
        passivity violation loss
    """
    # Convert to tensors if needed
    if not isinstance(g, torch.Tensor):
        g = torch.tensor(g, dtype=theta.dtype, device=theta.device)
    if not isinstance(L, torch.Tensor):
        L = torch.tensor(L, dtype=theta.dtype, device=theta.device)
    
    # Compute energy H = (1/2)m*L²*θ̇² + m*g*L*(1-cos(θ))
    # With m=1: H = (1/2)L²θ̇² + gL(1-cos(θ))
    kinetic = 0.5 * m * L**2 * theta_dot**2
    potential = m * g * L * (1 - torch.cos(theta))
    
    # Compute energy derivative Ḣ = ∂H/∂θ · θ̇ + ∂H/∂θ̇ · θ̈
    # ∂H/∂θ = m*g*L*sin(θ)
    # ∂H/∂θ̇ = m*L²*θ̇
    dH_dtheta = m * g * L * torch.sin(theta)
    dH_domega = m * L**2 * theta_dot
    
    dotH = dH_dtheta * theta_dot + dH_domega * theta_ddot
    
    # Penalize energy increase: relu(Ḣ + eps)²
    violation = torch.relu(dotH + eps)
    
    return torch.mean(violation ** 2)


def passivity_loss(
    theta: torch.Tensor,
    t: torch.Tensor,
    g: torch.Tensor,
    L: torch.Tensor,
    m: float = 1.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Passivity constraint loss: energy should be non-increasing (for damped systems).

    For undamped systems: dE/dt ≈ 0
    For damped systems: dE/dt ≤ 0

    Energy: E = (1/2)mL²θ_dot² + mgL(1-cos(θ))

    Args:
        theta: angle predictions [N, 1]
        t: time inputs [N, 1] (requires_grad=True)
        g: gravitational acceleration
        L: pendulum length
        m: pendulum mass
        eps: small tolerance

    Returns:
        passivity violation loss
    """
    theta_dot, theta_ddot = compute_derivatives(theta, t)
    return loss_passivity(theta, theta_dot, theta_ddot, g, L, m, eps)


def passivity_loss_with_damping(
    theta: torch.Tensor,
    t: torch.Tensor,
    g: torch.Tensor,
    L: torch.Tensor,
    damping: torch.Tensor,
    m: float = 1.0,
    tolerance: float = 1e-6,
) -> torch.Tensor:
    """
    Passivity constraint with expected damping: dE/dt = -damping * θ_dot²

    Args:
        theta: angle predictions [N, 1]
        t: time inputs [N, 1] (requires_grad=True)
        g: gravitational acceleration
        L: pendulum length
        damping: damping coefficient
        m: pendulum mass
        tolerance: tolerance for constraint

    Returns:
        passivity constraint violation
    """
    theta_dot, _ = compute_derivatives(theta, t)

    # Compute energy
    kinetic = 0.5 * m * L**2 * theta_dot**2
    potential = m * g * L * (1 - torch.cos(theta))
    energy = kinetic + potential

    # Compute energy derivative
    dE_dt = torch.autograd.grad(
        energy,
        t,
        grad_outputs=torch.ones_like(energy),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Expected energy dissipation
    expected_dE_dt = -damping * m * L**2 * theta_dot**2

    # Constraint: dE/dt should match expected dissipation
    residual = dE_dt - expected_dE_dt

    return torch.mean(residual**2)


def total_loss(
    theta: torch.Tensor,
    t: torch.Tensor,
    theta_obs: torch.Tensor,
    t_obs: torch.Tensor,
    g: Union[torch.Tensor, float],
    L: Union[torch.Tensor, float],
    theta0: float,
    omega0: float,
    lambda_data: float = 1.0,
    lambda_phys: float = 1.0,
    lambda_ic: float = 1.0,
    lambda_pass: float = 0.0,
    D: Optional[torch.Tensor] = None,
    m: float = 1.0,
    eps: float = 1e-5,
) -> Dict[str, torch.Tensor]:
    """
    Total loss aggregator.
    
    total = λ_data*loss_data + λ_phys*loss_phys + λ_ic*loss_ic + λ_pass*loss_passivity
    
    Args:
        theta: predicted angles at collocation points [N, 1]
        t: time at collocation points [N, 1] (requires_grad=True)
        theta_obs: observed angles [M, 1]
        t_obs: observation times [M, 1]
        g: gravitational acceleration
        L: pendulum length
        theta0: initial angle
        omega0: initial angular velocity
        lambda_data: weight for data loss
        lambda_phys: weight for physics loss
        lambda_ic: weight for initial condition loss
        lambda_pass: weight for passivity loss
        D: dissipation term [N, 1] (optional)
        m: pendulum mass
        eps: passivity tolerance
        
    Returns:
        dictionary of losses
    """
    # Compute derivatives
    theta_dot, theta_ddot = compute_derivatives(theta, t)
    
    # Data loss (evaluate network at observation points)
    # For simplicity, assume theta already contains predictions at obs points
    # In practice, you'd interpolate or evaluate network at t_obs
    l_data = loss_data(theta[:len(theta_obs)], theta_obs)
    
    # Physics loss
    l_phys = loss_phys(theta_ddot, theta, theta_dot, g, L, D)
    
    # Initial condition loss
    theta_hat0 = theta[0:1]
    omega_hat0 = theta_dot[0:1]
    l_ic = loss_ic(theta_hat0, omega_hat0, theta0, omega0)
    
    # Passivity loss
    l_pass = loss_passivity(theta, theta_dot, theta_ddot, g, L, m, eps)
    
    # Total loss
    total = (
        lambda_data * l_data +
        lambda_phys * l_phys +
        lambda_ic * l_ic +
        lambda_pass * l_pass
    )
    
    return {
        'data': l_data,
        'phys': l_phys,
        'ic': l_ic,
        'passivity': l_pass,
        'total': total,
    }


def combined_loss(
    theta: torch.Tensor,
    t: torch.Tensor,
    theta_data: torch.Tensor,
    g: torch.Tensor,
    L: torch.Tensor,
    damping: Optional[torch.Tensor],
    theta0: float,
    theta_dot0: float,
    weights: Dict[str, float],
    use_passivity: bool = False,
    m: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Combined loss with all components.

    Args:
        theta: angle predictions [N, 1]
        t: time inputs [N, 1] (requires_grad=True)
        theta_data: observed angles [N, 1]
        g: gravitational acceleration
        L: pendulum length
        damping: optional damping coefficient
        theta0: initial angle
        theta_dot0: initial angular velocity
        weights: loss weights {'physics': w1, 'data': w2, 'ic': w3, 'passivity': w4}
        use_passivity: whether to include passivity constraint
        m: pendulum mass

    Returns:
        dictionary of loss components and total loss
    """
    # Physics loss
    loss_physics = physics_loss(theta, t, g, L, damping)

    # Data loss
    loss_data = data_loss(theta, theta_data)

    # Initial condition loss (evaluate at t=0)
    # Need to recompute theta at t=0 with gradient tracking
    t0 = t[0:1].clone().detach().requires_grad_(True)
    # For testing, we'll use finite differences if autograd fails
    try:
        theta_t0 = theta[0:1]
        theta_dot_t0, _ = compute_derivatives(theta_t0, t0)
    except RuntimeError:
        # Fallback: use finite difference for initial velocity
        if len(t) > 1:
            dt = (t[1] - t[0]).item()
            theta_dot_t0 = (theta[1:2] - theta[0:1]) / dt
        else:
            theta_dot_t0 = torch.zeros_like(theta[0:1])
    loss_ic = initial_condition_loss(theta[0:1], theta_dot_t0, theta0, theta_dot0)

    # Total loss
    total = (
        weights.get("physics", 1.0) * loss_physics
        + weights.get("data", 1.0) * loss_data
        + weights.get("ic", 1.0) * loss_ic
    )

    losses = {
        "physics": loss_physics,
        "data": loss_data,
        "ic": loss_ic,
        "total": total,
    }

    # Passivity loss (optional)
    if use_passivity:
        if damping is not None:
            loss_pass = passivity_loss_with_damping(theta, t, g, L, damping, m)
        else:
            loss_pass = passivity_loss(theta, t, g, L, m)

        losses["passivity"] = loss_pass
        losses["total"] = losses["total"] + weights.get("passivity", 1.0) * loss_pass

    return losses


class LossComputer:
    """Utility class for computing and tracking losses."""

    def __init__(
        self,
        weights: Dict[str, float],
        use_passivity: bool = False,
        m: float = 1.0,
    ):
        """
        Initialize loss computer.

        Args:
            weights: loss weights
            use_passivity: whether to use passivity constraint
            m: pendulum mass
        """
        self.weights = weights
        self.use_passivity = use_passivity
        self.m = m

    def compute(
        self,
        theta: torch.Tensor,
        t: torch.Tensor,
        theta_data: torch.Tensor,
        g: torch.Tensor,
        L: torch.Tensor,
        damping: Optional[torch.Tensor],
        theta0: float,
        theta_dot0: float,
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss components."""
        return combined_loss(
            theta,
            t,
            theta_data,
            g,
            L,
            damping,
            theta0,
            theta_dot0,
            self.weights,
            self.use_passivity,
            self.m,
        )

