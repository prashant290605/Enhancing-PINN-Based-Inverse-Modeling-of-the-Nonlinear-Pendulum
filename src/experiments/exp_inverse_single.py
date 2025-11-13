"""Single inverse PINN experiment with/without passivity constraint."""

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict, Tuple

from src.data.generator import simulate_pendulum, add_noise, subsample, make_sparse_measurements
from src.models.pinn_inverse import PINN
from src.models.dissipation_net import DissipationNet
from src.models.train_inverse import create_trainer
from src.analysis.metrics import rmse, mse, mae, energy_drift, parameter_error
from src.viz.style import setup_plot_style


def generate_dataset(
    theta0_deg: float = 30.0,
    c: float = 0.05,
    n_sparse: int = 20,
    sigma: float = 0.01,
    g: float = 9.81,
    L: float = 1.0,
    t_max: float = 10.0,
    seed: int = 1337,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate one dataset with specified parameters.
    
    Args:
        theta0_deg: initial angle in degrees
        c: damping coefficient
        n_sparse: number of sparse observation points
        sigma: noise standard deviation
        g: gravitational acceleration
        L: pendulum length
        t_max: maximum time
        seed: random seed
        
    Returns:
        t_dense, theta_dense, omega_dense, t_sparse, theta_sparse_noisy, omega_sparse_noisy
    """
    np.random.seed(seed)
    
    # Convert to radians
    theta0 = np.radians(theta0_deg)
    omega0 = 0.0
    
    # Generate dense ground truth
    t_dense = np.linspace(0, t_max, 1000)
    t_dense, theta_dense, omega_dense = simulate_pendulum(
        theta0, omega0, g, L, c, t_dense, method="ivp"
    )
    
    # Make sparse measurements with velocity observations
    t_sparse, theta_sparse_noisy, omega_sparse_noisy = make_sparse_measurements(
        t_dense, theta_dense, omega_dense, n_sparse, noise_std=sigma, irregular=True, seed=seed
    )
    
    return t_dense, theta_dense, omega_dense, t_sparse, theta_sparse_noisy, omega_sparse_noisy


def train_inverse_pinn(
    t_obs: np.ndarray,
    theta_obs: np.ndarray,
    t_collocation: np.ndarray,
    theta0: float,
    omega0: float,
    g_true: float,
    L_true: float,
    c_true: float,
    use_passivity: bool,
    omega_obs: np.ndarray = None,
    use_velocity_obs: bool = True,
    dissipation_type: str = "viscous",
    n_epochs: int = 5000,
    device: str = "cpu",
) -> Tuple[PINN, Dict, Dict]:
    """
    Train inverse PINN with or without passivity.
    
    Args:
        t_obs: observation times
        theta_obs: observed angles
        t_collocation: collocation points for physics
        theta0: initial angle
        omega0: initial angular velocity
        g_true: true g (for reference)
        L_true: true L (for reference)
        c_true: true damping (for reference)
        use_passivity: whether to use passivity constraint
        omega_obs: observed angular velocities (optional)
        use_velocity_obs: whether to use velocity observations
        dissipation_type: "viscous" or "nn"
        n_epochs: number of training epochs
        device: device to train on
        
    Returns:
        model, history, params
    """
    # Convert to torch
    t_obs_torch = torch.tensor(t_obs, dtype=torch.float32).unsqueeze(-1)
    theta_obs_torch = torch.tensor(theta_obs, dtype=torch.float32).unsqueeze(-1)
    omega_obs_torch = torch.tensor(omega_obs, dtype=torch.float32).unsqueeze(-1) if omega_obs is not None else None
    t_colloc_torch = torch.tensor(t_collocation, dtype=torch.float32).unsqueeze(-1)
    
    # Create model
    model = PINN(
        hidden_layers=[32, 32, 32],
        activation="tanh",
        init_g=10.0,  # slightly off
        init_L=1.1,
        init_damping=0.08,
        learn_g=True,
        learn_L=True,
        learn_damping=(dissipation_type == "viscous"),
        use_fourier=True,
        num_frequencies=6,
    )
    
    # Create dissipation net if needed
    dissipation_net = None
    if dissipation_type == "nn":
        dissipation_net = DissipationNet(
            hidden_layers=[16, 16],
            activation="tanh",
            use_enhanced_features=True,
        )
    
    # Loss weights
    lambda_pass = 1.0 if use_passivity else 0.0
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        t_obs=t_obs_torch,
        theta_obs=theta_obs_torch,
        omega_obs=omega_obs_torch,
        t_collocation=t_colloc_torch,
        theta0=theta0,
        omega0=omega0,
        lambda_data=1.0,
        lambda_phys=10.0,
        lambda_ic=1.0,
        lambda_pass=lambda_pass,
        lambda_vel=1.0,
        use_velocity_obs=use_velocity_obs,
        dissipation_net=dissipation_net,
        learning_rate=1e-3,
        scheduler_type="cosine",
        n_epochs=n_epochs,
        device=device,
    )
    
    # Train
    history = trainer.train(n_epochs=n_epochs, verbose=True, save_best=False)
    
    # Get parameters
    params = model.get_parameters()
    
    return model, history, params


def plot_theta_vs_truth(
    t_dense: np.ndarray,
    theta_dense: np.ndarray,
    t_sparse: np.ndarray,
    theta_sparse: np.ndarray,
    model: PINN,
    title: str,
    save_path: Path,
):
    """Plot theta(t) vs ground truth."""
    setup_plot_style()
    
    # Predict on dense grid
    model.eval()
    with torch.no_grad():
        t_torch = torch.tensor(t_dense, dtype=torch.float32).unsqueeze(-1)
        theta_pred = model(t_torch).cpu().numpy().flatten()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(t_dense, theta_dense, 'k-', label='Ground Truth', linewidth=2)
    ax.plot(t_dense, theta_pred, 'r--', label='PINN Prediction', linewidth=2)
    ax.scatter(t_sparse, theta_sparse, c='blue', s=30, alpha=0.6, label='Observations', zorder=5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle θ (rad)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_parameter_traces(
    history: Dict,
    g_true: float,
    L_true: float,
    c_true: float,
    title: str,
    save_path: Path,
):
    """Plot parameter traces vs steps."""
    setup_plot_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    steps = np.arange(len(history['g']))
    
    # g
    axes[0].plot(steps, history['g'], 'b-', linewidth=2)
    axes[0].axhline(g_true, color='k', linestyle='--', label='True value')
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('g (m/s²)')
    axes[0].set_title('Gravitational Acceleration')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # L
    axes[1].plot(steps, history['L'], 'g-', linewidth=2)
    axes[1].axhline(L_true, color='k', linestyle='--', label='True value')
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('L (m)')
    axes[1].set_title('Pendulum Length')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # damping
    axes[2].plot(steps, history['damping'], 'r-', linewidth=2)
    axes[2].axhline(c_true, color='k', linestyle='--', label='True value')
    axes[2].set_xlabel('Training Step')
    axes[2].set_ylabel('c (1/s)')
    axes[2].set_title('Damping Coefficient')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_energy_drift(
    t_dense: np.ndarray,
    theta_dense: np.ndarray,
    omega_dense: np.ndarray,
    model: PINN,
    g: float,
    L: float,
    m: float,
    title: str,
    save_path: Path,
):
    """Plot energy drift."""
    setup_plot_style()
    
    # Compute true energy
    E_true = 0.5 * m * L**2 * omega_dense**2 + m * g * L * (1 - np.cos(theta_dense))
    
    # Predict and compute predicted energy
    model.eval()
    
    # Need gradients for computing derivatives
    t_torch = torch.tensor(t_dense, dtype=torch.float32).unsqueeze(-1).requires_grad_(True)
    theta_pred_torch = model(t_torch)
    
    # Compute derivatives
    from src.models.losses import compute_derivatives
    theta_dot_torch, _ = compute_derivatives(theta_pred_torch, t_torch)
    
    theta_pred = theta_pred_torch.detach().cpu().numpy().flatten()
    omega_pred = theta_dot_torch.detach().cpu().numpy().flatten()
    
    E_pred = 0.5 * m * L**2 * omega_pred**2 + m * g * L * (1 - np.cos(theta_pred))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(t_dense, E_true, 'k-', label='True Energy', linewidth=2)
    ax.plot(t_dense, E_pred, 'r--', label='PINN Energy', linewidth=2)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy (J)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_experiment(
    output_dir: Path,
    dissipation: str = "viscous",
    n_epochs: int = 5000,
    device: str = "cpu",
    use_velocity_obs: bool = True,
):
    """
    Run the full experiment pipeline.
    
    Args:
        output_dir: output directory
        dissipation: "viscous" or "nn"
        n_epochs: number of training epochs
        device: device to train on
        use_velocity_obs: whether to use velocity observations
    """
    print("=" * 80)
    print("INVERSE PINN SINGLE EXPERIMENT")
    print("=" * 80)
    
    # Create output directories
    output_dir = Path(output_dir)
    figs_dir = output_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    # True parameters
    g_true = 9.81
    L_true = 1.0
    c_true = 0.05
    m = 1.0
    
    # Generate dataset
    print("\n[1/5] Generating dataset (30°, c=0.05, 20 sparse points, σ=0.01)...")
    t_dense, theta_dense, omega_dense, t_sparse, theta_sparse_noisy, omega_sparse_noisy = generate_dataset(
        theta0_deg=30.0,
        c=c_true,
        n_sparse=20,
        sigma=0.01,
        g=g_true,
        L=L_true,
        t_max=10.0,
        seed=1337,
    )
    
    theta0 = theta_dense[0]
    omega0 = omega_dense[0]
    
    print(f"  Generated {len(t_sparse)} sparse observations")
    print(f"  Dense grid: {len(t_dense)} points")
    print(f"  Using velocity observations: {use_velocity_obs}")
    
    # Train without passivity
    print("\n[2/5] Training inverse PINN WITHOUT passivity...")
    model_no_pass, history_no_pass, params_no_pass = train_inverse_pinn(
        t_obs=t_sparse,
        theta_obs=theta_sparse_noisy,
        omega_obs=omega_sparse_noisy,
        t_collocation=t_dense,
        theta0=theta0,
        omega0=omega0,
        g_true=g_true,
        L_true=L_true,
        c_true=c_true,
        use_passivity=False,
        use_velocity_obs=use_velocity_obs,
        dissipation_type=dissipation,
        n_epochs=n_epochs,
        device=device,
    )
    
    print(f"\nLearned (No Passivity):")
    print(f"  g = {params_no_pass['g']:.4f} (true: {g_true:.4f})")
    print(f"  L = {params_no_pass['L']:.4f} (true: {L_true:.4f})")
    print(f"  c = {params_no_pass['damping']:.4f} (true: {c_true:.4f})")
    
    # Train with passivity
    print("\n[3/5] Training inverse PINN WITH passivity...")
    model_with_pass, history_with_pass, params_with_pass = train_inverse_pinn(
        t_obs=t_sparse,
        theta_obs=theta_sparse_noisy,
        omega_obs=omega_sparse_noisy,
        t_collocation=t_dense,
        theta0=theta0,
        omega0=omega0,
        g_true=g_true,
        L_true=L_true,
        c_true=c_true,
        use_passivity=True,
        use_velocity_obs=use_velocity_obs,
        dissipation_type=dissipation,
        n_epochs=n_epochs,
        device=device,
    )
    
    print(f"\nLearned (With Passivity):")
    print(f"  g = {params_with_pass['g']:.4f} (true: {g_true:.4f})")
    print(f"  L = {params_with_pass['L']:.4f} (true: {L_true:.4f})")
    print(f"  c = {params_with_pass['damping']:.4f} (true: {c_true:.4f})")
    
    # Generate plots
    print("\n[4/5] Generating plots...")
    
    # Theta vs truth
    plot_theta_vs_truth(
        t_dense, theta_dense, t_sparse, theta_sparse_noisy,
        model_no_pass,
        "Inverse PINN: θ(t) vs Ground Truth (No Passivity)",
        figs_dir / "theta_no_passivity.png"
    )
    
    plot_theta_vs_truth(
        t_dense, theta_dense, t_sparse, theta_sparse_noisy,
        model_with_pass,
        "Inverse PINN: θ(t) vs Ground Truth (With Passivity)",
        figs_dir / "theta_with_passivity.png"
    )
    
    # Parameter traces
    plot_parameter_traces(
        history_no_pass, g_true, L_true, c_true,
        "Parameter Convergence (No Passivity)",
        figs_dir / "params_no_passivity.png"
    )
    
    plot_parameter_traces(
        history_with_pass, g_true, L_true, c_true,
        "Parameter Convergence (With Passivity)",
        figs_dir / "params_with_passivity.png"
    )
    
    # Energy drift
    plot_energy_drift(
        t_dense, theta_dense, omega_dense,
        model_no_pass, g_true, L_true, m,
        "Energy Evolution (No Passivity)",
        figs_dir / "energy_no_passivity.png"
    )
    
    plot_energy_drift(
        t_dense, theta_dense, omega_dense,
        model_with_pass, g_true, L_true, m,
        "Energy Evolution (With Passivity)",
        figs_dir / "energy_with_passivity.png"
    )
    
    # Compute metrics
    print("\n[5/5] Computing metrics and saving comparison table...")
    
    # Predict on dense grid
    model_no_pass.eval()
    model_with_pass.eval()
    
    from src.models.losses import compute_derivatives
    
    t_torch = torch.tensor(t_dense, dtype=torch.float32).unsqueeze(-1).requires_grad_(True)
    
    # No passivity
    theta_pred_no_pass_torch = model_no_pass(t_torch)
    theta_dot_no_pass, _ = compute_derivatives(theta_pred_no_pass_torch, t_torch)
    theta_pred_no_pass = theta_pred_no_pass_torch.detach().cpu().numpy().flatten()
    omega_pred_no_pass = theta_dot_no_pass.detach().cpu().numpy().flatten()
    
    # With passivity
    theta_pred_with_pass_torch = model_with_pass(t_torch)
    theta_dot_with_pass, _ = compute_derivatives(theta_pred_with_pass_torch, t_torch)
    theta_pred_with_pass = theta_pred_with_pass_torch.detach().cpu().numpy().flatten()
    omega_pred_with_pass = theta_dot_with_pass.detach().cpu().numpy().flatten()
    
    # Compute metrics
    true_params = {'g': g_true, 'L': L_true, 'damping': c_true}
    
    # Parameter RMSE
    errors_no_pass = parameter_error(params_no_pass, true_params)
    param_rmse_no_pass = np.sqrt(
        errors_no_pass['g_abs_error']**2 +
        errors_no_pass['L_abs_error']**2 +
        errors_no_pass['damping_abs_error']**2
    ) / np.sqrt(3)
    
    errors_with_pass = parameter_error(params_with_pass, true_params)
    param_rmse_with_pass = np.sqrt(
        errors_with_pass['g_abs_error']**2 +
        errors_with_pass['L_abs_error']**2 +
        errors_with_pass['damping_abs_error']**2
    ) / np.sqrt(3)
    
    # Trajectory MSE
    traj_mse_no_pass = mse(theta_pred_no_pass, theta_dense)
    traj_mse_with_pass = mse(theta_pred_with_pass, theta_dense)
    
    # Energy drift (compute as max absolute change from initial energy)
    E_true = 0.5 * m * L_true**2 * omega_dense**2 + m * g_true * L_true * (1 - np.cos(theta_dense))
    
    E_pred_no_pass = 0.5 * m * L_true**2 * omega_pred_no_pass**2 + m * g_true * L_true * (1 - np.cos(theta_pred_no_pass))
    drift_no_pass = np.max(np.abs(E_pred_no_pass - E_pred_no_pass[0]))
    
    E_pred_with_pass = 0.5 * m * L_true**2 * omega_pred_with_pass**2 + m * g_true * L_true * (1 - np.cos(theta_pred_with_pass))
    drift_with_pass = np.max(np.abs(E_pred_with_pass - E_pred_with_pass[0]))
    
    # Create comparison table
    comparison = pd.DataFrame({
        'Method': ['No Passivity', 'With Passivity'],
        'Param RMSE': [param_rmse_no_pass, param_rmse_with_pass],
        'Traj MSE': [traj_mse_no_pass, traj_mse_with_pass],
        'Energy Drift': [drift_no_pass, drift_with_pass],
        'g': [params_no_pass['g'], params_with_pass['g']],
        'L': [params_no_pass['L'], params_with_pass['L']],
        'c': [params_no_pass['damping'], params_with_pass['damping']],
    })
    
    # Save to CSV
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    
    print("\nComparison Table:")
    print(comparison.to_string(index=False))
    
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"✓ Figures saved to: {figs_dir}")
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run single inverse PINN experiment")
    parser.add_argument(
        "--dissipation",
        type=str,
        default="viscous",
        choices=["viscous", "nn"],
        help="Dissipation type: viscous or nn"
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=5000,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/inverse_single",
        help="Output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu or cuda)"
    )
    parser.add_argument(
        "--use-velocity-obs",
        type=lambda x: x.lower() == 'true',
        default=True,
        help="Whether to use velocity observations (true or false)"
    )
    
    args = parser.parse_args()
    
    run_experiment(
        output_dir=Path(args.output_dir),
        dissipation=args.dissipation,
        n_epochs=args.n_epochs,
        device=args.device,
        use_velocity_obs=args.use_velocity_obs,
    )


if __name__ == "__main__":
    main()
