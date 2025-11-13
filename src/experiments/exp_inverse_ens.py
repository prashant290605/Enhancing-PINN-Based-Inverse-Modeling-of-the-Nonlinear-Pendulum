"""Ensemble inverse PINN experiment for uncertainty quantification."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict
from datetime import datetime

from src.data.generator import simulate_pendulum, add_noise, subsample, make_sparse_measurements
from src.models.pinn_inverse import PINN
from src.models.dissipation_net import DissipationNet
from src.models.ensemble import create_bootstrap_ensemble
from src.analysis.metrics import (
    compute_ensemble_parameter_metrics,
    compute_trajectory_ci_coverage,
    expected_calibration_error_params,
)
from src.viz.style import setup_plot_style


def plot_ensemble_trajectory(
    t: np.ndarray,
    theta_true: np.ndarray,
    theta_mean: np.ndarray,
    theta_std: np.ndarray,
    t_obs: np.ndarray,
    theta_obs: np.ndarray,
    title: str,
    save_path: Path,
):
    """Plot trajectory with ensemble mean ± 2σ band."""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot mean and uncertainty bands
    ax.plot(t, theta_true, 'k-', label='Ground Truth', linewidth=2)
    ax.plot(t, theta_mean, 'r-', label='Ensemble Mean', linewidth=2)
    ax.fill_between(
        t,
        theta_mean - 2 * theta_std,
        theta_mean + 2 * theta_std,
        alpha=0.3,
        color='red',
        label='Mean ± 2σ'
    )
    ax.scatter(t_obs, theta_obs, c='blue', s=30, alpha=0.6, label='Observations', zorder=5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle θ (rad)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_parameter_histograms(
    param_stats: Dict,
    true_params: Dict,
    save_path: Path,
):
    """Plot histograms of parameter estimates."""
    setup_plot_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    params = ['g', 'L', 'damping']
    labels = ['g (m/s²)', 'L (m)', 'c (1/s)']
    
    for ax, param, label in zip(axes, params, labels):
        values = param_stats[param]['values']
        true_val = true_params[param]
        mean_val = param_stats[param]['mean']
        
        ax.hist(values, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(true_val, color='k', linestyle='--', linewidth=2, label='True')
        ax.axvline(mean_val, color='r', linestyle='-', linewidth=2, label='Mean')
        
        ax.set_xlabel(label)
        ax.set_ylabel('Count')
        ax.set_title(f'{label}\nMean: {mean_val:.4f} ± {param_stats[param]["std"]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_reliability_diagram(
    param_stats: Dict,
    true_params: Dict,
    save_path: Path,
):
    """Plot reliability diagram for parameter CIs."""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Compute coverage at different confidence levels
    confidence_levels = np.linspace(0.1, 0.99, 20)
    
    coverages_g = []
    coverages_L = []
    coverages_c = []
    
    for conf in confidence_levels:
        # g
        from src.analysis.metrics import compute_parameter_ci_coverage
        cov_g = compute_parameter_ci_coverage(
            np.array(param_stats['g']['values']),
            true_params['g'],
            conf
        )['coverage']
        coverages_g.append(cov_g)
        
        # L
        cov_L = compute_parameter_ci_coverage(
            np.array(param_stats['L']['values']),
            true_params['L'],
            conf
        )['coverage']
        coverages_L.append(cov_L)
        
        # c
        cov_c = compute_parameter_ci_coverage(
            np.array(param_stats['damping']['values']),
            true_params['damping'],
            conf
        )['coverage']
        coverages_c.append(cov_c)
    
    # Plot
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    ax.plot(confidence_levels, coverages_g, 'o-', label='g', linewidth=2, markersize=6)
    ax.plot(confidence_levels, coverages_L, 's-', label='L', linewidth=2, markersize=6)
    ax.plot(confidence_levels, coverages_c, '^-', label='c', linewidth=2, markersize=6)
    
    ax.set_xlabel('Expected Coverage')
    ax.set_ylabel('Empirical Coverage')
    ax.set_title('Reliability Diagram for Parameter CIs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_ensemble_experiment(
    output_dir: Path,
    n_models: int = 7,
    theta0_deg: float = 30.0,
    c: float = 0.05,
    n_sparse: int = 20,
    sigma: float = 0.01,
    use_bootstrap: bool = True,
    use_velocity_obs: bool = True,
    dissipation: str = "viscous",
    n_epochs: int = 3000,
    seed: int = 1337,
    device: str = "cpu",
):
    """
    Run ensemble inverse PINN experiment.
    
    Args:
        output_dir: output directory
        n_models: number of models in ensemble
        theta0_deg: initial angle in degrees
        c: damping coefficient
        n_sparse: number of sparse observations
        sigma: noise standard deviation
        use_bootstrap: whether to use bootstrap sampling
        dissipation: dissipation type ("viscous" or "nn")
        n_epochs: number of training epochs
        seed: random seed
        device: device to train on
    """
    print("=" * 80)
    print("ENSEMBLE INVERSE PINN EXPERIMENT")
    print("=" * 80)
    
    # Create output directory with run_id
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir) / run_id
    figs_dir = output_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    # True parameters
    g_true = 9.81
    L_true = 1.0
    c_true = c
    m = 1.0
    
    # Generate dataset
    print(f"\n[1/6] Generating dataset...")
    print(f"  Amplitude: {theta0_deg}°, c={c_true}, {n_sparse} sparse points, σ={sigma}")
    
    np.random.seed(seed)
    theta0 = np.radians(theta0_deg)
    omega0 = 0.0
    
    # Dense ground truth
    t_dense = np.linspace(0, 10.0, 1000)
    t_dense, theta_dense, omega_dense = simulate_pendulum(
        theta0, omega0, g_true, L_true, c_true, t_dense, method="ivp"
    )
    
    # Sparse observations with velocity
    t_sparse, theta_sparse_noisy, omega_sparse_noisy = make_sparse_measurements(
        t_dense, theta_dense, omega_dense, n_sparse, noise_std=sigma, irregular=True, seed=seed
    )
    
    print(f"  Generated {len(t_sparse)} sparse observations")
    print(f"  Dense grid: {len(t_dense)} points")
    print(f"  Using velocity observations: {use_velocity_obs}")
    
    # Create ensemble
    print(f"\n[2/6] Creating ensemble of {n_models} models...")
    print(f"  Bootstrap: {use_bootstrap}, Dissipation: {dissipation}")
    
    model_config = {
        'hidden_layers': [32, 32, 32],
        'activation': 'tanh',
        'init_g': 10.0,
        'init_L': 1.1,
        'init_damping': 0.08,
        'learn_g': True,
        'learn_L': True,
        'learn_damping': (dissipation == "viscous"),
        'use_fourier': True,
        'num_frequencies': 6,
    }
    
    dissipation_net = None
    if dissipation == "nn":
        dissipation_net = DissipationNet(
            hidden_layers=[16, 16],
            activation='tanh',
            use_enhanced_features=True,
        )
    
    trainer_config = {
        'lambda_data': 1.0,
        'lambda_phys': 10.0,
        'lambda_ic': 1.0,
        'lambda_pass': 0.0,  # No passivity for ensemble
        'lambda_vel': 1.0,
        'use_velocity_obs': use_velocity_obs,
        'dissipation_net': dissipation_net,
        'learning_rate': 1e-3,
        'scheduler_type': 'cosine',
        'n_epochs': n_epochs,
        'device': device,
    }
    
    ensemble = create_bootstrap_ensemble(
        n_models=n_models,
        t_obs=t_sparse,
        theta_obs=theta_sparse_noisy,
        omega_obs=omega_sparse_noisy,
        t_collocation=t_dense,
        theta0=theta0,
        omega0=omega0,
        model_config=model_config,
        trainer_config=trainer_config,
        seed=seed,
        use_bootstrap=use_bootstrap,
    )
    
    # Train ensemble
    print(f"\n[3/6] Training ensemble ({n_epochs} epochs per model)...")
    histories = ensemble.train_all(n_epochs=n_epochs, verbose=True)
    
    # Get predictions
    print(f"\n[4/6] Computing ensemble predictions...")
    predictions = ensemble.predict(t_dense, return_std=True)
    
    theta_mean = predictions['theta_mean']
    theta_std = predictions['theta_std']
    theta_all = predictions['theta_all']
    
    # Get parameter statistics
    param_stats = ensemble.get_parameter_statistics()
    true_params = {'g': g_true, 'L': L_true, 'damping': c_true}
    
    print(f"\nEnsemble Parameter Estimates:")
    print(f"  g = {param_stats['g']['mean']:.4f} ± {param_stats['g']['std']:.4f} (true: {g_true:.4f})")
    print(f"  L = {param_stats['L']['mean']:.4f} ± {param_stats['L']['std']:.4f} (true: {L_true:.4f})")
    print(f"  c = {param_stats['damping']['mean']:.4f} ± {param_stats['damping']['std']:.4f} (true: {c_true:.4f})")
    
    # Compute metrics
    print(f"\n[5/6] Computing uncertainty metrics...")
    
    # Parameter CI coverage
    param_metrics = compute_ensemble_parameter_metrics(
        param_stats, true_params, confidence_levels=[0.90, 0.95]
    )
    
    # Trajectory CI coverage
    traj_coverage_90 = compute_trajectory_ci_coverage(theta_all, theta_dense, confidence=0.90)
    traj_coverage_95 = compute_trajectory_ci_coverage(theta_all, theta_dense, confidence=0.95)
    
    # ECE for parameters
    ece_g = expected_calibration_error_params(np.array(param_stats['g']['values']), g_true)
    ece_L = expected_calibration_error_params(np.array(param_stats['L']['values']), L_true)
    ece_c = expected_calibration_error_params(np.array(param_stats['damping']['values']), c_true)
    
    print(f"\nUncertainty Metrics:")
    print(f"  Parameter CI Coverage (90%):")
    print(f"    g: {param_metrics['g']['ci_90']['coverage']:.0f}")
    print(f"    L: {param_metrics['L']['ci_90']['coverage']:.0f}")
    print(f"    c: {param_metrics['damping']['ci_90']['coverage']:.0f}")
    print(f"  Parameter CI Coverage (95%):")
    print(f"    g: {param_metrics['g']['ci_95']['coverage']:.0f}")
    print(f"    L: {param_metrics['L']['ci_95']['coverage']:.0f}")
    print(f"    c: {param_metrics['damping']['ci_95']['coverage']:.0f}")
    print(f"  Trajectory Coverage (90%): {traj_coverage_90['coverage']:.4f}")
    print(f"  Trajectory Coverage (95%): {traj_coverage_95['coverage']:.4f}")
    print(f"  Parameter ECE:")
    print(f"    g: {ece_g:.4f}")
    print(f"    L: {ece_L:.4f}")
    print(f"    c: {ece_c:.4f}")
    
    # Generate plots
    print(f"\n[6/6] Generating plots...")
    
    # Trajectory with uncertainty bands
    plot_ensemble_trajectory(
        t_dense, theta_dense, theta_mean, theta_std,
        t_sparse, theta_sparse_noisy,
        f"Ensemble PINN: θ(t) with Uncertainty (N={n_models})",
        figs_dir / "ensemble_trajectory.png"
    )
    
    # Parameter histograms
    plot_parameter_histograms(
        param_stats, true_params,
        figs_dir / "parameter_histograms.png"
    )
    
    # Reliability diagram
    plot_reliability_diagram(
        param_stats, true_params,
        figs_dir / "reliability_diagram.png"
    )
    
    # Save metrics to CSV
    print(f"\nSaving results...")
    
    # Summary metrics
    summary = pd.DataFrame({
        'Parameter': ['g', 'L', 'c'],
        'Mean': [
            param_stats['g']['mean'],
            param_stats['L']['mean'],
            param_stats['damping']['mean']
        ],
        'Std': [
            param_stats['g']['std'],
            param_stats['L']['std'],
            param_stats['damping']['std']
        ],
        'True': [g_true, L_true, c_true],
        'Abs Error': [
            param_metrics['g']['abs_error'],
            param_metrics['L']['abs_error'],
            param_metrics['damping']['abs_error']
        ],
        'Rel Error (%)': [
            param_metrics['g']['rel_error'] * 100,
            param_metrics['L']['rel_error'] * 100,
            param_metrics['damping']['rel_error'] * 100
        ],
        'CI90 Coverage': [
            param_metrics['g']['ci_90']['coverage'],
            param_metrics['L']['ci_90']['coverage'],
            param_metrics['damping']['ci_90']['coverage']
        ],
        'CI95 Coverage': [
            param_metrics['g']['ci_95']['coverage'],
            param_metrics['L']['ci_95']['coverage'],
            param_metrics['damping']['ci_95']['coverage']
        ],
        'ECE': [ece_g, ece_L, ece_c],
    })
    
    summary.to_csv(output_dir / "parameter_metrics.csv", index=False)
    
    # Trajectory coverage metrics
    traj_metrics = pd.DataFrame({
        'Confidence': ['90%', '95%'],
        'Coverage': [
            traj_coverage_90['coverage'],
            traj_coverage_95['coverage']
        ],
        'Avg Width': [
            traj_coverage_90['avg_width'],
            traj_coverage_95['avg_width']
        ],
    })
    
    traj_metrics.to_csv(output_dir / "trajectory_coverage.csv", index=False)
    
    # Save individual parameter values
    param_values = pd.DataFrame({
        'Model': range(n_models),
        'g': param_stats['g']['values'],
        'L': param_stats['L']['values'],
        'c': param_stats['damping']['values'],
    })
    
    param_values.to_csv(output_dir / "parameter_values.csv", index=False)
    
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"✓ Figures saved to: {figs_dir}")
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run ensemble inverse PINN experiment")
    
    # Add config argument
    from src.configs.config_loader import add_config_argument
    add_config_argument(parser)
    
    parser.add_argument(
        "--n-models",
        type=int,
        default=None,
        help="Number of models in ensemble"
    )
    parser.add_argument(
        "--theta0",
        type=float,
        default=None,
        help="Initial angle in degrees"
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=None,
        help="Damping coefficient"
    )
    parser.add_argument(
        "--n-sparse",
        type=int,
        default=None,
        help="Number of sparse observations"
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=None,
        help="Noise standard deviation"
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Disable bootstrap sampling (use only different seeds)"
    )
    parser.add_argument(
        "--dissipation",
        type=str,
        default=None,
        choices=["viscous", "nn"],
        help="Dissipation type"
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=None,
        help="Number of training epochs per model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cpu or cuda)"
    )
    parser.add_argument(
        "--use-velocity-obs",
        type=lambda x: x.lower() == 'true',
        default=None,
        help="Whether to use velocity observations (true or false)"
    )
    
    args = parser.parse_args()
    
    # Load config with overrides
    from src.configs.config_loader import load_config_with_overrides
    config = load_config_with_overrides(args)
    
    # Extract values from config (with CLI overrides already applied)
    n_models = args.n_models if args.n_models is not None else config['ensemble']['n_models']
    theta0_deg = args.theta0 if args.theta0 is not None else config['initial_conditions']['theta0_deg']
    damping = args.damping if args.damping is not None else config['physics']['damping']
    n_sparse = args.n_sparse if args.n_sparse is not None else config['time']['n_points_sparse']
    noise = args.noise if args.noise is not None else config['data']['noise_std']
    dissipation = args.dissipation if args.dissipation is not None else config['dissipation']['type']
    n_epochs = args.n_epochs if args.n_epochs is not None else config['experiments']['inverse_ensemble']['n_epochs']
    output_dir = args.output_dir if args.output_dir is not None else config['paths']['output_dir'] + '/ensemble'
    seed = args.seed if args.seed is not None else config['seed']
    device = args.device if args.device is not None else config['device']
    use_bootstrap = (not args.no_bootstrap) and config['ensemble']['use_bootstrap']
    use_velocity_obs = args.use_velocity_obs if args.use_velocity_obs is not None else config.get('loss_weights', {}).get('use_velocity_obs', True)
    
    run_ensemble_experiment(
        output_dir=Path(output_dir),
        n_models=n_models,
        theta0_deg=theta0_deg,
        c=damping,
        n_sparse=n_sparse,
        sigma=noise,
        use_bootstrap=use_bootstrap,
        use_velocity_obs=use_velocity_obs,
        dissipation=dissipation,
        n_epochs=n_epochs,
        seed=seed,
        device=device,
    )


if __name__ == "__main__":
    main()
