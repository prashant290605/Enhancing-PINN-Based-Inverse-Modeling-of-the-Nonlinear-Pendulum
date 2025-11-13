"""Run experiment grid and aggregate results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from typing import Dict, List
from datetime import datetime

from src.experiments.grids import create_robustness_grid
from src.data.generator import simulate_pendulum, add_noise, subsample
from src.models.pinn_inverse import PINN
from src.models.ensemble import create_bootstrap_ensemble
from src.models.train_inverse import create_trainer
from src.analysis.metrics import rmse, mse, parameter_error
from src.viz.style import setup_plot_style


def run_single_inverse_experiment(
    config: Dict,
    output_dir: Path,
    n_epochs: int = 1000,
    device: str = "cpu",
) -> Dict:
    """
    Run a single inverse PINN experiment.
    
    Args:
        config: experiment configuration
        output_dir: output directory
        n_epochs: number of training epochs
        device: device to train on
        
    Returns:
        dictionary with results
    """
    # Extract parameters
    theta0_deg = config['theta0_deg']
    theta0 = config['theta0']
    damping = config['damping']
    noise = config['noise']
    n_sparse = config['n_sparse']
    use_passivity = config['use_passivity']
    
    # True parameters
    g_true = 9.81
    L_true = 1.0
    
    # Generate data
    np.random.seed(1337)
    omega0 = 0.0
    t_dense = np.linspace(0, 10.0, 1000)
    t_dense, theta_dense, omega_dense = simulate_pendulum(
        theta0, omega0, g_true, L_true, damping, t_dense, method="ivp"
    )
    
    # Sparse observations
    t_sparse, theta_sparse = subsample(t_dense, theta_dense, n_sparse, irregular=True, seed=1337)
    theta_sparse_noisy = add_noise(theta_sparse, noise, seed=1337)
    
    # Convert to torch
    import torch
    t_obs_torch = torch.tensor(t_sparse, dtype=torch.float32).unsqueeze(-1)
    theta_obs_torch = torch.tensor(theta_sparse_noisy, dtype=torch.float32).unsqueeze(-1)
    t_colloc_torch = torch.tensor(t_dense, dtype=torch.float32).unsqueeze(-1)
    
    # Create model
    model = PINN(
        hidden_layers=[32, 32, 32],
        activation='tanh',
        init_g=10.0,
        init_L=1.1,
        init_damping=0.08,
        learn_g=True,
        learn_L=True,
        learn_damping=True,
        use_fourier=True,
        num_frequencies=6,
    )
    
    # Create trainer
    lambda_pass = 1.0 if use_passivity else 0.0
    trainer = create_trainer(
        model=model,
        t_obs=t_obs_torch,
        theta_obs=theta_obs_torch,
        t_collocation=t_colloc_torch,
        theta0=theta0,
        omega0=omega0,
        lambda_data=1.0,
        lambda_phys=10.0,
        lambda_ic=1.0,
        lambda_pass=lambda_pass,
        dissipation_net=None,
        learning_rate=1e-3,
        scheduler_type='cosine',
        n_epochs=n_epochs,
        device=device,
    )
    
    # Train
    history = trainer.train(n_epochs=n_epochs, verbose=False, save_best=False)
    
    # Get parameters
    params = model.get_parameters()
    
    # Predict on dense grid
    model.eval()
    t_torch = torch.tensor(t_dense, dtype=torch.float32).unsqueeze(-1).requires_grad_(True)
    theta_pred_torch = model(t_torch)
    
    from src.models.losses import compute_derivatives
    theta_dot_torch, _ = compute_derivatives(theta_pred_torch, t_torch)
    
    theta_pred = theta_pred_torch.detach().cpu().numpy().flatten()
    omega_pred = theta_dot_torch.detach().cpu().numpy().flatten()
    
    # Compute metrics
    true_params = {'g': g_true, 'L': L_true, 'damping': damping}
    param_errors = parameter_error(params, true_params)
    
    traj_rmse = rmse(theta_pred, theta_dense)
    traj_mse = mse(theta_pred, theta_dense)
    
    # Energy drift
    m = 1.0
    E_pred = 0.5 * m * L_true**2 * omega_pred**2 + m * g_true * L_true * (1 - np.cos(theta_pred))
    energy_drift = np.max(np.abs(E_pred - E_pred[0]))
    
    # Results
    results = {
        'name': config['name'],
        'theta0_deg': theta0_deg,
        'damping': damping,
        'noise': noise,
        'n_sparse': n_sparse,
        'use_passivity': use_passivity,
        'experiment_type': 'inverse_single',
        'g_pred': params['g'],
        'L_pred': params['L'],
        'c_pred': params['damping'],
        'g_error': param_errors['g_abs_error'],
        'L_error': param_errors['L_abs_error'],
        'c_error': param_errors['damping_abs_error'],
        'g_rel_error': param_errors['g_rel_error'],
        'L_rel_error': param_errors['L_rel_error'],
        'c_rel_error': param_errors['damping_rel_error'],
        'traj_rmse': traj_rmse,
        'traj_mse': traj_mse,
        'energy_drift': energy_drift,
        'final_loss': history['total'][-1] if 'total' in history else np.nan,
    }
    
    return results


def run_ensemble_experiment(
    config: Dict,
    output_dir: Path,
    n_epochs: int = 1000,
    device: str = "cpu",
) -> Dict:
    """
    Run an ensemble experiment.
    
    Args:
        config: experiment configuration
        output_dir: output directory
        n_epochs: number of training epochs
        device: device to train on
        
    Returns:
        dictionary with results
    """
    # Extract parameters
    theta0_deg = config['theta0_deg']
    theta0 = config['theta0']
    damping = config['damping']
    noise = config['noise']
    n_sparse = config['n_sparse']
    n_models = config.get('n_models', 5)
    
    # True parameters
    g_true = 9.81
    L_true = 1.0
    
    # Generate data
    np.random.seed(1337)
    omega0 = 0.0
    t_dense = np.linspace(0, 10.0, 1000)
    t_dense, theta_dense, omega_dense = simulate_pendulum(
        theta0, omega0, g_true, L_true, damping, t_dense, method="ivp"
    )
    
    # Sparse observations
    t_sparse, theta_sparse = subsample(t_dense, theta_dense, n_sparse, irregular=True, seed=1337)
    theta_sparse_noisy = add_noise(theta_sparse, noise, seed=1337)
    
    # Create ensemble
    model_config = {
        'hidden_layers': [32, 32, 32],
        'activation': 'tanh',
        'init_g': 10.0,
        'init_L': 1.1,
        'init_damping': 0.08,
        'learn_g': True,
        'learn_L': True,
        'learn_damping': True,
        'use_fourier': True,
        'num_frequencies': 6,
    }
    
    trainer_config = {
        'lambda_data': 1.0,
        'lambda_phys': 10.0,
        'lambda_ic': 1.0,
        'lambda_pass': 1.0,  # Ensemble with passivity
        'dissipation_net': None,
        'learning_rate': 1e-3,
        'scheduler_type': 'cosine',
        'n_epochs': n_epochs,
        'device': device,
    }
    
    ensemble = create_bootstrap_ensemble(
        n_models=n_models,
        t_obs=t_sparse,
        theta_obs=theta_sparse_noisy,
        t_collocation=t_dense,
        theta0=theta0,
        omega0=omega0,
        model_config=model_config,
        trainer_config=trainer_config,
        seed=1337,
        use_bootstrap=True,
    )
    
    # Train
    ensemble.train_all(n_epochs=n_epochs, verbose=False)
    
    # Get predictions
    predictions = ensemble.predict(t_dense, return_std=True)
    theta_mean = predictions['theta_mean']
    theta_std = predictions['theta_std']
    theta_all = predictions['theta_all']
    
    # Get parameter statistics
    param_stats = ensemble.get_parameter_statistics()
    
    # Compute metrics
    true_params = {'g': g_true, 'L': L_true, 'damping': damping}
    
    traj_rmse = rmse(theta_mean, theta_dense)
    traj_mse = mse(theta_mean, theta_dense)
    
    # Trajectory coverage
    from src.analysis.metrics import compute_trajectory_ci_coverage
    traj_coverage_95 = compute_trajectory_ci_coverage(theta_all, theta_dense, confidence=0.95)
    
    # Parameter CI coverage
    from src.analysis.metrics import compute_parameter_ci_coverage
    g_ci = compute_parameter_ci_coverage(np.array(param_stats['g']['values']), g_true, 0.95)
    L_ci = compute_parameter_ci_coverage(np.array(param_stats['L']['values']), L_true, 0.95)
    c_ci = compute_parameter_ci_coverage(np.array(param_stats['damping']['values']), damping, 0.95)
    
    # Results
    results = {
        'name': config['name'],
        'theta0_deg': theta0_deg,
        'damping': damping,
        'noise': noise,
        'n_sparse': n_sparse,
        'use_passivity': True,
        'experiment_type': 'ensemble',
        'n_models': n_models,
        'g_pred': param_stats['g']['mean'],
        'L_pred': param_stats['L']['mean'],
        'c_pred': param_stats['damping']['mean'],
        'g_std': param_stats['g']['std'],
        'L_std': param_stats['L']['std'],
        'c_std': param_stats['damping']['std'],
        'g_error': abs(param_stats['g']['mean'] - g_true),
        'L_error': abs(param_stats['L']['mean'] - L_true),
        'c_error': abs(param_stats['damping']['mean'] - damping),
        'g_rel_error': abs(param_stats['g']['mean'] - g_true) / g_true,
        'L_rel_error': abs(param_stats['L']['mean'] - L_true) / L_true,
        'c_rel_error': abs(param_stats['damping']['mean'] - damping) / damping if damping > 0 else 0,
        'traj_rmse': traj_rmse,
        'traj_mse': traj_mse,
        'traj_coverage_95': traj_coverage_95['coverage'],
        'g_ci_coverage_95': g_ci['coverage'],
        'L_ci_coverage_95': L_ci['coverage'],
        'c_ci_coverage_95': c_ci['coverage'],
    }
    
    return results


def run_grid_experiments(
    grid: List[Dict],
    output_dir: Path,
    n_epochs: int = 1000,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Run all experiments in grid.
    
    Args:
        grid: list of experiment configurations
        output_dir: output directory
        n_epochs: number of training epochs
        device: device to train on
        
    Returns:
        DataFrame with all results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    print(f"Running {len(grid)} experiments...")
    print("=" * 80)
    
    for i, config in enumerate(grid):
        print(f"\n[{i+1}/{len(grid)}] Running: {config['name']}")
        
        try:
            if config['experiment_type'] == 'inverse_single':
                result = run_single_inverse_experiment(config, output_dir, n_epochs, device)
            elif config['experiment_type'] == 'ensemble':
                result = run_ensemble_experiment(config, output_dir, n_epochs, device)
            else:
                print(f"  Unknown experiment type: {config['experiment_type']}")
                continue
            
            results.append(result)
            print(f"  ✓ Complete: RMSE={result['traj_rmse']:.4f}, g_error={result['g_error']:.4f}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    df.to_csv(output_dir / "summary.csv", index=False)
    print(f"\n✓ Results saved to: {output_dir / 'summary.csv'}")
    
    return df


def plot_summary_results(df: pd.DataFrame, output_dir: Path):
    """
    Generate summary plots.
    
    Args:
        df: DataFrame with results
        output_dir: output directory
    """
    setup_plot_style()
    output_dir = Path(output_dir)
    figs_dir = output_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter for single experiments (not ensemble)
    df_single = df[df['experiment_type'] == 'inverse_single'].copy()
    
    if len(df_single) == 0:
        print("No single experiments to plot")
        return
    
    # Bar plot: RMSE comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group by configuration
    df_single['config'] = df_single.apply(
        lambda row: f"θ={int(row['theta0_deg'])}°, c={row['damping']:.2f}, σ={row['noise']:.2f}, n={int(row['n_sparse'])}",
        axis=1
    )
    
    configs = df_single['config'].unique()
    x = np.arange(len(configs))
    width = 0.35
    
    rmse_no_pass = []
    rmse_with_pass = []
    
    for config in configs:
        df_config = df_single[df_single['config'] == config]
        no_pass = df_config[df_config['use_passivity'] == False]['traj_rmse'].values
        with_pass = df_config[df_config['use_passivity'] == True]['traj_rmse'].values
        
        rmse_no_pass.append(no_pass[0] if len(no_pass) > 0 else 0)
        rmse_with_pass.append(with_pass[0] if len(with_pass) > 0 else 0)
    
    ax.bar(x - width/2, rmse_no_pass, width, label='No Passivity', color='steelblue')
    ax.bar(x + width/2, rmse_with_pass, width, label='With Passivity', color='coral')
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Trajectory RMSE')
    ax.set_title('Trajectory RMSE: With vs Without Passivity')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(figs_dir / "rmse_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Bar plot: Energy drift comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    drift_no_pass = []
    drift_with_pass = []
    
    for config in configs:
        df_config = df_single[df_single['config'] == config]
        no_pass = df_config[df_config['use_passivity'] == False]['energy_drift'].values
        with_pass = df_config[df_config['use_passivity'] == True]['energy_drift'].values
        
        drift_no_pass.append(no_pass[0] if len(no_pass) > 0 else 0)
        drift_with_pass.append(with_pass[0] if len(with_pass) > 0 else 0)
    
    ax.bar(x - width/2, drift_no_pass, width, label='No Passivity', color='steelblue')
    ax.bar(x + width/2, drift_with_pass, width, label='With Passivity', color='coral')
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Energy Drift')
    ax.set_title('Energy Drift: With vs Without Passivity')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(figs_dir / "energy_drift_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Coverage plot (if ensemble results exist)
    df_ensemble = df[df['experiment_type'] == 'ensemble'].copy()
    
    if len(df_ensemble) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Trajectory coverage
        df_ensemble['config'] = df_ensemble.apply(
            lambda row: f"θ={int(row['theta0_deg'])}°\nc={row['damping']:.2f}\nσ={row['noise']:.2f}",
            axis=1
        )
        
        configs_ens = df_ensemble['config'].unique()
        x_ens = np.arange(len(configs_ens))
        
        traj_cov = [df_ensemble[df_ensemble['config'] == c]['traj_coverage_95'].values[0] for c in configs_ens]
        
        axes[0].bar(x_ens, traj_cov, color='mediumseagreen')
        axes[0].axhline(0.95, color='k', linestyle='--', label='Expected (95%)')
        axes[0].set_xlabel('Configuration')
        axes[0].set_ylabel('Empirical Coverage')
        axes[0].set_title('Trajectory Coverage (95% CI)')
        axes[0].set_xticks(x_ens)
        axes[0].set_xticklabels(configs_ens, rotation=0, ha='center', fontsize=8)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_ylim([0, 1])
        
        # Parameter coverage
        g_cov = [df_ensemble[df_ensemble['config'] == c]['g_ci_coverage_95'].values[0] for c in configs_ens]
        L_cov = [df_ensemble[df_ensemble['config'] == c]['L_ci_coverage_95'].values[0] for c in configs_ens]
        c_cov = [df_ensemble[df_ensemble['config'] == c]['c_ci_coverage_95'].values[0] for c in configs_ens]
        
        width_cov = 0.25
        axes[1].bar(x_ens - width_cov, g_cov, width_cov, label='g', color='steelblue')
        axes[1].bar(x_ens, L_cov, width_cov, label='L', color='coral')
        axes[1].bar(x_ens + width_cov, c_cov, width_cov, label='c', color='mediumseagreen')
        axes[1].axhline(0.95, color='k', linestyle='--', label='Expected (95%)')
        axes[1].set_xlabel('Configuration')
        axes[1].set_ylabel('Empirical Coverage')
        axes[1].set_title('Parameter Coverage (95% CI)')
        axes[1].set_xticks(x_ens)
        axes[1].set_xticklabels(configs_ens, rotation=0, ha='center', fontsize=8)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(figs_dir / "coverage_plots.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Plots saved to: {figs_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run experiment grid")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full grid (otherwise run small subset)"
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=1000,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/summaries",
        help="Output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu or cuda)"
    )
    
    args = parser.parse_args()
    
    # Create grid
    grid = create_robustness_grid(full=args.full)
    
    print(f"Created grid with {len(grid)} experiments")
    print(f"Full grid: {args.full}")
    
    # Run experiments
    df = run_grid_experiments(
        grid,
        output_dir=args.output_dir,
        n_epochs=args.n_epochs,
        device=args.device,
    )
    
    # Generate plots
    plot_summary_results(df, output_dir=args.output_dir)
    
    print("\n" + "=" * 80)
    print("✓ Grid experiments complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

