"""Run ablation studies and method comparisons."""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import argparse
from typing import Dict

from src.data.generator import simulate_pendulum, add_noise, subsample
from src.models.pinn_inverse import PINN
from src.models.dissipation_net import DissipationNet
from src.models.ensemble import create_bootstrap_ensemble
from src.models.train_inverse import create_trainer
from src.analysis.metrics import rmse, mse, parameter_error
from src.models.losses import compute_derivatives


def run_ablation_no_passivity(
    t_sparse: np.ndarray,
    theta_sparse_noisy: np.ndarray,
    t_dense: np.ndarray,
    theta_dense: np.ndarray,
    omega_dense: np.ndarray,
    theta0: float,
    omega0: float,
    g_true: float,
    L_true: float,
    c_true: float,
    n_epochs: int = 1000,
    device: str = "cpu",
) -> Dict:
    """Ablation: Remove passivity term."""
    
    # Convert to torch
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
    
    # Create trainer WITHOUT passivity
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
        lambda_pass=0.0,  # NO PASSIVITY
        dissipation_net=None,
        learning_rate=1e-3,
        scheduler_type='cosine',
        n_epochs=n_epochs,
        device=device,
    )
    
    # Train
    trainer.train(n_epochs=n_epochs, verbose=False, save_best=False)
    
    # Evaluate
    return evaluate_model(model, t_dense, theta_dense, omega_dense, g_true, L_true, c_true)


def run_ablation_no_physics(
    t_sparse: np.ndarray,
    theta_sparse_noisy: np.ndarray,
    t_dense: np.ndarray,
    theta_dense: np.ndarray,
    omega_dense: np.ndarray,
    theta0: float,
    omega0: float,
    g_true: float,
    L_true: float,
    c_true: float,
    n_epochs: int = 1000,
    device: str = "cpu",
) -> Dict:
    """Ablation: Remove physics residual (data-only fit)."""
    
    # Convert to torch
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
    
    # Create trainer WITHOUT physics
    trainer = create_trainer(
        model=model,
        t_obs=t_obs_torch,
        theta_obs=theta_obs_torch,
        t_collocation=t_colloc_torch,
        theta0=theta0,
        omega0=omega0,
        lambda_data=1.0,
        lambda_phys=0.0,  # NO PHYSICS
        lambda_ic=1.0,
        lambda_pass=0.0,
        dissipation_net=None,
        learning_rate=1e-3,
        scheduler_type='cosine',
        n_epochs=n_epochs,
        device=device,
    )
    
    # Train
    trainer.train(n_epochs=n_epochs, verbose=False, save_best=False)
    
    # Evaluate
    return evaluate_model(model, t_dense, theta_dense, omega_dense, g_true, L_true, c_true)


def run_ablation_nn_dissipation(
    t_sparse: np.ndarray,
    theta_sparse_noisy: np.ndarray,
    t_dense: np.ndarray,
    theta_dense: np.ndarray,
    omega_dense: np.ndarray,
    theta0: float,
    omega0: float,
    g_true: float,
    L_true: float,
    c_true: float,
    n_epochs: int = 1000,
    device: str = "cpu",
) -> Dict:
    """Ablation: Switch from viscous to NN dissipation."""
    
    # Convert to torch
    t_obs_torch = torch.tensor(t_sparse, dtype=torch.float32).unsqueeze(-1)
    theta_obs_torch = torch.tensor(theta_sparse_noisy, dtype=torch.float32).unsqueeze(-1)
    t_colloc_torch = torch.tensor(t_dense, dtype=torch.float32).unsqueeze(-1)
    
    # Create model (don't learn viscous damping)
    model = PINN(
        hidden_layers=[32, 32, 32],
        activation='tanh',
        init_g=10.0,
        init_L=1.1,
        init_damping=0.08,
        learn_g=True,
        learn_L=True,
        learn_damping=False,  # Don't learn viscous
        use_fourier=True,
        num_frequencies=6,
    )
    
    # Create NN dissipation
    dissipation_net = DissipationNet(
        hidden_layers=[16, 16],
        activation='tanh',
        use_enhanced_features=True,
    )
    
    # Create trainer with NN dissipation
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
        lambda_pass=1.0,
        dissipation_net=dissipation_net,
        learning_rate=1e-3,
        scheduler_type='cosine',
        n_epochs=n_epochs,
        device=device,
    )
    
    # Train
    trainer.train(n_epochs=n_epochs, verbose=False, save_best=False)
    
    # Evaluate (note: c_pred will be from model, not NN)
    return evaluate_model(model, t_dense, theta_dense, omega_dense, g_true, L_true, c_true)


def run_deterministic_with_passivity(
    t_sparse: np.ndarray,
    theta_sparse_noisy: np.ndarray,
    t_dense: np.ndarray,
    theta_dense: np.ndarray,
    omega_dense: np.ndarray,
    theta0: float,
    omega0: float,
    g_true: float,
    L_true: float,
    c_true: float,
    n_epochs: int = 1000,
    device: str = "cpu",
) -> Dict:
    """Baseline: Deterministic PINN with passivity."""
    
    # Convert to torch
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
    
    # Create trainer WITH passivity
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
        lambda_pass=1.0,  # WITH PASSIVITY
        dissipation_net=None,
        learning_rate=1e-3,
        scheduler_type='cosine',
        n_epochs=n_epochs,
        device=device,
    )
    
    # Train
    trainer.train(n_epochs=n_epochs, verbose=False, save_best=False)
    
    # Evaluate
    return evaluate_model(model, t_dense, theta_dense, omega_dense, g_true, L_true, c_true)


def run_ensemble_with_passivity(
    t_sparse: np.ndarray,
    theta_sparse_noisy: np.ndarray,
    t_dense: np.ndarray,
    theta_dense: np.ndarray,
    omega_dense: np.ndarray,
    theta0: float,
    omega0: float,
    g_true: float,
    L_true: float,
    c_true: float,
    n_models: int = 5,
    n_epochs: int = 1000,
    device: str = "cpu",
) -> Dict:
    """Baseline: Ensemble with passivity."""
    
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
        'lambda_pass': 1.0,
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
    theta_all = predictions['theta_all']
    
    # Get parameter statistics
    param_stats = ensemble.get_parameter_statistics()
    
    # Compute metrics
    true_params = {'g': g_true, 'L': L_true, 'damping': c_true}
    
    traj_rmse_val = rmse(theta_mean, theta_dense)
    traj_mse_val = mse(theta_mean, theta_dense)
    
    # Energy drift
    m = 1.0
    # Use mean predictions for energy
    model_eval = ensemble.models[0]  # Use first model for derivative computation
    model_eval.eval()
    t_torch = torch.tensor(t_dense, dtype=torch.float32).unsqueeze(-1).requires_grad_(True)
    theta_torch = torch.tensor(theta_mean, dtype=torch.float32).unsqueeze(-1).requires_grad_(True)
    
    # Approximate derivatives from mean
    omega_pred = np.gradient(theta_mean, t_dense)
    E_pred = 0.5 * m * L_true**2 * omega_pred**2 + m * g_true * L_true * (1 - np.cos(theta_mean))
    energy_drift_val = np.max(np.abs(E_pred - E_pred[0]))
    
    # Parameter errors
    g_error = abs(param_stats['g']['mean'] - g_true)
    L_error = abs(param_stats['L']['mean'] - L_true)
    c_error = abs(param_stats['damping']['mean'] - c_true)
    param_rmse_val = np.sqrt((g_error**2 + L_error**2 + c_error**2) / 3)
    
    # Coverage
    from src.analysis.metrics import compute_trajectory_ci_coverage, compute_parameter_ci_coverage
    traj_coverage = compute_trajectory_ci_coverage(theta_all, theta_dense, confidence=0.95)
    g_ci = compute_parameter_ci_coverage(np.array(param_stats['g']['values']), g_true, 0.95)
    L_ci = compute_parameter_ci_coverage(np.array(param_stats['L']['values']), L_true, 0.95)
    c_ci = compute_parameter_ci_coverage(np.array(param_stats['damping']['values']), c_true, 0.95)
    
    avg_param_coverage = (g_ci['coverage'] + L_ci['coverage'] + c_ci['coverage']) / 3
    
    return {
        'traj_mse': traj_mse_val,
        'param_rmse': param_rmse_val,
        'energy_drift': energy_drift_val,
        'coverage_95': avg_param_coverage,
        'traj_coverage_95': traj_coverage['coverage'],
        'g_pred': param_stats['g']['mean'],
        'L_pred': param_stats['L']['mean'],
        'c_pred': param_stats['damping']['mean'],
    }


def evaluate_model(
    model: PINN,
    t_dense: np.ndarray,
    theta_dense: np.ndarray,
    omega_dense: np.ndarray,
    g_true: float,
    L_true: float,
    c_true: float,
) -> Dict:
    """Evaluate a trained model."""
    
    # Predict on dense grid
    model.eval()
    t_torch = torch.tensor(t_dense, dtype=torch.float32).unsqueeze(-1).requires_grad_(True)
    theta_pred_torch = model(t_torch)
    
    theta_dot_torch, _ = compute_derivatives(theta_pred_torch, t_torch)
    
    theta_pred = theta_pred_torch.detach().cpu().numpy().flatten()
    omega_pred = theta_dot_torch.detach().cpu().numpy().flatten()
    
    # Get parameters
    params = model.get_parameters()
    
    # Compute metrics
    true_params = {'g': g_true, 'L': L_true, 'damping': c_true}
    param_errors = parameter_error(params, true_params)
    
    traj_mse_val = mse(theta_pred, theta_dense)
    
    # Parameter RMSE
    g_error = param_errors['g_abs_error']
    L_error = param_errors['L_abs_error']
    c_error = param_errors['damping_abs_error']
    param_rmse_val = np.sqrt((g_error**2 + L_error**2 + c_error**2) / 3)
    
    # Energy drift
    m = 1.0
    E_pred = 0.5 * m * L_true**2 * omega_pred**2 + m * g_true * L_true * (1 - np.cos(theta_pred))
    energy_drift_val = np.max(np.abs(E_pred - E_pred[0]))
    
    return {
        'traj_mse': traj_mse_val,
        'param_rmse': param_rmse_val,
        'energy_drift': energy_drift_val,
        'coverage_95': np.nan,  # Not applicable for deterministic
        'traj_coverage_95': np.nan,
        'g_pred': params['g'],
        'L_pred': params['L'],
        'c_pred': params['damping'],
    }


def run_all_methods(
    output_dir: Path,
    theta0_deg: float = 30.0,
    c: float = 0.05,
    n_sparse: int = 20,
    sigma: float = 0.01,
    n_epochs: int = 1000,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Run all methods and ablations.
    
    Args:
        output_dir: output directory
        theta0_deg: initial angle in degrees
        c: damping coefficient
        n_sparse: number of sparse observations
        sigma: noise standard deviation
        n_epochs: number of training epochs
        device: device to train on
        
    Returns:
        DataFrame with results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # True parameters
    g_true = 9.81
    L_true = 1.0
    c_true = c
    
    # Generate data
    print("Generating dataset...")
    np.random.seed(1337)
    theta0 = np.radians(theta0_deg)
    omega0 = 0.0
    
    t_dense = np.linspace(0, 10.0, 1000)
    t_dense, theta_dense, omega_dense = simulate_pendulum(
        theta0, omega0, g_true, L_true, c_true, t_dense, method="ivp"
    )
    
    t_sparse, theta_sparse = subsample(t_dense, theta_dense, n_sparse, irregular=True, seed=1337)
    theta_sparse_noisy = add_noise(theta_sparse, sigma, seed=1337)
    
    print(f"Generated {len(t_sparse)} sparse observations")
    
    # Run all methods
    results = []
    
    print("\n[1/5] Running: Ablation - No Passivity")
    result = run_ablation_no_passivity(
        t_sparse, theta_sparse_noisy, t_dense, theta_dense, omega_dense,
        theta0, omega0, g_true, L_true, c_true, n_epochs, device
    )
    result['method'] = 'Ablation: No Passivity'
    results.append(result)
    print(f"  ✓ MSE={result['traj_mse']:.4f}, Param RMSE={result['param_rmse']:.4f}")
    
    print("\n[2/5] Running: Ablation - No Physics (Data Only)")
    result = run_ablation_no_physics(
        t_sparse, theta_sparse_noisy, t_dense, theta_dense, omega_dense,
        theta0, omega0, g_true, L_true, c_true, n_epochs, device
    )
    result['method'] = 'Ablation: No Physics'
    results.append(result)
    print(f"  ✓ MSE={result['traj_mse']:.4f}, Param RMSE={result['param_rmse']:.4f}")
    
    print("\n[3/5] Running: Ablation - NN Dissipation")
    result = run_ablation_nn_dissipation(
        t_sparse, theta_sparse_noisy, t_dense, theta_dense, omega_dense,
        theta0, omega0, g_true, L_true, c_true, n_epochs, device
    )
    result['method'] = 'Ablation: NN Dissipation'
    results.append(result)
    print(f"  ✓ MSE={result['traj_mse']:.4f}, Param RMSE={result['param_rmse']:.4f}")
    
    print("\n[4/5] Running: Deterministic + Passivity")
    result = run_deterministic_with_passivity(
        t_sparse, theta_sparse_noisy, t_dense, theta_dense, omega_dense,
        theta0, omega0, g_true, L_true, c_true, n_epochs, device
    )
    result['method'] = 'Deterministic + Passivity'
    results.append(result)
    print(f"  ✓ MSE={result['traj_mse']:.4f}, Param RMSE={result['param_rmse']:.4f}")
    
    print("\n[5/5] Running: Ensemble + Passivity")
    result = run_ensemble_with_passivity(
        t_sparse, theta_sparse_noisy, t_dense, theta_dense, omega_dense,
        theta0, omega0, g_true, L_true, c_true, 5, n_epochs, device
    )
    result['method'] = 'Ensemble + Passivity'
    results.append(result)
    print(f"  ✓ MSE={result['traj_mse']:.4f}, Param RMSE={result['param_rmse']:.4f}, Coverage={result['coverage_95']:.2f}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['method', 'traj_mse', 'param_rmse', 'energy_drift', 'coverage_95', 'traj_coverage_95', 'g_pred', 'L_pred', 'c_pred']
    df = df[cols]
    
    # Save
    df.to_csv(output_dir / "table_methods.csv", index=False)
    print(f"\n✓ Results saved to: {output_dir / 'table_methods.csv'}")
    
    return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run ablation studies and method comparisons")
    parser.add_argument(
        "--theta0",
        type=float,
        default=30.0,
        help="Initial angle in degrees"
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.05,
        help="Damping coefficient"
    )
    parser.add_argument(
        "--n-sparse",
        type=int,
        default=20,
        help="Number of sparse observations"
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.01,
        help="Noise standard deviation"
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
    
    print("=" * 80)
    print("ABLATION STUDIES AND METHOD COMPARISONS")
    print("=" * 80)
    
    df = run_all_methods(
        output_dir=Path(args.output_dir),
        theta0_deg=args.theta0,
        c=args.damping,
        n_sparse=args.n_sparse,
        sigma=args.noise,
        n_epochs=args.n_epochs,
        device=args.device,
    )
    
    print("\n" + "=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()

