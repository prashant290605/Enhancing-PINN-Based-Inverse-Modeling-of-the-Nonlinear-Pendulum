"""Baseline experiment: compare analytical vs nonlinear solutions."""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse

from src.data.generator import simulate_pendulum
from src.baseline.linear_small_angle import analytic_small_angle
from src.baseline.plots_baseline import plot_all_baseline_figures
from src.analysis.metrics import rmse, mae, max_error


def run_baseline_experiment(
    output_dir: Path = Path("outputs/baseline"),
    g: float = 9.81,
    L: float = 1.0,
    m: float = 1.0,
    t_span: tuple = (0.0, 10.0),
    n_points: int = 10000,
):
    """
    Run baseline comparison experiment.
    
    Generates:
    - Analytic vs nonlinear plots for θ₀ ∈ {5°, 10°, 15°} with c=0
    - Nonlinear comparisons for θ₀ ∈ {10°, 30°, 60°, 90°} with c ∈ {0, 0.05}
    - Error plots (analytic - nonlinear) vs time
    - Energy evolution plots for c=0 (flat) and c>0 (decreasing)
    
    Args:
        output_dir: output directory
        g: gravitational acceleration (m/s²)
        L: pendulum length (m)
        m: pendulum mass (kg)
        t_span: time span (s)
        n_points: number of time points
    """
    print("=" * 80)
    print("BASELINE EXPERIMENT: Analytical vs Nonlinear Solutions")
    print("=" * 80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Time grid
    t = np.linspace(t_span[0], t_span[1], n_points)
    
    # Configuration
    small_angles_deg = [5, 10, 15]  # For analytic vs nonlinear comparison
    large_angles_deg = [10, 30, 60, 90]  # For nonlinear comparisons
    damping_values = [0.0, 0.05]
    
    analytic_results = {}
    nonlinear_results = {}
    metrics_data = []
    
    print(f"\n[1/3] Computing analytical solutions for small angles...")
    # Analytical solutions (small angles, c=0 only)
    for theta0_deg in small_angles_deg:
        theta0_rad = np.radians(theta0_deg)
        theta_a = analytic_small_angle(theta0_rad, g, L, t)
        
        key = f"theta0_{theta0_deg}_c_0.00"
        analytic_results[key] = {
            't': t,
            'theta': theta_a,
            'omega': None,  # Not computed for analytic
        }
    
    print(f"[2/3] Computing nonlinear solutions...")
    # Nonlinear solutions
    all_angles = sorted(set(small_angles_deg + large_angles_deg))
    
    for theta0_deg in all_angles:
        theta0_rad = np.radians(theta0_deg)
        omega0 = 0.0
        
        for c in damping_values:
            print(f"  θ₀={theta0_deg}°, c={c:.2f}")
            
            t_sim, theta_n, omega_n = simulate_pendulum(
                theta0_rad, omega0, g, L, c, t, method="ivp"
            )
            
            key = f"theta0_{theta0_deg}_c_{c:.2f}"
            nonlinear_results[key] = {
                't': t_sim,
                'theta': theta_n,
                'omega': omega_n,
            }
            
            # Compute metrics for small angles with c=0
            if theta0_deg in small_angles_deg and c == 0.0:
                key_a = f"theta0_{theta0_deg}_c_0.00"
                if key_a in analytic_results:
                    theta_a = analytic_results[key_a]['theta']
                    
                    error_rmse = rmse(theta_a, theta_n)
                    error_mae = mae(theta_a, theta_n)
                    error_max = max_error(theta_a, theta_n)
                    
                    metrics_data.append({
                        'theta0_deg': theta0_deg,
                        'c': c,
                        'rmse_rad': error_rmse,
                        'mae_rad': error_mae,
                        'max_error_rad': error_max,
                        'rmse_deg': np.degrees(error_rmse),
                        'mae_deg': np.degrees(error_mae),
                        'max_error_deg': np.degrees(error_max),
                    })
    
    print(f"[3/3] Generating plots and saving metrics...")
    
    # Generate all plots
    plot_all_baseline_figures(
        analytic_results,
        nonlinear_results,
        output_dir,
        m=m,
        L=L,
        g=g,
    )
    
    # Save metrics CSV
    if metrics_data:
        df = pd.DataFrame(metrics_data)
        metrics_path = output_dir / "metrics.csv"
        df.to_csv(metrics_path, index=False, float_format='%.6f')
        print(f"✓ Metrics saved to {metrics_path}")
        
        print("\nSummary Errors (Analytic vs Nonlinear, c=0):")
        print(df.to_string(index=False))
    
    print(f"\n✓ All results saved to: {output_dir}")
    print("=" * 80)
    
    return {
        'analytic_results': analytic_results,
        'nonlinear_results': nonlinear_results,
        'metrics': metrics_data,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run baseline experiment: compare analytical vs nonlinear pendulum solutions"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/baseline",
        help="Output directory for figures and metrics"
    )
    parser.add_argument(
        "--g",
        type=float,
        default=9.81,
        help="Gravitational acceleration (m/s²)"
    )
    parser.add_argument(
        "--L",
        type=float,
        default=1.0,
        help="Pendulum length (m)"
    )
    parser.add_argument(
        "--m",
        type=float,
        default=1.0,
        help="Pendulum mass (kg)"
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=10.0,
        help="Maximum simulation time (s)"
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=10000,
        help="Number of time points (use dense grid for accuracy)"
    )
    
    args = parser.parse_args()
    
    run_baseline_experiment(
        output_dir=Path(args.output_dir),
        g=args.g,
        L=args.L,
        m=args.m,
        t_span=(0.0, args.t_max),
        n_points=args.n_points,
    )


if __name__ == "__main__":
    main()

