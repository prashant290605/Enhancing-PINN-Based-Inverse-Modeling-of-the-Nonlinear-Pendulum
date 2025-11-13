"""Demonstrate improved plot quality and naming conventions."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.viz.style import (
    setup_plot_style,
    save_plot_dual_format,
    generate_plot_name,
    format_axis_label,
)
from src.data.generator import simulate_pendulum


def demo_theta_vs_time_plot(output_dir: Path):
    """Demonstrate theta vs time plot with improved styling."""
    setup_plot_style()
    
    # Generate data
    theta0_deg = 30.0
    damping = 0.05
    theta0 = np.radians(theta0_deg)
    omega0 = 0.0
    g = 9.81
    L = 1.0
    
    t = np.linspace(0, 10, 1000)
    t, theta, omega = simulate_pendulum(theta0, omega0, g, L, damping, t, method="ivp")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(t, theta, 'b-', linewidth=2.5, label='PINN Prediction')
    ax.plot(t, theta, 'k--', linewidth=1.5, alpha=0.7, label='Ground Truth')
    
    # Use LaTeX-style labels
    ax.set_xlabel(format_axis_label('time', 's'), fontsize=14)
    ax.set_ylabel(format_axis_label('theta', 'rad'), fontsize=14)
    ax.set_title(r'Pendulum Angle vs Time ($\theta_0=30°$, $c=0.05$)', fontsize=16, weight='bold')
    
    # Legend outside if needed
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    # Generate consistent filename
    filename = generate_plot_name(
        'theta_vs_time',
        theta0_deg=theta0_deg,
        damping=damping,
        use_passivity=True
    )
    
    # Save in both formats
    save_path = output_dir / filename
    save_plot_dual_format(fig, save_path, formats=['png', 'pdf'])
    plt.close()
    
    print(f"✓ Saved: {filename}.png and {filename}.pdf")


def demo_param_histogram_plot(output_dir: Path):
    """Demonstrate parameter histogram with improved styling."""
    setup_plot_style()
    
    # Generate synthetic parameter estimates
    np.random.seed(1337)
    g_estimates = np.random.normal(9.81, 0.2, 50)
    
    theta0_deg = 30.0
    damping = 0.05
    n_models = 7
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.hist(g_estimates, bins=15, alpha=0.7, color='steelblue', edgecolor='black', linewidth=1.5)
    ax.axvline(9.81, color='red', linestyle='--', linewidth=2.5, label='True Value')
    ax.axvline(np.mean(g_estimates), color='green', linestyle='-', linewidth=2.5, label='Mean Estimate')
    
    # Use LaTeX-style labels
    ax.set_xlabel(format_axis_label('g', 'm_s2'), fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title(r'Ensemble Estimates of $g$ (N=7)', fontsize=16, weight='bold')
    
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Generate consistent filename
    filename = generate_plot_name(
        'param_hist_g',
        theta0_deg=theta0_deg,
        damping=damping,
        n_models=n_models
    )
    
    # Save in both formats
    save_path = output_dir / filename
    save_plot_dual_format(fig, save_path, formats=['png', 'pdf'])
    plt.close()
    
    print(f"✓ Saved: {filename}.png and {filename}.pdf")


def demo_energy_drift_plot(output_dir: Path):
    """Demonstrate energy drift comparison with improved styling."""
    setup_plot_style()
    
    # Generate data
    t = np.linspace(0, 10, 1000)
    
    # Simulate energy with and without passivity
    E_no_pass = 1.0 + 0.5 * np.sin(t) + 0.1 * t  # Drift
    E_with_pass = 1.0 * np.exp(-0.05 * t)  # Decay (passivity)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(t, E_no_pass, 'b-', linewidth=2.5, label='Without Passivity')
    ax.plot(t, E_with_pass, 'r-', linewidth=2.5, label='With Passivity')
    ax.axhline(1.0, color='k', linestyle=':', linewidth=1.5, alpha=0.5, label='Initial Energy')
    
    # Use LaTeX-style labels
    ax.set_xlabel(format_axis_label('time', 's'), fontsize=14)
    ax.set_ylabel(format_axis_label('energy', 'J'), fontsize=14)
    ax.set_title(r'Energy Evolution: $H(t) = \frac{1}{2}L^2\dot{\theta}^2 + gL(1-\cos\theta)$', 
                 fontsize=16, weight='bold')
    
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    # Generate consistent filename
    filename = generate_plot_name('energy_drift_comparison')
    
    # Save in both formats
    save_path = output_dir / filename
    save_plot_dual_format(fig, save_path, formats=['png', 'pdf'])
    plt.close()
    
    print(f"✓ Saved: {filename}.png and {filename}.pdf")


def demo_uncertainty_bands_plot(output_dir: Path):
    """Demonstrate uncertainty bands with improved styling."""
    setup_plot_style()
    
    # Generate data
    t = np.linspace(0, 10, 1000)
    theta_mean = 0.5 * np.cos(np.sqrt(9.81) * t) * np.exp(-0.05 * t)
    theta_std = 0.05 + 0.02 * t / 10  # Increasing uncertainty
    
    theta0_deg = 30.0
    damping = 0.05
    n_models = 5
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(t, theta_mean, 'b-', linewidth=2.5, label='Ensemble Mean')
    ax.fill_between(t, theta_mean - 2*theta_std, theta_mean + 2*theta_std,
                     alpha=0.3, color='blue', label=r'Mean $\pm 2\sigma$')
    
    # Use LaTeX-style labels
    ax.set_xlabel(format_axis_label('time', 's'), fontsize=14)
    ax.set_ylabel(format_axis_label('theta', 'rad'), fontsize=14)
    ax.set_title(r'Ensemble Prediction with Uncertainty ($N=5$)', fontsize=16, weight='bold')
    
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    # Generate consistent filename
    filename = generate_plot_name(
        'theta_uncertainty',
        theta0_deg=theta0_deg,
        damping=damping,
        n_models=n_models
    )
    
    # Save in both formats
    save_path = output_dir / filename
    save_plot_dual_format(fig, save_path, formats=['png', 'pdf'])
    plt.close()
    
    print(f"✓ Saved: {filename}.png and {filename}.pdf")


def main():
    """Generate demo plots."""
    output_dir = Path("outputs/demo_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("GENERATING PUBLICATION-QUALITY DEMO PLOTS")
    print("=" * 80)
    print()
    
    print("[1/4] Theta vs Time Plot")
    demo_theta_vs_time_plot(output_dir)
    
    print("\n[2/4] Parameter Histogram")
    demo_param_histogram_plot(output_dir)
    
    print("\n[3/4] Energy Drift Comparison")
    demo_energy_drift_plot(output_dir)
    
    print("\n[4/4] Uncertainty Bands")
    demo_uncertainty_bands_plot(output_dir)
    
    print()
    print("=" * 80)
    print(f"✓ All demo plots saved to: {output_dir}")
    print("=" * 80)
    print()
    print("Features demonstrated:")
    print("  • LaTeX-style labels with symbols (θ, ω, etc.)")
    print("  • Readable fonts (Times New Roman for serif)")
    print("  • Consistent naming convention")
    print("  • Dual format saving (.png and .pdf)")
    print("  • Publication-quality styling")
    print()


if __name__ == "__main__":
    main()

