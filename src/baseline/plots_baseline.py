"""Plotting utilities specifically for baseline experiments."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple


def plot_analytic_vs_nonlinear(
    t: np.ndarray,
    theta_analytic: np.ndarray,
    theta_nonlinear: np.ndarray,
    theta0_deg: float,
    c: float,
    save_path: Path,
):
    """
    Plot θ(t) analytic vs nonlinear comparison.
    
    Args:
        t: time array
        theta_analytic: analytical solution
        theta_nonlinear: nonlinear solution
        theta0_deg: initial angle in degrees
        c: damping coefficient
        save_path: path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to degrees for plotting
    theta_analytic_deg = np.degrees(theta_analytic)
    theta_nonlinear_deg = np.degrees(theta_nonlinear)
    
    ax.plot(t, theta_analytic_deg, 'b-', linewidth=2, label='Analytic (small-angle)', alpha=0.8)
    ax.plot(t, theta_nonlinear_deg, 'r--', linewidth=2, label='Nonlinear (numerical)', alpha=0.8)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('θ (degrees)', fontsize=12)
    ax.set_title(f'Analytic vs Nonlinear: θ₀={theta0_deg}°, c={c}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_nonlinear_comparisons(
    results: Dict[str, Dict[str, np.ndarray]],
    save_path: Path,
):
    """
    Plot nonlinear comparisons for different θ₀ and c values.
    
    Args:
        results: dict with keys like "theta0_10_c_0.00" containing {'t', 'theta', 'omega'}
        save_path: path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Extract unique theta0 values
    theta0_values = sorted(set([
        float(key.split('_')[1]) for key in results.keys()
    ]))
    
    for idx, theta0 in enumerate(theta0_values[:4]):  # Plot first 4
        ax = axes[idx]
        
        # Plot for different c values
        for key, data in results.items():
            if f"theta0_{int(theta0)}_" in key:
                c_val = float(key.split('_c_')[1])
                t = data['t']
                theta_deg = np.degrees(data['theta'])
                
                label = f"c={c_val:.2f}"
                linestyle = '-' if c_val == 0 else '--'
                ax.plot(t, theta_deg, linestyle, linewidth=2, label=label, alpha=0.8)
        
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('θ (degrees)', fontsize=11)
        ax.set_title(f'θ₀={int(theta0)}°', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_vs_time(
    t: np.ndarray,
    errors: Dict[float, np.ndarray],
    save_path: Path,
):
    """
    Plot difference (analytic - nonlinear) vs time for small angles.
    
    Args:
        t: time array
        errors: dict mapping theta0_deg to error array
        save_path: path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for theta0_deg, error in errors.items():
        error_deg = np.degrees(error)
        ax.plot(t, error_deg, linewidth=2, label=f'θ₀={theta0_deg}°', alpha=0.8)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Error (degrees)', fontsize=12)
    ax.set_title('Analytic - Nonlinear Error vs Time', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_energy_evolution(
    results: Dict[str, Dict[str, np.ndarray]],
    m: float,
    L: float,
    g: float,
    save_path: Path,
):
    """
    Plot energy H(t) = 0.5*m*L²*θ̇² + m*g*L*(1-cos(θ)).
    
    Args:
        results: dict with simulation results
        m: mass
        L: length
        g: gravitational acceleration
        save_path: path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Separate c=0 and c>0 cases
    c0_results = {k: v for k, v in results.items() if '_c_0.00' in k or '_c_0.0' in k}
    c_nonzero_results = {k: v for k, v in results.items() if '_c_0.00' not in k and '_c_0.0' not in k}
    
    # Plot c=0 (should be flat)
    for key, data in c0_results.items():
        theta0 = int(key.split('_')[1])
        t = data['t']
        theta = data['theta']
        omega = data['omega']
        
        # Compute energy
        kinetic = 0.5 * m * L**2 * omega**2
        potential = m * g * L * (1 - np.cos(theta))
        energy = kinetic + potential
        
        ax1.plot(t, energy, linewidth=2, label=f'θ₀={theta0}°', alpha=0.8)
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Energy (J)', fontsize=12)
    ax1.set_title('Energy Evolution (c=0, undamped)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot c>0 (should be decreasing)
    for key, data in c_nonzero_results.items():
        parts = key.split('_')
        theta0 = int(parts[1])
        c_val = float(key.split('_c_')[1])
        t = data['t']
        theta = data['theta']
        omega = data['omega']
        
        # Compute energy
        kinetic = 0.5 * m * L**2 * omega**2
        potential = m * g * L * (1 - np.cos(theta))
        energy = kinetic + potential
        
        ax2.plot(t, energy, linewidth=2, label=f'θ₀={theta0}°, c={c_val:.2f}', alpha=0.8)
    
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Energy (J)', fontsize=12)
    ax2.set_title('Energy Evolution (c>0, damped)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_all_baseline_figures(
    analytic_results: Dict[str, Dict],
    nonlinear_results: Dict[str, Dict],
    output_dir: Path,
    m: float = 1.0,
    L: float = 1.0,
    g: float = 9.81,
):
    """
    Generate all baseline figures.
    
    Args:
        analytic_results: analytical solutions
        nonlinear_results: nonlinear solutions
        output_dir: output directory for figures
        m: mass
        L: length
        g: gravitational acceleration
    """
    figs_dir = output_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Analytic vs nonlinear for small angles (5°, 10°, 15°) with c=0
    small_angles = [5, 10, 15]
    for theta0_deg in small_angles:
        key = f"theta0_{theta0_deg}_c_0.00"
        if key in analytic_results and key in nonlinear_results:
            t = analytic_results[key]['t']
            theta_a = analytic_results[key]['theta']
            theta_n = nonlinear_results[key]['theta']
            
            save_path = figs_dir / f"analytic_vs_nonlinear_theta0_{theta0_deg}_c_0.png"
            plot_analytic_vs_nonlinear(t, theta_a, theta_n, theta0_deg, 0.0, save_path)
    
    # 2. Nonlinear comparisons for different θ₀ and c
    plot_nonlinear_comparisons(nonlinear_results, figs_dir / "nonlinear_comparisons.png")
    
    # 3. Error vs time for small angles
    errors = {}
    for theta0_deg in small_angles:
        key = f"theta0_{theta0_deg}_c_0.00"
        if key in analytic_results and key in nonlinear_results:
            t = analytic_results[key]['t']
            error = analytic_results[key]['theta'] - nonlinear_results[key]['theta']
            errors[theta0_deg] = error
    
    if errors:
        plot_error_vs_time(t, errors, figs_dir / "error_vs_time.png")
    
    # 4. Energy evolution
    plot_energy_evolution(nonlinear_results, m, L, g, figs_dir / "energy_evolution.png")
    
    print(f"✓ All baseline figures saved to {figs_dir}")

