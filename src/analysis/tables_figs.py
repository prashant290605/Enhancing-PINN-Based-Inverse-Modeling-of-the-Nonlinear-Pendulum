"""Generate publication-quality tables and figures."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def create_results_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
    output_path: Optional[Path] = None,
    format: str = "latex",
) -> pd.DataFrame:
    """
    Create results table comparing different methods.

    Args:
        results: dict of {method_name: {metric_name: value}}
        metrics: list of metrics to include
        output_path: path to save table
        format: 'latex', 'csv', 'markdown'

    Returns:
        pandas DataFrame
    """
    # Create DataFrame
    data = []
    for method, method_results in results.items():
        row = {"Method": method}
        for metric in metrics:
            if metric in method_results:
                row[metric] = method_results[metric]
            else:
                row[metric] = np.nan
        data.append(row)

    df = pd.DataFrame(data)

    # Save if path provided
    if output_path is not None:
        if format == "latex":
            latex_str = df.to_latex(index=False, float_format="%.4f")
            output_path.write_text(latex_str)
        elif format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "markdown":
            markdown_str = df.to_markdown(index=False)
            output_path.write_text(markdown_str)

    return df


def create_parameter_table(
    results: Dict[str, Dict[str, float]],
    true_params: Dict[str, float],
    output_path: Optional[Path] = None,
    format: str = "latex",
) -> pd.DataFrame:
    """
    Create parameter estimation table.

    Args:
        results: dict of {method_name: {param_name: value}}
        true_params: true parameter values
        output_path: path to save table
        format: 'latex', 'csv', 'markdown'

    Returns:
        pandas DataFrame
    """
    data = []

    # Add true values row
    true_row = {"Method": "True"}
    true_row.update(true_params)
    data.append(true_row)

    # Add estimated values
    for method, params in results.items():
        row = {"Method": method}
        row.update(params)
        data.append(row)

    df = pd.DataFrame(data)

    # Save if path provided
    if output_path is not None:
        if format == "latex":
            latex_str = df.to_latex(index=False, float_format="%.4f")
            output_path.write_text(latex_str)
        elif format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "markdown":
            markdown_str = df.to_markdown(index=False)
            output_path.write_text(markdown_str)

    return df


def plot_predictions_comparison(
    t: np.ndarray,
    true_theta: np.ndarray,
    predictions: Dict[str, np.ndarray],
    title: str = "Predictions Comparison",
    save_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (12, 6),
) -> plt.Figure:
    """
    Plot comparison of predictions from different methods.

    Args:
        t: time array
        true_theta: true angles
        predictions: dict of {method_name: theta_predictions}
        title: plot title
        save_path: path to save figure
        figsize: figure size

    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot predictions
    axes[0].plot(t, true_theta, "k-", linewidth=2, label="True", alpha=0.7)

    for method, theta_pred in predictions.items():
        axes[0].plot(t, theta_pred, "--", linewidth=1.5, label=method, alpha=0.8)

    axes[0].set_xlabel("Time (s)", fontsize=12)
    axes[0].set_ylabel(r"$\theta$ (rad)", fontsize=12)
    axes[0].set_title("Predictions", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot errors
    for method, theta_pred in predictions.items():
        error = np.abs(theta_pred - true_theta)
        axes[1].semilogy(t, error, linewidth=1.5, label=method, alpha=0.8)

    axes[1].set_xlabel("Time (s)", fontsize=12)
    axes[1].set_ylabel(r"$|\theta - \theta_{true}|$ (rad)", fontsize=12)
    axes[1].set_title("Absolute Error", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which="both")

    fig.suptitle(title, fontsize=16, y=1.00)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_uncertainty_quantification(
    t: np.ndarray,
    true_theta: np.ndarray,
    theta_mean: np.ndarray,
    theta_std: np.ndarray,
    confidence: float = 0.95,
    title: str = "Uncertainty Quantification",
    save_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (12, 5),
) -> plt.Figure:
    """
    Plot predictions with uncertainty bands.

    Args:
        t: time array
        true_theta: true angles
        theta_mean: mean predictions
        theta_std: standard deviations
        confidence: confidence level
        title: plot title
        save_path: path to save figure
        figsize: figure size

    Returns:
        matplotlib figure
    """
    from scipy.stats import norm

    z = norm.ppf((1 + confidence) / 2)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot with uncertainty bands
    axes[0].plot(t, true_theta, "k-", linewidth=2, label="True", alpha=0.7)
    axes[0].plot(t, theta_mean, "b-", linewidth=2, label="Mean prediction", alpha=0.8)

    lower = theta_mean - z * theta_std
    upper = theta_mean + z * theta_std

    axes[0].fill_between(
        t.flatten(),
        lower.flatten(),
        upper.flatten(),
        alpha=0.3,
        label=f"{confidence*100:.0f}% CI",
    )

    axes[0].set_xlabel("Time (s)", fontsize=12)
    axes[0].set_ylabel(r"$\theta$ (rad)", fontsize=12)
    axes[0].set_title("Predictions with Uncertainty", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot uncertainty evolution
    axes[1].plot(t, theta_std, "b-", linewidth=2)
    axes[1].set_xlabel("Time (s)", fontsize=12)
    axes[1].set_ylabel(r"$\sigma_{\theta}$ (rad)", fontsize=12)
    axes[1].set_title("Prediction Uncertainty", fontsize=14)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, y=1.00)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_parameter_convergence(
    history: Dict[str, List[float]],
    true_params: Dict[str, float],
    title: str = "Parameter Convergence",
    save_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (12, 4),
) -> plt.Figure:
    """
    Plot parameter convergence during training.

    Args:
        history: training history with parameter values
        true_params: true parameter values
        title: plot title
        save_path: path to save figure
        figsize: figure size

    Returns:
        matplotlib figure
    """
    params = ["g", "L", "damping"]
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for i, param in enumerate(params):
        if param in history:
            axes[i].plot(history[param], linewidth=2)
            if param in true_params:
                axes[i].axhline(
                    y=true_params[param],
                    color="r",
                    linestyle="--",
                    linewidth=2,
                    label="True value",
                )
            axes[i].set_xlabel("Epoch", fontsize=12)
            axes[i].set_ylabel(param, fontsize=12)
            axes[i].set_title(f"Parameter: {param}", fontsize=14)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_loss_history(
    history: Dict[str, List[float]],
    title: str = "Training Loss",
    save_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (12, 5),
) -> plt.Figure:
    """
    Plot training loss history.

    Args:
        history: training history
        title: plot title
        save_path: path to save figure
        figsize: figure size

    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Total loss
    if "loss" in history:
        axes[0].semilogy(history["loss"], linewidth=2)
        axes[0].set_xlabel("Epoch", fontsize=12)
        axes[0].set_ylabel("Total Loss", fontsize=12)
        axes[0].set_title("Total Loss", fontsize=14)
        axes[0].grid(True, alpha=0.3, which="both")

    # Loss components
    loss_components = ["loss_physics", "loss_data", "loss_ic", "loss_passivity"]
    for component in loss_components:
        if component in history:
            label = component.replace("loss_", "").replace("_", " ").title()
            axes[1].semilogy(history[component], linewidth=2, label=label, alpha=0.8)

    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Loss", fontsize=12)
    axes[1].set_title("Loss Components", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which="both")

    fig.suptitle(title, fontsize=16, y=1.00)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_energy_comparison(
    t: np.ndarray,
    energies: Dict[str, np.ndarray],
    title: str = "Energy Comparison",
    save_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (12, 5),
) -> plt.Figure:
    """
    Plot energy evolution for different methods.

    Args:
        t: time array
        energies: dict of {method_name: energy_array}
        title: plot title
        save_path: path to save figure
        figsize: figure size

    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Absolute energy
    for method, energy in energies.items():
        axes[0].plot(t, energy, linewidth=2, label=method, alpha=0.8)

    axes[0].set_xlabel("Time (s)", fontsize=12)
    axes[0].set_ylabel("Energy (J)", fontsize=12)
    axes[0].set_title("Total Energy", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Relative energy drift
    for method, energy in energies.items():
        E0 = energy[0]
        drift = (energy - E0) / E0 * 100
        axes[1].plot(t, drift, linewidth=2, label=method, alpha=0.8)

    axes[1].set_xlabel("Time (s)", fontsize=12)
    axes[1].set_ylabel("Energy Drift (%)", fontsize=12)
    axes[1].set_title("Relative Energy Drift", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color="k", linestyle="--", linewidth=0.5)

    fig.suptitle(title, fontsize=16, y=1.00)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig



def plot_method_comparison(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot comparison of different methods.
    
    Args:
        df: DataFrame with columns ['method', 'traj_mse', 'param_rmse', 'energy_drift', 'coverage_95']
        save_path: path to save figure
        
    Returns:
        matplotlib Figure
    """
    from src.viz.style import setup_plot_style
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    methods = df['method'].values
    x = np.arange(len(methods))
    
    # Trajectory MSE
    ax = axes[0, 0]
    values = df['traj_mse'].values
    bars = ax.bar(x, values, color='steelblue', edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Trajectory MSE')
    ax.set_title('Trajectory Prediction Error')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Parameter RMSE
    ax = axes[0, 1]
    values = df['param_rmse'].values
    bars = ax.bar(x, values, color='coral', edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Parameter RMSE')
    ax.set_title('Parameter Estimation Error')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, val) in enumerate(zip(bars, values)):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Energy Drift
    ax = axes[1, 0]
    values = df['energy_drift'].values
    bars = ax.bar(x, values, color='mediumseagreen', edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Energy Drift')
    ax.set_title('Energy Conservation')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, val) in enumerate(zip(bars, values)):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Coverage (only for ensemble)
    ax = axes[1, 1]
    values = df['coverage_95'].values
    # Filter out NaN values
    valid_mask = ~np.isnan(values)
    valid_x = x[valid_mask]
    valid_values = values[valid_mask]
    valid_methods = methods[valid_mask]
    
    if len(valid_values) > 0:
        bars = ax.bar(valid_x, valid_values, color='mediumpurple', edgecolor='black', linewidth=1.5)
        ax.axhline(0.95, color='k', linestyle='--', linewidth=2, label='Expected (95%)')
        ax.set_ylabel('Parameter Coverage (95% CI)')
        ax.set_title('Uncertainty Quantification')
        ax.set_xticks(valid_x)
        ax.set_xticklabels(valid_methods, rotation=45, ha='right', fontsize=9)
        ax.set_ylim([0, 1.1])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, val) in enumerate(zip(bars, valid_values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No ensemble results', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle('Method Comparison: Ablations and Baselines', fontsize=16, y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def generate_method_comparison_table_and_plot(
    csv_path: Path,
    output_dir: Path,
):
    """
    Generate method comparison table and plot from CSV.
    
    Args:
        csv_path: path to table_methods.csv
        output_dir: output directory for plots
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Generate plot
    figs_dir = output_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    plot_method_comparison(df, save_path=figs_dir / "method_comparison.png")
    
    print(f"✓ Method comparison plot saved to: {figs_dir / 'method_comparison.png'}")
