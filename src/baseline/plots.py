"""Plotting utilities for baseline comparisons."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
from pathlib import Path


def plot_trajectory_comparison(
    t: np.ndarray,
    trajectories: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "Pendulum Trajectory Comparison",
    save_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (12, 5),
) -> plt.Figure:
    """
    Plot comparison of multiple trajectory solutions.

    Args:
        t: time array
        trajectories: dict of {label: (theta, theta_dot)}
        title: plot title
        save_path: path to save figure
        figsize: figure size

    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot theta
    for label, (theta, theta_dot) in trajectories.items():
        axes[0].plot(t, theta, label=label, linewidth=2)

    axes[0].set_xlabel("Time (s)", fontsize=12)
    axes[0].set_ylabel(r"$\theta$ (rad)", fontsize=12)
    axes[0].set_title("Angle", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot theta_dot
    for label, (theta, theta_dot) in trajectories.items():
        axes[1].plot(t, theta_dot, label=label, linewidth=2)

    axes[1].set_xlabel("Time (s)", fontsize=12)
    axes[1].set_ylabel(r"$\dot{\theta}$ (rad/s)", fontsize=12)
    axes[1].set_title("Angular Velocity", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_phase_portrait(
    trajectories: Dict[str, Tuple[np.ndarray, np.ndarray]],
    vector_field: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    title: str = "Phase Portrait",
    save_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (8, 8),
) -> plt.Figure:
    """
    Plot phase portrait with trajectories.

    Args:
        trajectories: dict of {label: (theta, theta_dot)}
        vector_field: optional (theta_grid, theta_dot_grid, dtheta, dtheta_dot)
        title: plot title
        save_path: path to save figure
        figsize: figure size

    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot vector field if provided
    if vector_field is not None:
        theta_grid, theta_dot_grid, dtheta, dtheta_dot = vector_field
        # Normalize vectors for better visualization
        magnitude = np.sqrt(dtheta**2 + dtheta_dot**2)
        magnitude[magnitude == 0] = 1  # Avoid division by zero
        ax.quiver(
            theta_grid,
            theta_dot_grid,
            dtheta / magnitude,
            dtheta_dot / magnitude,
            magnitude,
            alpha=0.3,
            cmap="gray",
        )

    # Plot trajectories
    for label, (theta, theta_dot) in trajectories.items():
        ax.plot(theta, theta_dot, label=label, linewidth=2)
        # Mark initial condition
        ax.plot(theta[0], theta_dot[0], "o", markersize=8)

    ax.set_xlabel(r"$\theta$ (rad)", fontsize=12)
    ax.set_ylabel(r"$\dot{\theta}$ (rad/s)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
    ax.axvline(x=0, color="k", linestyle="--", linewidth=0.5)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_energy(
    t: np.ndarray,
    energies: Dict[str, Dict[str, np.ndarray]],
    title: str = "Energy Evolution",
    save_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (12, 8),
) -> plt.Figure:
    """
    Plot energy components over time.

    Args:
        t: time array
        energies: dict of {label: {'kinetic': ..., 'potential': ..., 'total': ...}}
        title: plot title
        save_path: path to save figure
        figsize: figure size

    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Kinetic energy
    for label, energy_dict in energies.items():
        axes[0, 0].plot(t, energy_dict["kinetic"], label=label, linewidth=2)
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Energy (J)")
    axes[0, 0].set_title("Kinetic Energy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Potential energy
    for label, energy_dict in energies.items():
        axes[0, 1].plot(t, energy_dict["potential"], label=label, linewidth=2)
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Energy (J)")
    axes[0, 1].set_title("Potential Energy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Total energy
    for label, energy_dict in energies.items():
        axes[1, 0].plot(t, energy_dict["total"], label=label, linewidth=2)
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Energy (J)")
    axes[1, 0].set_title("Total Energy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Energy drift (relative to initial)
    for label, energy_dict in energies.items():
        E0 = energy_dict["total"][0]
        drift = (energy_dict["total"] - E0) / E0 * 100
        axes[1, 1].plot(t, drift, label=label, linewidth=2)
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Energy Drift (%)")
    axes[1, 1].set_title("Relative Energy Drift")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color="k", linestyle="--", linewidth=0.5)

    fig.suptitle(title, fontsize=16, y=1.00)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_error_comparison(
    t: np.ndarray,
    reference: Tuple[np.ndarray, np.ndarray],
    approximations: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "Error Comparison",
    save_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (12, 5),
) -> plt.Figure:
    """
    Plot errors relative to reference solution.

    Args:
        t: time array
        reference: (theta_ref, theta_dot_ref)
        approximations: dict of {label: (theta, theta_dot)}
        title: plot title
        save_path: path to save figure
        figsize: figure size

    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    theta_ref, theta_dot_ref = reference

    # Theta error
    for label, (theta, theta_dot) in approximations.items():
        error = np.abs(theta - theta_ref)
        axes[0].semilogy(t, error, label=label, linewidth=2)

    axes[0].set_xlabel("Time (s)", fontsize=12)
    axes[0].set_ylabel(r"$|\theta - \theta_{ref}|$ (rad)", fontsize=12)
    axes[0].set_title("Angle Error", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, which="both")

    # Theta_dot error
    for label, (theta, theta_dot) in approximations.items():
        error = np.abs(theta_dot - theta_dot_ref)
        axes[1].semilogy(t, error, label=label, linewidth=2)

    axes[1].set_xlabel("Time (s)", fontsize=12)
    axes[1].set_ylabel(r"$|\dot{\theta} - \dot{\theta}_{ref}|$ (rad/s)", fontsize=12)
    axes[1].set_title("Angular Velocity Error", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which="both")

    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig

