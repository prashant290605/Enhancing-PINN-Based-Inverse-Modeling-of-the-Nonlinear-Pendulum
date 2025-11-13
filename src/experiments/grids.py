"""Experiment grids and configuration management."""

from typing import Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    name: str
    seed: int = 1337

    # Data generation
    g: float = 9.81
    L: float = 1.0
    damping: float = 0.1
    theta0: float = 0.5236  # pi/6
    theta_dot0: float = 0.0
    t_span: tuple = (0.0, 10.0)
    n_points: int = 100
    noise_std: float = 0.01

    # Model architecture
    hidden_layers: List[int] = field(default_factory=lambda: [32, 32, 32])
    activation: str = "tanh"

    # Training
    n_epochs: int = 10000
    learning_rate: float = 1e-3
    optimizer: str = "adam"

    # Loss weights
    weight_physics: float = 1.0
    weight_data: float = 1.0
    weight_ic: float = 10.0
    weight_passivity: float = 0.0

    # Flags
    use_passivity: bool = False
    learn_g: bool = True
    learn_L: bool = True
    learn_damping: bool = True


def create_baseline_grid() -> List[Dict[str, Any]]:
    """
    Create experiment grid for baseline comparisons.

    Returns:
        list of experiment configurations
    """
    grid = []

    # Vary initial conditions
    theta0_values = [0.1, 0.3, 0.5, 1.0]  # radians

    for theta0 in theta0_values:
        config = {
            "name": f"baseline_theta0_{theta0:.1f}",
            "theta0": theta0,
            "seed": 1337,
        }
        grid.append(config)

    return grid


def create_inverse_single_grid() -> List[Dict[str, Any]]:
    """
    Create experiment grid for single inverse PINN experiments.

    Returns:
        list of experiment configurations
    """
    grid = []

    # Compare with/without passivity
    for use_passivity in [False, True]:
        for weight_passivity in [0.1, 1.0, 10.0] if use_passivity else [0.0]:
            config = {
                "name": f"inverse_passivity_{use_passivity}_w_{weight_passivity}",
                "use_passivity": use_passivity,
                "weight_passivity": weight_passivity,
                "seed": 1337,
            }
            grid.append(config)

    return grid


def create_inverse_ensemble_grid() -> List[Dict[str, Any]]:
    """
    Create experiment grid for ensemble experiments.

    Returns:
        list of experiment configurations
    """
    grid = []

    # Vary ensemble size
    n_models_values = [5, 10, 20]

    for n_models in n_models_values:
        for use_passivity in [False, True]:
            config = {
                "name": f"ensemble_n_{n_models}_passivity_{use_passivity}",
                "n_models": n_models,
                "use_passivity": use_passivity,
                "weight_passivity": 1.0 if use_passivity else 0.0,
                "seed": 1337,
            }
            grid.append(config)

    return grid


def create_noise_sensitivity_grid() -> List[Dict[str, Any]]:
    """
    Create experiment grid for noise sensitivity analysis.

    Returns:
        list of experiment configurations
    """
    grid = []

    noise_levels = [0.0, 0.01, 0.05, 0.1]

    for noise_std in noise_levels:
        for use_passivity in [False, True]:
            config = {
                "name": f"noise_{noise_std}_passivity_{use_passivity}",
                "noise_std": noise_std,
                "use_passivity": use_passivity,
                "weight_passivity": 1.0 if use_passivity else 0.0,
                "seed": 1337,
            }
            grid.append(config)

    return grid


def create_architecture_grid() -> List[Dict[str, Any]]:
    """
    Create experiment grid for architecture comparison.

    Returns:
        list of experiment configurations
    """
    grid = []

    architectures = {
        "shallow": [64],
        "default": [32, 32, 32],
        "deep": [32, 32, 32, 32, 32],
        "wide": [128, 128],
    }

    for arch_name, hidden_layers in architectures.items():
        config = {
            "name": f"arch_{arch_name}",
            "hidden_layers": hidden_layers,
            "seed": 1337,
        }
        grid.append(config)

    return grid


def create_robustness_grid(full: bool = False) -> List[Dict[str, Any]]:
    """
    Create experiment grid for robustness study.
    
    Grid dimensions:
    - amplitudes: [10°, 30°, 60°]
    - damping: [0.02, 0.05]
    - noise σ: [0.0, 0.01, 0.05]
    - sparsity (points): [10, 20]
    
    Args:
        full: if True, run full grid; if False, run small subset
        
    Returns:
        list of experiment configurations
    """
    import itertools
    import numpy as np
    
    if full:
        amplitudes_deg = [10.0, 30.0, 60.0]
        damping_values = [0.02, 0.05]
        noise_levels = [0.0, 0.01, 0.05]
        sparsity_values = [50, 100]  # UPDATED: was [10, 20]
    else:
        # Small subset for quick testing
        amplitudes_deg = [30.0]
        damping_values = [0.05]
        noise_levels = [0.0, 0.01]
        sparsity_values = [100]  # UPDATED: was [20]
    
    grid = []
    
    for amp, damp, noise, sparse in itertools.product(
        amplitudes_deg, damping_values, noise_levels, sparsity_values
    ):
        theta0_rad = np.radians(amp)
        
        # Deterministic inverse PINN without passivity
        config_no_pass = {
            'name': f'robust_amp{int(amp)}_d{damp:.2f}_n{noise:.2f}_s{sparse}_nopass',
            'theta0_deg': amp,
            'theta0': theta0_rad,
            'damping': damp,
            'noise': noise,
            'n_sparse': sparse,
            'use_passivity': False,
            'experiment_type': 'inverse_single',
        }
        grid.append(config_no_pass)
        
        # Deterministic inverse PINN with passivity
        config_with_pass = {
            'name': f'robust_amp{int(amp)}_d{damp:.2f}_n{noise:.2f}_s{sparse}_pass',
            'theta0_deg': amp,
            'theta0': theta0_rad,
            'damping': damp,
            'noise': noise,
            'n_sparse': sparse,
            'use_passivity': True,
            'experiment_type': 'inverse_single',
        }
        grid.append(config_with_pass)
        
        # Ensemble (with passivity)
        config_ensemble = {
            'name': f'robust_amp{int(amp)}_d{damp:.2f}_n{noise:.2f}_s{sparse}_ensemble',
            'theta0_deg': amp,
            'theta0': theta0_rad,
            'damping': damp,
            'noise': noise,
            'n_sparse': sparse,
            'use_passivity': True,
            'experiment_type': 'ensemble',
            'n_models': 5,
        }
        grid.append(config_ensemble)
    
    return grid


def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration."""
    return ExperimentConfig(name="default")


def merge_configs(base_config: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations.

    Args:
        base_config: base configuration
        override: override values

    Returns:
        merged configuration
    """
    merged = base_config.copy()
    merged.update(override)
    return merged

