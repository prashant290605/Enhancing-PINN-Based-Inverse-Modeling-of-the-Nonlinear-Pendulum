"""Configuration loading and management utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import argparse


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: path to YAML config file
        
    Returns:
        configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations, with override_config taking precedence.
    
    Args:
        base_config: base configuration
        override_config: configuration to override with
        
    Returns:
        merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def override_config_from_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Override config values from command-line arguments.
    
    Args:
        config: base configuration
        args: parsed command-line arguments
        
    Returns:
        updated configuration
    """
    # Map common CLI args to config keys
    arg_mapping = {
        'seed': ('seed',),
        'n_epochs': ('training', 'n_epochs'),
        'learning_rate': ('training', 'learning_rate'),
        'lr': ('training', 'learning_rate'),
        'theta0': ('initial_conditions', 'theta0_deg'),
        'damping': ('physics', 'damping'),
        'noise': ('data', 'noise_std'),
        'n_sparse': ('time', 'n_points_sparse'),
        'n_models': ('ensemble', 'n_models'),
        'use_passivity': ('passivity', 'use_passivity'),
        'weight_passivity': ('passivity', 'weight'),
        'device': ('device',),
        'output_dir': ('paths', 'output_dir'),
    }
    
    for arg_name, config_path in arg_mapping.items():
        if hasattr(args, arg_name):
            arg_value = getattr(args, arg_name)
            if arg_value is not None:
                # Navigate to nested config location
                current = config
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[config_path[-1]] = arg_value
    
    return config


def get_default_config() -> Dict[str, Any]:
    """
    Load default configuration.
    
    Returns:
        default configuration dictionary
    """
    default_path = Path(__file__).parent / "default.yaml"
    return load_config(default_path)


def save_config(config: Dict[str, Any], output_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: configuration dictionary
        output_path: path to save config
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def add_config_argument(parser: argparse.ArgumentParser):
    """
    Add --config argument to argument parser.
    
    Args:
        parser: argparse ArgumentParser
    """
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML config file (overrides defaults)'
    )


def load_config_with_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Load configuration with command-line overrides.
    
    This is the main function to use in experiment scripts.
    
    Args:
        args: parsed command-line arguments (must include --config)
        
    Returns:
        final configuration dictionary
    """
    # Start with default config
    config = get_default_config()
    
    # Override with custom config file if provided
    if hasattr(args, 'config') and args.config is not None:
        custom_config = load_config(args.config)
        config = merge_configs(config, custom_config)
    
    # Override with command-line arguments
    config = override_config_from_args(config, args)
    
    return config


def print_config(config: Dict[str, Any], indent: int = 0):
    """
    Pretty print configuration.
    
    Args:
        config: configuration dictionary
        indent: indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")


# Example usage functions
def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract training configuration."""
    return {
        'n_epochs': config['training']['n_epochs'],
        'learning_rate': config['training']['learning_rate'],
        'optimizer': config['training']['optimizer'],
        'scheduler': config['training']['scheduler'],
        'device': config['device'],
    }


def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model configuration."""
    return {
        'hidden_layers': config['model']['hidden_layers'],
        'activation': config['model']['activation'],
        'use_fourier': config['model']['use_fourier'],
        'num_frequencies': config['model']['num_frequencies'],
        'init_g': config['initial_params']['g'],
        'init_L': config['initial_params']['L'],
        'init_damping': config['initial_params']['damping'],
        'learn_g': config['trainable']['learn_g'],
        'learn_L': config['trainable']['learn_L'],
        'learn_damping': config['trainable']['learn_damping'],
    }


def get_loss_weights(config: Dict[str, Any]) -> Dict[str, float]:
    """Extract loss weights."""
    return {
        'lambda_data': config['loss_weights']['data'],
        'lambda_phys': config['loss_weights']['physics'],
        'lambda_ic': config['loss_weights']['ic'],
        'lambda_pass': config['loss_weights']['passivity'],
    }


def get_ensemble_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract ensemble configuration."""
    return {
        'n_models': config['ensemble']['n_models'],
        'method': config['ensemble']['method'],
        'use_bootstrap': config['ensemble']['use_bootstrap'],
        'confidence_levels': config['ensemble']['confidence_levels'],
    }

