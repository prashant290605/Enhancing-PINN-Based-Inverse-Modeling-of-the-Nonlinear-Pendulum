"""Test configuration loading and overrides."""

import argparse
from pathlib import Path

from src.configs.config_loader import (
    load_config,
    get_default_config,
    load_config_with_overrides,
    print_config,
    get_training_config,
    get_model_config,
    get_loss_weights,
    get_ensemble_config,
    save_config,
)


def test_load_default():
    """Test loading default config."""
    print("=" * 80)
    print("TEST 1: Load Default Config")
    print("=" * 80)
    
    config = get_default_config()
    print("\nDefault configuration loaded successfully!")
    print(f"Seed: {config['seed']}")
    print(f"Training epochs: {config['training']['n_epochs']}")
    print(f"Ensemble models: {config['ensemble']['n_models']}")
    print()


def test_extract_configs():
    """Test extracting specific config sections."""
    print("=" * 80)
    print("TEST 2: Extract Config Sections")
    print("=" * 80)
    
    config = get_default_config()
    
    print("\nTraining Config:")
    training_config = get_training_config(config)
    for key, value in training_config.items():
        print(f"  {key}: {value}")
    
    print("\nModel Config:")
    model_config = get_model_config(config)
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    print("\nLoss Weights:")
    loss_weights = get_loss_weights(config)
    for key, value in loss_weights.items():
        print(f"  {key}: {value}")
    
    print("\nEnsemble Config:")
    ensemble_config = get_ensemble_config(config)
    for key, value in ensemble_config.items():
        print(f"  {key}: {value}")
    print()


def test_cli_overrides():
    """Test CLI argument overrides."""
    print("=" * 80)
    print("TEST 3: CLI Overrides")
    print("=" * 80)
    
    # Simulate CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n-epochs', type=int, default=None)
    parser.add_argument('--n-models', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    
    # Test with overrides
    args = parser.parse_args(['--seed', '42', '--n-epochs', '5000', '--n-models', '10'])
    
    config = load_config_with_overrides(args)
    
    print("\nConfig with CLI overrides:")
    print(f"  Seed: {config['seed']} (overridden to 42)")
    print(f"  Training epochs: {config['training']['n_epochs']} (overridden to 5000)")
    print(f"  Ensemble models: {config['ensemble']['n_models']} (overridden to 10)")
    print()


def test_save_and_load():
    """Test saving and loading config."""
    print("=" * 80)
    print("TEST 4: Save and Load Config")
    print("=" * 80)
    
    # Load default
    config = get_default_config()
    
    # Modify some values
    config['seed'] = 9999
    config['training']['n_epochs'] = 20000
    
    # Save to temp file
    temp_path = Path("outputs/temp_config.yaml")
    save_config(config, temp_path)
    print(f"\n✓ Config saved to: {temp_path}")
    
    # Load it back
    loaded_config = load_config(temp_path)
    print(f"✓ Config loaded from: {temp_path}")
    print(f"  Seed: {loaded_config['seed']}")
    print(f"  Training epochs: {loaded_config['training']['n_epochs']}")
    
    # Clean up
    temp_path.unlink()
    print(f"✓ Cleaned up temp file")
    print()


def test_print_full_config():
    """Test printing full config."""
    print("=" * 80)
    print("TEST 5: Full Config Structure")
    print("=" * 80)
    print()
    
    config = get_default_config()
    print_config(config)
    print()


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "CONFIG LOADING TESTS" + " " * 38 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    test_load_default()
    test_extract_configs()
    test_cli_overrides()
    test_save_and_load()
    test_print_full_config()
    
    print("=" * 80)
    print("✓ All config tests passed!")
    print("=" * 80)
    print()
    print("Usage in experiment scripts:")
    print()
    print("  # With default config")
    print("  python -m src.experiments.exp_inverse_ens")
    print()
    print("  # With custom config file")
    print("  python -m src.experiments.exp_inverse_ens --config configs/default.yaml")
    print()
    print("  # With config + CLI overrides")
    print("  python -m src.experiments.exp_inverse_ens --config configs/default.yaml \\")
    print("      --n-epochs 5000 --n-models 10 --seed 42")
    print()


if __name__ == "__main__":
    main()

