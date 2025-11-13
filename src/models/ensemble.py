"""Ensemble methods for uncertainty quantification."""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

from .pinn_inverse import PINN, create_pinn
from .train_inverse import PINNTrainer, create_trainer


class PINNEnsemble:
    """Ensemble of PINN models for uncertainty quantification."""

    def __init__(
        self,
        models: List[PINN],
        trainers: Optional[List[PINNTrainer]] = None,
    ):
        """
        Initialize ensemble.

        Args:
            models: list of PINN models
            trainers: optional list of trainers
        """
        self.models = models
        self.trainers = trainers
        self.n_models = len(models)
        self.histories = []

    def train_all(
        self,
        n_epochs: int = 10000,
        verbose: bool = True,
    ) -> List[Dict[str, list]]:
        """
        Train all models in ensemble.

        Args:
            n_epochs: number of epochs
            verbose: show progress

        Returns:
            list of training histories
        """
        if self.trainers is None:
            raise ValueError("Trainers not provided")

        self.histories = []

        for i, trainer in enumerate(self.trainers):
            if verbose:
                print(f"\n{'='*80}")
                print(f"Training Ensemble Model {i+1}/{self.n_models}")
                print('='*80)

            history = trainer.train(n_epochs=n_epochs, verbose=verbose, save_best=False)
            self.histories.append(history)

        return self.histories

    def predict(
        self, t: np.ndarray, return_std: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Make ensemble predictions.

        Args:
            t: time inputs [N,]
            return_std: whether to return standard deviations

        Returns:
            dictionary with mean and std predictions
        """
        predictions = []

        for model in self.models:
            model.eval()
            
            # Need gradients enabled for derivative computation
            t_torch = torch.tensor(t, dtype=torch.float32).unsqueeze(-1).requires_grad_(True)
            theta_torch = model(t_torch)
            
            # Compute derivatives
            from .losses import compute_derivatives
            theta_dot_torch, theta_ddot_torch = compute_derivatives(theta_torch, t_torch)
            
            theta = theta_torch.detach().cpu().numpy().flatten()
            theta_dot = theta_dot_torch.detach().cpu().numpy().flatten()
            theta_ddot = theta_ddot_torch.detach().cpu().numpy().flatten()
                
            predictions.append({
                "theta": theta,
                "theta_dot": theta_dot,
                "theta_ddot": theta_ddot,
            })

        # Compute statistics
        theta_all = np.array([p["theta"] for p in predictions])
        theta_dot_all = np.array([p["theta_dot"] for p in predictions])
        theta_ddot_all = np.array([p["theta_ddot"] for p in predictions])

        result = {
            "theta_mean": np.mean(theta_all, axis=0),
            "theta_dot_mean": np.mean(theta_dot_all, axis=0),
            "theta_ddot_mean": np.mean(theta_ddot_all, axis=0),
        }

        if return_std:
            result["theta_std"] = np.std(theta_all, axis=0)
            result["theta_dot_std"] = np.std(theta_dot_all, axis=0)
            result["theta_ddot_std"] = np.std(theta_ddot_all, axis=0)

        # Store all predictions for further analysis
        result["theta_all"] = theta_all
        result["theta_dot_all"] = theta_dot_all
        result["theta_ddot_all"] = theta_ddot_all

        return result

    def get_parameter_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics of learned parameters across ensemble.

        Returns:
            dictionary with mean and std for each parameter
        """
        params_list = [model.get_parameters() for model in self.models]

        g_values = [p["g"] for p in params_list]
        L_values = [p["L"] for p in params_list]
        damping_values = [p["damping"] for p in params_list]

        return {
            "g": {"mean": np.mean(g_values), "std": np.std(g_values), "values": g_values},
            "L": {"mean": np.mean(L_values), "std": np.std(L_values), "values": L_values},
            "damping": {
                "mean": np.mean(damping_values),
                "std": np.std(damping_values),
                "values": damping_values,
            },
        }

    def save_ensemble(self, directory: Path):
        """Save all models in ensemble."""
        directory.mkdir(parents=True, exist_ok=True)

        for i, trainer in enumerate(self.trainers):
            path = directory / f"model_{i}.pt"
            trainer.save_checkpoint(path)

    def load_ensemble(self, directory: Path):
        """Load all models in ensemble."""
        for i, trainer in enumerate(self.trainers):
            path = directory / f"model_{i}.pt"
            trainer.load_checkpoint(path)


def create_bootstrap_ensemble(
    n_models: int,
    t_obs: np.ndarray,
    theta_obs: np.ndarray,
    t_collocation: np.ndarray,
    theta0: float,
    omega0: float,
    model_config: Dict,
    trainer_config: Dict,
    omega_obs: Optional[np.ndarray] = None,
    seed: int = 42,
    use_bootstrap: bool = True,
) -> PINNEnsemble:
    """
    Create ensemble using bootstrap sampling or different seeds.

    Args:
        n_models: number of models in ensemble
        t_obs: observation times
        theta_obs: observed angles
        t_collocation: collocation points
        theta0: initial angle
        omega0: initial angular velocity
        model_config: configuration for PINN models
        trainer_config: configuration for trainers
        omega_obs: observed angular velocities (optional)
        seed: random seed
        use_bootstrap: whether to use bootstrap sampling

    Returns:
        PINNEnsemble
    """
    models = []
    trainers = []

    for i in range(n_models):
        # Set seed for this model
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)
        
        # Bootstrap sample if requested
        if use_bootstrap:
            n_samples = len(t_obs)
            indices = np.random.choice(n_samples, n_samples, replace=True)
            t_obs_i = t_obs[indices]
            theta_obs_i = theta_obs[indices]
            omega_obs_i = omega_obs[indices] if omega_obs is not None else None
        else:
            t_obs_i = t_obs
            theta_obs_i = theta_obs
            omega_obs_i = omega_obs

        # Convert to torch
        t_obs_torch = torch.tensor(t_obs_i, dtype=torch.float32).unsqueeze(-1)
        theta_obs_torch = torch.tensor(theta_obs_i, dtype=torch.float32).unsqueeze(-1)
        omega_obs_torch = torch.tensor(omega_obs_i, dtype=torch.float32).unsqueeze(-1) if omega_obs_i is not None else None
        t_colloc_torch = torch.tensor(t_collocation, dtype=torch.float32).unsqueeze(-1)

        # Create model
        model = create_pinn(**model_config)

        # Create trainer with this model's data
        trainer = create_trainer(
            model=model,
            t_obs=t_obs_torch,
            theta_obs=theta_obs_torch,
            omega_obs=omega_obs_torch,
            t_collocation=t_colloc_torch,
            theta0=theta0,
            omega0=omega0,
            **trainer_config
        )

        models.append(model)
        trainers.append(trainer)

    return PINNEnsemble(models=models, trainers=trainers)


def compute_prediction_intervals(
    predictions: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute prediction intervals from ensemble predictions.

    Args:
        predictions: array of predictions [n_models, n_points, ...]
        confidence: confidence level

    Returns:
        lower_bound: lower confidence bound
        upper_bound: upper confidence bound
    """
    alpha = 1 - confidence
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)

    lower_bound = np.percentile(predictions, lower_percentile, axis=0)
    upper_bound = np.percentile(predictions, upper_percentile, axis=0)

    return lower_bound, upper_bound


def compute_coverage(
    predictions: np.ndarray,
    true_values: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Compute empirical coverage of prediction intervals.

    Args:
        predictions: array of predictions [n_models, n_points, ...]
        true_values: true values [n_points, ...]
        confidence: confidence level

    Returns:
        coverage: fraction of true values within prediction intervals
    """
    lower_bound, upper_bound = compute_prediction_intervals(predictions, confidence)

    # Check if true values are within bounds
    within_bounds = (true_values >= lower_bound) & (true_values <= upper_bound)

    coverage = np.mean(within_bounds)

    return coverage

