"""Trainer for inverse PINN models."""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Callable, Any
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json

from .pinn_inverse import PINN
from .losses import LossComputer, compute_derivatives, loss_data, loss_velocity, loss_ic, loss_phys, loss_passivity
from .dissipation_net import DissipationNet


class PINNTrainer:
    """Trainer for PINN inverse problems."""

    def __init__(
        self,
        model: PINN,
        t_obs: torch.Tensor,
        theta_obs: torch.Tensor,
        t_collocation: torch.Tensor,
        theta0: float,
        omega0: float,
        omega_obs: Optional[torch.Tensor] = None,
        lambda_data: float = 1.0,
        lambda_phys: float = 10.0,
        lambda_ic: float = 1.0,
        lambda_pass: float = 1.0,
        lambda_vel: float = 1.0,
        use_velocity_obs: bool = True,
        dissipation_net: Optional[DissipationNet] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cpu",
        log_dir: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: PINN model
            t_obs: observation times [M, 1]
            theta_obs: observed angles [M, 1]
            t_collocation: full time grid for physics collocation [N, 1]
            theta0: initial angle
            omega0: initial angular velocity
            omega_obs: observed angular velocities [M, 1] (optional)
            lambda_data: weight for data loss (default: 1.0)
            lambda_phys: weight for physics loss (default: 10.0)
            lambda_ic: weight for IC loss (default: 1.0)
            lambda_pass: weight for passivity loss (default: 1.0)
            lambda_vel: weight for velocity loss (default: 1.0)
            use_velocity_obs: whether to use velocity observations (default: True)
            dissipation_net: optional neural network for D(θ, θ̇) (default: None, use viscous)
            optimizer: optimizer (default: Adam with lr=1e-3)
            scheduler: learning rate scheduler (default: cosine decay)
            device: device to train on
            log_dir: directory for tensorboard logs
            checkpoint_dir: directory for saving checkpoints
        """
        self.model = model.to(device)
        self.dissipation_net = dissipation_net.to(device) if dissipation_net is not None else None
        self.device = device
        
        # Store data
        self.t_obs = t_obs.to(device)
        self.theta_obs = theta_obs.to(device)
        self.omega_obs = omega_obs.to(device) if omega_obs is not None else None
        self.t_collocation = t_collocation.to(device).requires_grad_(True)
        self.theta0 = theta0
        self.omega0 = omega0
        
        # Loss weights
        self.lambda_data = lambda_data
        self.lambda_phys = lambda_phys
        self.lambda_ic = lambda_ic
        self.lambda_pass = lambda_pass
        self.lambda_vel = lambda_vel
        self.use_velocity_obs = use_velocity_obs and (omega_obs is not None)

        # Optimizer (include dissipation net parameters if present)
        if optimizer is None:
            params = list(self.model.parameters())
            if self.dissipation_net is not None:
                params += list(self.dissipation_net.parameters())
            self.optimizer = optim.Adam(params, lr=1e-3)
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

        # Logging
        self.writer = None
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        
        # Checkpointing
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_loss = float('inf')
        self.best_epoch = 0

        self.history = {
            "loss": [],
            "loss_physics": [],
            "loss_data": [],
            "loss_ic": [],
            "loss_passivity": [],
            "g": [],
            "L": [],
            "damping": [],
        }

    def train_step(self) -> Dict[str, float]:
        """
        Single training step.

        Returns:
            dictionary of loss values
        """
        self.optimizer.zero_grad()

        # Forward pass on collocation points
        theta_colloc = self.model(self.t_collocation)
        theta_dot, theta_ddot = compute_derivatives(theta_colloc, self.t_collocation)
        
        # Forward pass on observation points
        theta_pred_obs = self.model(self.t_obs)
        
        # Data loss
        l_data = loss_data(theta_pred_obs, self.theta_obs)
        
        # Velocity loss (if enabled)
        if self.use_velocity_obs:
            # Compute velocity at observation points
            t_obs_grad = self.t_obs.clone().detach().requires_grad_(True)
            theta_obs_pred = self.model(t_obs_grad)
            omega_obs_pred, _ = compute_derivatives(theta_obs_pred, t_obs_grad)
            l_vel = loss_velocity(omega_obs_pred, self.omega_obs)
        else:
            l_vel = torch.tensor(0.0, device=self.device)
        
        # Physics loss (use collocation points)
        # Compute dissipation: either from NN or from viscous damping
        if self.dissipation_net is not None:
            # Use neural network dissipation
            D = self.dissipation_net(theta_colloc, theta_dot)
        elif self.model.learn_damping:
            # Use viscous damping
            D = self.model.damping * theta_dot
        else:
            D = None
        
        l_phys = loss_phys(theta_ddot, theta_colloc, theta_dot, self.model.g, self.model.L, D)
        
        # Initial condition loss
        t0 = self.t_collocation[0:1]
        theta_hat0 = self.model(t0)
        theta_dot0_tensor, _ = compute_derivatives(theta_hat0, t0)
        l_ic = loss_ic(theta_hat0, theta_dot0_tensor, self.theta0, self.omega0)
        
        # Passivity loss
        l_pass = loss_passivity(theta_colloc, theta_dot, theta_ddot, self.model.g, self.model.L)
        
        # Total loss
        total = (
            self.lambda_data * l_data +
            self.lambda_vel * l_vel +
            self.lambda_phys * l_phys +
            self.lambda_ic * l_ic +
            self.lambda_pass * l_pass
        )

        # Backward pass
        total.backward()
        self.optimizer.step()

        # Convert to float for logging
        loss_dict = {
            'data': l_data.item(),
            'velocity': l_vel.item() if self.use_velocity_obs else 0.0,
            'physics': l_phys.item(),
            'ic': l_ic.item(),
            'passivity': l_pass.item(),
            'total': total.item(),
        }

        return loss_dict

    def train(
        self,
        n_epochs: int = 10000,
        log_interval: int = 100,
        verbose: bool = True,
        save_best: bool = True,
    ) -> Dict[str, list]:
        """
        Train the model.

        Args:
            n_epochs: number of training epochs (5k-20k configurable)
            log_interval: logging interval
            verbose: whether to show progress bar
            save_best: whether to save best checkpoint

        Returns:
            training history
        """
        iterator = range(n_epochs)
        if verbose:
            iterator = tqdm(iterator, desc="Training PINN")

        for epoch in iterator:
            loss_dict = self.train_step()

            # Update history
            self.history["loss"].append(loss_dict["total"])
            self.history["loss_physics"].append(loss_dict["physics"])
            self.history["loss_data"].append(loss_dict["data"])
            self.history["loss_ic"].append(loss_dict["ic"])
            self.history["loss_passivity"].append(loss_dict["passivity"])

            # Log parameters
            params = self.model.get_parameters()
            self.history["g"].append(params["g"])
            self.history["L"].append(params["L"])
            self.history["damping"].append(params["damping"])

            # Tensorboard logging
            if self.writer is not None and epoch % log_interval == 0:
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f"loss/{k}", v, epoch)

                for k, v in params.items():
                    self.writer.add_scalar(f"params/{k}", v, epoch)
                
                # Log learning rate
                if self.scheduler is not None:
                    self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], epoch)

            # Update progress bar
            if verbose and epoch % log_interval == 0:
                iterator.set_postfix(
                    {
                        "loss": f"{loss_dict['total']:.4e}",
                        "g": f"{params['g']:.3f}",
                        "L": f"{params['L']:.3f}",
                        "c": f"{params['damping']:.4f}",
                    }
                )

            # Save best checkpoint
            if save_best and loss_dict['total'] < self.best_loss:
                self.best_loss = loss_dict['total']
                self.best_epoch = epoch
                if self.checkpoint_dir is not None:
                    self.save_checkpoint(Path(self.checkpoint_dir) / "best_model.pt")

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

        if self.writer is not None:
            self.writer.close()

        return self.history

    def predict(self, t: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions.

        Args:
            t: time inputs [N, 1]

        Returns:
            theta: angle predictions
            theta_dot: angular velocity predictions
            theta_ddot: angular acceleration predictions
        """
        self.model.eval()

        with torch.no_grad():
            t_tensor = t.to(self.device)
            t_tensor.requires_grad_(True)

            theta, theta_dot, theta_ddot = self.model.predict_with_derivatives(t_tensor)

            theta = theta.cpu().numpy()
            theta_dot = theta_dot.cpu().numpy()
            theta_ddot = theta_ddot.cpu().numpy()

        self.model.train()

        return theta, theta_dot, theta_ddot

    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "parameters": self.model.get_parameters(),
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]
        self.best_loss = checkpoint.get("best_loss", float('inf'))
        self.best_epoch = checkpoint.get("best_epoch", 0)

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    def save_results(self, path: Path, true_params: Optional[Dict[str, float]] = None):
        """
        Save final estimates and errors to JSON.
        
        Args:
            path: path to save JSON file
            true_params: dictionary of true parameter values (for error computation)
        """
        params = self.model.get_parameters()
        
        results = {
            "final_parameters": params,
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "final_loss": self.history["loss"][-1] if self.history["loss"] else None,
        }
        
        # Compute errors if true parameters provided
        if true_params is not None:
            errors = {}
            for key in ["g", "L", "damping"]:
                if key in true_params and key in params:
                    true_val = true_params[key]
                    pred_val = params[key]
                    abs_error = abs(pred_val - true_val)
                    rel_error = abs_error / true_val if true_val != 0 else float('inf')
                    errors[f"{key}_abs_error"] = abs_error
                    errors[f"{key}_rel_error"] = rel_error
                    errors[f"{key}_rel_error_pct"] = rel_error * 100
            
            results["errors"] = errors
            results["true_parameters"] = true_params
        
        # Save to JSON
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


def create_trainer(
    model: PINN,
    t_obs: torch.Tensor,
    theta_obs: torch.Tensor,
    t_collocation: torch.Tensor,
    theta0: float,
    omega0: float,
    omega_obs: Optional[torch.Tensor] = None,
    lambda_data: float = 1.0,
    lambda_phys: float = 10.0,
    lambda_ic: float = 1.0,
    lambda_pass: float = 1.0,
    lambda_vel: float = 1.0,
    use_velocity_obs: bool = True,
    dissipation_net: Optional[DissipationNet] = None,
    learning_rate: float = 1e-3,
    optimizer_type: str = "adam",
    scheduler_type: Optional[str] = "cosine",
    n_epochs: int = 10000,
    device: str = "cpu",
    log_dir: Optional[Path] = None,
    checkpoint_dir: Optional[Path] = None,
) -> PINNTrainer:
    """
    Factory function to create trainer.

    Args:
        model: PINN model
        t_obs: observation times [M, 1]
        theta_obs: observed angles [M, 1]
        t_collocation: full time grid for physics collocation [N, 1]
        theta0: initial angle
        omega0: initial angular velocity
        omega_obs: observed angular velocities [M, 1] (optional)
        lambda_data: weight for data loss (default: 1.0)
        lambda_phys: weight for physics loss (default: 10.0)
        lambda_ic: weight for IC loss (default: 1.0)
        lambda_pass: weight for passivity loss (default: 1.0)
        lambda_vel: weight for velocity loss (default: 1.0)
        use_velocity_obs: whether to use velocity observations (default: True)
        dissipation_net: optional neural network for D(θ, θ̇) (default: None)
        learning_rate: learning rate (default: 1e-3)
        optimizer_type: 'adam', 'sgd', 'lbfgs'
        scheduler_type: 'step', 'exponential', 'cosine', None (default: 'cosine')
        n_epochs: number of epochs for cosine scheduler (default: 10000)
        device: device
        log_dir: log directory
        checkpoint_dir: checkpoint directory

    Returns:
        PINNTrainer
    """
    # Collect parameters from model and dissipation net
    params = list(model.parameters())
    if dissipation_net is not None:
        params += list(dissipation_net.parameters())
    
    # Create optimizer
    if optimizer_type == "adam":
        optimizer = optim.Adam(params, lr=learning_rate)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9)
    elif optimizer_type == "lbfgs":
        optimizer = optim.LBFGS(params, lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    # Create scheduler
    scheduler = None
    if scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    elif scheduler_type == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    elif scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    return PINNTrainer(
        model=model,
        t_obs=t_obs,
        theta_obs=theta_obs,
        omega_obs=omega_obs,
        t_collocation=t_collocation,
        theta0=theta0,
        omega0=omega0,
        lambda_data=lambda_data,
        lambda_phys=lambda_phys,
        lambda_ic=lambda_ic,
        lambda_pass=lambda_pass,
        lambda_vel=lambda_vel,
        use_velocity_obs=use_velocity_obs,
        dissipation_net=dissipation_net,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
    )

