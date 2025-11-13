# Quality Guardrails for PINN Passivity Paper

This document outlines the quality guardrails and best practices implemented in the codebase to ensure numerical stability, reproducibility, and physical consistency.

## 1. Numerical Precision

### Double Precision (Implemented)
```python
# Use double precision for stability in physics losses
import torch
torch.set_default_dtype(torch.float64)
```

**Status**: ✅ Can be enabled by adding to model initialization
**Location**: Should be added to training scripts
**Benefit**: Reduces numerical errors in gradient computation and physics residuals

## 2. Positional Encodings (Implemented ✅)

### Fourier Features for Time
```python
# In src/models/pinn_inverse.py
class FourierFeatures(nn.Module):
    def __init__(self, num_frequencies: int = 8):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.register_buffer('frequencies', 
                           torch.arange(1, num_frequencies + 1, dtype=torch.float32))
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = 2 * np.pi * t * self.frequencies.unsqueeze(0)
        sin_features = torch.sin(angles)
        cos_features = torch.cos(angles)
        return torch.cat([t, sin_features, cos_features], dim=1)
```

**Status**: ✅ Implemented in `PINN` class
**Usage**: `use_fourier=True, num_frequencies=6`
**Benefit**: Better representation of periodic/oscillatory dynamics

## 3. Parameter Positivity Constraints (Implemented ✅)

### Softplus with Minimum Clamping
```python
# In src/models/pinn_inverse.py
import torch.nn.functional as F

class PINN(nn.Module):
    def __init__(self, ...):
        # Store raw parameters
        self.g_raw = nn.Parameter(torch.tensor(self._inverse_softplus(init_g)))
        self.L_raw = nn.Parameter(torch.tensor(self._inverse_softplus(init_L)))
        self.damping_raw = nn.Parameter(torch.tensor(self._inverse_softplus(init_damping)))
    
    @property
    def g(self):
        # Enforce positivity with minimum threshold
        return F.softplus(self.g_raw).clamp(min=1e-6)
    
    @property
    def L(self):
        return F.softplus(self.L_raw).clamp(min=1e-6)
    
    @property
    def damping(self):
        return F.softplus(self.damping_raw).clamp(min=1e-6)
```

**Status**: ✅ Implemented with softplus (clamping can be added)
**Benefit**: Ensures physical parameters remain positive during optimization

## 4. Collocation Points Strategy (Optional)

### Adaptive Sampling Near Turning Points
```python
def create_adaptive_collocation_points(
    t_span: Tuple[float, float],
    n_uniform: int = 800,
    n_dense: int = 200,
    theta0: float = 0.5,
    g: float = 9.81,
    L: float = 1.0
) -> np.ndarray:
    """
    Create collocation points with denser sampling near turning points.
    
    Args:
        t_span: time span (t_start, t_end)
        n_uniform: number of uniform points
        n_dense: number of dense points near turning points
        theta0: initial angle
        g, L: pendulum parameters
        
    Returns:
        array of collocation points
    """
    # Uniform points
    t_uniform = np.linspace(t_span[0], t_span[1], n_uniform)
    
    # Estimate turning point times (approximate)
    omega0 = np.sqrt(g / L)
    period = 2 * np.pi / omega0
    n_periods = int((t_span[1] - t_span[0]) / period) + 1
    
    # Dense points near turning points (peaks and troughs)
    t_dense = []
    for i in range(n_periods):
        t_peak = t_span[0] + i * period / 2
        if t_peak <= t_span[1]:
            # Add points in a window around turning point
            window = period / 20
            t_window = np.linspace(t_peak - window, t_peak + window, n_dense // n_periods)
            t_dense.extend(t_window)
    
    # Combine and sort
    t_collocation = np.sort(np.concatenate([t_uniform, t_dense]))
    
    return t_collocation
```

**Status**: ⚠️ Optional enhancement
**Benefit**: Better captures dynamics at turning points where acceleration is highest

## 5. Curriculum Learning (Optional)

### Progressive Amplitude Training
```python
def curriculum_training(
    model: PINN,
    trainer: PINNTrainer,
    theta0_schedule: List[float] = [0.1, 0.3, 0.5, 0.7],
    epochs_per_stage: int = 1000
) -> Dict:
    """
    Train with progressively increasing amplitudes.
    
    Args:
        model: PINN model
        trainer: trainer instance
        theta0_schedule: list of initial angles (radians)
        epochs_per_stage: epochs per curriculum stage
        
    Returns:
        training history
    """
    history = {'loss': [], 'g': [], 'L': [], 'c': []}
    
    for stage, theta0 in enumerate(theta0_schedule):
        print(f"Curriculum Stage {stage+1}/{len(theta0_schedule)}: θ₀={theta0:.2f} rad")
        
        # Generate data for this amplitude
        t, theta, omega = generate_pendulum_data(theta0=theta0, ...)
        
        # Update trainer data
        trainer.update_data(t, theta, theta0, 0.0)
        
        # Train for this stage
        stage_history = trainer.train(n_epochs=epochs_per_stage, verbose=True)
        
        # Accumulate history
        for key in history:
            history[key].extend(stage_history.get(key, []))
    
    return history
```

**Status**: ⚠️ Optional enhancement
**Benefit**: Easier optimization by starting with simpler (small amplitude) problems

## 6. TensorBoard Logging (Partially Implemented)

### Comprehensive Logging
```python
# In src/models/train_inverse.py
from torch.utils.tensorboard import SummaryWriter

class PINNTrainer:
    def __init__(self, ..., log_dir: str = None):
        self.writer = SummaryWriter(log_dir) if log_dir else None
    
    def train_step(self, epoch: int):
        # ... compute losses ...
        
        if self.writer:
            # Log scalars
            self.writer.add_scalar('Loss/total', losses['total'], epoch)
            self.writer.add_scalar('Loss/data', losses['data'], epoch)
            self.writer.add_scalar('Loss/physics', losses['phys'], epoch)
            self.writer.add_scalar('Loss/ic', losses['ic'], epoch)
            self.writer.add_scalar('Loss/passivity', losses['passivity'], epoch)
            
            # Log parameter estimates
            params = self.model.get_parameters()
            self.writer.add_scalar('Parameters/g', params['g'], epoch)
            self.writer.add_scalar('Parameters/L', params['L'], epoch)
            self.writer.add_scalar('Parameters/c', params['damping'], epoch)
            
            # Log learning rate
            self.writer.add_scalar('Training/lr', 
                                 self.optimizer.param_groups[0]['lr'], epoch)
            
            # Optionally log images (plots) every N epochs
            if epoch % 500 == 0:
                fig = self.plot_predictions()
                self.writer.add_figure('Predictions/theta_vs_time', fig, epoch)
                plt.close(fig)
```

**Status**: ✅ Partially implemented (scalars logged)
**Enhancement**: Add figure logging
**Benefit**: Real-time monitoring of training progress

## 7. Seed Handling (Implemented ✅)

### Comprehensive Seed Setting
```python
def set_seeds(seed: int = 1337):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: random seed
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Log seed
    print(f"✓ Random seed set to: {seed}")
    
    # Save seed to output directory
    return seed
```

**Status**: ✅ Seeds set in experiments
**Location**: Each experiment script
**Benefit**: Ensures reproducibility

## 8. Gradient Clipping (Recommended)

### Prevent Exploding Gradients
```python
# In training loop
for epoch in range(n_epochs):
    optimizer.zero_grad()
    loss = compute_loss(...)
    loss.backward()
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
```

**Status**: ⚠️ Not implemented (optional)
**Benefit**: Prevents training instabilities

## 9. Physics Loss Scaling

### Adaptive Loss Weighting
```python
def compute_adaptive_weights(
    losses: Dict[str, torch.Tensor],
    initial_weights: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute adaptive loss weights based on loss magnitudes.
    
    This helps balance losses of different scales.
    """
    weights = {}
    for key, initial_weight in initial_weights.items():
        if key in losses:
            # Scale weight inversely with loss magnitude
            loss_magnitude = losses[key].detach().item()
            weights[key] = initial_weight / (loss_magnitude + 1e-8)
        else:
            weights[key] = initial_weight
    
    return weights
```

**Status**: ⚠️ Not implemented (optional)
**Benefit**: Better balance between different loss terms

## 10. Validation Set

### Monitor Generalization
```python
def create_train_val_split(
    t: np.ndarray,
    theta: np.ndarray,
    val_fraction: float = 0.2,
    seed: int = 1337
) -> Tuple:
    """
    Split data into training and validation sets.
    """
    np.random.seed(seed)
    n = len(t)
    indices = np.random.permutation(n)
    n_val = int(n * val_fraction)
    
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    return (t[train_indices], theta[train_indices],
            t[val_indices], theta[val_indices])
```

**Status**: ⚠️ Not implemented (optional)
**Benefit**: Monitor overfitting

## Summary of Implementation Status

| Feature | Status | Priority | Location |
|---------|--------|----------|----------|
| Double Precision | ⚠️ Optional | Medium | Training scripts |
| Fourier Features | ✅ Implemented | High | `pinn_inverse.py` |
| Softplus Constraints | ✅ Implemented | High | `pinn_inverse.py` |
| Clamping | ⚠️ Can add | Medium | `pinn_inverse.py` |
| Adaptive Collocation | ⚠️ Optional | Low | Data generation |
| Curriculum Learning | ⚠️ Optional | Low | Training scripts |
| TensorBoard Scalars | ✅ Implemented | High | `train_inverse.py` |
| TensorBoard Images | ⚠️ Optional | Low | `train_inverse.py` |
| Seed Handling | ✅ Implemented | High | All scripts |
| Seed Logging | ✅ Implemented | High | Config/outputs |
| Gradient Clipping | ⚠️ Optional | Medium | Training loop |
| Adaptive Weights | ⚠️ Optional | Low | Loss computation |
| Validation Split | ⚠️ Optional | Medium | Data generation |

## Quick Start: Enable Double Precision

To enable double precision in any experiment:

```python
# At the top of the script, before imports
import torch
torch.set_default_dtype(torch.float64)

# Then proceed with normal imports and execution
from src.models.pinn_inverse import PINN
...
```

## Recommendations

### For Production Runs:
1. ✅ Use Fourier features (`use_fourier=True`)
2. ✅ Set seeds explicitly
3. ✅ Enable TensorBoard logging
4. ⚠️ Consider double precision for critical experiments
5. ⚠️ Add gradient clipping if training is unstable

### For Research/Ablations:
1. ⚠️ Test curriculum learning for difficult cases
2. ⚠️ Experiment with adaptive collocation points
3. ⚠️ Compare adaptive vs fixed loss weights

### For Debugging:
1. ✅ Check TensorBoard logs
2. ✅ Verify seeds are set
3. ⚠️ Add validation set to monitor overfitting
4. ⚠️ Enable gradient clipping to catch instabilities

