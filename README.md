# Physics-Informed Neural Networks with Passivity Constraints

## Enhancing PINN-Based Inverse Modeling of the Nonlinear Pendulum Using Passivity Constraints and Ensemble UQ

---

### Authors

Pranav Singh · Prashant Singh · Nishit Soni · Jaskaran Singh · Ishwar Sanjay · Harshdeep

*Developed as part of MA-515 (Scientific Machine Learning), IIT Ropar.*

---

## Project Overview

This repository contains the complete implementation and analysis of **Physics-Informed Neural Networks (PINNs)** applied to inverse parameter identification in nonlinear pendulum dynamics. The project introduces **passivity constraints** to enforce thermodynamic consistency during training and employs **bootstrap ensemble methods** for rigorous uncertainty quantification.

**Central research question:** Does enforcing thermodynamic passivity constraints improve parameter identifiability and training stability in inverse modeling of nonlinear dynamical systems under sparse, noisy observations?

### Key Features

- Full PINN implementation for inverse problems (nonlinear pendulum)
- Passivity constraints enforcing non-negative energy dissipation throughout training
- 25-model bootstrap ensembles for uncertainty quantification
- Sparse velocity observations (100 measurements) with additive noise
- 20,000 training epochs per model
- Comprehensive robustness grid experiments
- Publication-ready LaTeX report (~40 pages)

---

## Scientific Insight

The most significant findings of this work are not the parameter accuracy results — they are the structural insights revealed by the experimental pipeline:

- **Passivity constraints stabilize conservative parameter estimates** (g, L), reducing trajectory energy drift by ~71% relative to unconstrained PINN.
- **Damping coefficient estimation remains fundamentally ill-posed** under sparse velocity observations. Errors of 700–2100% persist regardless of constraint formulation — a structural limitation of the observation regime, not a modeling failure.
- **Ensemble UQ is severely miscalibrated** (8.7% empirical coverage vs. 95% nominal), revealing that bootstrap diversity alone is insufficient when systematic bias dominates.
- **Bias >> Variance** — the dominant source of uncertainty is not stochastic but structural, indicating the inverse problem requires richer observational data or stronger priors to be well-conditioned.

---

## Repository Structure

```
pinn_passivity_paper/
├── src/
│   ├── data/
│   │   ├── generator.py          # Data generation (analytical + nonlinear solvers)
│   │   └── utils.py              # Time grids, noise, batching
│   ├── baseline/
│   │   ├── linear_small_angle.py # Analytical solutions
│   │   ├── nonlinear_rk.py       # RK4 and solve_ivp
│   │   └── plots_baseline.py     # Baseline plotting
│   ├── models/
│   │   ├── pinn_inverse.py       # PINN architecture with Fourier features
│   │   ├── losses.py             # Physics, IC, passivity losses
│   │   ├── train_inverse.py      # Training loop with TensorBoard
│   │   ├── dissipation_net.py    # NN for nonparametric damping
│   │   └── ensemble.py           # Bootstrap ensemble implementation
│   ├── analysis/
│   │   ├── metrics.py            # RMSE, energy drift, coverage, ECE
│   │   └── tables_figs.py        # Figure/table generators
│   ├── experiments/
│   │   ├── exp_baseline.py       # Baseline experiments
│   │   ├── exp_inverse_single.py # Single PINN runs
│   │   ├── exp_inverse_ens.py    # Ensemble experiments
│   │   └── grids.py              # Robustness study grids
│   ├── configs/
│   │   ├── default.yaml          # All hyperparameters
│   │   └── config_loader.py      # Config management
│   └── viz/
│       └── style.py              # Publication-quality plotting
├── scripts/
│   ├── run_all.sh                # Full pipeline orchestration
│   └── generate_final_study.py  # Automated analysis report
├── tests/
│   ├── test_data.py
│   ├── test_losses.py
│   ├── test_models.py
│   └── test_metrics.py
├── outputs/
│   ├── baseline/
│   ├── inverse_single/
│   ├── ensemble/
│   ├── summaries/
│   ├── FINAL_STUDY.md
│   └── *.csv, *.png, *.json
├── report/
│   ├── COMPLETE_REPORT.tex
│   ├── figures/
│   └── *.md
├── requirements.txt
├── pyproject.toml
├── Makefile
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Required packages:** `torch`, `numpy`, `scipy`, `matplotlib`, `pandas`, `tqdm`, `tensorboard`, `pyyaml`, `pytest`, `ruff`, `black`

### 2. Run the Full Pipeline

```bash
# Full pipeline
bash scripts/run_all.sh --full

# Or step-by-step
make setup
make test
make run-baseline
make run-inverse
make run-ensemble
```

### 3. View Results

```bash
tensorboard --logdir outputs/
cat outputs/FINAL_STUDY.md
```

---

## Experiments

**Baseline**
```bash
python -m src.experiments.exp_baseline
```

**Single Inverse PINN**
```bash
python -m src.experiments.exp_inverse_single \
    --n-epochs 20000 \
    --n-sparse 100 \
    --noise 0.01 \
    --use-velocity-obs true
```

**Ensemble UQ**
```bash
python -m src.experiments.exp_inverse_ens \
    --n-models 25 \
    --n-epochs 20000 \
    --use-passivity true
```

**Robustness Grid**
```bash
python -m src.experiments.run_grid --full
```

---

## Results

### Parameter Estimation (Noisy Case, σ = 0.01)

| Method | g error | L error | c error | Trajectory RMSE | Energy Drift |
|--------|---------|---------|---------|-----------------|--------------|
| Standard PINN | 0.04% | 22.9% | 1032% | 0.327 | 0.00199 |
| Passivity PINN | 2.12% | **9.06%** | 696% | 0.327 | **0.00058** |
| Ensemble (25 models) | **1.44%** | 13.3% | 768% | 0.329 | 0.00061 |

Passivity constraints reduce energy drift by ~71% and improve length estimation, but damping identification remains catastrophically ill-posed across all methods — a consequence of the fundamental observability structure of the problem, not a training artifact.

---

## Configuration

Edit `configs/default.yaml` to modify the experimental setup:

```yaml
physics:
  g: 9.81
  L: 1.0
  c: 0.05

time:
  t_start: 0.0
  t_end: 10.0
  n_points_dense: 1000
  n_points_sparse: 100

model:
  hidden_dims: [32, 32, 32]
  activation: tanh
  n_fourier_features: 6

training:
  n_epochs: 20000
  lr: 0.001
  optimizer: adam

loss_weights:
  data: 1.0
  velocity: 1.0
  physics: 10.0
  ic: 1.0
  passivity: 1.0

ensemble:
  n_models: 25
  bootstrap: true
```

---

## Testing

```bash
pytest                          # Full suite
pytest tests/test_losses.py -v  # Specific module
pytest --cov=src tests/         # With coverage
```

---

## Report

The `report/` directory contains the complete LaTeX source (~40 pages, 20+ figures, 15+ tables, 13 references).

```bash
cd report/
pdflatex COMPLETE_REPORT.tex
pdflatex COMPLETE_REPORT.tex  # Run twice for cross-references
```

Alternatively, upload `COMPLETE_REPORT.tex` and `figures/` to Overleaf.

---

## Reproducibility

All experiments use a fixed seed of `1337`, deterministic algorithms where available, and logged hyperparameters with saved checkpoints.

```bash
export PYTHONHASHSEED=1337
bash scripts/run_all.sh --full
```

---

## Code Quality

```bash
black src/ tests/       # Format
ruff check src/ tests/  # Lint
mypy src/               # Type check (optional)
pre-commit install       # Install hooks
```

---

## Citation

```bibtex
@techreport{singh2024pinn,
  title={Enhancing PINN-Based Inverse Modeling of the Nonlinear Pendulum 
         Using Passivity Constraints and Ensemble UQ},
  author={Singh, Pranav and Singh, Prashant and Soni, Nishit and 
          Singh, Jaskaran and Sanjay, Ishwar and Harshdeep},
  institution={Indian Institute of Technology Ropar},
  year={2024},
  type={Course Project Report},
  number={MA-515}
}
```

---

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). Physics-informed neural networks. *Journal of Computational Physics.*
2. Karniadakis, G.E. et al. (2021). Physics-informed machine learning. *Nature Reviews Physics.*
3. Wang, S. et al. (2021). Understanding and mitigating gradient pathologies in PINNs. *SIAM Journal on Scientific Computing.*
4. Yang, L. et al. (2021). B-PINNs: Bayesian physics-informed neural networks. *Journal of Computational Physics.*

---

## License

Academic coursework — IIT Ropar, MA-515. Open for reference and adaptation with attribution.
