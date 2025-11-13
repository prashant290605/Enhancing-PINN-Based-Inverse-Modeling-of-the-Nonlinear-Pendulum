# Physics-Informed Neural Networks with Passivity Constraints

## Enhancing PINN-Based Inverse Modeling of the Nonlinear Pendulum Using Passivity Constraints and Ensemble UQ

**MA-515 Course Project | IIT Ropar**

### Authors
- Pranav Singh (2023MCB1308)
- Prashant Singh (2023MCB1309)
- Nishit Soni (2023MCB1304)
- Jaskaran Singh (2023MCB1297)
- Ishwar Sanjay (2023MCB1000)
- Harshdeep (2023MCB1200)

---

## 📋 Project Overview

This repository contains the complete implementation and analysis of **Physics-Informed Neural Networks (PINNs)** for inverse parameter identification in nonlinear pendulum dynamics, with novel **passivity constraints** for thermodynamic consistency and **bootstrap ensemble methods** for uncertainty quantification.

### Key Features:
- ✅ Full PINN implementation for inverse problems
- ✅ Passivity constraints enforcing energy dissipation
- ✅ 25-model bootstrap ensembles for UQ
- ✅ Sparse velocity observations (100 measurements)
- ✅ 20,000 training epochs per model
- ✅ Comprehensive experimental pipeline
- ✅ Publication-ready LaTeX report (~40 pages)

---

## 🗂️ Repository Structure

```
pinn_passivity_paper/
├── src/                          # Main source code
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
│   ├── test_data.py              # Data generation tests
│   ├── test_losses.py            # Loss function tests
│   ├── test_models.py            # PINN model tests
│   └── test_metrics.py           # Metrics tests
├── outputs/                      # Generated results
│   ├── baseline/                 # Baseline figures & metrics
│   ├── inverse_single/           # Single PINN results
│   ├── ensemble/                 # Ensemble UQ results
│   ├── summaries/                # Aggregated results
│   ├── FINAL_STUDY.md           # Comprehensive analysis
│   └── *.csv, *.png, *.json     # All experimental data
├── report/                       # LaTeX report
│   ├── COMPLETE_REPORT.tex       # Full report source
│   ├── figures/                  # Report figures
│   └── *.md                      # Documentation
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Poetry config (optional)
├── Makefile                     # Build targets
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

**Required packages:**
- `torch` (PyTorch)
- `numpy`, `scipy`, `matplotlib`, `pandas`
- `tqdm`, `tensorboard`, `pyyaml`
- `pytest`, `ruff`, `black`

### 2. Run the Full Pipeline

```bash
# Run complete experimental pipeline
bash scripts/run_all.sh --full

# Or step-by-step:
make setup        # Setup environment
make test         # Run tests
make run-baseline # Baseline experiments
make run-inverse  # Single inverse PINNs
make run-ensemble # Ensemble UQ experiments
```

### 3. View Results

```bash
# TensorBoard logs
tensorboard --logdir outputs/

# Generated analysis
cat outputs/FINAL_STUDY.md

# Figures
open outputs/analysis/*.png
```

---

## 🧪 Experiments

### Baseline Experiments
```bash
python -m src.experiments.exp_baseline
```
Generates analytical vs. numerical comparisons, validates solvers.

### Single Inverse PINN
```bash
python -m src.experiments.exp_inverse_single \
    --n-epochs 20000 \
    --n-sparse 100 \
    --noise 0.01 \
    --use-velocity-obs true
```

### Ensemble UQ
```bash
python -m src.experiments.exp_inverse_ens \
    --n-models 25 \
    --n-epochs 20000 \
    --use-passivity true
```

### Robustness Grid
```bash
python -m src.experiments.run_grid --full
```

---

## 📊 Key Results

### Parameter Estimation (Noisy Case, σ=0.01)

| Method | g error | L error | c error | Trajectory RMSE | Energy Drift |
|--------|---------|---------|---------|-----------------|--------------|
| **Standard PINN** | 0.04% | 22.9% | **1032%** | 0.327 | 0.00199 |
| **Passivity PINN** | 2.12% | **9.06%** | 696% | 0.327 | **0.00058** |
| **Ensemble (25)** | **1.44%** | 13.3% | 768% | 0.329 | 0.00061 |

### Key Findings:
- ✅ **Passivity improves conservative parameters** (g, L) by stabilizing estimates
- ❌ **Damping catastrophic** (700-2100% errors) - fundamentally ill-posed
- ⚠️ **Ensemble UQ severely miscalibrated** (8.7% coverage vs. 95% target)
- 📊 **Bias >> Variance** - systematic errors dominate

---

## 🔬 Configuration

Edit `configs/default.yaml`:

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

## 📈 Visualization

All plots are saved to `outputs/` with publication-quality formatting:

### Generated Figures:
- Trajectory comparisons (predicted vs. ground truth)
- Parameter evolution during training
- Energy dissipation plots
- Ensemble uncertainty bands
- Parameter distribution histograms
- Coverage reliability diagrams
- Grid robustness comparisons

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Specific test modules
pytest tests/test_data.py -v
pytest tests/test_losses.py -v
pytest tests/test_models.py -v

# With coverage
pytest --cov=src tests/
```

---

## 📝 Report

The `report/` folder contains the complete LaTeX source:

```bash
cd report/

# Compile PDF
pdflatex COMPLETE_REPORT.tex
pdflatex COMPLETE_REPORT.tex

# Or use Overleaf
# Upload COMPLETE_REPORT.tex and figures/ folder
```

**Report Contents:**
- 40+ pages of detailed analysis
- 20+ experimental figures
- 15+ result tables
- 13 embedded references
- Mathematical derivations
- Comprehensive discussion

---

## 🎯 Reproducibility

**Seed:** `1337` (fixed throughout)

All experiments use:
- Fixed random seeds
- Deterministic algorithms where possible
- Logged hyperparameters
- Saved checkpoints and configs

To reproduce exact results:
```bash
export PYTHONHASHSEED=1337
bash scripts/run_all.sh --full
```

---

## 🛠️ Development

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check (optional)
mypy src/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

---

## 📚 Citation

If you use this code or methodology, please cite:

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

## 🔗 References

1. **Raissi et al. (2019)** - Physics-Informed Neural Networks (JCP)
2. **Karniadakis et al. (2021)** - Physics-Informed Machine Learning (Nature Reviews)
3. **Wang et al. (2021)** - Gradient Pathologies in PINNs (SISC)
4. **Yang et al. (2021)** - Bayesian PINNs (JCP)

---

## 📄 License

This project is part of academic coursework at IIT Ropar (MA-515).

---

## 🤝 Contributing

This is a course project, but issues and suggestions are welcome!

---

## 📧 Contact

For questions or collaboration:
- **Institution:** Indian Institute of Technology Ropar
- **Course:** MA-515
- **Project Team:** See authors above

---

## 🎓 Acknowledgments

- **Course Instructor:** MA-515, IIT Ropar
- **Physics-Informed ML Community**
- **PyTorch Team**

---

**⭐ Star this repo if you find it useful!**

Last Updated: November 2024
