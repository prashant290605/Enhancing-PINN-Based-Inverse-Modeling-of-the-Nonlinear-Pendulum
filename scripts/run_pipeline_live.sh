#!/bin/bash

# run_pipeline_live.sh - Run pipeline with live output visible in terminal
# Usage: bash scripts/run_pipeline_live.sh

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  PINN Passivity Paper - Live Pipeline Run                 ║"
echo "║  Updated defaults: 20000 epochs, 100 points, 25 models    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

SEED=1337
N_EPOCHS=20000
N_SPARSE=100
N_MODELS=25
OUTPUT_BASE="outputs"

echo "Configuration:"
echo "  Seed:           ${SEED}"
echo "  Epochs:         ${N_EPOCHS}"
echo "  Sparse points:  ${N_SPARSE}"
echo "  Ensemble size:  ${N_MODELS}"
echo ""
echo "This will take approximately 1-2 hours..."
echo ""
read -p "Press ENTER to start or Ctrl+C to cancel..."
echo ""

# Activate venv
source venv/bin/activate

# Step 1: Baseline
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[1/4] Running Baseline Comparison..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -m src.experiments.exp_baseline \
    --output-dir ${OUTPUT_BASE}/baseline \
    --g 9.81 \
    --L 1.0 \
    --m 1.0 \
    --t-max 10.0 \
    --n-points 10000

echo ""
echo "✓ Baseline complete"
echo ""

# Step 2: Single Inverse PINN
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[2/4] Running Single Inverse PINN (${N_EPOCHS} epochs)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -m src.experiments.exp_inverse_single \
    --dissipation viscous \
    --n-epochs ${N_EPOCHS} \
    --output-dir ${OUTPUT_BASE}/inverse_single

echo ""
echo "✓ Inverse PINN complete"
echo ""

# Step 3: Ensemble
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[3/4] Running Ensemble (${N_MODELS} models, ${N_EPOCHS} epochs each)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "This will take the longest time..."
echo ""
python -m src.experiments.exp_inverse_ens \
    --n-models ${N_MODELS} \
    --theta0 30.0 \
    --damping 0.05 \
    --n-sparse ${N_SPARSE} \
    --noise 0.01 \
    --dissipation viscous \
    --n-epochs ${N_EPOCHS} \
    --output-dir ${OUTPUT_BASE}/ensemble \
    --seed ${SEED}

echo ""
echo "✓ Ensemble complete"
echo ""

# Step 4: Mini Grid
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[4/4] Running Mini Grid..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -m src.experiments.run_grid \
    --n-epochs ${N_EPOCHS} \
    --output-dir ${OUTPUT_BASE}/mini_grid \
    --device cpu

echo ""
echo "✓ Mini grid complete"
echo ""

# Generate summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Pipeline Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Results saved to: ${OUTPUT_BASE}/"
echo ""
echo "Generated files:"
echo "  PNG files: $(find ${OUTPUT_BASE} -name "*.png" 2>/dev/null | wc -l)"
echo "  CSV files: $(find ${OUTPUT_BASE} -name "*.csv" 2>/dev/null | wc -l)"
echo ""
echo "Next step: Regenerate analysis"
echo "  python scripts/generate_final_study.py"
echo ""

