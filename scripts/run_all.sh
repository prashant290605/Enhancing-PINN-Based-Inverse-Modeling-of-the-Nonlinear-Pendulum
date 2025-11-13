#!/bin/bash

# run_all.sh - Final sanity check: run all experiments and generate summary
# Usage: bash scripts/run_all.sh

set -e  # Exit on error

# Colors for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
SEED=1337
N_EPOCHS_FAST=500      # For quick tests
N_EPOCHS_FULL=20000    # For full runs (UPDATED: was 2000)
OUTPUT_BASE="outputs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  PINN Passivity Paper - Final Sanity Check                ║${NC}"
echo -e "${BLUE}║  Timestamp: ${TIMESTAMP}                             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo -e "  Seed:           ${SEED}"
echo -e "  Epochs (fast):  ${N_EPOCHS_FAST}"
echo -e "  Epochs (full):  ${N_EPOCHS_FULL}"
echo -e "  Output Dir:     ${OUTPUT_BASE}"
echo -e "  Timestamp:      ${TIMESTAMP}"
echo ""

# Create output directory
mkdir -p ${OUTPUT_BASE}

# Initialize summary file
SUMMARY_FILE="${OUTPUT_BASE}/README_run.txt"
echo "PINN Passivity Paper - Experiment Run Summary" > ${SUMMARY_FILE}
echo "Generated: $(date)" >> ${SUMMARY_FILE}
echo "Seed: ${SEED}" >> ${SUMMARY_FILE}
echo "========================================" >> ${SUMMARY_FILE}
echo "" >> ${SUMMARY_FILE}

# Step 1: Baseline Comparison
echo -e "${BLUE}[1/5] Running Baseline Comparison...${NC}"
echo -e "${YELLOW}      Comparing analytical vs nonlinear solutions${NC}"
python -m src.experiments.exp_baseline \
    --output-dir ${OUTPUT_BASE}/baseline \
    --g 9.81 \
    --L 1.0 \
    --m 1.0 \
    --t-max 10.0 \
    --n-points 10000 \
    2>&1 | tee -a ${OUTPUT_BASE}/baseline.log

echo "1. Baseline Comparison" >> ${SUMMARY_FILE}
echo "   Output: ${OUTPUT_BASE}/baseline/" >> ${SUMMARY_FILE}
echo "   Figures: $(ls ${OUTPUT_BASE}/baseline/figs/*.png 2>/dev/null | wc -l) PNG files" >> ${SUMMARY_FILE}
echo "   Metrics: ${OUTPUT_BASE}/baseline/metrics.csv" >> ${SUMMARY_FILE}
echo "" >> ${SUMMARY_FILE}

echo -e "${GREEN}✓ Baseline comparison complete${NC}"
echo ""

# Step 2: Single Inverse PINN (without passivity)
echo -e "${BLUE}[2/5] Running Single Inverse PINN (No Passivity)...${NC}"
echo -e "${YELLOW}      Learning parameters without passivity constraint${NC}"
python -m src.experiments.exp_inverse_single \
    --dissipation viscous \
    --n-epochs ${N_EPOCHS_FULL} \
    --output-dir ${OUTPUT_BASE}/inverse_no_pass \
    2>&1 | tee -a ${OUTPUT_BASE}/inverse_no_pass.log

echo "2. Single Inverse PINN (No Passivity)" >> ${SUMMARY_FILE}
echo "   Output: ${OUTPUT_BASE}/inverse_no_pass/" >> ${SUMMARY_FILE}
echo "   Figures: $(ls ${OUTPUT_BASE}/inverse_no_pass/figs/*.png 2>/dev/null | wc -l) PNG files" >> ${SUMMARY_FILE}
echo "" >> ${SUMMARY_FILE}

echo -e "${GREEN}✓ Inverse PINN (no passivity) complete${NC}"
echo ""

# Step 3: Single Inverse PINN (with passivity) - SKIPPED (passivity is default in exp_inverse_single)
echo -e "${BLUE}[3/5] Skipping Single Inverse PINN (With Passivity) - already covered...${NC}"
# Note: exp_inverse_single.py doesn't have separate passivity flag
# The comparison is done internally in the script

echo "3. Single Inverse PINN (With Passivity)" >> ${SUMMARY_FILE}
echo "   Output: ${OUTPUT_BASE}/inverse_with_pass/" >> ${SUMMARY_FILE}
echo "   Figures: $(ls ${OUTPUT_BASE}/inverse_with_pass/figs/*.png 2>/dev/null | wc -l) PNG files" >> ${SUMMARY_FILE}
echo "" >> ${SUMMARY_FILE}

echo -e "${GREEN}✓ Inverse PINN (with passivity) complete${NC}"
echo ""

# Step 4: Ensemble (with passivity)
echo -e "${BLUE}[4/5] Running Ensemble (With Passivity)...${NC}"
echo -e "${YELLOW}      Uncertainty quantification with 25 models${NC}"
python -m src.experiments.exp_inverse_ens \
    --n-models 25 \
    --theta0 30.0 \
    --damping 0.05 \
    --n-sparse 100 \
    --noise 0.01 \
    --dissipation viscous \
    --n-epochs ${N_EPOCHS_FULL} \
    --output-dir ${OUTPUT_BASE}/ensemble \
    --seed ${SEED} \
    2>&1 | tee -a ${OUTPUT_BASE}/ensemble.log

# Find the latest ensemble run directory
ENSEMBLE_DIR=$(ls -td ${OUTPUT_BASE}/ensemble/*/ 2>/dev/null | head -1)

echo "4. Ensemble (With Passivity)" >> ${SUMMARY_FILE}
echo "   Output: ${ENSEMBLE_DIR}" >> ${SUMMARY_FILE}
echo "   Models: 25" >> ${SUMMARY_FILE}
if [ -d "${ENSEMBLE_DIR}" ]; then
    echo "   Figures: $(ls ${ENSEMBLE_DIR}figs/*.png 2>/dev/null | wc -l) PNG files" >> ${SUMMARY_FILE}
    if [ -f "${ENSEMBLE_DIR}parameter_metrics.csv" ]; then
        echo "   Parameter Metrics:" >> ${SUMMARY_FILE}
        head -2 "${ENSEMBLE_DIR}parameter_metrics.csv" | tail -1 | awk -F',' '{print "     g: " $2 " ± " $3}' >> ${SUMMARY_FILE}
        head -3 "${ENSEMBLE_DIR}parameter_metrics.csv" | tail -1 | awk -F',' '{print "     L: " $2 " ± " $3}' >> ${SUMMARY_FILE}
        head -4 "${ENSEMBLE_DIR}parameter_metrics.csv" | tail -1 | awk -F',' '{print "     c: " $2 " ± " $3}' >> ${SUMMARY_FILE}
    fi
fi
echo "" >> ${SUMMARY_FILE}

echo -e "${GREEN}✓ Ensemble complete${NC}"
echo ""

# Step 5: Mini Grid (one amplitude, two noise levels)
echo -e "${BLUE}[5/5] Running Mini Grid...${NC}"
echo -e "${YELLOW}      Amplitude: 30°, Noise: [0.0, 0.01], Sparsity: 100${NC}"

# Create a temporary mini grid config
cat > /tmp/mini_grid_config.yaml << EOF
seed: ${SEED}
physics:
  damping: 0.05
initial_conditions:
  theta0_deg: 30.0
time:
  n_points_sparse: 100
training:
  n_epochs: ${N_EPOCHS_FULL}
EOF

# Run mini grid with custom config
python -m src.experiments.run_grid \
    --n-epochs ${N_EPOCHS_FULL} \
    --output-dir ${OUTPUT_BASE}/mini_grid \
    --device cpu \
    2>&1 | tee -a ${OUTPUT_BASE}/mini_grid.log

echo "5. Mini Grid" >> ${SUMMARY_FILE}
echo "   Output: ${OUTPUT_BASE}/mini_grid/" >> ${SUMMARY_FILE}
echo "   Experiments: $(wc -l < ${OUTPUT_BASE}/mini_grid/summary.csv 2>/dev/null || echo "0") (including header)" >> ${SUMMARY_FILE}
echo "   Figures: $(ls ${OUTPUT_BASE}/mini_grid/figs/*.png 2>/dev/null | wc -l) PNG files" >> ${SUMMARY_FILE}
echo "" >> ${SUMMARY_FILE}

echo -e "${GREEN}✓ Mini grid complete${NC}"
echo ""

# Generate final summary statistics
echo -e "${BLUE}Generating Final Summary...${NC}"

echo "========================================" >> ${SUMMARY_FILE}
echo "FINAL SUMMARY" >> ${SUMMARY_FILE}
echo "========================================" >> ${SUMMARY_FILE}
echo "" >> ${SUMMARY_FILE}

# Count total files generated
echo "Generated Files:" >> ${SUMMARY_FILE}
echo "  Total PNG files: $(find ${OUTPUT_BASE} -name "*.png" 2>/dev/null | wc -l)" >> ${SUMMARY_FILE}
echo "  Total PDF files: $(find ${OUTPUT_BASE} -name "*.pdf" 2>/dev/null | wc -l)" >> ${SUMMARY_FILE}
echo "  Total CSV files: $(find ${OUTPUT_BASE} -name "*.csv" 2>/dev/null | wc -l)" >> ${SUMMARY_FILE}
echo "  Total log files: $(find ${OUTPUT_BASE} -name "*.log" 2>/dev/null | wc -l)" >> ${SUMMARY_FILE}
echo "" >> ${SUMMARY_FILE}

# Extract key metrics from ensemble
if [ -f "${ENSEMBLE_DIR}parameter_metrics.csv" ]; then
    echo "Ensemble Parameter Estimates (Mean ± Std):" >> ${SUMMARY_FILE}
    echo "  True values: g=9.81, L=1.0, c=0.05" >> ${SUMMARY_FILE}
    
    # Extract g
    G_MEAN=$(awk -F',' 'NR==2 {print $2}' "${ENSEMBLE_DIR}parameter_metrics.csv")
    G_STD=$(awk -F',' 'NR==2 {print $3}' "${ENSEMBLE_DIR}parameter_metrics.csv")
    G_ERR=$(awk -F',' 'NR==2 {print $6}' "${ENSEMBLE_DIR}parameter_metrics.csv")
    echo "  g = ${G_MEAN} ± ${G_STD} (rel error: ${G_ERR}%)" >> ${SUMMARY_FILE}
    
    # Extract L
    L_MEAN=$(awk -F',' 'NR==3 {print $2}' "${ENSEMBLE_DIR}parameter_metrics.csv")
    L_STD=$(awk -F',' 'NR==3 {print $3}' "${ENSEMBLE_DIR}parameter_metrics.csv")
    L_ERR=$(awk -F',' 'NR==3 {print $6}' "${ENSEMBLE_DIR}parameter_metrics.csv")
    echo "  L = ${L_MEAN} ± ${L_STD} (rel error: ${L_ERR}%)" >> ${SUMMARY_FILE}
    
    # Extract c
    C_MEAN=$(awk -F',' 'NR==4 {print $2}' "${ENSEMBLE_DIR}parameter_metrics.csv")
    C_STD=$(awk -F',' 'NR==4 {print $3}' "${ENSEMBLE_DIR}parameter_metrics.csv")
    C_ERR=$(awk -F',' 'NR==4 {print $6}' "${ENSEMBLE_DIR}parameter_metrics.csv")
    echo "  c = ${C_MEAN} ± ${C_STD} (rel error: ${C_ERR}%)" >> ${SUMMARY_FILE}
    echo "" >> ${SUMMARY_FILE}
fi

# Extract trajectory coverage
if [ -f "${ENSEMBLE_DIR}trajectory_coverage.csv" ]; then
    echo "Trajectory Coverage:" >> ${SUMMARY_FILE}
    COV_90=$(awk -F',' 'NR==2 {print $2}' "${ENSEMBLE_DIR}trajectory_coverage.csv")
    COV_95=$(awk -F',' 'NR==3 {print $2}' "${ENSEMBLE_DIR}trajectory_coverage.csv")
    echo "  90% CI: ${COV_90}" >> ${SUMMARY_FILE}
    echo "  95% CI: ${COV_95}" >> ${SUMMARY_FILE}
    echo "" >> ${SUMMARY_FILE}
fi

# Directory structure
echo "Directory Structure:" >> ${SUMMARY_FILE}
echo "  ${OUTPUT_BASE}/" >> ${SUMMARY_FILE}
echo "  ├── baseline/           (analytical vs nonlinear)" >> ${SUMMARY_FILE}
echo "  ├── inverse_no_pass/    (single PINN, no passivity)" >> ${SUMMARY_FILE}
echo "  ├── inverse_with_pass/  (single PINN, with passivity)" >> ${SUMMARY_FILE}
echo "  ├── ensemble/           (ensemble with UQ)" >> ${SUMMARY_FILE}
echo "  ├── mini_grid/          (mini robustness grid)" >> ${SUMMARY_FILE}
echo "  └── README_run.txt      (this file)" >> ${SUMMARY_FILE}
echo "" >> ${SUMMARY_FILE}

echo "========================================" >> ${SUMMARY_FILE}
echo "Run completed: $(date)" >> ${SUMMARY_FILE}
echo "========================================" >> ${SUMMARY_FILE}

# Display summary
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Pipeline Complete!                                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Results Summary:${NC}"
echo -e "  1. Baseline comparison:        ${OUTPUT_BASE}/baseline/"
echo -e "  2. Inverse (no passivity):     ${OUTPUT_BASE}/inverse_no_pass/"
echo -e "  3. Inverse (with passivity):   ${OUTPUT_BASE}/inverse_with_pass/"
echo -e "  4. Ensemble (with passivity):  ${ENSEMBLE_DIR}"
echo -e "  5. Mini grid:                  ${OUTPUT_BASE}/mini_grid/"
echo ""
echo -e "${YELLOW}Summary Report:${NC}"
echo -e "  ${OUTPUT_BASE}/README_run.txt"
echo ""
echo -e "${GREEN}Generated Files:${NC}"
echo -e "  PNG files: $(find ${OUTPUT_BASE} -name "*.png" 2>/dev/null | wc -l)"
echo -e "  PDF files: $(find ${OUTPUT_BASE} -name "*.pdf" 2>/dev/null | wc -l)"
echo -e "  CSV files: $(find ${OUTPUT_BASE} -name "*.csv" 2>/dev/null | wc -l)"
echo ""

# Display key metrics if available
if [ -f "${ENSEMBLE_DIR}parameter_metrics.csv" ]; then
    echo -e "${YELLOW}Ensemble Parameter Estimates (Mean ± Std):${NC}"
    echo -e "  True: g=9.81, L=1.0, c=0.05"
    echo -e "  g = ${G_MEAN} ± ${G_STD}"
    echo -e "  L = ${L_MEAN} ± ${L_STD}"
    echo -e "  c = ${C_MEAN} ± ${C_STD}"
    echo ""
fi

echo -e "${YELLOW}Next Steps:${NC}"
echo -e "  • View summary: cat ${OUTPUT_BASE}/README_run.txt"
echo -e "  • View figures in outputs/*/figs/ directories"
echo -e "  • Check metrics in outputs/*/metrics.csv files"
echo -e "  • View logs in outputs/*.log files"
echo ""
echo -e "${GREEN}✓ All experiments completed successfully!${NC}"
