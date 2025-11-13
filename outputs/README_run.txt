PINN Passivity Paper - Experiment Run Summary
Generated: Sun Nov  9 16:21:04 IST 2025
Seed: 1337
========================================

1. Baseline Comparison
   Output: outputs/baseline/
   Figures:        6 PNG files
   Metrics: outputs/baseline/metrics.csv

2. Single Inverse PINN (No Passivity)
   Output: outputs/inverse_no_pass/
   Figures:        6 PNG files

3. Single Inverse PINN (With Passivity)
   Output: outputs/inverse_with_pass/
   Figures:        0 PNG files

4. Ensemble (With Passivity)
   Output: outputs/ensemble/20251109_162312/
   Models: 25
   Figures:        3 PNG files
   Parameter Metrics:
     g: 9.950849571228026 ± 0.1435322269551541
     L: 1.1327316427230836 ± 0.09531519244418205
     c: 0.43404212802648545 ± 0.8772868789041905

5. Mini Grid
   Output: outputs/mini_grid/
   Experiments:        7 (including header)
   Figures:        3 PNG files

========================================
FINAL SUMMARY
========================================

Generated Files:
  Total PNG files:       18
  Total PDF files:        0
  Total CSV files:        6
  Total log files:        4

Ensemble Parameter Estimates (Mean ± Std):
  True values: g=9.81, L=1.0, c=0.05
  g = 9.950849571228026 ± 0.1435322269551541 (rel error: 1.4357754457495004%)
  L = 1.1327316427230836 ± 0.09531519244418205 (rel error: 13.273164272308357%)
  c = 0.43404212802648545 ± 0.8772868789041905 (rel error: 768.0842560529709%)

Trajectory Coverage:
  90% CI: 0.004
  95% CI: 0.006

Directory Structure:
  outputs/
  ├── baseline/           (analytical vs nonlinear)
  ├── inverse_no_pass/    (single PINN, no passivity)
  ├── inverse_with_pass/  (single PINN, with passivity)
  ├── ensemble/           (ensemble with UQ)
  ├── mini_grid/          (mini robustness grid)
  └── README_run.txt      (this file)

========================================
Run completed: Sun Nov  9 17:02:12 IST 2025
========================================
