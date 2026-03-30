# Complete Reproduction Script for WavKAN-CL (PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "=== WavKAN-CL: Starting Complete Reproduction Pipeline ===" -ForegroundColor Cyan

# 1. Data Pipeline
Write-Host "[1/6] Processing Datasets (12-Lead, Multi-Class)" -ForegroundColor Yellow
python src\emit_ptbxl.py
python src\emit_chapman.py
python src\emit_georgia.py

# 2. Deep Learning Baseline & Domain Adaptation Automator
Write-Host "[2/6] Running Deep Learning Sweeps (Source, DANN, EWC)" -ForegroundColor Yellow
# This runs across 6 architectures x 5 seeds
python run_da_experiments.py --epochs 30

# 3. Traditional ML Baseline
Write-Host "[3/6] Running Wavelet + XGBoost Baseline" -ForegroundColor Yellow
python src\wavelet_xgb_baseline.py

# 4. Interpretability & Uncertainty
Write-Host "[4/6] Generating Interpretability & Uncertainty Metrics" -ForegroundColor Yellow
python src\analyze_wavelets.py --checkpoint experiments\checkpoints\multiclass_wavkan_seed42_best.pt --data_file data\ptbxl_signals.npy
python src\evaluate_uncertainty.py --model wavkan
python src\evaluate_uncertainty.py --model inception
python src\evaluate_clinical.py --model wavkan

# 5. Robustness Suite
Write-Host "[5/6] Running Stress Tests (6x8x6 matrix)" -ForegroundColor Yellow
python run_robustness_sweep.py

# 6. Deployment & Demographics
Write-Host "[6/6] Benchmarking Inference Limits & Algorithmic Fairness" -ForegroundColor Yellow
python src\benchmark_deployment.py
python src\evaluate_demographics.py

Write-Host "=== Pipeline Complete! All results saved to experiments\ ===" -ForegroundColor Green
