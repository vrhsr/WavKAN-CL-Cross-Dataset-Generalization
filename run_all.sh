#!/bin/bash
# Complete Reproduction Script for WavKAN-CL
set -e

echo "=== WavKAN-CL: Starting Complete Reproduction Pipeline ==="
echo "Estimated total runtime: ~192 GPU-hours on RTX 5080/4090"
echo "Estimated completion: $(date -d '+8 days')"

# 1. Data Pipeline
echo "[1/6] Processing Datasets (12-Lead, Multi-Class)"
python src/emit_ptbxl.py
python src/emit_chapman.py
python src/emit_georgia.py

# 2. Deep Learning Baseline & Domain Adaptation Automator
echo "[2/6] Running Deep Learning Sweeps (Source, DANN, EWC)"
# This runs across 6 architectures x 5 seeds
python run_da_experiments.py --epochs 30

# 3. Traditional ML Baseline
echo "[3/6] Running Wavelet + XGBoost Baseline"
python src/wavelet_xgb_baseline.py

# 4. Interpretability & Uncertainty
echo "[4/6] Generating Interpretability & Uncertainty Metrics"
python src/analyze_wavelets.py --checkpoint experiments/checkpoints/multiclass_wavkan_seed42_best.pt --data_file data/ptbxl_signals.npy
python src/evaluate_uncertainty.py --model wavkan
python src/evaluate_uncertainty.py --model inception
python src/evaluate_clinical.py --model wavkan

# 5. Robustness Suite
echo "[5/6] Running Stress Tests (6x8x6 matrix)"
python run_robustness_sweep.py

# 6. Deployment & Demographics
echo "[6/6] Benchmarking Inference Limits & Algorithmic Fairness"
python src/benchmark_deployment.py
python src/evaluate_demographics.py

echo "=== Pipeline Complete! All results saved to experiments/ ==="
