# Paper Strengthening Plan (Execution Checklist)

This checklist operationalizes the requested high-impact upgrades into executable tasks.

## Phase 1 (1-2 weeks)
- Add stronger DA/UDA/TTA baselines (e.g., CORAL, MMD, TENT, SHOT).
- Run calibration/uncertainty metrics (ECE, Brier, AUROC, AUPRC) with bootstrap CIs.
- Expand robustness suite to realistic ECG corruptions:
  - baseline wander
  - powerline interference
  - muscle artifact
  - motion artifact
  - lead dropout
  - sampling jitter
  - label flip stress

## Phase 2 (2-3 weeks)
- Extend to multi-lead experiments.
- Extend to multi-class or hierarchical label experiments.

## Phase 3 (1 week)
- Quantitative interpretability metrics:
  - faithfulness/deletion curves
  - localization to ECG intervals
  - cardiologist heuristic agreement
- Deployment benchmark with INT8 quantization and footprint constraints.

## Phase 4 (final)
- Consistency audit: README / manuscript / configs / checkpoints.
- Reproducibility package with exact commands and random seeds.

## New Utilities Added in This Repo
- `src/test_robustness.py`: now supports corruption sweeps beyond AWGN.
- `src/statistical_tests.py`: bootstrap CIs + paired tests + optional calibration report.
- `src/evaluate_uncertainty.py`: exports probability predictions for calibration metrics.
- `src/benchmark_deployment.py`: FP32 vs INT8 latency + model-size benchmarking.

## Suggested commands
```bash
# Robustness sweep with realistic corruptions
python -m src.test_robustness --model wavkan --corruptions awgn,baseline_wander,powerline,muscle,motion,lead_dropout,sampling_jitter,label_flip

# Export prediction probabilities
python -m src.evaluate_uncertainty --model wavkan --data_file data/ptbxl_processed.csv

# Enhanced stats (Welch + paired + bootstrap + calibration)
python -m src.statistical_tests

# Deployment benchmark
python -m src.benchmark_deployment
```
