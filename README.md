# WavKAN-CL: Cross-Dataset ECG Generalization via Kolmogorov-Arnold Networks

> **A systematic benchmark of KAN architectures for cross-hospital ECG transfer learning.**  
> Trains on MIT-BIH → evaluates on PTB-XL and CPSC 2018 under zero-shot, few-shot, and noise robustness protocols.

## Project Structure

```
├── src/
│   ├── models/           # WavKAN, Spline-KAN, ResNet-1D, ViT-1D, DANN, MLP
│   ├── emit_mitbih.py    # MIT-BIH preprocessor (original labels)
│   ├── emit_mitbih_rhythm.py   # MIT-BIH preprocessor (rhythm-harmonized)
│   ├── emit_ptbxl.py     # PTB-XL preprocessor (original labels)
│   ├── emit_ptbxl_rhythm.py    # PTB-XL preprocessor (rhythm-harmonized)
│   ├── emit_cpsc2018_rhythm.py # CPSC 2018 preprocessor (SNOMED-CT → rhythm)
│   ├── dataset.py        # HarmonizedDataset + SSLAugmentedDataset
│   ├── train.py          # Main training loop (source domain)
│   ├── train_dann.py     # DANN adversarial training
│   ├── train_ssl.py      # SimCLR self-supervised pretraining
│   ├── test_fewshot.py   # k-shot cross-domain adaptation
│   ├── test_robustness.py     # Gaussian noise stress testing
│   ├── test_forgetting.py     # Catastrophic forgetting measurement
│   ├── compute_mmd.py    # Maximum Mean Discrepancy metric
│   ├── compute_flops.py  # Parameter & FLOP counting
│   ├── analyze_wavelets.py    # Wavelet scale → frequency interpretability
│   ├── statistical_tests.py   # Bonferroni-corrected significance tests
│   ├── verify_harmonization.py # Dataset harmonization QA
│   └── plot_*.py, visualize_tsne.py  # Visualization scripts
├── run_rhythm_training.py  # Master pipeline (rhythm-harmonized)
├── experiments/            # Checkpoints, CSVs, plots
├── paper/                  # LaTeX manuscript + figures
└── data/                   # Processed datasets (not in VCS)
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset Preparation

### Step 1: Generate Rhythm-Harmonized Datasets

```bash
# MIT-BIH (source domain)
python -m src.emit_mitbih_rhythm

# PTB-XL (target domain 1)
python -m src.emit_ptbxl_rhythm

# CPSC 2018 (target domain 2) — requires PhysioNet credentials
export PHYSIONET_USER=your_user
export PHYSIONET_PASS=your_pass
python src/download_cpsc.py
python -m src.emit_cpsc2018_rhythm
```

### Step 2: Verify Harmonization
```bash
python -m src.verify_harmonization
```

## Training

### Full Pipeline (all models × seeds)
```bash
python run_rhythm_training.py
```

### Individual Models
```bash
python -m src.train --model wavkan --epochs 50 --seed 42
python -m src.train --model resnet --epochs 50 --seed 42
python -m src.train --model vit --epochs 50 --seed 42
python -m src.train --model spline_kan --epochs 50 --seed 42
python -m src.train --model mlp --epochs 50 --seed 42
```

### DANN (Domain-Adversarial)
```bash
python -m src.train_dann --epochs 50 --seed 42
```

### Self-Supervised Pre-training
```bash
python -m src.train_ssl --model wavkan --epochs 300
```

## Evaluation

```bash
# Zero-shot & Few-shot
python -m src.test_fewshot --model wavkan

# Noise Robustness
python -m src.test_robustness --model wavkan

# Catastrophic Forgetting
python -m src.test_forgetting --model wavkan --checkpoint experiments/wavkan_endpoint.pth

# Domain Alignment (MMD)
python -m src.compute_mmd --model wavkan --checkpoint experiments/wavkan_endpoint.pth

# Statistical Significance (Welch + paired + bootstrap CI)
python -m src.statistical_tests

# Export probabilities for calibration / uncertainty metrics
python -m src.evaluate_uncertainty --model wavkan --data_file data/ptbxl_processed.csv

# Deployment benchmark (FP32 vs INT8, footprint + latency)
python -m src.benchmark_deployment
```

## Models

| Model | Type | Params | Description |
|-------|------|--------|-------------|
| **WavKAN** | Wavelet KAN | ~114K | Mexican Hat / Morlet wavelet edges with constrained scale |
| **Spline-KAN** | B-Spline KAN | ~195K | Radial basis function edges |
| **ResNet-1D** | CNN | ~3.8M | 1D ResNet with 7×1 kernels |
| **ViT-1D** | Transformer | ~538K | Patch-based 1D Vision Transformer |
| **DANN** | Domain Adaptation | ~690K | Gradient reversal for domain alignment |
| **SimpleMLP** | MLP | ~106K | Parameter-matched ablation baseline |

## Label Harmonization

All datasets use a **rhythm-only** binary classification:
- **Normal (0):** Normal Sinus Rhythm
- **Abnormal (1):** AF, LBBB, RBBB, PAC, PVC, and other rhythm disorders

This excludes morphological abnormalities (MI, ST changes) to ensure the cross-dataset transfer measures genuine covariate shift, not label ontology mismatch.


## Model Size Note (Reproducibility)

You may see two WavKAN parameter numbers in this project:
- `~114K` in the compact reference setup (original WavKAN-CL profile).
- `~471K` in the wider benchmark setup (`hidden_dim=64`) used for fairer capacity comparison with stronger baselines in the manuscript.

Always report the exact architecture hyperparameters (`hidden_dim`, `depth`, wavelet type, stem usage) alongside parameter count in tables and scripts.

