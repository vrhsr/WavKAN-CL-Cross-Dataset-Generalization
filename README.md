# WavKAN-CL: Cross-Hospital ECG Generalization

This repository contains the implementation for the thesis project: **"Architectures with wavelet-based inductive bias exhibit significantly lower cross-dataset performance degradation than convolutional and attention-based models in ECG classification."**

## 1. Project Structure

*   `data/`: Raw and processed datasets (MIT-BIH, PTB-XL).
*   `src/`: Source code.
    *   `emit_mitbih.py`: Downloads and harmonizes MIT-BIH (100Hz, Lead II).
    *   `emit_ptbxl.py`: Downloads and harmonizes PTB-XL (100Hz, Lead II).
    *   `dataset.py`: PyTorch Dataset loader for harmonized CSVs.
    *   `losses.py`: NT-Xent contrastive loss for SimCLR.
    *   `models/`: Model definitions (WavKAN, ResNet, ViT, Spline-KAN, MLP).
    *   `train.py`: Main training and evaluation loop (with validation split & loss logging).
    *   `train_ssl.py`: Self-supervised contrastive pre-training (SimCLR).
    *   `test_fewshot.py`: Few-shot adaptation evaluation (stratified sampling).
    *   `test_robustness.py`: Noise stress-testing at varying SNR levels.
    *   `verify_harmonization.py`: Statistical verification of domain alignment.
    *   `visualize_analysis.py`: Confusion matrices and basis function plots.
*   `experiments/`: Model checkpoints, result CSVs, and plots.

## 2. Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Data Generation**:
    ```bash
    python src/emit_mitbih.py
    python src/emit_ptbxl.py
    ```

## 3. Usage

### Full Pipeline
```powershell
experiments/run_project.ps1
```

### Train Individual Models
```bash
python -m src.train --model wavkan --epochs 50 --seed 42
python -m src.train --model resnet --epochs 50 --seed 42
python -m src.train --model vit --epochs 50 --seed 42
python -m src.train --model spline_kan --epochs 50 --seed 42
python -m src.train --model mlp --epochs 50 --seed 42
```

### Self-Supervised Pre-training (SimCLR)
```bash
python -m src.train_ssl --model wavkan --epochs 300
python -m src.train_ssl --model spline_kan --epochs 300
```

### Few-Shot Evaluation
```bash
python -m src.test_fewshot --model wavkan
python -m src.test_fewshot --model wavkan --pretrained_path experiments/ssl/wavkan_pretrained.pth
```

### Robustness Testing
```bash
python -m src.test_robustness --model wavkan
```

## 4. Models
*   **WavKAN-CL** (~114k params): Wavelet Kolmogorov-Arnold Network.
*   **Spline-KAN** (~195k params): B-Spline KAN baseline.
*   **ResNet-1D** (~3.8M params): Standard CNN baseline.
*   **ViT-1D** (~538k params): Vision Transformer baseline.
*   **SimpleMLP** (~106k params): Parameter-matched ablation baseline.

## 5. Key Results (500-Shot Adaptation F1)

| Model | Baseline | SSL Pre-trained |
|:---|:---|:---|
| **Spline-KAN** | 0.688 | **0.803** |
| **WavKAN** | 0.653 | **0.793** |
| **ViT** | 0.596 | — |
| **ResNet** | 0.592 | — |
| **MLP** | 0.190 | — |
