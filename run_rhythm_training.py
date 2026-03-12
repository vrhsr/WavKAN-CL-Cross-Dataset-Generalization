"""
Phase A: Full Rhythm-Harmonized Training Pipeline for Kaggle/Colab.

Upload your entire project to Kaggle/Colab, then run this script.
It trains all 6 models × 5 seeds on the rhythm-harmonized datasets,
runs zero-shot + few-shot evaluation, and saves all results.

Usage (Kaggle/Colab):
    !python run_rhythm_training.py

Prerequisites:
    - data/mitbih_rhythm_processed.csv (from emit_mitbih_rhythm.py)
    - data/ptbxl_rhythm_processed.csv (from remap_ptbxl_rhythm.py)
"""
import subprocess
import sys
import os
import shutil

# Configuration
MIT_FILE = 'data/mitbih_rhythm_processed.csv'
PTB_FILE = 'data/ptbxl_rhythm_processed.csv'
SEEDS = [42, 123, 456, 789, 2024]
EPOCHS = 50
BATCH_SIZE = 64
NUM_WORKERS = 2

# Models to train via train.py
MODELS = ['wavkan', 'spline_kan', 'resnet', 'vit', 'mlp']

# Model-specific settings
MODEL_CONFIGS = {
    'wavkan':     {'lr': 1e-3, 'hidden_dim': 128, 'wavelet_type': 'morlet', 'depth': 3},
    'spline_kan': {'lr': 1e-3},
    'resnet':     {'lr': 1e-3},
    'vit':        {'lr': 5e-4},
    'mlp':        {'lr': 1e-3},
}


def rename_if_exists(src, dst):
    """Safely rename a file, creating directories as needed."""
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst) if os.path.dirname(dst) else '.', exist_ok=True)
        shutil.move(src, dst)
        return True
    return False


def run_training(model, seed):
    """Run training for a single model and seed."""
    config = MODEL_CONFIGS.get(model, {})
    
    cmd = [
        sys.executable, '-m', 'src.train',
        '--model', model,
        '--mit_file', MIT_FILE,
        '--ptb_file', PTB_FILE,
        '--seed', str(seed),
        '--epochs', str(EPOCHS),
        '--batch_size', str(BATCH_SIZE),
        '--num_workers', str(NUM_WORKERS),
        '--lr', str(config.get('lr', 1e-3)),
    ]
    
    # WavKAN-specific arguments
    if model == 'wavkan':
        cmd.extend(['--hidden_dim', str(config.get('hidden_dim', 128))])
        cmd.extend(['--wavelet_type', config.get('wavelet_type', 'morlet')])
        cmd.extend(['--depth', str(config.get('depth', 3))])
    
    print(f"\n{'='*60}")
    print(f"  Training: {model} | Seed: {seed}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"  ⚠️  Training failed for {model} seed {seed}")
        return False
    
    # Rename outputs to include seed suffix for rhythm variant
    rename_if_exists(
        f'experiments/{model}_endpoint.pth',
        f'experiments/rhythm_{model}_seed{seed}.pth'
    )
    rename_if_exists(
        f'experiments/{model}_history.csv',
        f'experiments/rhythm_{model}_history_seed{seed}.csv'
    )
    rename_if_exists(
        f'experiments/zeroshot_{model}_seed{seed}.csv',
        f'experiments/rhythm_zeroshot_{model}_seed{seed}.csv'
    )
    
    print(f"  ✅ Saved: rhythm_{model}_seed{seed}.pth")
    return True


def run_fewshot(model, seed):
    """Run few-shot evaluation for a single model and seed."""
    checkpoint = f'experiments/rhythm_{model}_seed{seed}.pth'
    if not os.path.exists(checkpoint):
        print(f"  Skipping few-shot for {model} seed {seed} — no checkpoint")
        return
    
    config = MODEL_CONFIGS.get(model, {})
    
    cmd = [
        sys.executable, '-m', 'src.test_fewshot',
        '--model', model,
        '--ptb_file', PTB_FILE,
        '--pretrained_path', checkpoint,
    ]
    
    # WavKAN-specific args
    if model == 'wavkan':
        cmd.extend(['--hidden_dim', str(config.get('hidden_dim', 128))])
        cmd.extend(['--wavelet_type', config.get('wavelet_type', 'morlet')])
        cmd.extend(['--depth', str(config.get('depth', 3))])
    
    print(f"  Few-shot: {model} | Seed: {seed}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        # test_fewshot.py saves to experiments/fewshot_{model}.csv (no seed suffix)
        rename_if_exists(
            f'experiments/fewshot_{model}.csv',
            f'experiments/rhythm_fewshot_{model}_seed{seed}.csv'
        )


def run_dann_training(seed):
    """Run DANN training which has its own separate script."""
    cmd = [
        sys.executable, '-m', 'src.train_dann',
        '--source_file', MIT_FILE,
        '--target_file', PTB_FILE,
        '--seed', str(seed),
        '--epochs', str(EPOCHS),
        '--batch_size', str(BATCH_SIZE),
    ]
    
    print(f"\n{'='*60}")
    print(f"  Training DANN | Seed: {seed}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"  ⚠️  DANN training failed for seed {seed}")
        return False
    
    rename_if_exists(
        f'experiments/dann_endpoint.pth',
        f'experiments/rhythm_dann_seed{seed}.pth'
    )
    rename_if_exists(
        f'experiments/dann_history.csv',
        f'experiments/rhythm_dann_history_seed{seed}.csv'
    )
    
    print(f"  ✅ Saved: rhythm_dann_seed{seed}.pth")
    return True


def run_dann_fewshot(seed):
    """Run few-shot for DANN."""
    checkpoint = f'experiments/rhythm_dann_seed{seed}.pth'
    if not os.path.exists(checkpoint):
        return
    
    cmd = [
        sys.executable, '-m', 'src.test_fewshot',
        '--model', 'dann',
        '--ptb_file', PTB_FILE,
        '--pretrained_path', checkpoint,
    ]
    
    print(f"  Few-shot: dann | Seed: {seed}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        rename_if_exists(
            f'experiments/fewshot_dann.csv',
            f'experiments/rhythm_fewshot_dann_seed{seed}.csv'
        )


def main():
    os.makedirs('experiments', exist_ok=True)
    
    print("=" * 60)
    print("  RHYTHM-HARMONIZED FULL TRAINING PIPELINE")
    print("=" * 60)
    print(f"  Source: {MIT_FILE}")
    print(f"  Target: {PTB_FILE}")
    print(f"  Models: {MODELS + ['dann']}")
    print(f"  Seeds:  {SEEDS}")
    print(f"  Total runs: {(len(MODELS) + 1) * len(SEEDS)}")
    print("=" * 60)
    
    # Phase 1: Training + Zero-Shot
    print("\n\n📊 PHASE 1: Training + Zero-Shot Evaluation")
    for model in MODELS:
        for seed in SEEDS:
            run_training(model, seed)
    
    # DANN has separate training script
    for seed in SEEDS:
        run_dann_training(seed)
    
    # Phase 2: Few-Shot Evaluation
    print("\n\n📊 PHASE 2: Few-Shot Evaluation")
    for model in MODELS:
        for seed in SEEDS:
            run_fewshot(model, seed)
    
    for seed in SEEDS:
        run_dann_fewshot(seed)
    
    print("\n\n" + "=" * 60)
    print("  ✅ ALL DONE!")
    print("  Results saved with 'rhythm_' prefix in experiments/")
    print("  Download: experiments/rhythm_*.csv and experiments/rhythm_*.pth")
    print("=" * 60)


if __name__ == '__main__':
    main()
