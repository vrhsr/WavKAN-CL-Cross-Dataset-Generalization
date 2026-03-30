import os
import json
import argparse
from gc import collect
import torch

from src import train_multiclass
from src import train_wavkan_dann
from src import train_ewc
from src.utils import set_seed

# Experiment Space
MODELS = ['wavkan', 'spline_kan', 'resnet', 'inception', 'xresnet', 'vit', 'simple_mlp']
SEEDS = [42, 43, 44, 45, 46]

class DictObj:
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
               setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
               setattr(self, key, DictObj(val) if isinstance(val, dict) else val)
               
def run_experiments():
    parser = argparse.ArgumentParser("DA Experiment Runner")
    parser.add_argument('--source', type=str, default='data/ptbxl_signals.npy')
    parser.add_argument('--target', type=str, default='data/chapman_signals.npy')
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()

    os.makedirs('experiments/runs', exist_ok=True)
    os.makedirs('experiments/checkpoints', exist_ok=True)

    # 1. Source-Only Baseline (Zero-Shot)
    for model in MODELS:
        for seed in SEEDS:
            run_id = f"source_only_{model}_seed{seed}"
            run_file = f"experiments/runs/{run_id}.json"
            
            if os.path.exists(run_file):
                print(f"Skipping {run_id} (already done)")
                continue
                
            batch_size = 64
            if model == 'vit': batch_size = 32
            elif model in ['inception', 'xresnet']: batch_size = 48
            
            print(f"\n[{run_id}] Starting... (Batch: {batch_size})")
            cfg = {
                'data_file': args.source,
                'target_file': args.target,
                'model': model,
                'hidden_dim': 64,
                'in_channels': 12,
                'batch_size': batch_size,
                'lr': 1e-3,
                'epochs': args.epochs,
                'seed': seed,
                'patience': 10
            }
            
            try:
                # Assuming train_multiclass has been updated to evaluate zero-shot on target_file
                # and take these arguments gracefully
                train_multiclass.main(DictObj(cfg))
            except Exception as e:
                print(f"Failed {run_id}: {e}")
                
            collect()
            torch.cuda.empty_cache()
            
    # 2. DANN (Domain Adaptation)
    for model in MODELS:
        for seed in SEEDS:
            run_id = f"dann_{model}_seed{seed}"
            run_file = f"experiments/runs/{run_id}.json"
            
            if os.path.exists(run_file):
                print(f"Skipping {run_id} (already done)")
                continue

            batch_size = 64
            if model == 'vit': batch_size = 32
            elif model in ['inception', 'xresnet']: batch_size = 48
            
            print(f"\n[{run_id}] Starting... (Batch: {batch_size})")
            cfg = {
                'source_file': args.source,
                'target_file': args.target,
                'model': model,
                'hidden_dim': 64,
                'in_channels': 12,
                'batch_size': batch_size,
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'epochs': args.epochs,
                'seed': seed
            }
            try:
                train_wavkan_dann.main(DictObj(cfg))
            except Exception as e:
                print(f"Failed {run_id}: {e}")
                
            collect()
            torch.cuda.empty_cache()

    # 3. EWC (Continual Learning)
    for model in MODELS:
        for seed in SEEDS:
            run_id = f"ewc_{model}_seed{seed}"
            run_file = f"experiments/runs/{run_id}.json"
            
            if os.path.exists(run_file):
                print(f"Skipping {run_id} (already done)")
                continue
                
            batch_size = 64
            if model == 'vit': batch_size = 32
            elif model in ['inception', 'xresnet']: batch_size = 48
            
            print(f"\n[{run_id}] Starting... (Batch: {batch_size})")
            cfg = {
                'source_file': args.source,
                'target_file': args.target,
                'checkpoint': f"experiments/checkpoints/source_only_{model}_seed{seed}_best.pt",
                'model': model,
                'hidden_dim': 64,
                'in_channels': 12,
                'batch_size': batch_size,
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'epochs': args.epochs,
                'ewc_lambda': 5000.0,
                'seed': seed
            }
            try:
                train_ewc.main(DictObj(cfg))
            except Exception as e:
                print(f"Failed {run_id}: {e}")
                
            collect()
            torch.cuda.empty_cache()
            
    print("\nAll automated experiments launched via direct calls.")

if __name__ == "__main__":
    run_experiments()
