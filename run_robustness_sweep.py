import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.metrics import f1_score

from src.dataset import HarmonizedDataset
from src.models.wavkan import WavKANClassifier
from src.models.spline_kan import SplineKANClassifier
from src.models.baselines import ResNet1D, ViT1D, SimpleMLP, InceptionTime, XResNet1D

def evaluate_noise(model, dataset, device, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return f1_macro

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Robustness Sweep on device: {device}")
    
    models = ['wavkan', 'spline_kan', 'resnet', 'inception', 'vit', 'simple_mlp']
    corruptions = ['awgn', 'baseline_wander', 'powerline', 'muscle', 'motion', 'lead_dropout', 'sampling_jitter', 'label_flip']
    snrs = [None, 20, 15, 10, 5, 0] # None represents clean
    seeds = [42, 43, 44, 45, 46]
    
    os.makedirs('experiments/runs', exist_ok=True)
    out_file = args.out_file
    
    # Load existing results to resume
    if os.path.exists(out_file):
        with open(out_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = []
        
    def check_done(m, c, s, seed):
        for res in all_results:
            if res['model'] == m and res['corruption'] == c and res['snr'] == s and res['seed'] == seed:
                return True
        return False
        
    num_classes = 5
    in_channels = 12
    
    total_evals = len(models) * len(corruptions) * len(snrs) * len(seeds)
    print(f"Total Sweep Evaluations: {total_evals}")
    
    for seed in seeds:
        for model_name in models:
            ckpt_path = f"experiments/checkpoints/multiclass_{model_name}_seed{seed}_best.pt"
            if not os.path.exists(ckpt_path):
                # Optionally fallback
                ckpt_path = f"experiments/checkpoints/source_only_{model_name}_seed{seed}_best.pt"
                if not os.path.exists(ckpt_path):
                    print(f"Skipping {model_name} seed {seed} (No Checkpoint)")
                    continue
                
            # Initialize model
            if model_name == 'wavkan': model = WavKANClassifier(input_dim=1000, num_classes=num_classes, hidden_dim=64, in_channels=in_channels).to(device)
            elif model_name == 'spline_kan': model = SplineKANClassifier(input_dim=1000, num_classes=num_classes, hidden_dim=64, in_channels=in_channels).to(device)
            elif model_name == 'resnet': model = ResNet1D(in_channels=in_channels, num_classes=num_classes, seq_len=1000).to(device)
            elif model_name == 'vit': model = ViT1D(seq_len=1000, num_classes=num_classes, in_channels=in_channels).to(device)
            elif model_name == 'simple_mlp': model = SimpleMLP(input_dim=1000, num_classes=num_classes, in_channels=in_channels).to(device)
            elif model_name == 'inception': model = InceptionTime(in_channels=in_channels, num_classes=num_classes).to(device)
            elif model_name == 'xresnet': model = XResNet1D(in_channels=in_channels, num_classes=num_classes).to(device)
            
            model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state_dict'] if 'model_state_dict' in torch.load(ckpt_path, map_location=device) else torch.load(ckpt_path, map_location=device))
            model.eval()
            
            for corruption in corruptions:
                for snr in snrs:
                    if check_done(model_name, corruption, snr, seed):
                        continue
                        
                    label_path = args.test_file.replace('signals', 'labels') if '.npy' in args.test_file else None
                    dataset = HarmonizedDataset(args.test_file, label_path=label_path, noise_snr_db=snr, corruption_type=corruption)
                    
                    f1 = evaluate_noise(model, dataset, device, batch_size=args.batch_size)
                    
                    res_dict = {
                        'model': model_name,
                        'seed': seed,
                        'corruption': corruption,
                        'snr': snr if snr is not None else 'clean',
                        'f1_macro': float(f1)
                    }
                    all_results.append(res_dict)
                    print(f"[{model_name} | Seed {seed} | {corruption} | SNR {snr}] F1: {f1:.4f}")
                    
                    # Incremental save
                    with open(out_file, 'w') as f:
                        json.dump(all_results, f, indent=2)

    print(f"Sweep complete. Results in {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='data/ptbxl_signals.npy')
    parser.add_argument('--out_file', type=str, default='experiments/runs/robustness_matrix.json')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    main(args)
