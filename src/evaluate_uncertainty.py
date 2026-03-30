import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score
import json
import os
from tqdm import tqdm

from src.dataset import HarmonizedDataset
from src.models.wavkan import WavKANClassifier
from src.models.baselines import ResNet1D, ViT1D, SimpleMLP, InceptionTime
from src.models.spline_kan import SplineKANClassifier


def build_model(model_name, device):
    in_channels = 12
    num_classes = 5
    if model_name == 'wavkan': return WavKANClassifier(input_dim=1000, num_classes=num_classes, hidden_dim=64, in_channels=in_channels).to(device)
    if model_name == 'resnet': return ResNet1D(in_channels=in_channels, num_classes=num_classes, seq_len=1000).to(device)
    if model_name == 'vit': return ViT1D(seq_len=1000, num_classes=num_classes, in_channels=in_channels).to(device)
    if model_name == 'spline_kan': return SplineKANClassifier(input_dim=1000, num_classes=num_classes, hidden_dim=64, in_channels=in_channels).to(device)
    if model_name == 'mlp': return SimpleMLP(input_dim=1000, num_classes=num_classes, in_channels=in_channels).to(device)
    if model_name == 'inception': return InceptionTime(in_channels=in_channels, num_classes=num_classes).to(device)
    raise ValueError(f'Unknown model: {model_name}')

def enable_dropout(model):
    """Enable dropout layers during test time for MC Dropout."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    label_path = args.data_file.replace('signals', 'labels') if '.npy' in args.data_file else None
    dataset = HarmonizedDataset(args.data_file, label_path=label_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.model, device)
    ckpt = args.checkpoint or f'experiments/checkpoints/multiclass_{args.model}_seed42_best.pt'
    
    if not os.path.exists(ckpt):
        ckpt = f'experiments/checkpoints/source_only_{args.model}_seed42_best.pt'
        
    state = torch.load(ckpt, map_location=device)['model_state_dict'] if 'model_state_dict' in torch.load(ckpt) else torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    
    # 1. Base Evaluation
    model.eval()
    all_y_true = []
    
    print(f"Running MC Dropout Predictions (N={args.n_passes})...")
    mc_probs = [] # (N_passes, Num_samples, Num_classes)
    
    for pass_idx in range(args.n_passes):
        model.eval()
        enable_dropout(model) # Keep dropout active
        
        pass_probs = []
        with torch.no_grad():
            for x, y in tqdm(loader, desc=f"Pass {pass_idx+1}/{args.n_passes}", leave=False):
                x = x.to(device).float()
                
                logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy()
                pass_probs.append(probs)
                
                if pass_idx == 0:
                    all_y_true.extend(y.numpy())
                    
        mc_probs.append(np.vstack(pass_probs))
        
    mc_probs = np.array(mc_probs) # (N, S, C)
    all_y_true = np.array(all_y_true)
    
    mean_probs = np.mean(mc_probs, axis=0) # (S, C)
    std_probs = np.std(mc_probs, axis=0)   # (S, C) => Entropy/Uncertainty proxy
    
    # Macro F1 at full retention
    preds = (mean_probs >= 0.5).astype(int)
    full_f1 = f1_score(all_y_true, preds, average='macro', zero_division=0)
    
    # Selective Prediction Curve
    # Reject the top K% most uncertain samples (highest std_probs mean across classes)
    sample_uncertainty = np.mean(std_probs, axis=1) # (S,)
    
    retention_fractions = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]
    selective_curve = []
    
    for frac in retention_fractions:
        cutoff_idx = int(len(sample_uncertainty) * frac)
        if cutoff_idx == 0:
            break
            
        # Indices of least uncertain samples
        retained_indices = np.argsort(sample_uncertainty)[:cutoff_idx]
        
        ret_preds = preds[retained_indices]
        ret_true = all_y_true[retained_indices]
        
        f1 = f1_score(ret_true, ret_preds, average='macro', zero_division=0)
        selective_curve.append({
            'retention': frac,
            'f1_macro': float(f1)
        })
        print(f"Retention {frac*100}% -> F1: {f1:.4f}")
        
    # Brier Score Calibration
    brier_scores = []
    for c in range(all_y_true.shape[1]):
        if len(np.unique(all_y_true[:, c])) > 1:
            brier_scores.append(brier_score_loss(all_y_true[:, c], mean_probs[:, c]))
    macro_brier = float(np.mean(brier_scores))
    
    output = {
        'model': args.model,
        'full_f1_macro': float(full_f1),
        'macro_brier_score': macro_brier,
        'selective_prediction_curve': selective_curve
    }
    
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, 'w') as f:
        json.dump(output, f, indent=2)
        
    print(f"\nSaved Uncertainty Results to {args.out_file}")
    print(f"Macro Brier Score: {macro_brier:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data_file', default='data/ptbxl_signals.npy')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--out_file', default='experiments/runs/uncertainty_eval.json')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_passes', type=int, default=50, help="Number of MC Dropout forward passes")
    main(parser.parse_args())
