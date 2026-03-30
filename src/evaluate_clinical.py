import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
import json
import os
from tqdm import tqdm

from src.dataset import HarmonizedDataset
from src.models.wavkan import WavKANClassifier
from src.models.baselines import ResNet1D, ViT1D, SimpleMLP, InceptionTime
from src.models.spline_kan import SplineKANClassifier
from src.models.wavkan_multiscale import MultiScaleWavKANClassifier
from src.models.dann import DANN

def build_model(model_name, device):
    in_channels = 12
    num_classes = 5
    if model_name == 'wavkan': return WavKANClassifier(input_dim=1000, num_classes=num_classes, hidden_dim=64, in_channels=in_channels).to(device)
    if model_name == 'wavkan_multiscale': return MultiScaleWavKANClassifier(input_dim=1000, num_classes=num_classes, hidden_dim=64, in_channels=in_channels).to(device)
    if model_name == 'resnet': return ResNet1D(in_channels=in_channels, num_classes=num_classes, seq_len=1000).to(device)
    if model_name == 'vit': return ViT1D(seq_len=1000, num_classes=num_classes, in_channels=in_channels).to(device)
    if model_name == 'spline_kan': return SplineKANClassifier(input_dim=1000, num_classes=num_classes, hidden_dim=64, in_channels=in_channels).to(device)
    if model_name == 'mlp': return SimpleMLP(input_dim=1000, num_classes=num_classes, in_channels=in_channels).to(device)
    if model_name == 'inception': return InceptionTime(in_channels=in_channels, num_classes=num_classes).to(device)
    if model_name == 'dann': return DANN(in_channels=in_channels, num_classes=num_classes, feature_dim=256).to(device)
    raise ValueError(f'Unknown model: {model_name}')

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Clinical Metrics Evaluation on {device}")
    
    label_path = args.data_file.replace('signals', 'labels') if '.npy' in args.data_file else None
    dataset = HarmonizedDataset(args.data_file, label_path=label_path)
    # Split into Val / Test since we need a threshold-tuning set
    from sklearn.model_selection import train_test_split
    val_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.5, random_state=42)
    
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    test_subset = torch.utils.data.Subset(dataset, test_idx)
    
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.model, device)
    ckpt = args.checkpoint or f'experiments/checkpoints/multiclass_{args.model}_seed42_best.pt'
    if not os.path.exists(ckpt):
        ckpt = f'experiments/checkpoints/source_only_{args.model}_seed42_best.pt'
        
    state = torch.load(ckpt, map_location=device)['model_state_dict'] if 'model_state_dict' in torch.load(ckpt) else torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    
    def get_predictions(loader):
        all_y = []
        all_p = []
        with torch.no_grad():
            for x, y in tqdm(loader, leave=False):
                x = x.to(device).float()
                if args.model == 'dann':
                    logits = model.predict(x)
                else:
                    logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_p.extend(probs)
                all_y.extend(y.numpy())
        return np.array(all_y), np.array(all_p)
        
    y_val, p_val = get_predictions(val_loader)
    y_test, p_test = get_predictions(test_loader)
    
    num_classes = 5
    optimal_thresholds = np.zeros(num_classes)
    
    clinical_metrics = {}
    
    # 1. Optimize Thresholds on Validation Set
    print("\n[Optimal Thresholds & Clinical Operating Points]")
    for c in range(num_classes):
        fpr, tpr, thresholds = roc_curve(y_val[:, c], p_val[:, c])
        
        # Find threshold maximizing F1
        best_f1 = 0
        best_t = 0.5
        for t in thresholds:
            preds = (p_val[:, c] >= t).astype(int)
            f = f1_score(y_val[:, c], preds, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_t = t
                
        optimal_thresholds[c] = best_t
        
        # Metric: Sensitivity @ 95% Specificity
        idx_spec = np.where((1 - fpr) >= 0.95)[0]
        sens_at_95spec = tpr[idx_spec[-1]] if len(idx_spec) > 0 else 0.0
        
        # Metric: Specificity @ 95% Sensitivity
        idx_sens = np.where(tpr >= 0.95)[0]
        spec_at_95sens = (1 - fpr[idx_sens[0]]) if len(idx_sens) > 0 else 0.0
        
        clinical_metrics[f'Class_{c}'] = {
            'optimal_threshold': float(best_t),
            'sensitivity_at_95_specificity': float(sens_at_95spec),
            'specificity_at_95_sensitivity': float(spec_at_95sens),
            'val_f1_optimal': float(best_f1)
        }
        print(f"Class {c} -> Opt Thresh: {best_t:.3f} | Sens@95%Spec: {sens_at_95spec:.3f} | Spec@95%Sens: {spec_at_95sens:.3f}")
        
    # 2. Evaluate on Test Set
    pred_05 = (p_test >= 0.5).astype(int)
    pred_opt = (p_test >= optimal_thresholds).astype(int)
    
    f1_05 = f1_score(y_test, pred_05, average='macro', zero_division=0)
    f1_opt = f1_score(y_test, pred_opt, average='macro', zero_division=0)
    
    try:
        auroc = roc_auc_score(y_test, p_test, average='macro', multi_class='ovr')
    except:
        auroc = 0.0
        
    print(f"\n[Test Evaluation]")
    print(f"Macro AUROC (Threshold-Free): {auroc:.4f}")
    print(f"Macro F1 (Fixed 0.5):         {f1_05:.4f}")
    print(f"Macro F1 (Optimal Thresholds):{f1_opt:.4f}")
    
    output = {
        'model': args.model,
        'macro_auroc': float(auroc),
        'macro_f1_fixed_05': float(f1_05),
        'macro_f1_optimal': float(f1_opt),
        'per_class_clinical': clinical_metrics
    }
    
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, 'w') as f:
        json.dump(output, f, indent=2)
        
    print(f"\nSaved Clinical Evaluation to {args.out_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data_file', default='data/ptbxl_signals.npy')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--out_file', default='experiments/runs/clinical_eval.json')
    parser.add_argument('--batch_size', type=int, default=64)
    main(parser.parse_args())
