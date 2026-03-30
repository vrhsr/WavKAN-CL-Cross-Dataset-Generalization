import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score
import os
import json

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

def evaluate_subset(model, dataset, indices, device, batch_size=64):
    if len(indices) == 0:
        return np.nan
        
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device).float(), y.to(device).float()
            
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    return f1_score(all_labels, all_preds, average='macro', zero_division=0)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading metadata from {args.metadata_csv}...")
    
    df_meta = pd.read_csv(args.metadata_csv)
    
    # We will evaluate on strat_fold == 10 (Test set) if it exists
    if 'strat_fold' in df_meta.columns:
        test_indices = df_meta[df_meta['strat_fold'] == 10].index.tolist()
        if len(test_indices) == 0:
            test_indices = list(range(len(df_meta)))
    else:
        test_indices = list(range(len(df_meta)))
        
    print(f"Evaluating on {len(test_indices)} test samples...")
    
    label_path = args.data_file.replace('signals', 'labels') if '.npy' in args.data_file else None
    dataset = HarmonizedDataset(args.data_file, label_path=label_path)
    
    # Define demographic groups
    unique_sex = df_meta['sex'].dropna().unique()
    
    # Age groups
    age_groups = {
        '<50': lambda x: x < 50,
        '50-70': lambda x: 50 <= x <= 70,
        '>70': lambda x: x > 70
    }
    
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    results = []
    
    for model_name in models:
        print(f"\nEvaluating Demographics for {model_name}...")
        ckpt = f'experiments/checkpoints/multiclass_{model_name}_seed42_best.pt'
        if not os.path.exists(ckpt):
             ckpt = f'experiments/checkpoints/source_only_{model_name}_seed42_best.pt'
             
        model = build_model(model_name, device)
        try:
            state = torch.load(ckpt, map_location=device)['model_state_dict'] if 'model_state_dict' in torch.load(ckpt, map_location=device) else torch.load(ckpt, map_location=device)
            model.load_state_dict(state, strict=False)
        except Exception as e:
            print(f"Skipping {model_name}: {e}")
            continue
            
        res_dict = {'model': model_name}
        
        # Overall Test F1
        overall_f1 = evaluate_subset(model, dataset, test_indices, device, args.batch_size)
        res_dict['overall_f1'] = overall_f1
        
        # Sex Subgroups
        sex_f1s = []
        for s in unique_sex:
            indices = [i for i in test_indices if df_meta.iloc[i]['sex'] == s]
            f1 = evaluate_subset(model, dataset, indices, device, args.batch_size)
            res_dict[f'sex_{s}_f1'] = f1
            if not np.isnan(f1): sex_f1s.append(f1)
            
        if len(sex_f1s) > 1:
            res_dict['fairness_gap_sex'] = max(sex_f1s) - min(sex_f1s)
            
        # Age Subgroups
        age_f1s = []
        for age_name, age_fn in age_groups.items():
            indices = [i for i in test_indices if pd.notnull(df_meta.iloc[i]['age']) and age_fn(df_meta.iloc[i]['age'])]
            f1 = evaluate_subset(model, dataset, indices, device, args.batch_size)
            res_dict[f'age_{age_name}_f1'] = f1
            if not np.isnan(f1): age_f1s.append(f1)
            
        if len(age_f1s) > 1:
            res_dict['fairness_gap_age'] = max(age_f1s) - min(age_f1s)
            
        results.append(res_dict)
        
    df_results = pd.DataFrame(results)
    
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    df_results.to_csv(args.out_file, index=False)
    
    print("\n--- Demographics Fairness Results ---")
    print(df_results.to_string(index=False))
    print(f"\nSaved fairness benchmark to {args.out_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default='wavkan,spline_kan,resnet,vit,mlp,inception')
    parser.add_argument('--data_file', default='data/ptbxl_signals.npy')
    parser.add_argument('--metadata_csv', default='data/ptbxl_metadata.csv')
    parser.add_argument('--out_file', default='experiments/runs/demographics_fairness.csv')
    parser.add_argument('--batch_size', type=int, default=64)
    main(parser.parse_args())
