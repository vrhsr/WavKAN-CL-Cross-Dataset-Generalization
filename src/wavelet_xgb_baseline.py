import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, roc_auc_score
import pywt
import argparse
import json
from src.dataset import HarmonizedDataset
from torch.utils.data import DataLoader
from src.utils import set_seed

def extract_cwt_features(signals, scales=np.arange(1, 15)):
    """
    Extracts statistical features from CWT coefficients for 12-lead ECG.
    signals shape: (batch_size, 12, length)
    """
    batch_size, num_leads, length = signals.shape
    features = []
    
    for i in tqdm(range(batch_size), desc="Extracting CWT features", leave=False):
        sample_features = []
        for lead in range(num_leads):
            sig = signals[i, lead, :]
            # CWT
            coeffs, _ = pywt.cwt(sig, scales, 'cmor1.5-1.0')
            coeffs = np.abs(coeffs)
            
            # Sub-divide into frequency bands (e.g. 3 bands: HF, MF, LF based on scales)
            band1 = coeffs[:5, :]
            band2 = coeffs[5:10, :]
            band3 = coeffs[10:, :]
            
            for band in [band1, band2, band3]:
                sample_features.extend([
                    np.mean(band),
                    np.std(band),
                    np.max(band),
                    np.min(band),
                    np.sum(band**2) # Energy
                ])
                
        features.append(sample_features)
        
    return np.array(features)

def evaluate_xgb(model, X, Y):
    preds = model.predict(X)
    probs_list = model.predict_proba(X)
    
    # predict_proba returns a list of shape (num_classes, num_samples, 2). We need (num_samples, num_classes)
    probs = np.zeros((X.shape[0], len(probs_list)))
    for i, p in enumerate(probs_list):
        if p.shape[1] == 2:
            probs[:, i] = p[:, 1]
        else:
            probs[:, i] = p[:, 0]
            
    f1_macro = f1_score(Y, preds, average='macro', zero_division=0)
    try:
        auroc_macro = roc_auc_score(Y, probs, average='macro', multi_class='ovr')
    except ValueError:
        auroc_macro = 0.0
        
    return f1_macro, auroc_macro

def get_data_arrays(dataset, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_x = []
    all_y = []
    for x, y in loader:
        all_x.append(x.numpy())
        all_y.append(y.numpy())
    return np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0)

def main(args):
    print(f"=== CWT + XGBoost Baseline (Cross-Domain) ===")
    
    # 1. Load Datasets
    print(f"Loading Source Dataset: {args.source_file}")
    source_label_path = args.source_file.replace('signals', 'labels') if '.npy' in args.source_file else None
    source_dataset = HarmonizedDataset(args.source_file, label_path=source_label_path)
    
    X_source, Y_source = get_data_arrays(source_dataset)
    X_source_ft = extract_cwt_features(X_source)
    
    target_datasets = {}
    for tgt_file in args.target_files:
        print(f"Loading Target Dataset: {tgt_file}")
        tgt_label_path = tgt_file.replace('signals', 'labels') if '.npy' in tgt_file else None
        tgt_dataset = HarmonizedDataset(tgt_file, label_path=tgt_label_path)
        X_tgt, Y_tgt = get_data_arrays(tgt_dataset)
        X_tgt_ft = extract_cwt_features(X_tgt)
        target_datasets[tgt_file] = (X_tgt_ft, Y_tgt)
        
    seeds = [42, 43, 44, 45, 46] if args.seeds == 'all' else [args.seeds]
    
    all_results = []
    
    for seed in seeds:
        print(f"\n--- Training Seed {seed} ---")
        set_seed(seed)
        
        base_xgb = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        model = MultiOutputClassifier(base_xgb)
        
        print("Training XGBoost MultiOutput Classifier...")
        model.fit(X_source_ft, Y_source)
        
        # Eval Source
        src_f1, src_auc = evaluate_xgb(model, X_source_ft, Y_source)
        print(f"[Source] PTB-XL F1: {src_f1:.4f}, AUC: {src_auc:.4f}")
        
        seed_results = {
            'seed': seed,
            'source_f1': src_f1,
            'source_auc': src_auc,
            'targets': {}
        }
        
        # Eval Targets
        for tgt_file, (X_tgt_ft, Y_tgt) in target_datasets.items():
            tgt_f1, tgt_auc = evaluate_xgb(model, X_tgt_ft, Y_tgt)
            tgt_name = os.path.basename(tgt_file).replace('.npy', '').replace('.csv', '')
            print(f"[Target] {tgt_name} F1: {tgt_f1:.4f}, AUC: {tgt_auc:.4f}")
            seed_results['targets'][tgt_name] = {
                'f1': tgt_f1,
                'auc': tgt_auc
            }
            
        all_results.append(seed_results)
        
    # Aggregate and Save
    os.makedirs('experiments/runs', exist_ok=True)
    out_path = 'experiments/runs/wavelet_xgb_baseline_results.json'
    
    final_output = {
        'config': vars(args),
        'runs': all_results
    }
    
    with open(out_path, 'w') as f:
        json.dump(final_output, f, indent=2)
        
    print(f"\nSaved XGBoost Baseline Results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file', type=str, default='data/ptbxl_signals.npy')
    parser.add_argument('--target_files', nargs='+', default=['data/chapman_signals.npy', 'data/georgia_signals.npy'])
    parser.add_argument('--seeds', type=str, default='all', help="'all' or an integer")
    
    args = parser.parse_args()
    if args.seeds != 'all':
        args.seeds = int(args.seeds)
        
    main(args)
