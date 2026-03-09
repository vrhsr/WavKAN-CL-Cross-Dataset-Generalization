"""
Aggregate per-seed experiment results into Mean ± Std summary tables.
Reads experiments/fewshot_*_seed*.csv and experiments/robustness_*_seed*.csv
Produces summary CSVs for the paper.
"""
import os
import glob
import pandas as pd
import numpy as np


def aggregate_fewshot():
    """Aggregate few-shot results across seeds."""
    models = ['wavkan', 'resnet', 'vit', 'spline_kan', 'mlp']
    summary = {}
    
    for model in models:
        pattern = f"experiments/fewshot_{model}_seed*.csv"
        files = sorted(glob.glob(pattern))
        
        if not files:
            print(f"  No seed files found for {model} (pattern: {pattern})")
            continue
            
        dfs = [pd.read_csv(f, index_col=0) for f in files]
        combined = pd.concat(dfs)
        
        means = combined.mean()
        stds = combined.std()
        
        row = {}
        for col in combined.columns:
            row[f"{col}_mean"] = round(means[col], 4)
            row[f"{col}_std"] = round(stds[col], 4)
            row[col] = f"{means[col]:.3f} ± {stds[col]:.3f}"
        
        summary[model] = row
        print(f"  {model}: {len(files)} seeds aggregated")
    
    if summary:
        df = pd.DataFrame(summary).T
        df.to_csv("experiments/results_fewshot_summary.csv")
        print(f"\n  Saved: experiments/results_fewshot_summary.csv")
        print(df.to_string())
    

def aggregate_robustness():
    """Aggregate robustness results across seeds."""
    models = ['wavkan', 'resnet', 'vit', 'spline_kan']
    summary = {}
    
    for model in models:
        pattern = f"experiments/robustness_{model}_seed*.csv"
        files = sorted(glob.glob(pattern))
        
        if not files:
            continue
            
        dfs = [pd.read_csv(f, index_col=0) for f in files]
        combined = pd.concat(dfs)
        
        means = combined.mean()
        stds = combined.std()
        
        row = {}
        for col in combined.columns:
            row[col] = f"{means[col]:.3f} ± {stds[col]:.3f}"
        
        summary[model] = row
        print(f"  {model}: {len(files)} seeds aggregated")
    
    if summary:
        df = pd.DataFrame(summary).T
        df.to_csv("experiments/results_robustness_summary.csv")
        print(f"\n  Saved: experiments/results_robustness_summary.csv")
        print(df.to_string())


def aggregate_ssl():
    """Aggregate SSL few-shot results."""
    models = ['wavkan', 'spline_kan']
    summary = {}
    
    for model in models:
        path = f"experiments/fewshot_{model}_ssl.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            row = {col: f"{df[col].values[0]:.4f}" for col in df.columns}
            summary[f"{model}_ssl"] = row
            print(f"  {model}_ssl: loaded")
    
    if summary:
        df = pd.DataFrame(summary).T
        df.to_csv("experiments/results_ssl_summary.csv")
        print(f"\n  Saved: experiments/results_ssl_summary.csv")
        print(df.to_string())


def print_paper_table():
    """Print formatted table ready for paper."""
    print("\n" + "=" * 70)
    print("  PAPER-READY RESULTS TABLE")
    print("=" * 70)
    
    fewshot_path = "experiments/results_fewshot_summary.csv"
    if os.path.exists(fewshot_path):
        df = pd.read_csv(fewshot_path, index_col=0)
        # Filter to display columns
        display_cols = [c for c in df.columns if not c.endswith('_mean') and not c.endswith('_std')]
        if display_cols:
            print("\n  Table: Few-Shot Adaptation (F1, Mean ± Std, n=5)")
            print(df[display_cols].to_string())
    
    ssl_path = "experiments/results_ssl_summary.csv"
    if os.path.exists(ssl_path):
        df = pd.read_csv(ssl_path, index_col=0)
        print("\n  Table: SSL Pre-trained Few-Shot (F1)")
        print(df.to_string())
    
    robustness_path = "experiments/results_robustness_summary.csv"
    if os.path.exists(robustness_path):
        df = pd.read_csv(robustness_path, index_col=0)
        print("\n  Table: Noise Robustness (F1, Mean ± Std, n=5)")
        print(df.to_string())


def aggregate_zeroshot():
    """Aggregate zero-shot results across seeds."""
    models = ['wavkan', 'resnet', 'vit', 'spline_kan', 'mlp']
    summary = {}
    
    for model in models:
        pattern = f"experiments/zeroshot_{model}_seed*.csv"
        files = sorted(glob.glob(pattern))
        
        if not files:
            print(f"  No zero-shot seed files found for {model}")
            continue
            
        dfs = [pd.read_csv(f) for f in files]
        combined = pd.concat(dfs)
        
        f1_mean = combined['zero_shot_f1'].mean()
        f1_std = combined['zero_shot_f1'].std()
        acc_mean = combined['zero_shot_acc'].mean()
        auc_mean = combined['zero_shot_auc'].mean()
        
        summary[model] = {
            'f1': f"{f1_mean:.3f} ± {f1_std:.3f}",
            'f1_mean': round(f1_mean, 4),
            'f1_std': round(f1_std, 4),
            'acc_mean': round(acc_mean, 4),
            'auc_mean': round(auc_mean, 4)
        }
        print(f"  {model}: {len(files)} seeds, F1={f1_mean:.4f}±{f1_std:.4f}")
    
    if summary:
        df = pd.DataFrame(summary).T
        df.to_csv("experiments/results_zeroshot_summary.csv")
        print(f"\n  Saved: experiments/results_zeroshot_summary.csv")


if __name__ == "__main__":
    print("=" * 50)
    print("  AGGREGATING EXPERIMENT RESULTS")
    print("=" * 50)
    
    print("\n--- Zero-Shot Results ---")
    aggregate_zeroshot()
    
    print("\n--- Few-Shot Results ---")
    aggregate_fewshot()
    
    print("\n--- Robustness Results ---")
    aggregate_robustness()
    
    print("\n--- SSL Results ---")
    aggregate_ssl()
    
    print_paper_table()
    
    print("\n\nDone. All summary CSVs saved to experiments/")
