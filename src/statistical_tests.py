"""
Statistical Significance Testing for WavKAN-CL Paper.
Implements Welch's t-test and Cohen's d effect size across 5 seeds.
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy import stats

def cohens_d(group1, group2):
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def interpret_d(d):
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.8:
        return "medium"
    else:
        return "large"

def load_fewshot_seeds(model, k_shot='500-shot'):
    """Load F1 scores for a model across all seeds for a given k-shot setting."""
    pattern = f"experiments/fewshot_{model}_seed*.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        return np.array([])
    
    values = []
    for f in files:
        df = pd.read_csv(f, index_col=0)
        if k_shot in df.columns:
            values.append(df[k_shot].values[0])
    return np.array(values)

def load_robustness_seeds(model, snr_col='Clean'):
    """Load F1 scores for a model across all seeds for a given SNR level."""
    pattern = f"experiments/robustness_{model}_seed*.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        return np.array([])
    
    values = []
    for f in files:
        df = pd.read_csv(f, index_col=0)
        if snr_col in df.columns:
            values.append(df[snr_col].values[0])
    return np.array(values)

def load_zeroshot_seeds(model):
    """Load zero-shot F1 scores for a model across all seeds."""
    pattern = f"experiments/zeroshot_{model}_seed*.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        return np.array([])
    
    values = []
    for f in files:
        df = pd.read_csv(f)
        if 'zero_shot_f1' in df.columns:
            values.append(df['zero_shot_f1'].values[0])
    return np.array(values)


def run_pairwise_tests(target_model, baselines, metric_loader, metric_name, **kwargs):
    """Run Welch's t-test between target_model and each baseline."""
    results = []
    target_scores = metric_loader(target_model, **kwargs)
    
    if len(target_scores) < 2:
        print(f"  Insufficient data for {target_model} ({metric_name})")
        return results
    
    for baseline in baselines:
        baseline_scores = metric_loader(baseline, **kwargs)
        
        if len(baseline_scores) < 2:
            print(f"  Insufficient data for {baseline} ({metric_name})")
            continue
        
        # Welch's t-test (unequal variance)
        t_stat, p_value = stats.ttest_ind(target_scores, baseline_scores, equal_var=False)
        d = cohens_d(target_scores, baseline_scores)
        
        results.append({
            'comparison': f"{target_model} vs {baseline}",
            'metric': metric_name,
            'target_mean': f"{np.mean(target_scores):.4f}",
            'target_std': f"{np.std(target_scores):.4f}",
            'baseline_mean': f"{np.mean(baseline_scores):.4f}",
            'baseline_std': f"{np.std(baseline_scores):.4f}",
            't_statistic': round(t_stat, 4),
            'p_value': round(p_value, 6),
            'significant': "Yes" if p_value < 0.05 else "No",
            'cohens_d': round(d, 4),
            'effect_size': interpret_d(d)
        })
    
    return results


def main():
    print("=" * 70)
    print("  STATISTICAL SIGNIFICANCE TESTING")
    print("  Welch's t-test (two-sided, unequal variance)")
    print("  Cohen's d effect size")
    print("=" * 70)
    
    baselines = ['resnet', 'vit', 'spline_kan', 'mlp']
    all_results = []
    
    # 1. Zero-Shot F1     
    print("\n--- Zero-Shot Transfer F1 ---")
    results = run_pairwise_tests('wavkan', baselines, load_zeroshot_seeds, 'zero_shot_f1')
    all_results.extend(results)
    for r in results:
        sig = "***" if r['p_value'] < 0.001 else ("**" if r['p_value'] < 0.01 else ("*" if r['p_value'] < 0.05 else "ns"))
        print(f"  {r['comparison']}: t={r['t_statistic']}, p={r['p_value']} {sig}, d={r['cohens_d']} ({r['effect_size']})")
    
    # 2. Few-shot F1 (10-shot and 500-shot)
    for k in ['10-shot', '500-shot']:
        print(f"\n--- {k} Adaptation F1 ---")
        results = run_pairwise_tests('wavkan', baselines, load_fewshot_seeds, k, k_shot=k)
        all_results.extend(results)
        for r in results:
            sig = "***" if r['p_value'] < 0.001 else ("**" if r['p_value'] < 0.01 else ("*" if r['p_value'] < 0.05 else "ns"))
            print(f"  {r['comparison']}: t={r['t_statistic']}, p={r['p_value']} {sig}, d={r['cohens_d']} ({r['effect_size']})")
    
    # 3. Noise Robustness (Clean vs 0dB)
    for snr in ['Clean', '0dB']:
        print(f"\n--- Robustness F1 ({snr}) ---")
        results = run_pairwise_tests('wavkan', ['resnet', 'vit', 'spline_kan'], 
                                     load_robustness_seeds, f'robustness_{snr}', snr_col=snr)
        all_results.extend(results)
        for r in results:
            sig = "***" if r['p_value'] < 0.001 else ("**" if r['p_value'] < 0.01 else ("*" if r['p_value'] < 0.05 else "ns"))
            print(f"  {r['comparison']}: t={r['t_statistic']}, p={r['p_value']} {sig}, d={r['cohens_d']} ({r['effect_size']})")
    
    # 4. KAN comparison: WavKAN vs Spline-KAN specifically
    print("\n--- KAN Architecture Comparison (WavKAN vs Spline-KAN) ---")
    for k in ['10-shot', '100-shot', '500-shot']:
        wav_scores = load_fewshot_seeds('wavkan', k)
        spl_scores = load_fewshot_seeds('spline_kan', k)
        if len(wav_scores) >= 2 and len(spl_scores) >= 2:
            t, p = stats.ttest_ind(wav_scores, spl_scores, equal_var=False)
            d = cohens_d(wav_scores, spl_scores)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            print(f"  {k}: WavKAN={np.mean(wav_scores):.4f}±{np.std(wav_scores):.4f} vs "
                  f"Spline={np.mean(spl_scores):.4f}±{np.std(spl_scores):.4f} | "
                  f"p={p:.6f} {sig} | d={d:.4f} ({interpret_d(d)})")
    
    # Save all results
    if all_results:
        df = pd.DataFrame(all_results)
        save_path = "experiments/statistical_tests.csv"
        df.to_csv(save_path, index=False)
        print(f"\n  All statistical tests saved to {save_path}")
        
        # Also save a LaTeX-ready table
        print("\n" + "=" * 70)
        print("  LaTeX-Ready Statistical Comparison Table")
        print("=" * 70)
        for r in all_results:
            print(f"  {r['comparison']} & {r['metric']} & "
                  f"{r['target_mean']} $\\pm$ {r['target_std']} & "
                  f"{r['baseline_mean']} $\\pm$ {r['baseline_std']} & "
                  f"{r['p_value']} & {r['cohens_d']} \\\\")


if __name__ == "__main__":
    main()
