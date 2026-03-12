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


def run_pairwise_tests(target_model, baselines, metric_loader, metric_name, alpha=0.05, **kwargs):
    """Run Welch's t-test between target_model and each baseline."""
    results = []
    target_scores = metric_loader(target_model, **kwargs)
    
    if len(target_scores) < 2:
        return results
    
    for baseline in baselines:
        baseline_scores = metric_loader(baseline, **kwargs)
        
        if len(baseline_scores) < 2:
            continue
        
        # Welch's t-test (unequal variance)
        t_stat, p_value = stats.ttest_ind(target_scores, baseline_scores, equal_var=False)
        d = cohens_d(target_scores, baseline_scores)
        
        results.append({
            'comparison': f"{target_model} vs {baseline}",
            'metric': metric_name,
            'target_mean': np.mean(target_scores),
            'target_std': np.std(target_scores),
            'baseline_mean': np.mean(baseline_scores),
            'baseline_std': np.std(baseline_scores),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': "Yes" if p_value < alpha else "No",
            'cohens_d': d,
            'effect_size': interpret_d(d)
        })
    
    return results


def main():
    print("=" * 70)
    print("  STATISTICAL SIGNIFICANCE TESTING (BONFERRONI CORRECTED)")
    print("  Welch's t-test (two-sided, unequal variance)")
    print("  Cohen's d effect size")
    print("=" * 70)
    
    baselines = ['resnet', 'vit', 'spline_kan', 'mlp']
    
    # 1. First Pass: Count comparisons for Bonferroni
    num_comparisons = 0
    # Zero-shot
    num_comparisons += len(baselines)
    # Few-shot (10, 500)
    num_comparisons += 2 * len(baselines)
    # Robustness (Clean, 0dB)
    num_comparisons += 2 * 3
    # KAN Comparison
    num_comparisons += 3
    
    alpha_base = 0.05
    alpha_adj = alpha_base / num_comparisons
    print(f"Total Comparisons: {num_comparisons}")
    print(f"Base Alpha: {alpha_base:.4f} -> Adjusted Alpha: {alpha_adj:.6f}")
    print("-" * 70)

    all_results = []
    
    # 2. Run Tests
    # 2.1 Zero-Shot
    print("\n[Zero-Shot Transfer F1]")
    results = run_pairwise_tests('wavkan', baselines, load_zeroshot_seeds, 'zero_shot_f1', alpha=alpha_adj)
    all_results.extend(results)
    
    # 2.2 Few-shot
    for k in ['10-shot', '500-shot']:
        print(f"\n[{k} Adaptation F1]")
        results = run_pairwise_tests('wavkan', baselines, load_fewshot_seeds, k, alpha=alpha_adj, k_shot=k)
        all_results.extend(results)
    
    # 2.3 Noise Robustness
    for snr in ['Clean', '0dB']:
        print(f"\n[Robustness F1 ({snr})]")
        results = run_pairwise_tests('wavkan', ['resnet', 'vit', 'spline_kan'], 
                                     load_robustness_seeds, f'robustness_{snr}', alpha=alpha_adj, snr_col=snr)
        all_results.extend(results)
    
    # 2.4 KAN Specific
    print("\n[WavKAN vs Spline-KAN]")
    for k in ['10-shot', '100-shot', '500-shot']:
        wav_scores = load_fewshot_seeds('wavkan', k)
        spl_scores = load_fewshot_seeds('spline_kan', k)
        if len(wav_scores) >= 2 and len(spl_scores) >= 2:
            t, p = stats.ttest_ind(wav_scores, spl_scores, equal_var=False)
            d = cohens_d(wav_scores, spl_scores)
            all_results.append({
                'comparison': 'wavkan vs spline_kan',
                'metric': f'kan_{k}',
                'target_mean': np.mean(wav_scores),
                'target_std': np.std(wav_scores),
                'baseline_mean': np.mean(spl_scores),
                'baseline_std': np.std(spl_scores),
                't_statistic': t,
                'p_value': p,
                'significant': "Yes" if p < alpha_adj else "No",
                'cohens_d': d,
                'effect_size': interpret_d(d)
            })

    # Display & Save
    if all_results:
        df = pd.DataFrame(all_results)
        save_path = "experiments/statistical_tests.csv"
        df.to_csv(save_path, index=False)
        
        print("\nSignificant Results (p < {:.6f}):".format(alpha_adj))
        for _, r in df[df['significant'] == 'Yes'].iterrows():
             print(f"  {r['metric']}: {r['comparison']} (p={r['p_value']:.6f}, d={r['cohens_d']:.2f})")

        # LaTeX Export
        print("\n" + "=" * 70)
        print("  LaTeX Table Snippet")
        print("=" * 70)
        for _, r in df.iterrows():
            marker = "*" if r['significant'] == 'Yes' else ""
            print(f"  {r['comparison'].replace('_','')} & {r['metric']} & "
                  f"{r['target_mean']:.3f} $\\pm$ {r['target_std']:.3f} & "
                  f"{r['baseline_mean']:.3f} $\\pm$ {r['baseline_std']:.3f} & "
                  f"{r['p_value']:.4f}{marker} & {r['cohens_d']:.2f} \\\\")


if __name__ == "__main__":
    main()
