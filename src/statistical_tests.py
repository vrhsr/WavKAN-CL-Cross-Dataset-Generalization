"""
Enhanced statistical testing for WavKAN-CL experiments.

Includes:
- Welch's t-test
- Paired t-test (when seed alignment is available)
- Cohen's d
- Bootstrap confidence intervals for mean difference
- Optional calibration report (ECE/Brier/AUROC/AUPRC with bootstrap CI)
"""
import glob
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(1, (n1 + n2 - 2)))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def interpret_d(d):
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.8:
        return "medium"
    return "large"


def bootstrap_mean_diff_ci(a, b, n_boot=5000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        a_s = rng.choice(a, size=len(a), replace=True)
        b_s = rng.choice(b, size=len(b), replace=True)
        diffs.append(np.mean(a_s) - np.mean(b_s))
    lo = np.percentile(diffs, (100 - ci) / 2)
    hi = np.percentile(diffs, 100 - (100 - ci) / 2)
    return float(lo), float(hi)


def bootstrap_metric_ci(y_true, y_prob, metric_fn, n_boot=2000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    vals = []
    for _ in range(n_boot):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        yt = y_true[sample_idx]
        yp = y_prob[sample_idx]
        # Avoid degenerate bootstrap sample for AUC metrics
        if len(np.unique(yt)) < 2:
            continue
        vals.append(metric_fn(yt, yp))
    if not vals:
        return np.nan, np.nan
    lo = np.percentile(vals, (100 - ci) / 2)
    hi = np.percentile(vals, 100 - (100 - ci) / 2)
    return float(lo), float(hi)


def expected_calibration_error(y_true, y_prob, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if np.sum(mask) == 0:
            continue
        acc = np.mean(y_true[mask])
        conf = np.mean(y_prob[mask])
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def _load_scores_with_fallback(seed_pattern, legacy_pattern, value_col, index_col=None):
    rows = _load_seeded_scores(seed_pattern, value_col, index_col=index_col)
    if rows:
        return rows

    legacy_rows = []
    for f in sorted(glob.glob(legacy_pattern)):
        if 'seed' in f:
            continue
        df = pd.read_csv(f, index_col=index_col)
        if value_col in df.columns:
            legacy_rows.append((None, float(df[value_col].values[0])))
    return legacy_rows


def _load_seeded_scores(pattern, value_col, index_col=None):
    rows = []
    for f in sorted(glob.glob(pattern)):
        seed = int(f.split('seed')[-1].split('.csv')[0]) if 'seed' in f else None
        df = pd.read_csv(f, index_col=index_col)
        if value_col in df.columns:
            rows.append((seed, float(df[value_col].values[0])))
    return rows


def load_fewshot_seeds(model, k_shot='500-shot'):
    rows = _load_scores_with_fallback(f"experiments/fewshot_{model}_seed*.csv", f"experiments/fewshot_{model}.csv", k_shot, index_col=0)
    return np.array([v for _, v in rows]), {s: v for s, v in rows}


def load_robustness_seeds(model, snr_col='awgn:Clean'):
    rows = _load_scores_with_fallback(f"experiments/robustness_{model}_seed*.csv", f"experiments/robustness_{model}.csv", snr_col, index_col=0)
    if not rows:
        rows = _load_scores_with_fallback(f"experiments/robustness_{model}_norm_seed*.csv", f"experiments/robustness_{model}_norm.csv", snr_col, index_col=0)
    return np.array([v for _, v in rows]), {s: v for s, v in rows}


def load_zeroshot_seeds(model):
    rows = _load_seeded_scores(f"experiments/zeroshot_{model}_seed*.csv", 'zero_shot_f1')
    return np.array([v for _, v in rows]), {s: v for s, v in rows}


def _paired_test(map_a, map_b):
    shared = sorted(set(map_a.keys()) & set(map_b.keys()))
    if len(shared) < 2:
        return np.nan, np.nan, len(shared)
    a = np.array([map_a[s] for s in shared])
    b = np.array([map_b[s] for s in shared])
    t, p = stats.ttest_rel(a, b)
    return float(t), float(p), len(shared)


def run_pairwise_tests(target_model, baselines, metric_loader, metric_name, alpha=0.05, **kwargs):
    results = []
    target_scores, target_seed_map = metric_loader(target_model, **kwargs)
    if len(target_scores) < 2:
        return results

    for baseline in baselines:
        baseline_scores, baseline_seed_map = metric_loader(baseline, **kwargs)
        if len(baseline_scores) < 2:
            continue

        t_stat, p_value = stats.ttest_ind(target_scores, baseline_scores, equal_var=False)
        d = cohens_d(target_scores, baseline_scores)
        ci_lo, ci_hi = bootstrap_mean_diff_ci(target_scores, baseline_scores)
        t_paired, p_paired, n_paired = _paired_test(target_seed_map, baseline_seed_map)

        results.append({
            'comparison': f"{target_model} vs {baseline}",
            'metric': metric_name,
            'target_mean': np.mean(target_scores),
            'target_std': np.std(target_scores),
            'baseline_mean': np.mean(baseline_scores),
            'baseline_std': np.std(baseline_scores),
            't_welch': t_stat,
            'p_welch': p_value,
            'p_significant': "Yes" if p_value < alpha else "No",
            'cohens_d': d,
            'effect_size': interpret_d(d),
            'diff_ci95_lo': ci_lo,
            'diff_ci95_hi': ci_hi,
            't_paired': t_paired,
            'p_paired': p_paired,
            'n_paired': n_paired,
        })

    return results


def calibration_report(pred_csv='experiments/predictions_eval.csv', n_boot=2000):
    """Generate uncertainty/calibration report if prediction file is available.

    Expected columns: y_true, y_prob
    """
    if not glob.glob(pred_csv):
        return None
    df = pd.read_csv(pred_csv)
    if not {'y_true', 'y_prob'}.issubset(df.columns):
        return None

    y_true = df['y_true'].values.astype(int)
    y_prob = df['y_prob'].values.astype(float)

    row = {
        'n': len(df),
        'ece': expected_calibration_error(y_true, y_prob),
        'brier': brier_score_loss(y_true, y_prob),
        'auroc': roc_auc_score(y_true, y_prob),
        'auprc': average_precision_score(y_true, y_prob),
    }
    for name, fn in [
        ('ece', expected_calibration_error),
        ('brier', brier_score_loss),
        ('auroc', roc_auc_score),
        ('auprc', average_precision_score),
    ]:
        lo, hi = bootstrap_metric_ci(y_true, y_prob, fn, n_boot=n_boot)
        row[f'{name}_ci95_lo'] = lo
        row[f'{name}_ci95_hi'] = hi

    out = pd.DataFrame([row])
    out_path = 'experiments/calibration_report.csv'
    out.to_csv(out_path, index=False)
    return out_path


def main():
    print("=" * 70)
    print("  ENHANCED STATISTICAL TESTING")
    print("  Welch + Paired tests + Bootstrap CI")
    print("=" * 70)

    baselines = ['resnet', 'vit', 'spline_kan', 'mlp']
    num_comparisons = len(baselines) + 2 * len(baselines) + 2 * 3 + 3
    alpha_adj = 0.05 / num_comparisons
    print(f"Total Comparisons: {num_comparisons}")
    print(f"Adjusted Alpha (Bonferroni): {alpha_adj:.6f}")

    all_results = []
    all_results.extend(run_pairwise_tests('wavkan', baselines, load_zeroshot_seeds, 'zero_shot_f1', alpha=alpha_adj))

    for k in ['10-shot', '500-shot']:
        all_results.extend(run_pairwise_tests('wavkan', baselines, load_fewshot_seeds, k, alpha=alpha_adj, k_shot=k))

    for snr in ['awgn:Clean', 'awgn:0dB']:
        all_results.extend(run_pairwise_tests('wavkan', ['resnet', 'vit', 'spline_kan'],
                                              load_robustness_seeds, f'robustness_{snr}', alpha=alpha_adj, snr_col=snr))

    # KAN specific
    for k in ['10-shot', '100-shot', '500-shot']:
        wav_scores, wav_map = load_fewshot_seeds('wavkan', k)
        spl_scores, spl_map = load_fewshot_seeds('spline_kan', k)
        if len(wav_scores) >= 2 and len(spl_scores) >= 2:
            t_w, p_w = stats.ttest_ind(wav_scores, spl_scores, equal_var=False)
            ci_lo, ci_hi = bootstrap_mean_diff_ci(wav_scores, spl_scores)
            t_p, p_p, n_p = _paired_test(wav_map, spl_map)
            all_results.append({
                'comparison': 'wavkan vs spline_kan',
                'metric': f'kan_{k}',
                'target_mean': np.mean(wav_scores),
                'target_std': np.std(wav_scores),
                'baseline_mean': np.mean(spl_scores),
                'baseline_std': np.std(spl_scores),
                't_welch': t_w,
                'p_welch': p_w,
                'p_significant': "Yes" if p_w < alpha_adj else "No",
                'cohens_d': cohens_d(wav_scores, spl_scores),
                'effect_size': interpret_d(cohens_d(wav_scores, spl_scores)),
                'diff_ci95_lo': ci_lo,
                'diff_ci95_hi': ci_hi,
                't_paired': t_p,
                'p_paired': p_p,
                'n_paired': n_p,
            })

    if all_results:
        df = pd.DataFrame(all_results)
        save_path = 'experiments/statistical_tests.csv'
        df.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")
        print(df[['metric', 'comparison', 'p_welch', 'p_paired', 'diff_ci95_lo', 'diff_ci95_hi']].head(12))

    cal_path = calibration_report()
    if cal_path:
        print(f"Saved calibration report: {cal_path}")
    else:
        print("Calibration report skipped (experiments/predictions_eval.csv missing).")


if __name__ == '__main__':
    main()
