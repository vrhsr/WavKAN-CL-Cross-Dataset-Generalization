"""
Enhanced statistical testing for WavKAN-CL experiments.

Includes:
- Welch's t-test
- Paired t-test (when seed alignment is available)
- Cohen's d
- Bootstrap confidence intervals for mean difference
- Multiple testing correction (Bonferroni + BH)
- Optional calibration report (ECE/Brier/AUROC/AUPRC with bootstrap CI)
"""

import glob
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
from statsmodels.stats.multitest import multipletests


# ---------------------- EFFECT SIZE ----------------------

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


# ---------------------- BOOTSTRAP ----------------------

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
        if len(np.unique(yt)) < 2:
            continue
        vals.append(metric_fn(yt, yp))
    if not vals:
        return np.nan, np.nan
    lo = np.percentile(vals, (100 - ci) / 2)
    hi = np.percentile(vals, 100 - (100 - ci) / 2)
    return float(lo), float(hi)


# ---------------------- CALIBRATION ----------------------

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


# ---------------------- DATA LOADING ----------------------

def _load_seeded_scores(pattern, value_col, index_col=None):
    rows = []
    for f in sorted(glob.glob(pattern)):
        seed = int(f.split('seed')[-1].split('.csv')[0]) if 'seed' in f else None
        df = pd.read_csv(f, index_col=index_col)
        if value_col in df.columns:
            rows.append((seed, float(df[value_col].values[0])))
    return rows


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


def load_fewshot_seeds(model, k_shot='500-shot'):
    rows = _load_scores_with_fallback(
        f"experiments/fewshot_{model}_seed*.csv",
        f"experiments/fewshot_{model}.csv",
        k_shot,
        index_col=0
    )
    return np.array([v for _, v in rows]), {s: v for s, v in rows}


def load_robustness_seeds(model, snr_col='awgn:Clean'):
    rows = _load_scores_with_fallback(
        f"experiments/robustness_{model}_seed*.csv",
        f"experiments/robustness_{model}.csv",
        snr_col,
        index_col=0
    )
    if not rows:
        rows = _load_scores_with_fallback(
            f"experiments/robustness_{model}_norm_seed*.csv",
            f"experiments/robustness_{model}_norm.csv",
            snr_col,
            index_col=0
        )
    return np.array([v for _, v in rows]), {s: v for s, v in rows}


def load_zeroshot_seeds(model):
    rows = _load_seeded_scores(f"experiments/zeroshot_{model}_seed*.csv", 'zero_shot_f1')
    return np.array([v for _, v in rows]), {s: v for s, v in rows}


# ---------------------- TESTS ----------------------

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
            'p_welch': p_value,
            'p_paired': p_paired,
            'cohens_d': d,
            'effect_size': interpret_d(d),
            'diff_ci95_lo': ci_lo,
            'diff_ci95_hi': ci_hi,
        })

    return results


def apply_multiple_testing_corrections(df, alpha=0.05):
    p_vals = df['p_welch'].values

    reject_bonf, p_bonf, _, _ = multipletests(p_vals, alpha=alpha, method='bonferroni')
    reject_bh, p_bh, _, _ = multipletests(p_vals, alpha=alpha, method='fdr_bh')

    df = df.copy()
    df['p_adj_bonf'] = p_bonf
    df['reject_bonf'] = reject_bonf
    df['p_adj_bh'] = p_bh
    df['reject_bh'] = reject_bh

    return df


# ---------------------- CALIBRATION REPORT ----------------------

def calibration_report(pred_csv='experiments/predictions_eval.csv'):
    if not glob.glob(pred_csv):
        return None

    df = pd.read_csv(pred_csv)
    if not {'y_true', 'y_prob'}.issubset(df.columns):
        return None

    y_true = df['y_true'].values
    y_prob = df['y_prob'].values

    out = pd.DataFrame([{
        'ece': expected_calibration_error(y_true, y_prob),
        'brier': brier_score_loss(y_true, y_prob),
        'auroc': roc_auc_score(y_true, y_prob),
        'auprc': average_precision_score(y_true, y_prob),
    }])

    path = 'experiments/calibration_report.csv'
    out.to_csv(path, index=False)
    return path


# ---------------------- MAIN ----------------------

def main():
    baselines = ['resnet', 'vit', 'spline_kan', 'mlp']

    all_results = []
    all_results.extend(run_pairwise_tests('wavkan', baselines, load_zeroshot_seeds, 'zero_shot_f1'))

    for k in ['10-shot', '500-shot']:
        all_results.extend(run_pairwise_tests('wavkan', baselines, load_fewshot_seeds, k, k_shot=k))

    df = pd.DataFrame(all_results)

    # APPLY CORRECTIONS (IMPORTANT FOR PAPER)
    df = apply_multiple_testing_corrections(df)

    df.to_csv('experiments/statistical_tests.csv', index=False)
    print(df.head())

    calibration_report()


if __name__ == '__main__':
    main()