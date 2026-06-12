"""
standard/extras.py — Optional Analysis Extras
==============================================
Run after standard/run_analysis.py to generate additional outputs.
All settings are controlled from config.py.

Usage:
    python -m standard.extras
"""

import os
import sys
import glob
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder

import config
from standard.preprocessing import (
    load_experiment, bin_features, filter_mass_range,
    filter_low_variance, filter_low_abundance, preprocess,
)
from standard.pipeline import compute_vip_1comp, fit_plsda
from shared.classifier_comparison_standard import (
    random_forest, svm_classify, gradient_boosting,
    logistic_regression, lda_classify, ridge_classify,
    feature_importance_analysis,
    make_groups, make_preprocessor, auto_n_splits,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, 'results', 'standard')


# ── Shared data loading ────────────────────────────────────────────────────────

def _load_and_preprocess(experiment_name):
    """
    Load, filter, and preprocess one experiment.
    Returns X_binned, X, X_norm, y_labels, sample_names, mz, X_filt_raw.
    X_binned is the m/z-filtered binned matrix (before preprocessing) — needed
    for GroupKFold CV. X is fully preprocessed (for descriptive analyses).
    """
    experiment_dir = os.path.join(BASE_DIR, 'data', experiment_name)
    assert os.path.isdir(experiment_dir), (
        f"Experiment folder not found: {experiment_dir!r}"
    )

    X_raw, y_labels, sample_names, mz = load_experiment(experiment_dir)

    X_binned, mz = bin_features(X_raw, mz, bin_width=config.BIN_WIDTH)
    X_binned, mz = filter_mass_range(X_binned, mz,
                                     mz_min=config.MZ_MIN, mz_max=config.MZ_MAX)

    if config.VARIANCE_PERCENTILE > 0:
        X_filt, mz = filter_low_variance(X_binned, mz.copy(),
                                         percentile=config.VARIANCE_PERCENTILE)
    else:
        X_filt, mz = X_binned.copy(), mz.copy()

    if config.ABUNDANCE_PERCENTILE > 0:
        X_filt, mz = filter_low_abundance(X_filt, mz,
                                          percentile=config.ABUNDANCE_PERCENTILE)

    X_filt_raw = X_filt.copy()
    X_norm = preprocess(X_filt.copy(), normalization=config.NORMALIZATION,
                        log_transform=config.LOG_TRANSFORM, scaling='none')
    X = preprocess(X_filt, normalization=config.NORMALIZATION,
                   log_transform=config.LOG_TRANSFORM, scaling=config.SCALING)

    return X_binned, X, X_norm, y_labels, sample_names, mz, X_filt_raw


# ── 1. Summary Report ──────────────────────────────────────────────────────────

def run_summary_report(X, y_labels, mz, safe_name, out_dir,
                       classifier_results=None):
    """Save a plain-text summary of the run."""
    os.makedirs(out_dir, exist_ok=True)
    le      = LabelEncoder().fit(y_labels)
    classes = le.classes_

    lines = [
        f"Ambient MS Metabolomics Toolkit — Summary Report",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Experiment: {safe_name}",
        "=" * 60,
        f"Samples      : {X.shape[0]}",
        f"Features     : {X.shape[1]}  (after filtering)",
        f"m/z range    : {mz.min():.1f} – {mz.max():.1f} Da",
        f"Groups       : {list(classes)}",
        "",
        "Config",
        f"  Normalization : {config.NORMALIZATION}",
        f"  Log transform : {config.LOG_TRANSFORM}",
        f"  Scaling       : {config.SCALING}",
        f"  Bin width     : {config.BIN_WIDTH} Da",
        f"  Random seed   : {config.RANDOM_SEED}",
        "",
    ]

    for g in classes:
        n = (y_labels == g).sum()
        lines.append(f"  {g:30s}  n={n}")

    if classifier_results:
        lines += ["", "Classifier Results (mean ± std test accuracy)"]
        for name, (test_accs, train_accs) in classifier_results.items():
            lines.append(
                f"  {name:25s}  test={test_accs.mean():.3f}±{test_accs.std():.3f}  "
                f"train={train_accs.mean():.3f}±{train_accs.std():.3f}"
            )

    txt_path = os.path.join(out_dir, f'summary_{safe_name}.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved → {txt_path}")


# ── 2. Within-group variation plot ─────────────────────────────────────────────

def run_variation_plot(X, y_labels, mz, safe_name, out_dir):
    """Violin plot of per-feature RSD within each group."""
    os.makedirs(out_dir, exist_ok=True)
    le      = LabelEncoder().fit(y_labels)
    classes = le.classes_

    records = []
    for cls in classes:
        Xg   = X[y_labels == cls]
        mean = Xg.mean(axis=0)
        mean[mean == 0] = 1e-12
        rsd  = Xg.std(axis=0) / mean
        for r in rsd:
            records.append({'Group': cls, 'RSD': r})

    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 1.4), 5))
    sns.violinplot(data=df, x='Group', y='RSD', palette='colorblind', ax=ax,
                   inner='box', cut=0)
    ax.set_title(f'Within-group feature variation (RSD) — {safe_name}')
    ax.set_ylabel('Relative Standard Deviation')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    out_path = os.path.join(out_dir, f'variation_{safe_name}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")


# ── 3. Feature correlation heatmap ─────────────────────────────────────────────

def run_correlation_heatmap(X, y_labels, mz, vip_scores, safe_name, out_dir,
                            n_features=20):
    """Pairwise Pearson correlation heatmap for the top N VIP features."""
    os.makedirs(out_dir, exist_ok=True)
    top_idx  = np.argsort(vip_scores)[::-1][:n_features]
    top_mz   = mz[top_idx]
    X_top    = X[:, top_idx]
    labels   = [f"{v:.2f}" for v in top_mz]

    corr = np.corrcoef(X_top.T)

    fig, ax = plt.subplots(figsize=(max(8, n_features * 0.5), max(6, n_features * 0.5)))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                xticklabels=labels, yticklabels=labels,
                annot=n_features <= 15, fmt='.2f', linewidths=0.3, ax=ax)
    ax.set_title(f'Feature Correlation (top {n_features} VIP) — {safe_name}')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f'correlation_{safe_name}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")


# ── 4. Permutation test ────────────────────────────────────────────────────────

def run_permutation_test(X_binned, y_labels, sample_names, safe_name, out_dir,
                         n_permutations=100, random_state=None):
    """
    Label-permutation significance test using leave-one-biological-replicate-out
    GroupKFold cross-validation (same scheme as run_analysis.py).

    For each of N permutations, y_labels is randomly shuffled and the Random
    Forest classifier is evaluated with the grouped, leak-free CV pipeline.
    The observed accuracy (from run_analysis.py) is compared against this null
    distribution to compute an empirical p-value.

    p-value = fraction of permuted accuracies >= observed accuracy.
    p < 0.05 confirms the classifier is learning real signal.
    """
    import warnings
    if random_state is None:
        random_state = config.RANDOM_SEED

    os.makedirs(out_dir, exist_ok=True)

    # Build GroupKFold components
    groups     = make_groups(y_labels, sample_names)
    prep_steps = make_preprocessor(
        normalization=config.NORMALIZATION, log_transform=config.LOG_TRANSFORM,
        scaling=config.SCALING, variance_percentile=config.VARIANCE_PERCENTILE,
        abundance_percentile=config.ABUNDANCE_PERCENTILE,
    )
    n_splits = auto_n_splits(y_labels, groups, desired=config.CV_FOLDS)

    # Observed accuracy
    obs_test, _ = random_forest(X_binned, y_labels, n_splits=n_splits,
                                groups=groups, prep_steps=prep_steps,
                                random_state=random_state)
    obs_acc = obs_test.mean()
    print(f"  Observed accuracy : {obs_acc:.3f}")
    print(f"  Running {n_permutations} permutations …")

    rng  = np.random.default_rng(random_state)
    perm_accs = []
    for i in range(n_permutations):
        y_perm = rng.permutation(y_labels)
        # Groups stay the same — we only shuffle labels
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                perm_test, _ = random_forest(X_binned, y_perm,
                                             n_splits=n_splits,
                                             groups=groups,
                                             prep_steps=prep_steps,
                                             random_state=random_state + i)
            perm_accs.append(perm_test.mean())
        except Exception:
            perm_accs.append(float('nan'))

        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{n_permutations}")

    perm_accs  = np.array(perm_accs)
    valid      = ~np.isnan(perm_accs)
    p_value    = (perm_accs[valid] >= obs_acc).sum() / valid.sum()

    print(f"  Permuted mean acc : {perm_accs[valid].mean():.3f} ± "
          f"{perm_accs[valid].std():.3f}")
    print(f"  p-value           : {p_value:.4f}  "
          f"({'significant' if p_value < 0.05 else 'NOT significant'} at α=0.05)")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(perm_accs[valid], bins=20, color='steelblue', alpha=0.7,
            edgecolor='white', label='Permuted accuracy')
    ax.axvline(obs_acc, color='crimson', linewidth=2,
               label=f'Observed ({obs_acc:.3f})')
    ax.set_xlabel('Mean CV Accuracy')
    ax.set_ylabel('Count')
    ax.set_title(f'Permutation Test — {safe_name}\np = {p_value:.4f}')
    ax.legend(fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f'permutation_test_{safe_name}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")

    # Save null distribution
    np.save(os.path.join(out_dir, f'permutation_null_{safe_name}.npy'), perm_accs)

    return obs_acc, perm_accs, p_value


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    experiment_name = config.EXPERIMENT.strip()
    safe_name       = experiment_name.replace(' ', '_').replace(':', '')

    print(f"\nExtras — {experiment_name}")
    print(f"Output → {OUT_DIR}")

    X_binned, X, X_norm, y_labels, sample_names, mz, X_filt_raw = \
        _load_and_preprocess(experiment_name)

    # Load saved classifier results if available
    results_path = os.path.join(OUT_DIR, f'classifier_results_{safe_name}.npz')
    classifier_results = None
    if os.path.exists(results_path):
        data = np.load(results_path)
        names = set(k.replace('__test', '').replace('__train', '') for k in data.files)
        classifier_results = {
            n: (data[f'{n}__test'], data[f'{n}__train'])
            for n in names if f'{n}__test' in data.files
        }

    if config.RUN_SUMMARY_REPORT:
        print("\n[extras] Summary report")
        run_summary_report(X, y_labels, mz, safe_name, OUT_DIR, classifier_results)

    if config.RUN_VARIATION_PLOT:
        print("\n[extras] Within-group variation plot")
        run_variation_plot(X, y_labels, mz, safe_name, OUT_DIR)

    if config.RUN_CORRELATION_HEATMAP:
        print("\n[extras] Feature correlation heatmap")
        vip = compute_vip_1comp(X, y_labels)
        run_correlation_heatmap(X, y_labels, mz, vip, safe_name, OUT_DIR,
                                n_features=config.N_CORR_FEATURES)

    if config.RUN_PERMUTATION_TEST:
        print(f"\n[extras] Permutation test ({config.N_PERMUTATIONS} permutations, "
              f"GroupKFold — this may take a few minutes)")
        run_permutation_test(X_binned, y_labels, sample_names, safe_name, OUT_DIR,
                             n_permutations=config.N_PERMUTATIONS)

    print("\nExtras done.")


if __name__ == '__main__':
    main()
