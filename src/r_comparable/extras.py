"""
r_comparable/extras.py — Optional Analysis Extras (R-comparable pipeline)
==========================================================================
Run after r_comparable/run_analysis.py. Identical to standard/extras.py but
uses r_comparable preprocessing (MetaboAnalyst-offset bin labels) and writes
outputs to results/r_comparable/.

Usage:
    python -m src.r_comparable.extras
"""

import os
import sys
import glob
import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, _ROOT)  # config.py lives here
sys.path.insert(0, _SRC)   # src/ packages take priority

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

import config
from r_comparable.preprocessing import (
    load_experiment, bin_features, filter_mass_range,
    filter_low_variance, filter_low_abundance, preprocess,
)
from r_comparable.pipeline import compute_vip_1comp, fit_plsda
from shared.classifier_comparison import (
    random_forest,
    feature_importance_analysis,
    make_groups, make_preprocessor, auto_n_splits,
)

# Re-use the helper functions from standard.extras (they are pipeline-agnostic)
from standard.extras import (
    run_summary_report,
    run_variation_plot,
    run_correlation_heatmap,
)

BASE_DIR = _ROOT
OUT_DIR  = os.path.join(BASE_DIR, 'results', 'r_comparable')


def _load_and_preprocess(experiment_name):
    """Load, filter, and preprocess using r_comparable pipeline."""
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


def run_permutation_test(X_binned, y_labels, sample_names, safe_name, out_dir,
                         n_permutations=100, random_state=None):
    """GroupKFold permutation test using r_comparable classifier."""
    import warnings
    if random_state is None:
        random_state = config.RANDOM_SEED

    os.makedirs(out_dir, exist_ok=True)

    groups     = make_groups(y_labels, sample_names)
    prep_steps = make_preprocessor(
        normalization=config.NORMALIZATION, log_transform=config.LOG_TRANSFORM,
        scaling=config.SCALING, variance_percentile=config.VARIANCE_PERCENTILE,
        abundance_percentile=config.ABUNDANCE_PERCENTILE,
    )
    n_splits = auto_n_splits(y_labels, groups, desired=config.CV_FOLDS)

    obs_test, _ = random_forest(X_binned, y_labels, n_splits=n_splits,
                                groups=groups, prep_steps=prep_steps,
                                random_state=random_state)
    obs_acc = obs_test.mean()
    print(f"  Observed accuracy : {obs_acc:.3f}")
    print(f"  Running {n_permutations} permutations ...")

    rng = np.random.default_rng(random_state)
    perm_accs = []
    for i in range(n_permutations):
        y_perm = rng.permutation(y_labels)
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

    perm_accs = np.array(perm_accs)
    valid     = ~np.isnan(perm_accs)
    p_value   = (perm_accs[valid] >= obs_acc).sum() / valid.sum()

    print(f"  Permuted mean acc : {perm_accs[valid].mean():.3f} +/- "
          f"{perm_accs[valid].std():.3f}")
    print(f"  p-value           : {p_value:.4f}  "
          f"({'significant' if p_value < 0.05 else 'NOT significant'} at a=0.05)")

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
    print(f"  Saved -> {out_path}")

    np.save(os.path.join(out_dir, f'permutation_null_{safe_name}.npy'), perm_accs)
    return obs_acc, perm_accs, p_value


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    experiment_name = config.EXPERIMENT.strip()
    safe_name       = experiment_name.replace(' ', '_').replace(':', '')

    print(f"\nExtras (R-comparable) -- {experiment_name}")
    print(f"Output -> {OUT_DIR}")

    X_binned, X, X_norm, y_labels, sample_names, mz, X_filt_raw = \
        _load_and_preprocess(experiment_name)

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
              f"GroupKFold -- this may take a few minutes)")
        run_permutation_test(X_binned, y_labels, sample_names, safe_name, OUT_DIR,
                             n_permutations=config.N_PERMUTATIONS)

    print("\nExtras done.")


if __name__ == '__main__':
    main()
