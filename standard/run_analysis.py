import os
import sys
# Ensure repo root is on the path so config.py and sibling packages are found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
from standard.preprocessing import (load_experiment, bin_features,
    filter_mass_range, filter_low_variance, filter_low_abundance, preprocess)
from standard.pipeline import compute_vip_1comp, fit_plsda, plot_scores_3d, plot_vip
from shared.classifier_comparison_standard import (
    random_forest, svm_classify, gradient_boosting,
    logistic_regression, lda_classify, ridge_classify,
    plot_accuracy_comparison, feature_importance_analysis,
    make_groups, make_preprocessor, auto_n_splits,
)
from shared.visualization import plot_spectrum_with_features

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT = config.EXPERIMENT


def main():
    experiment_dir  = os.path.join(BASE_DIR, 'data', EXPERIMENT)
    experiment_name = EXPERIMENT.strip()
    safe_name       = experiment_name.replace(' ', '_').replace(':', '')

    out_dir = os.path.join(BASE_DIR, 'results', 'standard')
    os.makedirs(out_dir, exist_ok=True)

    assert os.path.isdir(experiment_dir), (
        f"Experiment folder not found: {experiment_dir!r}\n"
        f"Check that EXPERIMENT in config.py matches a folder name inside data/.\n"
        f"Available: {[d for d in os.listdir(os.path.join(BASE_DIR, 'data')) if os.path.isdir(os.path.join(BASE_DIR, 'data', d)) and not d.startswith('.')]}"
    )

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"\n[1/12] Loading data: {experiment_name!r}")
    X_raw, y_labels, sample_names, mz = load_experiment(experiment_dir)
    print(f"  Raw samples : {X_raw.shape[0]}")
    print(f"  Raw features: {X_raw.shape[1]}")

    # ── 2. Bin ────────────────────────────────────────────────────────────────
    # X_binned is the per-sample binned matrix BEFORE any filtering/scaling.
    # The classifier CV consumes THIS matrix and fits all preprocessing inside
    # each fold (no leakage). Binning is per-sample so it is leak-free here.
    X_binned, mz = bin_features(X_raw, mz, bin_width=config.BIN_WIDTH)
    print(f"\n[2/12] Binning ({config.BIN_WIDTH} Da bins): {X_binned.shape[1]} features")

    # ── 2b. m/z range filter ─────────────────────────────────────────────────
    # Applied before descriptive filtering AND before CV, so both paths see
    # the same feature set.
    X_binned, mz = filter_mass_range(X_binned, mz,
                                     mz_min=config.MZ_MIN, mz_max=config.MZ_MAX)
    print(f"  After m/z range filter ({config.MZ_MIN}–{config.MZ_MAX} Da): "
          f"{X_binned.shape[1]} features")
    mz_binned = mz.copy()

    # ── 3. Filter (FULL-DATA copy — for the descriptive PLS-DA/VIP/ensemble) ──
    # These outputs are reported models fit on all data, so full-data
    # preprocessing is correct for them (it is NOT used for the CV accuracies).
    print(f"\n[3/12] Filtering (descriptive, full-data)")
    if config.VARIANCE_PERCENTILE > 0:
        X_filt, mz = filter_low_variance(X_binned, mz.copy(), percentile=config.VARIANCE_PERCENTILE)
        print(f"  After variance filter ({config.VARIANCE_PERCENTILE}%): {X_filt.shape[1]} features")
    else:
        X_filt, mz = X_binned.copy(), mz.copy()
        print(f"  Variance filter disabled")
    if config.ABUNDANCE_PERCENTILE > 0:
        X_filt, mz = filter_low_abundance(X_filt, mz, percentile=config.ABUNDANCE_PERCENTILE)
        print(f"  After abundance filter ({config.ABUNDANCE_PERCENTILE}%): {X_filt.shape[1]} features")
    else:
        print(f"  Abundance filter disabled")

    X_filt_raw = X_filt.copy()

    print("  Groups:")
    for g, count in zip(*np.unique(y_labels, return_counts=True)):
        print(f"    {g:25s}  n={count}")

    # ── 4. Preprocess (FULL-DATA — descriptive only) ──────────────────────────
    print(f"\n[4/12] Preprocessing (descriptive, full-data)")
    print(f"  Normalization     : {config.NORMALIZATION}")
    print(f"  Transformation    : {config.LOG_TRANSFORM}")
    print(f"  Scaling           : {config.SCALING}")
    X_norm = preprocess(X_filt.copy(), normalization=config.NORMALIZATION,
                        log_transform=config.LOG_TRANSFORM, scaling='none')
    X = preprocess(X_filt, normalization=config.NORMALIZATION,
                   log_transform=config.LOG_TRANSFORM, scaling=config.SCALING)

    # ── Grouping + leak-free preprocessor for cross-validation ────────────────
    groups   = make_groups(y_labels, sample_names)
    n_groups = len(set(groups))
    prep_steps = make_preprocessor(
        normalization=config.NORMALIZATION, log_transform=config.LOG_TRANSFORM,
        scaling=config.SCALING, variance_percentile=config.VARIANCE_PERCENTILE,
        abundance_percentile=config.ABUNDANCE_PERCENTILE,
    )
    n_splits = auto_n_splits(y_labels, groups, desired=config.CV_FOLDS)
    print(f"\n  CV scheme: leave-one-biological-replicate-out "
          f"(StratifiedGroupKFold, {n_splits} folds, {n_groups} colonies)")

    # ── 5. PLS-DA (descriptive) ───────────────────────────────────────────────
    print(f"\n[5/12] Fitting PLS-DA ({config.N_PLSDA_COMPONENTS} components)")
    pls, T, y, Y, classes = fit_plsda(X, y_labels, config.N_PLSDA_COMPONENTS)
    print(f"  Classes: {list(classes)}")

    # ── 6. VIP scores (descriptive) ───────────────────────────────────────────
    print(f"\n[6/12] Computing VIP scores (1 component, top {config.N_TOP_VIP})")
    vip = compute_vip_1comp(X, y_labels)
    plot_scores_3d(T, pls, y_labels, classes, experiment_name,
                   out_path=os.path.join(out_dir, f"plsda_scores_3d_{safe_name}.html"))
    plot_vip(vip, mz, X_filt_raw, y_labels, config.N_TOP_VIP, experiment_name,
             out_path=os.path.join(out_dir, f"vip_scores_{safe_name}.png"))

    # ── 7–11. Classifiers (CORRECTED: grouped + leak-free) ────────────────────
    all_classifiers = {
        'Random Forest':       (config.USE_RANDOM_FOREST,       random_forest),
        'SVM':                 (config.USE_SVM,                 svm_classify),
        'Gradient Boosting':   (config.USE_GRADIENT_BOOSTING,   gradient_boosting),
        'Logistic Regression': (config.USE_LOGISTIC_REGRESSION, logistic_regression),
        'LDA':                 (config.USE_LDA,                 lda_classify),
        'Ridge':               (config.USE_RIDGE,               ridge_classify),
    }

    results = {}
    step = 7
    for name, (enabled, fn) in all_classifiers.items():
        if enabled:
            print(f"\n[{step}/12] {name}")
            test_accs, train_accs = fn(
                X_binned, y_labels, n_splits=n_splits,
                groups=groups, prep_steps=prep_steps,
            )
            results[name] = (test_accs, train_accs)
            print(f"  Test  accuracy: {test_accs.mean():.3f} ± {test_accs.std():.3f}")
            print(f"  Train accuracy: {train_accs.mean():.3f} ± {train_accs.std():.3f}")
            step += 1

    results_path = os.path.join(out_dir, f"classifier_results_{safe_name}.npz")
    np.savez(results_path,
             **{f"{name}__test":  test_accs for name, (test_accs, _) in results.items()},
             **{f"{name}__train": train_accs for name, (_, train_accs) in results.items()})
    print(f"\n  Classifier results saved → {results_path}")

    # ── 11. Plot comparison (chance line at 1/n_classes) ──────────────────────
    print("\n[11/12] Plot Comparison")
    plot_accuracy_comparison(results, experiment_name,
                             out_path=os.path.join(out_dir, f"classifier_comparison_{safe_name}.png"),
                             chance=1.0 / len(classes))

    # ── 12. Feature Importance (descriptive, full-data) ───────────────────────
    print("\n[12/12] Feature Importance Overlap Analysis")
    overlap_df, counts = feature_importance_analysis(
        X, y_labels, mz, safe_name, out_dir,
        top_n=config.TOP_N_FEATURES, X_norm=X_norm, log_transform=config.LOG_TRANSFORM,
    )
    plot_spectrum_with_features(X_binned, mz_binned, y_labels, overlap_df, experiment_name,
                                out_path=os.path.join(out_dir, f"spectrum_features_{safe_name}.png"))

    print("\nDone.")


if __name__ == '__main__':
    main()
