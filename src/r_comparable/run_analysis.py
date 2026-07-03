"""
r_comparable/run_analysis.py — R-compatible MS Metabolomics Pipeline
======================================================================
Identical methodology to standard/run_analysis.py but uses MetaboAnalyst-
offset bin labels for direct comparison with R-based PLS-DA packages.

Cross-validation uses StratifiedGroupKFold (leave-one-biological-replicate-out)
with preprocessing fitted inside each fold — see shared/classifier_comparison.py
for the full methodological rationale.

Usage:
    python -m src.r_comparable.run_analysis
"""

import os
import sys
# Ensure repo root is on the path so config.py and sibling packages are found
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, _ROOT)  # config.py lives here
sys.path.insert(0, _SRC)   # src/ packages take priority

import numpy as np

import config
from r_comparable.preprocessing import (load_experiment, bin_features,
    filter_mass_range, filter_snr_floor,
    filter_low_variance, filter_low_abundance, preprocess)
from r_comparable.pipeline import (compute_vip, compute_vip_1comp, fit_plsda,
                                   plot_scores_3d, plot_vip, evaluate_plsda_q2,
                                   adaptive_n_components, optimize_plsda_components)
from shared.classifier_comparison import (
    random_forest, svm_classify, gradient_boosting,
    logistic_regression, lda_classify, ridge_classify,
    plot_accuracy_comparison, feature_importance_analysis,
    make_preprocessor, auto_n_splits,
)
from shared.classifier_comparison_standard import grouped_permutation_importance
from shared.grouping import make_groups, check_batch_confound
from shared.visualization import plot_spectrum_with_features
from shared.runtime import resolve_experiment_dir, write_run_manifest
from shared.reporting import run_quality_control, run_classification_benchmark

BASE_DIR   = _ROOT
EXPERIMENT = config.EXPERIMENT

# ── Per-pipeline default resolution ──────────────────────────────────────────
# The r_comparable pipeline must reproduce MetaboAnalyst behaviour out of the
# box: log10, autoscale, no SNR floor.  If the user has NOT explicitly set
# these keys in config.json we override to the MetaboAnalyst-matching values.
# If they HAVE set them, their choice is honoured for both pipelines.
_json_keys       = getattr(config, '_json_keys', set())
_LOG_TRANSFORM   = config.LOG_TRANSFORM  if 'LOG_TRANSFORM'      in _json_keys else 'log10'
_SCALING         = config.SCALING        if 'SCALING'            in _json_keys else 'autoscale'
_SNR_ENABLED     = config.SNR_FLOOR_ENABLED if 'SNR_FLOOR_ENABLED' in _json_keys else False


def main():
    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = os.path.join(BASE_DIR, 'results', 'r_comparable')
    os.makedirs(out_dir, exist_ok=True)
    supp_dir = os.path.join(out_dir, getattr(config, 'SUPPLEMENTARY_SUBDIR',
                                             'supplementary'))

    # C-3: resolve the experiment safely (placeholder/missing -> bundled sample).
    experiment_dir, experiment_name, _used_fallback = resolve_experiment_dir(
        BASE_DIR, EXPERIMENT)
    experiment_name = experiment_name.strip()
    safe_name       = experiment_name.replace(' ', '_').replace(':', '')

    # Provenance manifest for this run.
    write_run_manifest(out_dir, BASE_DIR, config_module=config,
                       experiment_name=experiment_name,
                       experiment_dir=experiment_dir, pipeline='r_comparable')

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"\n[1/12] Loading data: {experiment_name!r}")
    # validate=False: defer the in-loader pre-flight so the explicit confound
    # guardrail below owns the failure message (clear n=1 explanation), exactly
    # as the standard pipeline does.
    X_raw, y_labels, sample_names, mz = load_experiment(experiment_dir,
                                                        validate=False)
    print(f"  Raw samples : {X_raw.shape[0]}")
    print(f"  Raw features: {X_raw.shape[1]}")

    # ── 1b. Guardrail: enforce a validatable design BEFORE any modelling ──────
    # Build the leave-one-biological-replicate-out groups now (single source of
    # truth, reused below) and HARD-FAIL on an n=1 design so cross-validation can
    # never emit a misleading ~1.0 score on a confounded study.
    groups    = make_groups(y_labels, sample_names, validate=False)
    n_groups  = len(set(groups))
    n_classes = int(np.unique(y_labels).size)
    check_batch_confound(y_labels, groups, names=sample_names, enforce=True)

    # Context-aware PLS-DA dimensionality: cap latent variables to the design
    # (binary -> <=2; never more than n_groups-1) so R^2Y is not forced to 1.0.
    n_comp_eff = adaptive_n_components(n_classes, n_groups,
                                       config.N_PLSDA_COMPONENTS)
    if n_comp_eff != config.N_PLSDA_COMPONENTS:
        print(f"  [adaptive] PLS-DA components capped "
              f"{config.N_PLSDA_COMPONENTS} -> {n_comp_eff} "
              f"(n_classes={n_classes}, biological groups={n_groups})")

    # ── 2. Bin ────────────────────────────────────────────────────────────────
    # X_binned is the per-sample binned matrix BEFORE any filtering/scaling.
    # The classifier CV consumes THIS matrix and fits all preprocessing inside
    # each fold (no leakage). Binning is per-sample so it is leak-free here.
    X_binned, mz = bin_features(X_raw, mz, bin_width=config.BIN_WIDTH)
    print(f"\n[2/12] Binning ({config.BIN_WIDTH} Da bins): {X_binned.shape[1]} features")

    # ── 2b. m/z range filter ─────────────────────────────────────────────────
    # Applied before descriptive filtering AND before CV so both paths see
    # the same feature set.
    X_binned, mz = filter_mass_range(X_binned, mz,
                                     mz_min=config.MZ_MIN, mz_max=config.MZ_MAX)
    print(f"  After m/z range filter ({config.MZ_MIN}-{config.MZ_MAX} Da): "
          f"{X_binned.shape[1]} features")
    mz_binned = mz.copy()

    # ── 2c. SNR floor (optional; OFF by default for r_comparable) ────────────
    # The SNR floor is label-aware (supervised), so applying it to the matrix
    # that feeds cross-validation leaks test-fold labels into feature selection.
    # It is therefore applied ONLY to the descriptive copy (X_desc) below; inside
    # the CV the in-pipeline SNRFloor transformer re-fits it per training fold.
    # X_binned (mass-range-filtered, RAW) is fed UNTOUCHED to the grouped CV.
    X_desc, mz_desc = X_binned.copy(), mz_binned.copy()
    if _SNR_ENABLED:
        print(f"\n  SNR floor filter (descriptive copy only; re-fit per-fold in CV) "
              f"(threshold={config.SNR_THRESHOLD}, "
              f"noise_q={config.NOISE_QUANTILE}, "
              f"min_frac={config.MIN_FRACTION_IN_GROUP})")
        X_desc, mz_desc = filter_snr_floor(
            X_desc, mz_desc, y_labels,
            snr_threshold=config.SNR_THRESHOLD,
            noise_quantile=config.NOISE_QUANTILE,
            min_fraction=config.MIN_FRACTION_IN_GROUP,
        )

    # ── 3. Filter (full-data copy — for descriptive PLS-DA/VIP/ensemble) ─────
    print(f"\n[3/12] Filtering (descriptive, full-data)")
    if config.VARIANCE_PERCENTILE > 0:
        X_filt, mz = filter_low_variance(X_desc, mz_desc.copy(),
                                         percentile=config.VARIANCE_PERCENTILE)
        print(f"  After variance filter ({config.VARIANCE_PERCENTILE}%): "
              f"{X_filt.shape[1]} features")
    else:
        X_filt, mz = X_desc.copy(), mz_desc.copy()
        print("  Variance filter disabled")

    if config.ABUNDANCE_PERCENTILE > 0:
        X_filt, mz = filter_low_abundance(X_filt, mz,
                                          percentile=config.ABUNDANCE_PERCENTILE)
        print(f"  After abundance filter ({config.ABUNDANCE_PERCENTILE}%): "
              f"{X_filt.shape[1]} features")
    else:
        print("  Abundance filter disabled")

    X_filt_raw = X_filt.copy()

    print("  Groups:")
    for g, count in zip(*np.unique(y_labels, return_counts=True)):
        print(f"    {g:25s}  n={count}")

    # ── 4. Preprocess (full-data — descriptive only) ──────────────────────────
    print(f"\n[4/12] Preprocessing (descriptive, full-data)")
    print(f"  Normalization  : {config.NORMALIZATION}")
    print(f"  Transformation : {_LOG_TRANSFORM}")
    print(f"  Scaling        : {_SCALING}")
    X_norm = preprocess(X_filt.copy(), normalization=config.NORMALIZATION,
                        log_transform=_LOG_TRANSFORM, scaling='none')
    # Single staged pass yields both the model matrix (normalised+transformed+scaled)
    # and the LINEAR normalised matrix (pre-transform) the QC metrics require.
    _stages = preprocess(X_filt, normalization=config.NORMALIZATION,
                         log_transform=_LOG_TRANSFORM, scaling=_SCALING,
                         return_stages=True)
    X        = _stages['scaled']
    X_linear = _stages['normalized']

    # ── 4b. Analytical QC (technical %CV, biological %CV, Broadhurst D-ratio) ──
    if getattr(config, 'RUN_QC_METRICS', False):
        print("\n[4b] Quality-control metrics (technical %CV, biological %CV, D-ratio)")
        _qc_table, _qc_summary = run_quality_control(
            X_linear, mz, groups, y_labels, supp_dir, safe_name, config)
        print(f"    median technical %CV = {_qc_summary['median_technical_cv_pct']:.1f}%  |  "
              f"median D-ratio = {_qc_summary['median_dratio_pct']:.1f}%  |  "
              f"features passing both = {_qc_summary['n_pass_both']}/{_qc_summary['n_features']}")

    # ── Leak-free preprocessor for cross-validation ──────────────────────────
    # `groups` / `n_groups` were built and validated in step 1b above.
    prep_steps = make_preprocessor(
        normalization=config.NORMALIZATION, log_transform=_LOG_TRANSFORM,
        scaling=_SCALING, variance_percentile=config.VARIANCE_PERCENTILE,
        abundance_percentile=config.ABUNDANCE_PERCENTILE,
        prevalence_threshold=0.0,        # R-comparable path: no prevalence filter
        snr_floor_enabled=_SNR_ENABLED,  # OFF by default for r_comparable
        snr_threshold=config.SNR_THRESHOLD,
        noise_quantile=config.NOISE_QUANTILE,
        min_fraction_in_group=config.MIN_FRACTION_IN_GROUP,
    )
    n_splits = auto_n_splits(y_labels, groups, desired=config.CV_FOLDS)
    print(f"\n  CV scheme: leave-one-biological-replicate-out "
          f"(StratifiedGroupKFold, {n_splits} folds, {n_groups} colonies)")

    # ── 5. PLS-DA (descriptive) ───────────────────────────────────────────────
    print(f"\n[5/12] Fitting PLS-DA ({n_comp_eff} components)")
    pls, T, y, Y, classes = fit_plsda(X, y_labels, n_comp_eff)
    print(f"  Classes: {list(classes)}")

    # ── 6. VIP scores (descriptive) ───────────────────────────────────────────
    # Intentional MetaboAnalyst-parity asymmetry (NOT a bug): VIP aggregates over
    # PLSDA_VIP_NUM_COMPONENTS latent variables (default 1) while the scores plot
    # uses N_PLSDA_COMPONENTS. Raise the flag to match the plot dimensions.
    _vip_nc = getattr(config, 'PLSDA_VIP_NUM_COMPONENTS', 1)
    print(f"\n[6/12] Computing VIP scores ({_vip_nc} component(s), top {config.N_TOP_VIP})")
    if _vip_nc == 1:
        print(f"  [note] VIP uses 1 latent variable for MetaboAnalyst parity; the "
              f"scores plot uses {config.N_PLSDA_COMPONENTS} components — an "
              f"intentional asymmetry (set PLSDA_VIP_NUM_COMPONENTS to change it).")
    vip = compute_vip(X, y_labels, n_components=_vip_nc)
    plot_scores_3d(T, pls, y_labels, classes, experiment_name,
                   out_path=os.path.join(out_dir, f"plsda_scores_3d_{safe_name}.html"))
    plot_vip(vip, mz, X_filt_raw, y_labels, config.N_TOP_VIP, experiment_name,
             out_path=os.path.join(out_dir, f"vip_scores_{safe_name}.png"))

    # ── 6b. PLS-DA R^2Y + Q^2 + permutation null (descriptive model quality) ──
    if getattr(config, 'RUN_PLSDA_Q2', False):
        print(f"\n  PLS-DA R^2Y / Q^2 (+ {config.N_Q2_PERMUTATIONS}-permutation null)")
        plsda_qual = evaluate_plsda_q2(
            X, y_labels, n_comp_eff, groups=groups,
            n_splits=n_splits, n_perm=config.N_Q2_PERMUTATIONS,
            random_state=config.RANDOM_SEED,
        )
        print(f"    R^2Y = {plsda_qual['r2y']:.3f}  (apparent fit)   "
              f"permuted p = {plsda_qual['r2y_p']:.4f}")
        print(f"    Q^2  = {plsda_qual['q2']:.3f}  (cross-validated) "
              f"permuted p = {plsda_qual['q2_p']:.4f} "
              f"({'significant' if plsda_qual['q2_p'] < 0.05 else 'NOT significant'} "
              f"at a=0.05)")

    # ── 7–11. Classifiers (grouped + leak-free) ───────────────────────────────
    all_classifiers = {
        'Random Forest':       (config.USE_RANDOM_FOREST,       random_forest),
        'SVM':                 (config.USE_SVM,                 svm_classify),
        'Gradient Boosting':   (config.USE_GRADIENT_BOOSTING,   gradient_boosting),
        'Logistic Regression': (config.USE_LOGISTIC_REGRESSION, logistic_regression),
        'LDA':                 (config.USE_LDA,                 lda_classify),
        'Ridge':               (config.USE_RIDGE,               ridge_classify),
    }

    n_repeats = getattr(config, 'N_CV_REPEATS', 1)
    print(f"  Repeated grouped CV: {n_repeats} repeat(s) x {n_splits} folds")
    results = {}
    balanced = {}
    step = 7
    for name, (enabled, fn) in all_classifiers.items():
        if enabled:
            print(f"\n[{step}/12] {name}")
            metrics = fn(
                X_binned, y_labels, n_splits=n_splits,
                groups=groups, prep_steps=prep_steps,
                n_repeats=n_repeats, return_metrics=True,
                n_biological=n_groups,
            )
            test_accs  = metrics['test_accuracy']
            train_accs = metrics['train_accuracy']
            bal_accs   = metrics['test_balanced_accuracy']
            results[name]  = (test_accs, train_accs)
            balanced[name] = bal_accs
            print(f"  Test  accuracy         : {test_accs.mean():.3f} +/- {test_accs.std():.3f}")
            print(f"  Test  balanced accuracy: {bal_accs.mean():.3f} +/- {bal_accs.std():.3f}")
            print(f"  Train accuracy         : {train_accs.mean():.3f} +/- {train_accs.std():.3f}")
            step += 1

    # ── Save classifier results for reuse in extras.py ───────────────────────
    results_path = os.path.join(out_dir, f"classifier_results_{safe_name}.npz")
    np.savez(results_path,
             **{f"{name}__test":  test_accs for name, (test_accs, _) in results.items()},
             **{f"{name}__train": train_accs for name, (_, train_accs) in results.items()},
             **{f"{name}__balanced": balanced[name] for name in results})
    print(f"\n  Classifier results saved -> {results_path}")

    # ── 11. Plot comparison (chance line at 1/n_classes) ──────────────────────
    print("\n[11/12] Plot Comparison")
    plot_accuracy_comparison(results, experiment_name,
                             out_path=os.path.join(out_dir,
                                 f"classifier_comparison_{safe_name}.png"),
                             chance=1.0 / len(classes))

    # ── 11b. Classification benchmark (pooled leak-free OOF) ──────────────────
    # Macro-F1, per-class recall/precision and the Analyst-standard confusion
    # matrix, from pooled out-of-fold predictions on the SAME grouped, per-fold
    # preprocessing pipeline as the accuracy path (no preprocessing leakage).
    if getattr(config, 'RUN_CLASSIFICATION_BENCHMARK', False):
        print("\n[11b] Classification benchmark (macro-F1, per-class recall, confusion matrix)")
        enabled_names = [name for name, (enabled, _) in all_classifiers.items()
                         if enabled]
        run_classification_benchmark(
            X_binned, y_labels, groups, prep_steps, n_splits,
            enabled_names, supp_dir, safe_name, experiment_name, config)

    # ── 12. Feature Importance (descriptive, full-data) ───────────────────────
    print("\n[12/12] Feature Importance Overlap Analysis")
    overlap_df, counts = feature_importance_analysis(
        X, y_labels, mz, safe_name, out_dir,
        top_n=config.TOP_N_FEATURES, X_norm=X_norm,
        log_transform=_LOG_TRANSFORM,
        X_filt_raw=X_filt_raw, groups=groups,
        normalization=config.NORMALIZATION, scaling=_SCALING,
        univariate_test=getattr(config, 'UNIVARIATE_TEST', 'auto'),
        compute_stability=getattr(config, 'RUN_FEATURE_STABILITY', False),
        n_boot=getattr(config, 'N_BOOTSTRAP', 200),
        compute_overlap_null=getattr(config, 'RUN_OVERLAP_PERMUTATION', False),
        n_overlap_perm=getattr(config, 'N_OVERLAP_PERMUTATIONS', 200),
        random_state=config.RANDOM_SEED,
    )
    # ── 12b. Permutation importance (B2: counters tree impurity bias) ─────────
    if getattr(config, 'RUN_PERMUTATION_IMPORTANCE', True):
        print("\n[12b] Grouped permutation importance (RF, GB; held-out folds)")
        perm_imp_df = grouped_permutation_importance(
            X_binned, y_labels, groups, prep_steps, mz_binned,
            models=('rf', 'gb'), n_splits=n_splits, n_repeats=1,
            n_perm_repeats=getattr(config, 'N_PERM_IMPORTANCE_REPEATS', 5),
            random_state=config.RANDOM_SEED, n_jobs=getattr(config, 'N_JOBS', -1),
        )
        perm_path = os.path.join(out_dir, f"permutation_importance_{safe_name}.csv")
        perm_imp_df.to_csv(perm_path, index=False, encoding='utf-8')
        print(f"  Saved permutation importance -> {perm_path}")
        perm_cols = [c for c in perm_imp_df.columns if c.endswith('_mean')]
        overlap_df = overlap_df.merge(perm_imp_df[['mz'] + perm_cols],
                                      on='mz', how='left')
        overlap_csv = os.path.join(out_dir, f"feature_overlap_{safe_name}.csv")
        overlap_df.to_csv(overlap_csv, index=False, encoding='utf-8')
        print(f"  Candidate table updated with permutation importance -> {overlap_csv}")

    plot_spectrum_with_features(X_binned, mz_binned, y_labels, overlap_df,
                                experiment_name,
                                out_path=os.path.join(out_dir,
                                    f"spectrum_features_{safe_name}.png"))

    print("\nDone.")


if __name__ == '__main__':
    main()
