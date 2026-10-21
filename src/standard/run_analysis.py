import os
import sys
# src/standard/run_analysis.py -- insert both SRC and ROOT so
# `from standard.x import` and `import config` both resolve.
_SRC  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../src
_ROOT = os.path.dirname(_SRC)                                           # repo root
sys.path.insert(0, _ROOT)  # config.py lives here
sys.path.insert(0, _SRC)   # standard/, shared/ live here (checked first)


def _parse_cli():
    """`--config FILE` selects the settings file (default: ./config.json);
    `--experiment NAME` overrides EXPERIMENT. Must run before `import config`."""
    import argparse
    ap = argparse.ArgumentParser(description='Standard analysis (ambient MS metabolomics toolkit).')
    ap.add_argument('--config', help='JSON settings file (default: config.json in the repository root)')
    ap.add_argument('--experiment', help='data folder name under data/ (overrides EXPERIMENT)')
    a, _ = ap.parse_known_args()
    if a.config:
        os.environ['TOOLKIT_CONFIG'] = os.path.abspath(a.config)
    if a.experiment:
        os.environ['TOOLKIT_EXPERIMENT'] = a.experiment


if __name__ == '__main__':
    _parse_cli()

import time
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd

import config
if os.environ.get('TOOLKIT_EXPERIMENT'):
    config.EXPERIMENT = os.environ['TOOLKIT_EXPERIMENT']
from standard.preprocessing import (load_experiment, bin_features,
    filter_mass_range, filter_snr_floor, filter_prevalence,
    filter_low_variance, filter_low_abundance, preprocess)
from standard.pipeline import (compute_vip, compute_vip_1comp, fit_plsda,
                               plot_scores_3d, plot_vip, evaluate_plsda_q2,
                               adaptive_n_components, optimize_plsda_components)
from shared.classifier_comparison_standard import (
    RandomForest, svm_classify, gradient_boosting,
    logistic_regression, lda_classify, ridge_classify,
    plot_accuracy_comparison, feature_importance_analysis,
    make_preprocessor, auto_n_splits, grouped_permutation_importance,
)
from shared.grouping import make_groups, check_batch_confound, resolve_groups
from shared.visualization import plot_spectrum_with_features
from shared.runtime import resolve_experiment_dir, write_run_manifest
from shared.reporting import run_quality_control, run_classification_benchmark
from shared.pca import run_pca
from shared.loco_cv import loco_table

BASE_DIR   = _ROOT
EXPERIMENT = config.EXPERIMENT


def main():
    t_start = time.time()
    # Resolve the experiment folder; an invalid EXPERIMENT raises a clear error
    # (the original silent fallback to the sample data needs ALLOW_SAMPLE_FALLBACK).
    experiment_dir, experiment_name, _used_fallback = resolve_experiment_dir(
        BASE_DIR, config.EXPERIMENT)
    experiment_name = experiment_name.strip()
    safe_name       = experiment_name.replace(' ', '_').replace(':', '')

    # v2: one output folder per experiment: results/standard/<experiment>/
    out_dir = os.path.join(BASE_DIR, 'results', 'standard', safe_name)
    os.makedirs(out_dir, exist_ok=True)
    supp_dir = os.path.join(out_dir, getattr(config, 'SUPPLEMENTARY_SUBDIR',
                                             'supplementary'))
    print(f"  Settings file : {getattr(config, 'CONFIG_FILE', 'config.json')}")
    print(f"  Output folder : {out_dir}")

    # Provenance: capture environment + parameters for this run.
    write_run_manifest(out_dir, BASE_DIR, config_module=config,
                       experiment_name=experiment_name,
                       experiment_dir=experiment_dir, pipeline='standard')

    # -- 1. Load -----------------------------------------------------------------
    print(f"\n[1/12] Loading data: {experiment_name!r}")
    # validate=False: defer the in-loader pre-flight so the explicit confound
    # guardrail below owns the failure message (clear n=1 explanation).
    _resample = getattr(config, 'SPECTRAL_RESAMPLING', 'linspace5000')
    X_raw, y_labels, sample_names, mz = load_experiment(experiment_dir,
                                                        validate=False,
                                                        resample=_resample)
    raw_mz_grid = np.asarray(mz, float)
    print(f"  Spectral grid: {_resample!r} ({raw_mz_grid.size} points per spectrum)")
    print(f"  Raw samples : {X_raw.shape[0]}")
    print(f"  Raw features: {X_raw.shape[1]}")

    # -- 1b. Guardrail: enforce a validatable design BEFORE any modelling --------
    # Build the leave-one-biological-replicate-out groups now (single source of
    # truth, reused below) and HARD-FAIL on an n=1 design so cross-validation can
    # never emit a misleading ~1.0 score on a confounded study.
    groups   = resolve_groups(y_labels, sample_names, experiment_dir,
                              getattr(config, 'METADATA_MANIFEST', None),
                              getattr(config, 'EXPECTED_TECH_REPS', None))
    n_groups = len(set(groups))
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

    # -- 2. Bin ------------------------------------------------------------------
    # X_binned is the per-sample binned matrix BEFORE any filtering/scaling.
    # The classifier CV consumes THIS matrix and fits all preprocessing inside
    # each fold (no leakage). Binning is per-sample, so it is leak-free here.
    X_binned, mz_binned = bin_features(X_raw, mz, bin_width=config.BIN_WIDTH)
    print(f"\n[2/12] Binning ({config.BIN_WIDTH} Da bins): {X_binned.shape[1]} features")

    # -- 2b. m/z range filter -----------------------------------------------------
    X_binned, mz_binned = filter_mass_range(
        X_binned, mz_binned, mz_min=config.MZ_MIN, mz_max=config.MZ_MAX
    )
    print(f"  After m/z range filter ({config.MZ_MIN}--{config.MZ_MAX} Da): "
          f"{X_binned.shape[1]} features")

    # -- 2c/2d. Supervised filters -- DESCRIPTIVE COPY ONLY (no CV leak) ----------
    # The SNR floor and the prevalence filter are BOTH label-aware (supervised):
    # they keep a feature based on its per-group detection/SNR using y_labels.
    # Applying either to the matrix that feeds cross-validation would select
    # features using the labels of samples that later appear in the test folds.
    #
    # X_binned (mass-range-filtered, RAW) is fed UNTOUCHED to the grouped CV.
    # A2: the SNR floor is now a DESCRIPTIVE-ONLY filter — it is applied here to
    # the separate descriptive copy (X_desc) used by the full-data PLS-DA / VIP /
    # ensemble, and is NOT a step inside the CV pipeline (snr_floor_enabled=False
    # in make_preprocessor below, unless SNR_FLOOR_IN_CV is explicitly set), so it
    # never dynamically alters the per-fold feature space. The PrevalenceFilter
    # remains in the CV pipeline where it is re-fit per training fold (leak-free).
    # Both descriptive filters run on raw counts BEFORE any log-transform so zeros
    # stay zeros.
    X_desc, mz_desc = X_binned.copy(), mz_binned.copy()

    if config.SNR_FLOOR_ENABLED:
        print(f"\n  SNR floor filter (descriptive copy only; not used inside CV) "
              f"(threshold={config.SNR_THRESHOLD}, "
              f"noise_q={config.NOISE_QUANTILE}, "
              f"min_frac={config.MIN_FRACTION_IN_GROUP})")
        X_desc, mz_desc = filter_snr_floor(
            X_desc, mz_desc, y_labels,
            snr_threshold=config.SNR_THRESHOLD,
            noise_quantile=config.NOISE_QUANTILE,
            min_fraction=config.MIN_FRACTION_IN_GROUP,
        )

    if config.PREVALENCE_THRESHOLD > 0.0:
        X_desc, mz_desc = filter_prevalence(
            X_desc, mz_desc, y_labels,
            threshold=config.PREVALENCE_THRESHOLD,
        )

    # -- 3. Filter (FULL-DATA copy -- for the descriptive PLS-DA/VIP/ensemble) --
    # These outputs are reported models fit on all data, so full-data
    # preprocessing is correct for them (it is NOT used for the CV accuracies).
    print(f"\n[3/12] Filtering (descriptive, full-data)")
    if config.VARIANCE_PERCENTILE > 0:
        X_filt, mz = filter_low_variance(X_desc, mz_desc, percentile=config.VARIANCE_PERCENTILE)
        print(f"  After variance filter ({config.VARIANCE_PERCENTILE}%): {X_filt.shape[1]} features")
    else:
        X_filt, mz = X_desc.copy(), mz_desc.copy()
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

    # -- 4. Preprocess (FULL-DATA -- descriptive only) ---------------------------
    print(f"\n[4/12] Preprocessing (descriptive, full-data)")
    print(f"  Normalization     : {config.NORMALIZATION}")
    print(f"  Transformation    : {config.LOG_TRANSFORM}")
    print(f"  Scaling           : {config.SCALING}")
    X_norm = preprocess(X_filt.copy(), normalization=config.NORMALIZATION,
                        log_transform=config.LOG_TRANSFORM, scaling='none')
    # Single staged pass yields both the model matrix (normalised+transformed+scaled)
    # and the LINEAR normalised matrix (pre-transform) the QC metrics require.
    _stages = preprocess(X_filt, normalization=config.NORMALIZATION,
                         log_transform=config.LOG_TRANSFORM,
                         scaling=config.SCALING, return_stages=True)
    X        = _stages['scaled']
    X_linear = _stages['normalized']

    # -- 4a. PCA (unsupervised; v2) ---------------------------------------------
    if getattr(config, 'RUN_PCA', True):
        print("\n[4a] PCA (unsupervised; labels used only to describe the fitted structure)")
        _ev = run_pca(X, y_labels, groups, mz, sample_names, out_dir, safe_name,
                      experiment_name, n_components=getattr(config, 'N_PCA_COMPONENTS', 5))
        for _, r in _ev.head(3).iterrows():
            print(f"    {r.pc}: {r.explained_variance_ratio * 100:5.1f}% of variance; score variance "
                  f"condition-plate {r.frac_condition_plate:.2f} / colony {r.frac_colony:.2f} / "
                  f"technical {r.frac_technical:.2f}")

    # -- 4b. Analytical QC (technical %CV, biological %CV, technical/total SD ratio) ----
    # Computed on X_linear (linear, normalised, pre-transform) with the colony
    # grouping so technical vs biological dispersion are separated correctly.
    if getattr(config, 'RUN_QC_METRICS', False):
        print("\n[4b] Quality-control metrics (technical %CV, biological %CV, technical/total SD ratio)")
        _qc_table, _qc_summary = run_quality_control(
            X_linear, mz, groups, y_labels, supp_dir, safe_name, config)
        print(f"    median technical %CV = {_qc_summary['median_technical_cv_pct']:.1f}%  |  "
              f"median technical/total SD ratio = {_qc_summary['median_dratio_pct']:.1f}%  |  "
              f"features passing both = {_qc_summary['n_pass_both']}/{_qc_summary['n_features']}")

    # -- Leak-free preprocessor for cross-validation -----------------------------
    # `groups` / `n_groups` were built and validated in step 1b above.
    prep_steps = make_preprocessor(
        normalization=config.NORMALIZATION, log_transform=config.LOG_TRANSFORM,
        scaling=config.SCALING, variance_percentile=config.VARIANCE_PERCENTILE,
        abundance_percentile=config.ABUNDANCE_PERCENTILE,
        prevalence_threshold=config.PREVALENCE_THRESHOLD,
        # A2: the SNR floor is a DESCRIPTIVE-ONLY filter (applied to X_desc above).
        # Keep it OUT of the CV folds so it never dynamically alters the per-fold
        # feature space, unless SNR_FLOOR_IN_CV is explicitly turned on.
        snr_floor_enabled=(config.SNR_FLOOR_ENABLED
                           and getattr(config, 'SNR_FLOOR_IN_CV', False)),
        snr_threshold=config.SNR_THRESHOLD,
        noise_quantile=config.NOISE_QUANTILE,
        min_fraction_in_group=config.MIN_FRACTION_IN_GROUP,
    )
    n_splits = auto_n_splits(y_labels, groups, desired=config.CV_FOLDS)
    print(f"\n  CV scheme: leave-one-biological-replicate-out "
          f"(StratifiedGroupKFold, {n_splits} folds, {n_groups} colonies)")

    # -- 5. PLS-DA (descriptive) -------------------------------------------------
    print(f"\n[5/12] Fitting PLS-DA ({n_comp_eff} components)")
    pls, T, y, Y, classes = fit_plsda(X, y_labels, n_comp_eff)
    print(f"  Classes: {list(classes)}")

    # -- 6. VIP scores (descriptive) ---------------------------------------------
    # Component asymmetry (intentional MetaboAnalyst-parity choice, NOT a bug):
    # the VIP ranking aggregates over PLSDA_VIP_NUM_COMPONENTS latent variables
    # (default 1, = MetaboAnalyst component-1 VIP) while the 3-D scores plot above
    # uses N_PLSDA_COMPONENTS (8). Raise PLSDA_VIP_NUM_COMPONENTS to match the plot.
    _vip_nc = getattr(config, 'PLSDA_VIP_NUM_COMPONENTS', 1)
    print(f"\n[6/12] Computing VIP scores ({_vip_nc} component(s), top {config.N_TOP_VIP})")
    if _vip_nc == 1:
        print(f"  [note] VIP uses 1 latent variable for MetaboAnalyst parity; the "
              f"scores plot uses {config.N_PLSDA_COMPONENTS} components. This 1-vs-"
              f"{config.N_PLSDA_COMPONENTS} asymmetry is intentional — set "
              f"PLSDA_VIP_NUM_COMPONENTS to aggregate VIP over more components.")
    vip = compute_vip(X, y_labels, n_components=_vip_nc)
    plot_scores_3d(T, pls, y_labels, classes, experiment_name,
                   out_path=os.path.join(out_dir, f"plsda_scores_3d_{safe_name}.html"))
    plot_vip(vip, mz, X_filt_raw, y_labels, config.N_TOP_VIP, experiment_name,
             out_path=os.path.join(out_dir, f"vip_scores_{safe_name}.png"))

    # -- 6b. PLS-DA R^2Y + Q^2 + permutation null (descriptive model quality) ----
    if getattr(config, 'RUN_PLSDA_Q2', False):
        print(f"\n  PLS-DA R^2Y / Q^2 (+ {config.N_Q2_PERMUTATIONS}-permutation null)")
        plsda_qual = evaluate_plsda_q2(
            X, y_labels, n_comp_eff, groups=groups,
            n_splits=n_splits, n_perm=config.N_Q2_PERMUTATIONS,
            random_state=config.RANDOM_SEED,
        )
        print(f"    [legacy] Q^2 here is cross-validated on the globally preprocessed matrix "
              f"(not leak-free); use the grouped Q^2 profile below.")
        print(f"    R^2Y = {plsda_qual['r2y']:.3f}  (apparent fit)   "
              f"shuffled-label exceedance = {plsda_qual['r2y_p']:.3f}")
        print(f"    Q^2  = {plsda_qual['q2']:.3f}  shuffled-label exceedance = "
              f"{plsda_qual['q2_p']:.3f} (exploratory; not a p-value)")

    # -- 6c. Multi-component VIP with leak-free, colony-grouped Q^2 (v2) ----------
    vip_multi, a_star = None, None
    if getattr(config, 'RUN_MULTICOMPONENT_VIP', True):
        from standard.pipeline import (grouped_q2_profile, VALIDATED_PLS_TOL,
                                       VALIDATED_PLS_MAX_ITER, VIP_REFERENCE_LABEL,
                                       VIP_MULTICOMPONENT_LABEL)
        _maxc = int(max(1, min(getattr(config, 'PLS_MAX_COMPONENTS', 8), n_groups - 1)))
        print(f"\n[6c] Grouped Q^2 profile (1..{_maxc} LVs; StratifiedGroupKFold "
              f"{getattr(config, 'Q2_N_SPLITS', 3)} folds x {getattr(config, 'Q2_N_REPEATS', 5)} repeats; "
              f"preprocessing refitted per fold)")
        _q2 = grouped_q2_profile(X_binned, y_labels, groups, prep_steps, max_components=_maxc,
                                 n_splits=getattr(config, 'Q2_N_SPLITS', 3),
                                 n_repeats=getattr(config, 'Q2_N_REPEATS', 5),
                                 random_state=config.RANDOM_SEED)
        a_star = int(_q2['best_n_components'])
        pd.DataFrame({'n_components': _q2['components'], 'grouped_q2_mean': _q2['q2'],
                      'grouped_q2_sd': _q2['q2_repeats'].std(axis=0)}).to_csv(
            os.path.join(out_dir, f'plsda_q2_profile_{safe_name}.csv'), index=False)
        vip_multi = compute_vip(X, y_labels, n_components=a_star, tol=VALIDATED_PLS_TOL,
                                max_iter=VALIDATED_PLS_MAX_ITER)
        print(f"    Q^2 by LV: " + ', '.join(f"{a}:{q:.3f}" for a, q in zip(_q2['components'], _q2['q2'])))
        print(f"    A* = {a_star} (argmax grouped Q^2)")
        _vip_tab = pd.DataFrame({'mz': mz, 'vip_comp1': vip,
                                 'vip_comp1_rank': pd.Series(-vip).rank(method='min').astype(int).values,
                                 f'vip_multi_A{a_star}': vip_multi,
                                 'vip_multi_rank': pd.Series(-vip_multi).rank(method='min').astype(int).values})
        _vip_tab.to_csv(os.path.join(out_dir, f'vip_{safe_name}.csv'), index=False)
        print(f"    vip_comp1 = {VIP_REFERENCE_LABEL}; vip_multi = {VIP_MULTICOMPONENT_LABEL}")

    # -- 7-11. Classifiers (CORRECTED: grouped + leak-free) ----------------------
    all_classifiers = {
        'Random Forest':       (config.USE_RANDOM_FOREST,       RandomForest),
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

    results_path = os.path.join(out_dir, f"classifier_results_{safe_name}.npz")
    np.savez(results_path,
             **{f"{name}__test":  test_accs for name, (test_accs, _) in results.items()},
             **{f"{name}__train": train_accs for name, (_, train_accs) in results.items()},
             **{f"{name}__balanced": balanced[name] for name in results})
    print(f"\n  Classifier results saved -> {results_path}")

    # -- 10b. Principal accuracy table + supplementary leave-one-colony-out (v2) --
    _acc = pd.DataFrame([dict(classifier=n, cv=f'{n_splits}-fold colony-grouped x {n_repeats}',
                              test_accuracy_mean=float(t.mean()), test_accuracy_sd=float(t.std()),
                              train_accuracy_mean=float(tr.mean()),
                              balanced_accuracy_mean=float(balanced[n].mean()))
                         for n, (t, tr) in results.items()])
    if getattr(config, 'RUN_LOCO_CV', True):
        print("\n[10b] Leave-one-colony-out (supplementary; pooled held-out predictions)")
        _abbr = {'Random Forest': 'rf', 'SVM': 'svm', 'Gradient Boosting': 'gb',
                 'Logistic Regression': 'lr', 'LDA': 'lda', 'Ridge': 'ridge'}
        _loco, _conf = loco_table(X_binned, y_labels, groups, prep_steps,
                                  methods=[_abbr[n] for n in results], n_biological=n_groups,
                                  seed=config.RANDOM_SEED)
        _acc = _acc.merge(_loco, on='classifier', how='left')
        os.makedirs(supp_dir, exist_ok=True)
        for _n, _c in _conf.items():
            _c.to_csv(os.path.join(supp_dir, f"loco_confusion_{safe_name}_{_n.replace(' ', '_')}.csv"))
        for _, r in _loco.iterrows():
            print(f"    {r.classifier:22s} LOCO accuracy {r.loco_accuracy:.3f}  "
                  f"balanced {r.loco_balanced_accuracy:.3f}  macro-F1 {r.loco_macro_f1:.3f}")
    _acc['chance_level'] = 1.0 / len(classes)
    _acc['n_spectra'] = len(y_labels); _acc['n_colonies'] = n_groups
    _acc.to_csv(os.path.join(out_dir, f'classifier_accuracy_{safe_name}.csv'), index=False)

    # -- 11. Plot comparison (chance line at 1/n_classes) ------------------------
    print("\n[11/12] Plot Comparison")
    plot_accuracy_comparison(results, experiment_name,
                             out_path=os.path.join(out_dir, f"classifier_comparison_{safe_name}.png"),
                             chance=1.0 / len(classes))

    # -- 11b. Classification benchmark (pooled leak-free OOF) ---------------------
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

    # -- 12. Feature Importance (descriptive, full-data) -------------------------
    print("\n[12/12] Feature Importance Overlap Analysis")
    overlap_df, counts = feature_importance_analysis(
        X, y_labels, mz, safe_name, out_dir,
        top_n=config.TOP_N_FEATURES, X_norm=X_norm, log_transform=config.LOG_TRANSFORM,
        X_filt_raw=X_filt_raw, groups=groups,
        normalization=config.NORMALIZATION, scaling=config.SCALING,
        univariate_test=getattr(config, 'UNIVARIATE_TEST', 'auto'),
        compute_stability=getattr(config, 'RUN_FEATURE_STABILITY', False),
        n_boot=getattr(config, 'N_BOOTSTRAP', 200),
        compute_overlap_null=getattr(config, 'RUN_OVERLAP_PERMUTATION', False),
        n_overlap_perm=getattr(config, 'N_OVERLAP_PERMUTATIONS', 200),
        random_state=config.RANDOM_SEED,
    )
    # -- 12b. Permutation importance (B2: counters tree impurity bias) -----------
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
        # Map candidate rows to their held-out permutation importance by m/z and
        # re-save the candidate table with the supplementary columns.
        perm_cols = [c for c in perm_imp_df.columns if c.endswith('_mean')]
        overlap_df = overlap_df.merge(perm_imp_df[['mz'] + perm_cols],
                                      on='mz', how='left')
        overlap_csv = os.path.join(out_dir, f"feature_overlap_{safe_name}.csv")
        overlap_df.to_csv(overlap_csv, index=False, encoding='utf-8')
        print(f"  Candidate table updated with permutation importance -> {overlap_csv}")

    # -- 12c. Annotate rankings: multi-component VIP + spectral signal groups (v2) --
    _rank_path = os.path.join(out_dir, f'feature_rankings_{safe_name}.csv')
    _ovl_path = os.path.join(out_dir, f'feature_overlap_{safe_name}.csv')
    _ann = pd.DataFrame({'mz': mz})
    if vip_multi is not None:
        _ann['vip_multi'] = vip_multi
        _ann['vip_multi_rank'] = pd.Series(-vip_multi).rank(method='min').astype(int).values
        _ann['vip_multi_A'] = a_star
    if getattr(config, 'RUN_SIGNAL_GROUPS', True):
        from shared.feature_groups import build_signal_groups
        _map, _pairs = build_signal_groups(X_binned, mz_binned, X_raw, raw_mz_grid, groups,
                                           bin_width=config.BIN_WIDTH,
                                           r_min=getattr(config, 'SIGNAL_R_MIN', 0.9))
        _map.to_csv(os.path.join(out_dir, f'signal_groups_{safe_name}.csv'), index=False)
        _pairs.to_csv(os.path.join(supp_dir, f'signal_group_evidence_{safe_name}.csv'), index=False)
        _sig = _map.set_index('mz_bin')
        _ann['signal_group'] = _sig.reindex(mz)['signal_group'].values
        _ann['signal_mz'] = _sig.reindex(mz)['signal_mz'].values
        _ann['n_bins_in_signal'] = pd.Series(_ann['signal_group']).map(
            _map.groupby('signal_group').size()).values
        print(f"\n[12c] Signal groups: {_map.signal_group.nunique()} signals from {len(_map)} bins "
              f"({int((_map.groupby('signal_group').size() > 1).sum())} multi-bin signals)")
    if _ann.shape[1] > 1:
        for _path in (_rank_path, _ovl_path):
            if os.path.exists(_path):
                _t = pd.read_csv(_path)
                _t = _t.drop(columns=[c for c in _ann.columns if c != 'mz' and c in _t.columns])
                _t = _t.merge(_ann, on='mz', how='left')
                _t.to_csv(_path, index=False, encoding='utf-8')
        overlap_df = pd.read_csv(_ovl_path)

    plot_spectrum_with_features(X_binned, mz_binned, y_labels, overlap_df, experiment_name,
                                out_path=os.path.join(out_dir, f"spectrum_features_{safe_name}.png"))

    print(f"\nDone in {time.time() - t_start:.0f} s. Results: {out_dir}")


if __name__ == '__main__':
    main()
