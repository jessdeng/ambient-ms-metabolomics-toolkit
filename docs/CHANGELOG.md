# Changelog

## [Unreleased] — Leakage hardening, univariate FDR & selection stability

### Fixed (data leakage / pseudoreplication)
- **Permutation test (standard pipeline).** `standard/extras.py::run_permutation_test`
  previously ran on the globally-preprocessed full-data `X` with plain
  `StratifiedKFold`, leaking preprocessing/feature-selection and splitting a
  colony's technical replicates across folds. It now runs the **entire**
  preprocessing + model inside `StratifiedGroupKFold` (colony-grouped,
  leak-free), permuting labels outside the pipeline — matching the
  `r_comparable` path. The observed accuracy is no longer inflated.
- **Global supervised SNR floor.** `run_analysis.py` (both pipelines) applied the
  label-aware `filter_snr_floor` to the matrix fed into cross-validation. The
  filter is now applied only to the descriptive copy; the CV input is the raw
  binned matrix and the in-pipeline `SNRFloor` transformer re-fits it per fold.

### Added (statistical rigor)
- `src/shared/feature_stats.py`: univariate fold-change + test (Welch t / ANOVA;
  Wilcoxon / Kruskal-Wallis) with **Benjamini-Hochberg FDR**; colony-bootstrap
  **selection-frequency stability**; and a **label-permutation null** for the
  cross-method overlap counts.
- `feature_overlap_<experiment>.csv` now includes `fold_change`,
  `log2_fold_change`, `p_value`, `q_value_BH`, `univariate_test`,
  `selection_frequency`, and `overlap_null_freq`; rows are sorted by BH q-value.
- **PLS-DA Q² + permuted-Q²** logged in both `run_analysis.py` pipelines
  (`compute_plsda_q2` / `evaluate_plsda_q2` in `standard/pipeline.py`).
- **Repeated grouped CV** with **balanced accuracy** reported per classifier
  (`N_CV_REPEATS`); balanced scores saved to the results `.npz`.

### Performance
- **Parallel bootstrap stability & overlap null.** Both loops now run across all
  cores via `joblib.Parallel(n_jobs=N_JOBS)` (default `-1`). Each replicate draws
  from an independent `np.random.SeedSequence(seed).spawn(...)` child stream, so
  results are **bit-for-bit reproducible regardless of `n_jobs`** or worker
  scheduling. New `N_JOBS` config flag; added `joblib` to requirements.
- The stability/null ensemble now **respects the `USE_*` config flags**
  (`enabled_methods_from_config`): a globally disabled model (e.g.
  `USE_GRADIENT_BOOSTING=False`) is omitted from these fits. The observed
  consensus in `feature_importance_analysis` uses the **same** enabled set, so
  `n_methods` and the permutation null keep a matching denominator (the printed
  "≥ k of N methods" adapts to N). All six importance columns are still written.

### Fixed (defensive / reproducibility)
- `filter_low_variance` now zero-guards the division by the feature mean
  (no more inf/NaN RSD corrupting the percentile threshold).
- Permutation p-values use the `+1/+1` estimator (Phipson & Smyth 2010) in
  **both** pipelines — they can no longer return exactly 0.
- New config flags: `N_CV_REPEATS`, `UNIVARIATE_TEST`, `RUN_FEATURE_STABILITY`,
  `N_BOOTSTRAP`, `RUN_OVERLAP_PERMUTATION`, `N_OVERLAP_PERMUTATIONS`,
  `RUN_PLSDA_Q2`, `N_Q2_PERMUTATIONS`. Added `scipy` to requirements.

## [Unreleased] — Per-group attribution for ensemble features

### Added
- `feature_overlap_<experiment>.csv` now includes per-group attribution columns
  that answer the question "which condition is this feature coming from?":
  - `mean_<group>` — mean log-normalised intensity in each group (univariate,
    raw-biology view)
  - `ridge_<group>` — signed one-vs-rest Ridge coefficient in each group
    (multivariate, model-attribution view; positive = associated with that
    group, negative = suppressed)
  - `top_condition_mean` — group with the highest mean intensity
  - `top_condition_ridge` — group with the largest positive Ridge coefficient
  - `mean_margin` — top mean ÷ second-highest mean; values close to 1.0 flag
    features that are similarly abundant in two or more groups
  - `ridge_direction` — `'elevated'`, `'suppressed'`, or `'mixed'` based on
    the sign pattern of the Ridge coefficients across groups

  Reporting both `mean` and `ridge` makes it easy to spot the interesting case
  where they disagree: when a feature ranks highly because of its multivariate
  structure rather than raw abundance.

### Changed
- `feature_importance_analysis()` (in both `shared/classifier_comparison.py`
  and `shared/classifier_comparison_standard.py`) now accepts an optional
  `X_norm` argument: the same matrix as `X` but normalised + log-transformed
  only (no scaling). This is what the `mean_<group>` columns are computed on,
  so values are interpretable on the raw log-intensity axis. If `X_norm` is
  not provided the function still runs and falls back to the scaled matrix
  (with a warning) for backward compatibility.
- `standard/run_analysis.py` and `r_comparable/run_analysis.py` now compute
  `X_norm` alongside `X` in the preprocessing step and pass it through.
- `scripts/condition_abundance.py` simplified: it no longer re-fits Ridge or
  re-loads the experiment. It reads the per-group columns directly from
  `feature_overlap_<experiment>.csv` and produces the side-by-side
  mean-abundance / Ridge-coefficient heatmap for high-confidence features
  (n_methods ≥ MIN_N_METHODS).
- `README.md` — output-files table updated to describe the enriched CSV.
- `NOTES.md` — feature-importance section restructured to document the new
  columns and how to interpret `top_condition_mean` vs `top_condition_ridge`
  agreement and disagreement.

### Migration notes
- Old `feature_overlap_<experiment>.csv` files generated before this change
  will not have the new columns. Re-run `python -m standard.run_analysis`
  (or the r_comparable equivalent) to regenerate them.
- `scripts/condition_abundance.py` will detect missing columns and print a
  message asking the user to re-run the main pipeline.
