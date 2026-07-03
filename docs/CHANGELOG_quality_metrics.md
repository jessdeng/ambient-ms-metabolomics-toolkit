# Changelog — reproducibility & benchmarking metrics

Additive changes only. No existing function, threshold, or published CV number was
modified, so all previously generated results remain reproducible.

## New module: `src/shared/quality_metrics.py`

Analytical reproducibility (chemometric QC):

- `technical_cv(X, tech_groups)` — per-feature technical %CV across technical
  replicates of each biological unit (colony), pooled across units by robust
  median (mean optional). SD uses `ddof=1`.
- `biological_cv(X, bio_groups, class_labels=None)` — per-feature biological %CV
  across colony means; computed within-class then pooled when `class_labels` is
  given, so the treatment effect does not inflate the estimate.
- `dispersion_ratio(X, tech_groups, class_labels=None)` — Broadhurst D-ratio,
  `100 · σ_technical / σ_total`, with pooled technical SD = RMS of within-unit SDs
  and total SD taken within-class when labels are supplied.
- `qc_report(X, mz, tech_groups, class_labels)` — per-feature table
  (technical CV, biological CV, D-ratio, pass flags) plus a study-level summary
  (median CVs, fraction of features ≤20%/≤30% CV and ≤50% D-ratio).
- Acceptance thresholds exposed as module constants: `CV_ACCEPT_PCT=30`,
  `CV_STRICT_PCT=20`, `DRATIO_ACCEPT_PCT=50` (Broadhurst et al. 2018, Metabolomics
  14:72; Dunn et al. 2011, Nat. Protoc. 6:1060).

Chemometric-rigor details:

- **Scale contract enforced.** CV is only defined on linear-scale intensities;
  the functions raise if the input contains negatives (the signature of a log or
  mean-centre step). Pass the matrix after TIC/median/PQN normalisation but before
  glog/log and before autoscaling.
- **Edge cases handled explicitly.** Zero/negative-mean features → CV = NaN (never
  `inf`); singleton units contribute a mean but no SD (no `ddof<=0`); an all-NaN
  pooled slice returns NaN rather than warning.

Classification benchmarking (multi-class, small-n):

- `grouped_oof_predictions(...)` — pooled out-of-fold predictions via
  `cross_val_predict` on the **same** per-fold `Pipeline(prep_steps + [clf])` and
  `StratifiedGroupKFold` as the accuracy path, so predictions are leak-free.
- `classification_metrics(y_true, y_pred, class_names)` — macro-F1, weighted-F1,
  accuracy, balanced accuracy, per-class recall/precision/F1/support, and the
  pooled confusion matrix. Answers referee item E-2 (macro-F1 + confusion matrix).
- `plot_confusion_matrix(...)` — Analyst-standard figure via `plot_style`
  (300 DPI, embedded fonts, tight bbox, colour-blind-safe `cividis`, fully
  labelled 'Predicted condition' / 'True condition' axes, row-normalised rate +
  raw count per cell).

## New tests: `tests/test_quality_metrics.py`

14 tests, all passing under `pytest -W error::RuntimeWarning` (no warnings):

- Technical/biological CV and D-ratio checked against hand-computed closed-form
  values on tiny designs.
- Edge cases: zero-mean feature → NaN, singleton units excluded, no-replication
  raises, negative (log/autoscaled) input rejected.
- Classification metrics against a known confusion structure; end-to-end
  leak-free grouped OOF prediction on separable synthetic clusters; figure file
  is written.

---

## Chemometric refactor — preprocessing unification, leakage isolation & orchestrator wiring

This phase eliminated the structural drift risk between the descriptive and
cross-validation preprocessing paths, isolated the linear matrix the QC metrics
require, removed the last leaky code path, and wired the QC + benchmarking suite
into both orchestrators. Numeric equivalence of the reported model matrix was held
to `0.0` max abs diff across all 120 normalisation × transform × scaling
combinations, so previously generated results remain reproducible.

### `src/standard/preprocessing.py` — single source of truth + linear-matrix access

- Extracted the normalisation, transformation and scaling mathematics into six
  canonical primitives — `fit_normalization`/`apply_normalization`,
  `fit_transform_params`/`apply_transform`, `fit_scaling`/`apply_scaling` — plus
  the `variance_keep_mask` / `abundance_keep_mask` feature-selection helpers.
- `preprocess()` now delegates to these primitives; its public signature and
  default return (the scaled model matrix) are unchanged.
- `preprocess(..., return_stages=True)` returns `{'normalized', 'transformed',
  'scaled'}`. The `'normalized'` stage is the linear, non-negative intensity
  matrix (normalisation applied, **before** glog/log and scaling) required by the
  QC metrics.
- `normalize_only(X, normalization)` returns that same linear matrix directly.
- `filter_low_variance` / `filter_low_abundance` delegate to the shared masks, so
  the descriptive filters and the per-fold CV filters select features identically.

### `src/shared/classifier_comparison_standard.py` and `classifier_comparison.py`

- The per-fold transformer classes (`Normalizer`, `LogTransform`, `Scaler`,
  `VarianceFilter`, `AbundanceFilter`) now delegate to the preprocessing
  primitives instead of re-implementing the arithmetic. Verified: the global
  `preprocess()` and the per-fold transformer chain are byte-identical
  (max abs diff `0.0` over 120 method combinations).
- **Data-leakage path removed.** `_run_cv_legacy` (ungrouped `StratifiedKFold` on
  a pre-preprocessed matrix) and the `_dispatch` fallback were deleted and
  replaced with `_require_grouped`, which **raises** when `groups`/`prep_steps`
  are absent. There is no ungrouped fallback; a pseudoreplicated /
  preprocessing-leaked estimate can no longer be produced. The now-unused
  `StratifiedKFold` and `accuracy_score` imports were dropped from both files.

### `config.py` — surfaced thresholds and toggles

- QC block: `RUN_QC_METRICS`, `QC_AGGREGATE`, `MIN_TECH_REPLICATES`,
  `EXPORT_QC_TABLE`, and the acceptance thresholds `CV_ACCEPT_PCT`,
  `CV_STRICT_PCT`, `DRATIO_ACCEPT_PCT` (surfaced from `quality_metrics` for
  one-place Methods reporting).
- CV design guard: `MIN_BIO_GROUPS_PER_CLASS`.
- Benchmark block: `RUN_CLASSIFICATION_BENCHMARK`, `EXPORT_BENCHMARK_TABLE`,
  `EXPORT_CONFUSION_MATRIX`, `CONFUSION_MATRIX_NORMALIZE`.
- Output routing: `SUPPLEMENTARY_SUBDIR`. Every key is `config.json`-overridable.

### New module: `src/shared/reporting.py` (orchestrator-facing helpers)

- `run_quality_control(X_linear, mz, groups, y_labels, out_dir, safe_name,
  config)` — technical %CV, biological %CV and the Broadhurst D-ratio on the
  linear matrix, written as `qc_features_<exp>.csv` + `qc_summary_<exp>.csv`.
- `run_classification_benchmark(X_binned, y_labels, groups, prep_steps, n_splits,
  enabled_names, out_dir, safe_name, experiment_name, config)` — pooled leak-free
  OOF per classifier (matching small-sample guards) → macro-F1, per-class
  recall/precision into `classification_metrics_<exp>.csv`, plus one
  Analyst-standard confusion-matrix figure per classifier.

### `src/standard/run_analysis.py` and `src/r_comparable/run_analysis.py`

- Step 4 now takes a single staged `preprocess(..., return_stages=True)` pass,
  extracting both the model matrix (`'scaled'`) and the linear QC matrix
  (`'normalized'`).
- New step 4b calls `run_quality_control`; new step 11b calls
  `run_classification_benchmark`. Both write to
  `results/<pipeline>/<SUPPLEMENTARY_SUBDIR>/`.

### Tests

- New `tests/test_pipeline_integration.py` (QC scale contract, 120-case
  preprocessing-drift equivalence, leakage/grouping, degenerate-feature clean-run,
  leak-free OOF benchmark).
- `tests/test_preprocessing.py`: three assertions that referenced the transformers'
  former private attributes (`.half_`, `.lambda_`) updated to the delegated
  `.params_` dict (`params_['half_min']` / `params_['lambda']`) — same behavioural
  assertion, current API.

Full suite after the refactor: **231 passed**.
