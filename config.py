"""
config.py — Pipeline Configuration
====================================
Default settings for all pipeline parameters. You can override any value by
editing config.json in the repository root — no Python editing required.

If config.json exists, its values are loaded and applied on top of these
defaults. Only the keys you want to change need to appear in config.json.

For guidance on which settings to choose, see the README.

Reference: van den Berg et al. (2006) BMC Genomics 7:142
           doi: 10.1186/1471-2164-7-142
"""

import os as _os
import json as _json

# ── Reproducibility ─────────────────────────────────────────────────────────────
# Fixed random seed for all stochastic classifiers and cross-validation splits.
# Keep this value to reproduce paper results exactly. Change it (e.g. to 0 or 99)
# to verify that findings are robust across seeds — results should be similar.
RANDOM_SEED = 42

# ── Experiment ─────────────────────────────────────────────────────────────────
# The name of your experiment folder inside data/. Include spaces/special
# characters exactly as they appear.
#
# Out-of-the-box safety (C-3): if this is left as the placeholder below, or points
# at a folder with no .csv/.txt data, run_analysis does NOT crash — it falls back
# to the bundled sample experiment (SAMPLE_EXPERIMENT) and prints a warning, so a
# fresh clone runs end-to-end. Set this to your own folder to analyse real data.
EXPERIMENT = 'your_experiment_folder'   # folder name inside data/

# Bundled demo dataset used as the fallback when EXPERIMENT is unset/missing.
SAMPLE_EXPERIMENT = 'sample_experiment'


# ── m/z Range ──────────────────────────────────────────────────────────────────
# Features outside this range are removed before any analysis.
# Default (100–1000 Da) covers the informative range for most small-molecule MS.
# Adjust to match the range where your instrument produces reliable signal.
MZ_MIN = 100   # lower m/z cutoff (Da)
MZ_MAX = 1000  # upper m/z cutoff (Da)


# ── Binning ────────────────────────────────────────────────────────────────────
# Groups nearby m/z values into fixed-width windows and sums their intensities.
# 0.5 Da is appropriate for low-resolution instruments (e.g. triple quadrupole)
# to account for m/z drift. Use a smaller value (e.g. 0.1) for higher-resolution
# instruments with stable m/z calibration.
BIN_WIDTH = 0.5  # bin width in Da


# ── Filtering ──────────────────────────────────────────────────────────────────
# Removes uninformative features before statistics.
# Increase the percentile to remove more features; set to 0 to disable.
VARIANCE_PERCENTILE = 25  # remove bottom X% of features by variance (0 = off)
ABUNDANCE_PERCENTILE = 5  # remove bottom X% of features by abundance (0 = off)

# Prevalence filter — removes features driven by imputed (half-min) values.
# A feature is kept only if it shows genuine detection (raw intensity > 0) in
# at least this fraction of samples within at least one condition group.
# 0.5 = 50 % (lenient); 0.8 = 80 % (strict). Set to 0.0 to disable.
PREVALENCE_THRESHOLD = 0.5

# SNR floor — removes bins whose signal never rises above the per-sample
# baseline noise in any condition group.  Applied on raw binned intensities
# BEFORE normalization.  Does NOT cut by m/z — genuinely low-abundance or
# single-condition features are retained as long as they exceed the noise
# floor in at least min_fraction of samples within at least one group.
#
# Per-sample robust noise estimate: sigma_i = 1.4826 * MAD of all bins in
# sample i that fall below its noise_quantile-th percentile.  This uses only
# within-sample values — no cross-sample or cross-group information.
#
# A feature is kept if SNR >= snr_threshold in >= min_fraction of samples
# within AT LEAST ONE group (same group-aware, discovery-preserving logic as
# the prevalence filter).
#
# Standard pipeline default: ENABLED (recommended for ambient MS baseline
# suppression without sacrificing genuine low-abundance features).
# R-comparable pipeline default: DISABLED (preserves MetaboAnalyst parity).
SNR_FLOOR_ENABLED      = True   # standard default; r_comparable defaults False
SNR_THRESHOLD          = 3      # minimum SNR to count a sample as detected
NOISE_QUANTILE         = 60     # percentile used to identify the noise region
MIN_FRACTION_IN_GROUP  = 0.5    # min fraction of group samples that must pass

# A2 — SNR-floor routing. The SNR floor is label/group-aware (supervised), so by
# default it is a DESCRIPTIVE-ONLY filter: it is applied to the full-data copy
# used for PLS-DA / VIP / the ensemble feature list, and is NOT a step inside the
# cross-validation folds (so it never dynamically alters the per-fold feature
# space). Set this True only if you deliberately want the SNR floor re-fit per
# training fold inside the CV pipeline (still leak-free, but reintroduces per-fold
# feature-set asymmetry). Reported CV accuracies use SNR_FLOOR_IN_CV = False.
SNR_FLOOR_IN_CV        = False


# ── Normalization ──────────────────────────────────────────────────────────────
# Corrects for differences in sample amount, injection volume, or instrument
# sensitivity between runs. Choose based on your experimental setup.
#
# Options:
#   'tic'      — divide each sample by its total ion current (sum of all
#                intensities) and rescale to the median TIC. The most common
#                approach for ambient MS and direct infusion experiments.
#                Default and recommended for most ambient MS workflows.
#
#   'median'   — divide each sample by its median feature intensity and rescale
#                to the global median. More robust than TIC when a small number
#                of very abundant features dominate the signal.
#
#   'pqn'      — Probabilistic Quotient Normalization. Compares each sample to
#                a reference spectrum (median of all samples) and estimates a
#                dilution factor. Handles sample-to-sample variation well.
#                Recommended if TIC gives poor results.
#
#   'quantile' — forces all samples to have the same intensity distribution.
#                Very aggressive normalization — use only if you are confident
#                that systematic distributional differences between samples are
#                technical, not biological.
#
#   'none'     — no normalization applied. Use only if your samples were
#                collected under strictly identical conditions and you trust
#                the raw signal intensities.
NORMALIZATION = 'tic'


# ── Transformation ─────────────────────────────────────────────────────────────
# Transformation compresses the wide dynamic range of MS data so that
# high-intensity features do not dominate the analysis.
# Options:
#   'glog'   — generalized-log / arcsinh transform: asinh(x / lambda_).
#              Linear near zero (does not amplify additive baseline noise),
#              log-like at high intensity (stabilises log-normal regime).
#              Tolerates exact zeros without the half-min offset hack.
#              lambda_ is the 5th percentile of positive values — a robust
#              noise-floor estimate fitted from training data in CV.
#              Recommended for ambient MS with a significant baseline; this
#              is the new standard-pipeline default.
#   'log10'  — log base-10 with half-minimum offset; MetaboAnalyst default.
#              R-comparable pipeline default for MetaboAnalyst parity.
#   'log2'   — similar to log10 but with base 2; common in transcriptomics
#   'sqrt'   — square root transform; gentler compression than log
#   'none'   — no transformation applied
LOG_TRANSFORM = 'glog'   # standard default; r_comparable defaults 'log10'


# ── Scaling ────────────────────────────────────────────────────────────────────
# Scaling adjusts for differences in the magnitude of features so that all
# features contribute equally to the analysis regardless of their raw intensity.
#
# Van den Berg et al. (2006) compared scaling methods for metabolomics and found
# that autoscaling and range scaling best removed the dependence of feature
# rankings on average concentration. Choose based on your biological question:
#
#   'autoscale'  — subtract mean, divide by std dev. All features equally
#                  weighted. Best for comparing relative changes across features.
#                  Default and recommended for exploratory analysis.
#
#   'pareto'     — divide by square root of std dev. Large fold changes are
#                  reduced but the data is not made dimensionless. A compromise
#                  between autoscaling and no scaling.
#
#   'range'      — divide by the biological range (max - min). Scales features
#                  relative to their observed biological variation. Good when
#                  the biological range is meaningful.
#
#   'vast'       — variable stability scaling. Downweights features with high
#                  relative variation. Useful when you want to focus on stable,
#                  consistently-changing features.
#
#   'level'      — divide by the mean. Converts intensities to fold changes
#                  relative to the average. Useful for identifying biomarkers
#                  with large relative changes.
#
#   'none'       — no scaling applied. High-abundance features will dominate
#                  the analysis. Not recommended unless you have a specific reason.
SCALING = 'autoscale'


# ── PLS-DA ─────────────────────────────────────────────────────────────────────
# Number of components for the PLS-DA scores plot.
# More components capture more variation but can overfit.
# 8 is a reasonable default for exploratory analysis.
# Note: VIP scores always use 1 component regardless of this setting.
N_PLSDA_COMPONENTS = 8


# ── VIP Plot ───────────────────────────────────────────────────────────────────
# How many top VIP features to show in the dot plot and heatmap.
N_TOP_VIP = 30

# Number of PLS-DA latent variables the VIP score aggregates over.
# Default 1 = exact MetaboAnalyst component-1 parity (this is why VIP uses 1 LV
# while the 3-D scores plot uses N_PLSDA_COMPONENTS=8 — an intentional asymmetry,
# NOT a bug). Set to N_PLSDA_COMPONENTS to make the VIP ranking aggregate over the
# same latent space the scores plot visualises. The ensemble/bootstrap/null VIP
# always stays at 1 LV for MetaboAnalyst-comparable consensus.
PLSDA_VIP_NUM_COMPONENTS = 1


# ── Classifiers ────────────────────────────────────────────────────────────────
# Set to True to include a classifier, False to skip it.
# Disabling slow classifiers (SVM, Gradient Boosting) can speed up the pipeline
# on large datasets.
USE_RANDOM_FOREST       = True
USE_SVM                 = True
USE_GRADIENT_BOOSTING   = True
USE_LOGISTIC_REGRESSION = True
USE_LDA                 = True
USE_RIDGE               = True


# ── Cross-validation ───────────────────────────────────────────────────────────
# Number of folds for stratified k-fold cross-validation.
# 5 is standard. Use a smaller number (e.g. 3) if you have very few samples.
CV_FOLDS = 5

# Repeated grouped CV — with only a few colonies per class each single split is
# noisy (one test fold = one colony). Repeating the grouped CV with different
# shuffles and averaging stabilises the estimate. 1 = no repeats (original
# behaviour); 10 is a good default for small designs.
N_CV_REPEATS = 10

# ── Univariate statistics & FDR ──────────────────────────────────────────────────
# Per-feature significance testing added alongside the ensemble ranking.
#   'auto'     — Welch's t-test (2 groups) / one-way ANOVA (>2 groups)
#   'ttest'    — same parametric tests as 'auto'
#   'wilcoxon' — Wilcoxon rank-sum (2 groups) / Kruskal-Wallis (>2 groups)
# Benjamini-Hochberg FDR q-values are always reported next to the raw p-values.
UNIVARIATE_TEST = 'auto'

# Replicate handling for the univariate test. True (default) averages each feature
# within its biological group (colony) BEFORE testing, so technical replicates are
# not pseudoreplicated and n = colonies per class (consistent with the grouped CV).
# Set False to test at the raw per-spectrum level (anti-conservative; for comparison
# only). The chosen unit is recorded in the table's 'univariate_test' column.
UNIVARIATE_GROUP_AGGREGATE = True

# ── Feature-selection stability (bootstrap) ───────────────────────────────────────
# Resamples colonies with replacement, re-fits the ensemble, and reports how
# often each feature is selected — out-of-sample evidence the candidate list is
# reproducible. Slower (re-fits the ensemble N_BOOTSTRAP times). Set False to skip.
RUN_FEATURE_STABILITY = True
N_BOOTSTRAP = 200

# ── Overlap permutation null ──────────────────────────────────────────────────────
# Permutes labels and recomputes the cross-method overlap to calibrate how many
# features would corroborate by chance (the linear methods are collinear, so the
# raw overlap overstates independence). Reports observed vs null + empirical p.
RUN_OVERLAP_PERMUTATION = True
N_OVERLAP_PERMUTATIONS = 200

# ── PLS-DA Q² validation ──────────────────────────────────────────────────────────
# Cross-validated Q² for the descriptive PLS-DA plus a permuted-Q² null, so the
# VIP-based prioritisation is reported with a calibrated model-quality metric.
RUN_PLSDA_Q2 = True
N_Q2_PERMUTATIONS = 200

# ── Parallelism ───────────────────────────────────────────────────────────────────
# Cores for the parallel bootstrap-stability and overlap-null loops (joblib).
# -1 = use all available cores; 1 = serial (useful for debugging). Results are
# identical regardless of this value (each replicate has an independent seed).
N_JOBS = -1


# ── Feature Importance Overlap ─────────────────────────────────────────────────
# How many top features to take from each method when looking for overlap.
# Features appearing in at least 2 of the 5 interpretable methods are reported.
TOP_N_FEATURES = 50

# ── Permutation importance (B2 — counters tree impurity bias) ──────────────────
# Computes sklearn.inspection.permutation_importance for RF and GB on the HELD-OUT
# CV folds (grouped by colony), a model-agnostic alternative to impurity-based
# feature_importances_. Saved to permutation_importance_<exp>.csv and merged into
# the candidate table by m/z. Set False to skip (saves a little time).
RUN_PERMUTATION_IMPORTANCE = True
N_PERM_IMPORTANCE_REPEATS  = 5    # permutation repeats inside permutation_importance


# ── Extras ─────────────────────────────────────────────────────────────────────
# Settings for extras.py — run this file separately after run_analysis.py.

# Summary report — saves a .txt file with settings, feature counts, and results
RUN_SUMMARY_REPORT = True

# Within-group variation plot — violin plot of feature intensities per group
RUN_VARIATION_PLOT = True

# Feature correlation heatmap — pairwise correlations between top VIP features
RUN_CORRELATION_HEATMAP = True
N_CORR_FEATURES = 20  # how many top VIP features to include

# Cross-experiment comparison — finds overlapping top features between two
# experiments. Both folders must be in the same directory as the .py files.
# Set RUN_COMPARISON = False to skip.
RUN_COMPARISON = False
EXPERIMENT_A = 'your_experiment_folder_A'  # ← change to first experiment name
EXPERIMENT_B = 'your_experiment_folder_B'  # ← change to second experiment name

# Reproducibility report — runs classifiers twice with different random seeds
# and reports how stable the top features are across runs. Slow on large datasets.
RUN_REPRODUCIBILITY = False

# VIP-filtered classifier comparison — runs all classifiers on only features
# with VIP > threshold. Connects the PLS-DA and ML steps, reduces overfitting
# caused by high-dimensional MS data. Recommended for small datasets.
RUN_VIP_FILTERED_CLASSIFIERS = False
VIP_FILTER_THRESHOLD = 1.0  # keep features with VIP > this value

# Permutation test — shuffles class labels 100 times to verify that
# classification accuracy is significantly above chance. The most rigorous
# validation for small MS datasets. Slow — allow ~2-5 minutes.
# p < 0.05 confirms the model is learning real signal, not memorising noise.
RUN_PERMUTATION_TEST = True   # set False to skip (saves ~2–5 min)
N_PERMUTATIONS = 100

# ── Ensemble confidence threshold ────────────────────────────────────────────────
# Features with n_methods >= this value are reported as "high confidence" by
# condition_abundance.py and condition_spectrum.py.
# The paper reports results for features corroborated by 4 or more methods.
# Set to 2 to inspect all overlap features.
HIGH_CONFIDENCE_N_METHODS = 4


# ── Load overrides from config.json (if present) ─────────────────────────────
# Any key in config.json that matches a variable name above will override it.
# Keys starting with '_' (like "_comment") are ignored.
#
# _json_keys: set of keys explicitly provided in config.json.  The r_comparable
# pipeline reads this to distinguish "user set this" from "config.py default",
# so it can apply its own MetaboAnalyst-matching defaults when the user has not
# explicitly overridden LOG_TRANSFORM, SCALING, or SNR_FLOOR_ENABLED.
_json_keys: set = set()
_config_json_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'config.json')
if _os.path.exists(_config_json_path):
    with open(_config_json_path) as _f:
        _overrides = _json.load(_f)
    _g = globals()
    for _k, _v in _overrides.items():
        if not _k.startswith('_') and _k in _g:
            _g[_k] = _v
            _json_keys.add(_k)
    del _g, _k, _v, _f, _overrides
del _config_json_path, _os, _json
