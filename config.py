"""
config.py — default settings
============================
Every setting the analysis uses, with its default. Override any of them in a
JSON settings file (config.json in the repository root, or the file passed with
--config); keys starting with '_' are comments. Unknown keys are reported.

The analysis answers one question: which m/z features differ between the
conditions, consistently across colonies, and look like real ions — i.e. which
are worth chemical follow-up. Each condition was grown on one plate/enclosure,
so condition and plate are aliased: every result is a condition-plate
association, never a proven light effect.
"""

import os as _os
import json as _json

# ── Data ────────────────────────────────────────────────────────────────────────
EXPERIMENT = 'sample_experiment'      # folder name under data/, or 'all' for every
                                      # experiment there (same as --all)
METADATA_MANIFEST = 'metadata_manifest.csv'   # file -> class + colony; used if present
EXPECTED_TECH_REPS = None             # e.g. 3: stop if a colony has another count
ALLOW_UNEQUAL_COLONIES = False        # false: stop if the colonies do not all have the same
                                      # number of spectra (usually one colony named two ways,
                                      # which would put its sprays in both training and test)

# ── Spectra → features ──────────────────────────────────────────────────────────
MZ_MIN = 100                          # Da
MZ_MAX = 2000                         # Da
BIN_WIDTH = 0.5                       # Da; unit-resolution QqQ, absorbs m/z drift
BIN_LABELS = 'mean_mz'                # 'mean_mz' (mean m/z of the bin's points),
                                      # 'centroid' (intensity-weighted mean m/z) or
                                      # 'lower_edge' (MetaboAnalyst's labels)

# ── Feature filters (label-free; MetaboAnalyst defaults) ─────────────────────────
VARIANCE_PERCENTILE = 25              # drop the lowest-RSD X% of bins (0 = off)
ABUNDANCE_PERCENTILE = 5              # drop the lowest-mean X% of bins (0 = off)

# ── Scaling for the multivariate rankers ────────────────────────────────────────
# Spectra are always TIC-normalised and log10-transformed. 'autoscale' weights
# every bin equally (MetaboAnalyst default); 'pareto' keeps some intensity weight.
SCALING = 'autoscale'
LOG_OFFSET = 'global'                 # value added before log10 so zeros stay finite:
                                      # 'global' (half the smallest positive value) or
                                      # 'per_bin' (half each bin's own smallest positive value)

# ── Prioritisation ──────────────────────────────────────────────────────────────
TOP_N_FEATURES = 50                   # top-N of each ranker
MIN_RANKERS = 2                       # candidate = top-N in >= this many of the two
                                      # voting rankers (univariate F, component-1 VIP)
RF_SEEDS = 100                        # seeds averaged for the random-forest importance
GB_SEEDS = 100                        # ... and for the gradient-boosting importance. Both
                                      # break ties between equal splits at random; fewer
                                      # seeds are faster but the top-N lists move more
RANDOM_SEED = 42

# ── Method comparison (six importance definitions) ──────────────────────────────
# Component-1 VIP, Ridge, logistic regression, linear SVM, random forest and
# gradient boosting each rank the bins; n_methods counts top-N memberships (0-6).
# The count is descriptive (the three linear models are closely related), so the
# follow-up list uses one vote per model family (linear, tree, PLS):
CONSENSUS_RULE = 'n_families_topk'    # top-N in >= CONSENSUS_MIN_FAMILIES_TOPK families;
                                      # or 'worst_family_rank' (<= CONSENSUS_MAX_WORST_FAMILY_RANK)
CONSENSUS_MIN_FAMILIES_TOPK = 2
CONSENSUS_MAX_WORST_FAMILY_RANK = 50
MIN_METHODS_OVERLAP = 2               # feature_overlap table: top-N in >= this many methods
N_RANKING_PERMUTATIONS = 500          # shuffles of the condition labels among colonies used as
                                      # the reference for the agreement between univariate F
                                      # and VIP (0 = skip). Both are fitted to the same colony
                                      # means, so they agree far more than unrelated lists would

# ── Classifier comparison (condition-plate discriminability) ────────────────────
# Six models (RF, SVM, GB, LR, LDA, Ridge), default settings, folds that hold out
# whole colonies. Produces classifier_comparison_<exp>.png and _accuracy_<exp>.csv.
RUN_CLASSIFIERS = True
CV_FOLDS = 5                          # reduced to the smallest colonies-per-condition
N_CV_REPEATS = 10                     # repeats with different colony splits

# ── Permutation test (could the accuracy arise with no real difference?) ────────
# Condition labels are shuffled among colonies and the whole cross-validation is
# repeated. The smallest attainable p-value is 1 / (N_PERMUTATIONS + 1).
RUN_PERMUTATION = True
N_PERMUTATIONS = 200
PERMUTATION_MODELS = 'Random Forest, SVM, Gradient Boosting, Logistic Regression, LDA, Ridge'
                                      # all six. Gradient Boosting takes nearly all of the
                                      # time (about 33 of 34 minutes for 45 spectra on an
                                      # 8-core laptop; about 2 minutes without it): remove
                                      # it from this list for a quicker run. The run
                                      # also tests the best of these models against the best
                                      # in each shuffle, which allows for picking the winner
N_JOBS = -1                           # parallel workers for the seed-averaged importances and
                                      # the permutation test (-1 = all cores)

# ── Descriptive outputs ─────────────────────────────────────────────────────────
N_PCA_COMPONENTS = 5
N_TOP_VIP = 30                        # features shown in the VIP plot
SIGNAL_R_MIN = 0.9                    # adjacent-bin correlation to test for one peak


# ── Load overrides from a JSON settings file ────────────────────────────────────
# Order: the file named by --config / TOOLKIT_CONFIG, else config.json and then,
# if present, config.local.json (your own settings; ignored by Git).
def _apply(path):
    with open(path) as f:
        overrides = _json.load(f)
    unknown = []
    for k, v in overrides.items():
        if k.startswith('_'):
            continue
        if k in globals() and k.isupper():
            globals()[k] = v
        else:
            unknown.append(k)
    if unknown:
        print(f"  [config] warning: unrecognised setting(s) ignored in {path}: {unknown}")
    return unknown


_here = _os.path.dirname(_os.path.abspath(__file__))
UNKNOWN_CONFIG_KEYS = []
if _os.environ.get('TOOLKIT_CONFIG'):
    CONFIG_FILE = _os.path.abspath(_os.environ['TOOLKIT_CONFIG'])
    if not _os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_FILE}")
    UNKNOWN_CONFIG_KEYS += _apply(CONFIG_FILE)
else:
    CONFIG_FILE = _os.path.join(_here, 'config.json')
    if _os.path.exists(CONFIG_FILE):
        UNKNOWN_CONFIG_KEYS += _apply(CONFIG_FILE)
    _local = _os.path.join(_here, 'config.local.json')
    if _os.path.exists(_local):
        UNKNOWN_CONFIG_KEYS += _apply(_local)
        CONFIG_FILE = _local
    del _local
del _here, _apply, _os, _json
