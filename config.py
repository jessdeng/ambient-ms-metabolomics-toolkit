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
EXPERIMENT = 'sample_experiment'      # folder name under data/
METADATA_MANIFEST = 'metadata_manifest.csv'   # file -> class + colony; used if present
EXPECTED_TECH_REPS = None             # e.g. 3: stop if a colony has another count

# ── Spectra → features ──────────────────────────────────────────────────────────
MZ_MIN = 100                          # Da
MZ_MAX = 2000                         # Da
BIN_WIDTH = 0.5                       # Da; unit-resolution QqQ, absorbs m/z drift
BIN_LABELS = 'mean_mz'                # 'mean_mz' (where the signal sits) or
                                      # 'lower_edge' (MetaboAnalyst's labels)

# ── Feature filters (label-free; MetaboAnalyst defaults) ─────────────────────────
VARIANCE_PERCENTILE = 25              # drop the lowest-RSD X% of bins (0 = off)
ABUNDANCE_PERCENTILE = 5              # drop the lowest-mean X% of bins (0 = off)

# ── Scaling for the multivariate rankers ────────────────────────────────────────
# Spectra are always TIC-normalised and log10-transformed. 'autoscale' weights
# every bin equally (MetaboAnalyst default); 'pareto' keeps some intensity weight.
SCALING = 'autoscale'

# ── Prioritisation ──────────────────────────────────────────────────────────────
TOP_N_FEATURES = 50                   # top-N of each ranker
MIN_RANKERS = 2                       # candidate = top-N in >= this many of the two
                                      # voting rankers (univariate F, component-1 VIP)
RF_SEEDS = 25                         # random forest: sensitivity column, not a voter
RANDOM_SEED = 42

# ── Descriptive outputs ─────────────────────────────────────────────────────────
N_PCA_COMPONENTS = 5
N_TOP_VIP = 30                        # features shown in the VIP plot
SIGNAL_R_MIN = 0.9                    # adjacent-bin correlation to test for one peak


# ── Load overrides from a JSON settings file ────────────────────────────────────
_config_json_path = _os.environ.get('TOOLKIT_CONFIG') or _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), 'config.json')
CONFIG_FILE = _os.path.abspath(_config_json_path)
UNKNOWN_CONFIG_KEYS = []
if _os.path.exists(_config_json_path):
    with open(_config_json_path) as _f:
        _overrides = _json.load(_f)
    for _k, _v in _overrides.items():
        if _k.startswith('_'):
            continue
        if _k in globals() and _k.isupper():
            globals()[_k] = _v
        else:
            UNKNOWN_CONFIG_KEYS.append(_k)
    if UNKNOWN_CONFIG_KEYS:
        print(f"  [config] warning: unrecognised setting(s) ignored in {CONFIG_FILE}: "
              f"{UNKNOWN_CONFIG_KEYS}")
    del _f, _overrides
elif _os.environ.get('TOOLKIT_CONFIG'):
    raise FileNotFoundError(f"Configuration file not found: {_config_json_path}")
del _config_json_path, _os, _json
