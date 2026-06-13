"""
condition_spectrum.py
=====================
Plots the mean MS spectrum per condition with high-confidence ensemble features
highlighted as vertical tick marks.

Reads feature_overlap_<experiment>.csv and the raw binned intensity matrix from
the results folder, then calls shared.visualization.plot_spectrum_with_features.

Usage (from repo root):
    python -m src.scripts.condition_spectrum

Configure PIPELINE below to match the pipeline you ran.
"""

import os
import sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, _ROOT)  # config.py lives here
sys.path.insert(0, _SRC)   # src/ packages take priority
import config

# ── Configuration ─────────────────────────────────────────────────────────────
PIPELINE      = 'standard'    # 'standard' or 'r_comparable'
MIN_N_METHODS = config.HIGH_CONFIDENCE_N_METHODS   # set in config.py
INTERACTIVE   = True         # True → Plotly HTML (notebook); False → Matplotlib PNG
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR   = _ROOT
OUTPUT_DIR = os.path.join(BASE_DIR, 'results', PIPELINE)


def _load_pipeline(pipeline):
    """Return (preprocessing module, bin_features function) for the given pipeline."""
    if pipeline == 'r_comparable':
        from r_comparable.preprocessing import (
            load_experiment, bin_features, filter_mass_range,
            filter_low_variance, filter_low_abundance,
        )
    else:
        from standard.preprocessing import (
            load_experiment, bin_features, filter_mass_range,
            filter_low_variance, filter_low_abundance,
        )
    return load_experiment, bin_features, filter_mass_range, filter_low_variance, filter_low_abundance


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    experiment = config.EXPERIMENT.strip()
    safe_name  = experiment.replace(' ', '_').replace(':', '')

    overlap_path = os.path.join(OUTPUT_DIR, f'feature_overlap_{safe_name}.csv')
    if not os.path.exists(overlap_path):
        print(f"feature_overlap CSV not found: {overlap_path}")
        print("Run run_analysis.py first.")
        return

    overlap_df = pd.read_csv(overlap_path)
    print(f"Loaded {len(overlap_df)} candidate features from {overlap_path}")

    # Load experiment data through the appropriate pipeline
    load_experiment, bin_features, filter_mass_range, filter_low_variance, filter_low_abundance = \
        _load_pipeline(PIPELINE)

    experiment_dir = os.path.join(BASE_DIR, 'data', experiment)
    if not os.path.isdir(experiment_dir):
        print(f"Experiment data not found at: {experiment_dir}")
        print("Place your data inside data/<EXPERIMENT>/ and re-run.")
        return

    X_raw, y_labels, sample_names, mz = load_experiment(experiment_dir)

    X_binned, mz = bin_features(X_raw, mz, bin_width=config.BIN_WIDTH)
    X_binned, mz = filter_mass_range(X_binned, mz,
                                     mz_min=config.MZ_MIN, mz_max=config.MZ_MAX)

    from shared.visualization import plot_spectrum_with_features

    ext      = 'html' if INTERACTIVE else 'png'
    out_path = os.path.join(OUTPUT_DIR, f'condition_spectrum_{safe_name}.{ext}')
    plot_spectrum_with_features(
        X_binned, mz, y_labels, overlap_df,
        experiment_name=f'{experiment} ({PIPELINE})',
        out_path=out_path,
        min_n_methods=MIN_N_METHODS,
        interactive=INTERACTIVE,
    )
    print(f"Saved -> {out_path}")


if __name__ == '__main__':
    main()
