"""
r_comparable/preprocessing.py — R-compatible Data Loading and Preprocessing
=============================================================================
Identical to standard/preprocessing.py except that bin_features uses the
MetaboAnalyst offset convention for bin labels: the label for each bin is the
lower edge of the bin interval (e.g. a bin spanning 100.0–100.5 Da is
labelled 100.0). This matches the bin labels produced by MetaboAnalyst's
processRawMSData() and allows direct numeric comparison of reported m/z values.

All other functions (load_experiment, filter_*, preprocess) are identical to
standard/preprocessing.py.

Usage:
    from r_comparable.preprocessing import (
        load_experiment, bin_features, filter_mass_range,
        filter_low_variance, filter_low_abundance, preprocess
    )
"""

# Re-export everything from standard.preprocessing, then override bin_features.
from standard.preprocessing import (
    load_experiment,
    filter_mass_range,
    filter_prevalence,
    filter_snr_floor,
    filter_low_variance,
    filter_low_abundance,
    preprocess,
)

import numpy as np


def bin_features(X, mz, bin_width=0.5):
    """
    Aggregate features into fixed-width m/z bins by summing intensities.

    R-comparable version: bin labels are the **lower edge** of each bin
    interval, matching the MetaboAnalyst convention. This allows m/z values
    reported here to be compared directly with MetaboAnalyst output.

    Parameters
    ----------
    X         : ndarray (n_samples, n_features)
    mz        : ndarray (n_features,)
    bin_width : float — width of each bin in Da

    Returns
    -------
    X_binned  : ndarray (n_samples, n_bins)
    mz_binned : ndarray (n_bins,)  — lower edge m/z per bin
    """
    mz_min  = np.floor(mz.min() / bin_width) * bin_width
    mz_max  = np.ceil(mz.max()  / bin_width) * bin_width
    edges   = np.arange(mz_min, mz_max + bin_width, bin_width)
    bin_idx = np.digitize(mz, edges) - 1   # 0-indexed

    n_bins    = len(edges) - 1
    X_binned  = np.zeros((X.shape[0], n_bins), dtype=float)
    mz_binned = edges[:n_bins].copy()      # lower edge — MetaboAnalyst convention

    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            X_binned[:, b] = X[:, mask].sum(axis=1)

    # Drop bins with no signal
    has_data  = X_binned.sum(axis=0) > 0
    X_binned  = X_binned[:, has_data]
    mz_binned = mz_binned[has_data]

    return X_binned, mz_binned
