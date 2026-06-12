"""
standard/preprocessing.py — Data Loading and Preprocessing
============================================================
Loads raw ambient MS data from per-group CSV/TXT files, bins features into
fixed-width m/z windows (bin labels = mean m/z of actual values in bin),
and applies the configurable normalization → transformation → scaling chain.

Usage:
    from standard.preprocessing import (
        load_experiment, bin_features, filter_mass_range,
        filter_low_variance, filter_low_abundance, preprocess
    )
"""

import os
import numpy as np
import pandas as pd


# ── File loading ───────────────────────────────────────────────────────────────

def _read_sample(path):
    """Read one CSV or TXT sample file. Returns (mz_array, intensity_array)."""
    ext = os.path.splitext(path)[1].lower()
    sep = ',' if ext == '.csv' else '\t'
    df = pd.read_csv(path, sep=sep, engine='python')

    # Normalise column names
    df.columns = [c.strip().lower().replace('/', '').replace(' ', '') for c in df.columns]

    # Locate m/z column
    mz_candidates = ['mz', 'masscharge', 'moverz', 'mass']
    mz_col = next((c for c in mz_candidates if c in df.columns), None)
    if mz_col is None:
        raise ValueError(
            f"Could not find m/z column in {path}. "
            f"Expected one of: mz, Mass/Charge. Found: {list(df.columns)}"
        )

    # Locate intensity column
    int_candidates = ['int', 'intensity', 'intensities', 'abundance']
    int_col = next((c for c in int_candidates if c in df.columns), None)
    if int_col is None:
        raise ValueError(
            f"Could not find intensity column in {path}. "
            f"Expected one of: int, Intensity. Found: {list(df.columns)}"
        )

    mz  = df[mz_col].astype(float).values
    ints = df[int_col].astype(float).values
    return mz, ints


def load_experiment(experiment_dir):
    """
    Load all samples from an experiment directory.

    Directory structure:
        experiment_dir/
            Group1/
                sample1.csv
                sample2.csv
            Group2/
                ...

    Returns
    -------
    X_raw        : ndarray (n_samples, n_features)  — intensity matrix
    y_labels     : ndarray of str                   — group label per sample
    sample_names : list of str                      — filename per sample
    mz           : ndarray                          — common m/z axis
    """
    groups = sorted([
        d for d in os.listdir(experiment_dir)
        if os.path.isdir(os.path.join(experiment_dir, d)) and not d.startswith('.')
    ])
    if not groups:
        raise ValueError(f"No group subfolders found in {experiment_dir!r}")

    all_mz, all_ints, all_labels, all_names = [], [], [], []

    for group in groups:
        group_dir = os.path.join(experiment_dir, group)
        files = sorted([
            f for f in os.listdir(group_dir)
            if f.lower().endswith(('.csv', '.txt')) and not f.startswith('.')
        ])
        for fname in files:
            mz, ints = _read_sample(os.path.join(group_dir, fname))
            all_mz.append(mz)
            all_ints.append(ints)
            all_labels.append(group)
            all_names.append(fname)

    # Build a common m/z axis (union of all sample m/z values, sorted)
    common_mz = np.unique(np.concatenate(all_mz))

    # Align every sample onto the common axis
    n_samples  = len(all_ints)
    n_features = len(common_mz)
    X_raw = np.zeros((n_samples, n_features), dtype=float)

    for i, (mz, ints) in enumerate(zip(all_mz, all_ints)):
        idx = np.searchsorted(common_mz, mz)
        # Only write positions that land within bounds (guards against float drift)
        valid = (idx < n_features) & (common_mz[np.clip(idx, 0, n_features - 1)] == mz)
        X_raw[i, idx[valid]] = ints[valid]

    return X_raw, np.array(all_labels), all_names, common_mz


# ── Binning ────────────────────────────────────────────────────────────────────

def bin_features(X, mz, bin_width=0.5):
    """
    Aggregate features into fixed-width m/z bins by summing intensities.

    Standard version: bin labels are the **mean of actual m/z values** within
    each bin — use these for accurate database lookup.

    Parameters
    ----------
    X         : ndarray (n_samples, n_features)
    mz        : ndarray (n_features,)
    bin_width : float — width of each bin in Da

    Returns
    -------
    X_binned  : ndarray (n_samples, n_bins)
    mz_binned : ndarray (n_bins,)  — mean m/z per bin
    """
    mz_min  = np.floor(mz.min() / bin_width) * bin_width
    mz_max  = np.ceil(mz.max()  / bin_width) * bin_width
    edges   = np.arange(mz_min, mz_max + bin_width, bin_width)
    bin_idx = np.digitize(mz, edges) - 1   # 0-indexed

    n_bins    = len(edges) - 1
    X_binned  = np.zeros((X.shape[0], n_bins), dtype=float)
    mz_binned = np.zeros(n_bins, dtype=float)

    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            X_binned[:, b] = X[:, mask].sum(axis=1)
            mz_binned[b]   = mz[mask].mean()       # mean of actual m/z values
        else:
            mz_binned[b] = edges[b] + bin_width / 2.0

    # Drop bins with no data
    has_data  = mz_binned > 0
    X_binned  = X_binned[:, has_data]
    mz_binned = mz_binned[has_data]

    return X_binned, mz_binned


# ── Range filter ───────────────────────────────────────────────────────────────

def filter_mass_range(X, mz, mz_min=100, mz_max=1000):
    """
    Remove features outside [mz_min, mz_max].

    Returns X_filtered, mz_filtered.
    """
    mask = (mz >= mz_min) & (mz <= mz_max)
    return X[:, mask], mz[mask]


# ── Feature-level filters ──────────────────────────────────────────────────────

def filter_low_variance(X, mz, percentile=25):
    """
    Remove features in the bottom `percentile` by relative standard deviation
    (RSD = std / mean). Set percentile=0 to disable.
    """
    if percentile <= 0:
        return X, mz
    mean = X.mean(axis=0)
    mean[mean == 0] = 1e-12
    rsd  = X.std(axis=0) / mean
    keep = rsd > np.percentile(rsd, percentile)
    return X[:, keep], mz[keep]


def filter_low_abundance(X, mz, percentile=5):
    """
    Remove features in the bottom `percentile` by mean intensity.
    Set percentile=0 to disable.
    """
    if percentile <= 0:
        return X, mz
    m    = X.mean(axis=0)
    keep = m > np.percentile(m, percentile)
    return X[:, keep], mz[keep]


# ── Preprocessing chain ────────────────────────────────────────────────────────

def _normalize(X, method):
    """Sample-level normalisation (operates row-wise)."""
    X = X.astype(float)
    if method == 'none':
        return X
    if method == 'tic':
        rs = X.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        median_tic = np.median(X.sum(axis=1))
        return X / rs * median_tic
    if method == 'median':
        rm = np.median(X, axis=1, keepdims=True)
        rm[rm == 0] = 1
        global_median = np.median(np.median(X, axis=1))
        return X / rm * global_median
    if method == 'pqn':
        rs  = X.sum(axis=1, keepdims=True); rs[rs == 0] = 1
        Xn  = X / rs
        ref = np.median(Xn, axis=0); ref[ref == 0] = 1
        q   = Xn / ref
        d   = np.median(q, axis=1, keepdims=True); d[d == 0] = 1
        return Xn / d
    if method == 'quantile':
        sorted_means = np.sort(X, axis=1).mean(axis=0)
        ranks = np.argsort(np.argsort(X, axis=1), axis=1)
        return sorted_means[ranks]
    raise ValueError(f"Unknown normalization: '{method}'")


def _transform(X, method):
    """Feature-level transformation (log, sqrt, etc.)."""
    X = X.astype(float)
    if method == 'none':
        return X
    # Half-minimum pseudo-count avoids log(0)
    pos = X[X > 0]
    half = pos.min() / 2 if pos.size else 1e-6
    if method == 'log10':
        return np.log10(X + half)
    if method == 'log2':
        return np.log2(X + half)
    if method == 'sqrt':
        return np.sqrt(X)
    raise ValueError(f"Unknown log_transform: '{method}'")


def _scale(X, method):
    """Feature-level scaling (operates column-wise)."""
    X = X.astype(float)
    if method == 'none':
        return X
    mean = X.mean(axis=0)
    std  = X.std(axis=0, ddof=1)
    std[std == 0] = 1
    if method == 'autoscale':
        return (X - mean) / std
    if method == 'pareto':
        return (X - mean) / np.sqrt(std)
    if method == 'range':
        rng = X.max(axis=0) - X.min(axis=0)
        rng[rng == 0] = 1
        return (X - mean) / rng
    if method == 'vast':
        return ((X - mean) / std) * (mean / (std + 1e-10))
    if method == 'level':
        lvl = np.abs(mean); lvl[lvl == 0] = 1
        return (X - mean) / lvl
    raise ValueError(f"Unknown scaling: '{method}'")


def preprocess(X, normalization='tic', log_transform='log10', scaling='autoscale'):
    """
    Apply the full preprocessing chain in order:
        normalise → transform → scale

    Parameters mirror config.py settings. Operates in-place on a copy — pass
    X.copy() if you need the original unchanged.
    """
    X = _normalize(X, normalization)
    X = _transform(X, log_transform)
    X = _scale(X, scaling)
    return X
