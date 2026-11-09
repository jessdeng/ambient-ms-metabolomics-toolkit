"""
Spectra → feature matrix, in the order the physics of ambient MS suggests.

1. load      every file's (m/z, intensity) profile; the instrument grid is kept
2. bin       sum all points into fixed-width m/z bins (unit-resolution QqQ)
3. range     keep MZ_MIN..MZ_MAX
4. TIC       divide each spectrum by its own total ion current over ALL bins in
             range — corrects spray-to-spray differences in how much material
             was sampled. Per-spectrum, so no information is shared between
             samples.
5. filter    drop low-RSD and low-abundance bins (label-free)
6. log10     with a half-minimum offset for zeros (spray noise is multiplicative)

Scaling (autoscale / Pareto) is applied later, to the colony-level matrix the
multivariate rankers use.
"""

import os
import numpy as np
import pandas as pd

_METADATA_FILENAMES = {'metadata_manifest.csv'}


def load_experiment(experiment_dir):
    """Read every .csv/.txt spectrum under ``experiment_dir``.

    Each file needs an m/z column ('mz' or 'Mass/Charge') and an intensity column
    ('int' or 'Intensity'). If all files share one m/z axis it is used as is;
    otherwise every spectrum is linearly interpolated onto a grid with the median
    native step over the m/z range common to all files.

    Returns (X, names, mz): X (n_spectra, n_points), names = paths relative to
    ``experiment_dir`` (POSIX), mz = the common axis.
    """
    experiment_dir = os.path.abspath(experiment_dir)
    found = []
    for dirpath, dirnames, filenames in os.walk(experiment_dir):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith('.'))
        for fname in sorted(filenames):
            if not fname.lower().endswith(('.csv', '.txt')) or fname.startswith('.'):
                continue
            if dirpath == experiment_dir and fname.lower() in _METADATA_FILENAMES:
                continue
            path = os.path.join(dirpath, fname)
            found.append((os.path.relpath(path, experiment_dir).replace(os.sep, '/'), path))
    if not found:
        raise ValueError(f"No .csv/.txt spectra found under {experiment_dir!r}.")
    found.sort()

    names, mzs, ints = [], [], []
    for rel, path in found:
        sep = '\t' if path.lower().endswith('.txt') else ','
        df = pd.read_csv(path, sep=sep, encoding='utf-8-sig')
        cols = {c.strip().lower(): c for c in df.columns}
        mz_col = cols.get('mz') or cols.get('mass/charge') or cols.get('m/z')
        int_col = cols.get('int') or cols.get('intensity')
        if mz_col is None or int_col is None:
            raise ValueError(f"{rel}: no m/z / intensity columns (found {list(df.columns)})")
        names.append(rel)
        mzs.append(df[mz_col].to_numpy(float))
        ints.append(df[int_col].to_numpy(float))

    ref = mzs[0]
    if all(m.shape == ref.shape and np.allclose(m, ref, rtol=0, atol=1e-9) for m in mzs):
        X, mz = np.vstack(ints), ref
    else:
        lo, hi = max(m.min() for m in mzs), min(m.max() for m in mzs)
        step = float(np.median(np.diff(ref)))
        mz = np.arange(lo, hi + step / 2, step)
        X = np.vstack([np.interp(mz, m, i) for m, i in zip(mzs, ints)])
    validate_spectra(X, mz, names)
    return X, names, mz


def validate_spectra(X, mz, names=None):
    """Stop on input that would silently corrupt every downstream number."""
    if X.ndim != 2 or X.shape[1] != mz.size:
        raise ValueError(f"spectra {X.shape} do not match the m/z axis ({mz.size} points)")
    if not np.all(np.diff(mz) > 0):
        raise ValueError("m/z values must be strictly increasing within every file")
    who = lambda rows: [names[i] for i in rows[:5]] if names is not None else rows[:5].tolist()
    bad = np.flatnonzero(~np.isfinite(X).all(axis=1))
    if bad.size:
        raise ValueError(f"non-finite intensities in {who(bad)}")
    neg = np.flatnonzero((X < 0).any(axis=1))
    if neg.size:
        raise ValueError(f"negative intensities in {who(neg)} (baseline-subtracted export?)")
    empty = np.flatnonzero(X.sum(axis=1) <= 0)
    if empty.size:
        raise ValueError(f"empty spectra: {who(empty)}")


def bin_features(X, mz, bin_width=0.5, labels='mean_mz'):
    """Sum intensities into fixed-width bins on a grid aligned to multiples of
    ``bin_width`` (so bins are comparable across experiments).

    labels='mean_mz'   : each bin is labelled with the mean m/z of the points in it
    labels='lower_edge': each bin is labelled with its lower edge (MetaboAnalyst)
    Empty bins are dropped. Returns (X_binned, mz_binned).
    """
    X, mz = np.asarray(X, float), np.asarray(mz, float)
    start = np.floor(mz.min() / bin_width + 1e-9) * bin_width
    idx = np.floor((mz - start) / bin_width + 1e-9).astype(int)     # bin of each point
    bins, first = np.unique(idx, return_index=True)                 # mz is sorted
    Xb = np.add.reduceat(X, first, axis=1)
    if labels == 'mean_mz':
        mzb = np.add.reduceat(mz, first) / np.diff(np.r_[first, mz.size])
    elif labels == 'lower_edge':
        mzb = start + bins * bin_width
    else:
        raise ValueError("BIN_LABELS must be 'mean_mz' or 'lower_edge'")
    keep = Xb.sum(axis=0) > 0
    return Xb[:, keep], mzb[keep]


def filter_mass_range(X, mz, mz_min=100, mz_max=2000):
    keep = (mz >= mz_min) & (mz <= mz_max)
    return X[:, keep], mz[keep]


def tic_normalise(X):
    """Each spectrum divided by its total ion current (fractions of TIC)."""
    X = np.asarray(X, float)
    return X / X.sum(axis=1, keepdims=True)


def feature_filter_mask(X_lin, variance_percentile=25, abundance_percentile=5):
    """Keep-mask for the label-free filters on the linear, TIC-normalised matrix:
    relative standard deviation above its ``variance_percentile`` and mean
    intensity above its ``abundance_percentile`` (both computed over all bins)."""
    mean = X_lin.mean(axis=0)
    keep = np.ones(X_lin.shape[1], dtype=bool)
    if variance_percentile > 0:
        rsd = np.divide(X_lin.std(axis=0), mean, out=np.zeros_like(mean), where=mean > 0)
        keep &= rsd > np.percentile(rsd, variance_percentile)
    if abundance_percentile > 0:
        keep &= mean > np.percentile(mean, abundance_percentile)
    return keep


def log10_transform(X_lin):
    """log10 with an offset of half the smallest positive value (zeros stay finite)."""
    pos = X_lin[X_lin > 0]
    return np.log10(X_lin + (pos.min() / 2 if pos.size else 1e-12))


def scale(X, method='autoscale'):
    """Column-centre and divide by the SD ('autoscale') or sqrt(SD) ('pareto')."""
    sd = X.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    if method == 'autoscale':
        return (X - X.mean(axis=0)) / sd
    if method == 'pareto':
        return (X - X.mean(axis=0)) / np.sqrt(sd)
    raise ValueError("SCALING must be 'autoscale' or 'pareto'")


def build_matrices(X_raw, mz_raw, bin_width=0.5, labels='mean_mz', mz_min=100,
                   mz_max=2000, variance_percentile=25, abundance_percentile=5):
    """Run steps 2-6. Returns a dict with
    'binned' (raw binned, in range), 'mz_binned', 'linear' (TIC-normalised,
    filtered), 'log' (log10 of 'linear'), 'mz' (filtered bins)."""
    Xb, mzb = bin_features(X_raw, mz_raw, bin_width, labels)
    Xb, mzb = filter_mass_range(Xb, mzb, mz_min, mz_max)
    Xn = tic_normalise(Xb)
    keep = feature_filter_mask(Xn, variance_percentile, abundance_percentile)
    return {'binned': Xb, 'mz_binned': mzb, 'linear': Xn[:, keep],
            'log': log10_transform(Xn[:, keep]), 'mz': mzb[keep]}
