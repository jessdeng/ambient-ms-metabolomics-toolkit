"""
Spectra → feature matrix, in the order the physics of ambient MS suggests.

1. load      every file's (m/z, intensity) profile; the instrument grid is kept
2. bin       sum all points into fixed-width m/z bins that lie entirely inside
             MZ_MIN..MZ_MAX (unit-resolution QqQ)
3. TIC       divide each spectrum by its own total ion current over ALL bins in
             range — corrects spray-to-spray differences in how much material
             was sampled. Per-spectrum, so no information is shared between
             samples.
4. filter    drop low-RSD and low-abundance bins (label-free)
5. log10     with a half-minimum offset for zeros (spray noise is multiplicative)

Scaling (autoscale / Pareto) is applied later, to the colony-level matrix the
multivariate rankers use. Inside cross-validation, steps 4-5 and the scaling are
fitted on the training spectra only (FoldPreprocessor).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

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


def bin_spectra(X: ArrayLike, mz: ArrayLike, bin_width: float = 0.5, mz_min: float = 100.0,
                mz_max: float = 2000.0, labels: str = 'mean_mz') -> tuple[FloatArray, FloatArray]:
    """Sum profile points into the half-open bins [mz_min + k*w, mz_min + (k+1)*w).

    Only bins that lie entirely inside [mz_min, mz_max) exist, so every bin spans
    the same width (no part-filled bin at either limit) and the grid is the same
    for every experiment run with the same settings. Bins that hold no
    instrument point, or no intensity in any spectrum, are dropped.

    labels='mean_mz'   : mean m/z of the instrument points in the bin
    labels='centroid'  : intensity-weighted mean m/z of the summed signal
    labels='lower_edge': the bin's lower edge (MetaboAnalyst)
    Returns (X_binned, mz_binned).
    """
    X, mz = np.asarray(X, float), np.asarray(mz, float)
    if X.ndim != 2 or X.shape[1] != mz.size:
        raise ValueError(f"spectra {X.shape} do not match the m/z axis ({mz.size} points)")
    if bin_width <= 0 or mz_max <= mz_min:
        raise ValueError("need BIN_WIDTH > 0 and MZ_MAX > MZ_MIN")
    if labels not in ('mean_mz', 'centroid', 'lower_edge'):
        raise ValueError("BIN_LABELS must be 'mean_mz', 'centroid' or 'lower_edge'")
    if np.any(np.diff(mz) <= 0):
        raise ValueError("m/z values must be strictly increasing")
    n_bins = int(np.floor((mz_max - mz_min) / bin_width + 1e-9))
    k = np.floor((mz - mz_min) / bin_width + 1e-9).astype(np.int64)     # bin of each point
    inside = (k >= 0) & (k < n_bins)
    if not inside.any():
        raise ValueError(f"no instrument points between m/z {mz_min} and {mz_max}")
    k, Xi, mzi = k[inside], X[:, inside], mz[inside]
    occupied, first = np.unique(k, return_index=True)                   # mz is sorted
    Xb = np.add.reduceat(Xi, first, axis=1)
    total = Xi.sum(axis=0)
    weight = np.add.reduceat(total, first)
    keep = weight > 0
    if labels == 'mean_mz':
        mzb = (np.add.reduceat(mzi, first) / np.diff(np.r_[first, mzi.size]))[keep]
    elif labels == 'centroid':
        mzb = np.add.reduceat(total * mzi, first)[keep] / weight[keep]
    else:
        mzb = mz_min + occupied[keep] * bin_width
    return Xb[:, keep], mzb


def tic_normalise(X):
    """Each spectrum divided by its total ion current (fractions of TIC)."""
    X = np.asarray(X, float)
    tic = X.sum(axis=1, keepdims=True)
    if np.any(tic <= 0):
        raise ValueError(f"spectra with no signal in the m/z range: rows "
                         f"{np.flatnonzero(tic.ravel() <= 0)[:5].tolist()}")
    return X / tic


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


def log_offset(X_lin: ArrayLike, how: str = 'global') -> FloatArray:
    """Offset added before log10 so that zeros stay finite, one value per bin.

    how='global' : half the smallest positive value of the whole matrix
    how='per_bin': half each bin's own smallest positive value, so a zero sits
                   just below that bin's other values instead of 1-2 log units
                   below them (bins with no positive value get the global one)
    """
    X = np.asarray(X_lin, float)
    if how not in ('global', 'per_bin'):
        raise ValueError("LOG_OFFSET must be 'global' or 'per_bin'")
    pos = X[X > 0]
    overall = pos.min() / 2 if pos.size else 1e-12
    if how == 'global':
        return np.full(X.shape[1], overall)
    smallest = np.where(X > 0, X, np.inf).min(axis=0)
    return np.where(np.isfinite(smallest), smallest / 2, overall)


def log10_transform(X_lin, offset='global'):
    """log10 after adding ``log_offset(X_lin, offset)`` (zeros stay finite)."""
    return np.log10(X_lin + log_offset(X_lin, offset))


def scale(X, method='autoscale'):
    """Column-centre and divide by the SD ('autoscale') or sqrt(SD) ('pareto')."""
    sd = X.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    if method == 'autoscale':
        return (X - X.mean(axis=0)) / sd
    if method == 'pareto':
        return (X - X.mean(axis=0)) / np.sqrt(sd)
    raise ValueError("SCALING must be 'autoscale' or 'pareto'")


@dataclass
class FoldPreprocessor:
    """The label-free filters, the log10 offset and the scaling, fitted on the
    training spectra only and then applied unchanged to held-out spectra.

    Input is the TIC-normalised matrix (TIC is per spectrum and needs no fit).
    Bins that are constant in the training spectra are always dropped.
    """
    variance_percentile: float = 25.0
    abundance_percentile: float = 5.0
    scaling: str = 'autoscale'
    offset: str = 'global'
    keep_: NDArray[np.bool_] = field(init=False, repr=False)
    offset_: FloatArray = field(init=False, repr=False)
    mean_: FloatArray = field(init=False, repr=False)
    scale_: FloatArray = field(init=False, repr=False)

    def fit(self, X_train: ArrayLike) -> FoldPreprocessor:
        X = self._check(X_train)
        if self.scaling not in ('autoscale', 'pareto'):
            raise ValueError("SCALING must be 'autoscale' or 'pareto'")
        keep = feature_filter_mask(X, self.variance_percentile, self.abundance_percentile)
        keep &= X.std(axis=0) > 0
        if not keep.any():
            raise ValueError("the filters removed every bin")
        off = log_offset(X[:, keep], self.offset)
        L = np.log10(X[:, keep] + off)
        sd = L.std(axis=0, ddof=1)
        sd = np.where(sd > 0, sd, 1.0)
        self.keep_, self.offset_, self.mean_ = keep, off, L.mean(axis=0)
        self.scale_ = np.sqrt(sd) if self.scaling == 'pareto' else sd
        return self

    def transform(self, X: ArrayLike) -> FloatArray:
        X = self._check(X)
        return (np.log10(X[:, self.keep_] + self.offset_) - self.mean_) / self.scale_

    @staticmethod
    def _check(X: ArrayLike) -> FloatArray:
        X = np.asarray(X, float)
        if X.ndim != 2 or not np.isfinite(X).all() or (X < 0).any():
            raise ValueError("expected a finite, non-negative 2-D matrix of TIC-normalised spectra")
        return X


def build_matrices(X_raw, mz_raw, bin_width=0.5, labels='mean_mz', mz_min=100,
                   mz_max=2000, variance_percentile=25, abundance_percentile=5,
                   log_offset_rule='global'):
    """Run steps 2-5. Returns a dict with
    'binned' (raw binned, in range), 'mz_binned', 'linear' (TIC-normalised,
    filtered), 'log' (log10 of 'linear'), 'mz' (filtered bins)."""
    Xb, mzb = bin_spectra(X_raw, mz_raw, bin_width, mz_min, mz_max, labels)
    Xn = tic_normalise(Xb)
    keep = feature_filter_mask(Xn, variance_percentile, abundance_percentile)
    return {'binned': Xb, 'mz_binned': mzb, 'linear': Xn[:, keep],
            'log': log10_transform(Xn[:, keep], log_offset_rule), 'mz': mzb[keep]}
