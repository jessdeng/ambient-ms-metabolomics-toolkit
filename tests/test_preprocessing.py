"""Spectra -> feature matrix: loading, binning, TIC, filters, log, scaling."""
import os, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from standard.preprocessing import (load_experiment, validate_spectra, bin_features,
                                    filter_mass_range, tic_normalise, feature_filter_mask,
                                    log10_transform, scale, build_matrices)

MZ = np.round(np.arange(100.0, 110.0001, 0.1), 1)


def _slow_bins(X, mz, w):
    start = np.floor(mz.min() / w + 1e-9) * w
    idx = np.floor((mz - start) / w + 1e-9).astype(int)
    out = {b: X[:, idx == b].sum(1) for b in np.unique(idx)}
    return np.column_stack([out[b] for b in sorted(out)])


def test_binning_matches_reference_and_labels():
    X = np.random.default_rng(0).random((3, MZ.size))
    Xb, mzb = bin_features(X, MZ, 0.5)
    np.testing.assert_allclose(Xb, _slow_bins(X, MZ, 0.5))
    assert np.allclose(Xb.sum(1), X.sum(1))                     # nothing lost
    assert mzb[0] == pytest.approx(MZ[MZ < 100.5].mean())       # mean m/z of the bin
    _, edges = bin_features(X, MZ, 0.5, labels='lower_edge')
    assert np.allclose(edges[:3], [100.0, 100.5, 101.0])        # grid on multiples of 0.5
    with pytest.raises(ValueError):
        bin_features(X, MZ, 0.5, labels='centre')


def test_tic_filters_log_scale():
    rng = np.random.default_rng(1)
    X = rng.lognormal(3, 1, (6, 40)); X[:, 0] = 0
    Xn = tic_normalise(X)
    assert np.allclose(Xn.sum(1), 1)
    keep = feature_filter_mask(Xn, 25, 5)
    assert 25 <= keep.sum() <= 30 and not keep[0]               # all-zero bin dropped
    L = log10_transform(Xn)
    assert np.isfinite(L).all()
    A, P = scale(L, 'autoscale'), scale(L, 'pareto')
    assert np.allclose(A.mean(0), 0) and np.allclose(A[:, 1:].std(0, ddof=1), 1)
    assert np.allclose(P.mean(0), 0)
    with pytest.raises(ValueError):
        scale(L, 'vast')


def test_tic_uses_all_bins_before_filtering():
    X = np.array([[1., 1., 98.], [1., 1., 8.]])
    M = build_matrices(X, np.array([100.0, 101.0, 102.0]), 0.5, 'mean_mz', 100, 200, 0, 0)
    np.testing.assert_allclose(M['linear'], X / X.sum(1, keepdims=True))


@pytest.mark.parametrize('bad, msg', [
    (lambda X: np.where(X > 0.5, np.nan, X), 'non-finite'),
    (lambda X: X - 1, 'negative'),
    (lambda X: np.vstack([np.zeros(X.shape[1]), X[1:]]), 'empty'),
])
def test_validate_spectra_rejects_bad_input(bad, msg):
    X = np.random.default_rng(2).random((3, MZ.size))
    with pytest.raises(ValueError, match=msg):
        validate_spectra(bad(X), MZ)
    with pytest.raises(ValueError, match='increasing'):
        validate_spectra(X, MZ[::-1])


def test_load_experiment_shared_and_differing_axes(tmp_path):
    for i, (cond, mz) in enumerate([('A', MZ), ('A', MZ), ('B', MZ + 0.05)]):
        (tmp_path / cond).mkdir(exist_ok=True)
        pd.DataFrame({'mz': mz, 'int': np.arange(mz.size) + 1.0}).to_csv(tmp_path / cond / f'c{i}T1.csv', index=False)
    pd.DataFrame({'file_path': ['x']}).to_csv(tmp_path / 'metadata_manifest.csv', index=False)
    X, names, mz = load_experiment(str(tmp_path))
    assert names == ['A/c0T1.csv', 'A/c1T1.csv', 'B/c2T1.csv']   # manifest skipped
    assert mz.min() >= 100.05 - 1e-9 and X.shape == (3, mz.size)   # common range, interpolated


def test_mass_range():
    X = np.ones((2, 5)); mz = np.array([90., 100., 150., 2000., 2100.])
    _, kept = filter_mass_range(X, mz, 100, 2000)
    assert kept.tolist() == [100., 150., 2000.]
