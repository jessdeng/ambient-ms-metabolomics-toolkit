"""Spectra -> feature matrix: loading, binning, TIC, filters, log, scaling."""
import os, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from standard.preprocessing import (load_experiment, validate_spectra, bin_spectra,
                                    tic_normalise, feature_filter_mask, log_offset,
                                    log10_transform, scale, build_matrices, FoldPreprocessor)

MZ = np.round(np.arange(100.0, 110.0001, 0.1), 1)


def _slow_bins(X, mz, w, lo, hi):
    n = int(np.floor((hi - lo) / w + 1e-9))
    idx = np.floor((mz - lo) / w + 1e-9).astype(int)
    return np.column_stack([X[:, idx == b].sum(1) for b in range(n) if (idx == b).any()])


def test_binning_matches_reference_and_labels():
    X = np.random.default_rng(0).random((3, MZ.size))
    Xb, mzb = bin_spectra(X, MZ, 0.5, 100, 110)
    np.testing.assert_allclose(Xb, _slow_bins(X, MZ, 0.5, 100, 110))
    assert Xb.shape[1] == 20                                    # [100, 110): the point at 110.0 is outside
    assert np.allclose(Xb.sum(1), X[:, MZ < 110].sum(1))        # nothing inside the range lost
    assert mzb[0] == pytest.approx(MZ[MZ < 100.5].mean())       # mean m/z of the bin
    _, edges = bin_spectra(X, MZ, 0.5, 100, 110, labels='lower_edge')
    assert np.allclose(edges[:3], [100.0, 100.5, 101.0])        # grid starts at mz_min
    _, cen = bin_spectra(X, MZ, 0.5, 100, 110, labels='centroid')
    w = X[:, MZ < 100.5].sum(0)
    assert cen[0] == pytest.approx((w * MZ[MZ < 100.5]).sum() / w.sum())   # intensity-weighted
    assert ((cen >= edges) & (cen < edges + 0.5)).all()
    with pytest.raises(ValueError):
        bin_spectra(X, MZ, 0.5, 100, 110, labels='centre')


@pytest.mark.parametrize('width', [0.5, 0.7, 1.0])
def test_every_bin_spans_the_full_width(width):
    mz = np.round(np.arange(100.0, 2000.0001, 0.1), 1)
    X = np.ones((2, mz.size))
    Xb, edges = bin_spectra(X, mz, width, 100, 2000, labels='lower_edge')
    assert np.allclose(Xb, Xb[0, 0]) and Xb[0, 0] == pytest.approx(width / 0.1)   # same points in every bin
    assert edges[0] == 100 and edges[-1] + width <= 2000 + 1e-9


def test_binning_drops_empty_bins_and_points_outside_the_range():
    X = np.array([[1., 2, 0, 0, 5, 9], [3, 4, 0, 0, 6, 9]])
    mz = np.array([100.0, 100.1, 100.6, 100.7, 103.2, 250.0])
    Xb, edges = bin_spectra(X, mz, 0.5, 100, 104, labels='lower_edge')
    np.testing.assert_allclose(Xb, [[3, 5], [7, 6]])            # all-zero bin and m/z 250 gone
    assert edges.tolist() == [100.0, 103.0]
    with pytest.raises(ValueError, match='no instrument points'):
        bin_spectra(X, mz, 0.5, 300, 400)


def test_tic_filters_log_scale():
    rng = np.random.default_rng(1)
    X = rng.lognormal(3, 1, (6, 40)); X[:, 0] = 0
    Xn = tic_normalise(X)
    assert np.allclose(Xn.sum(1), 1)
    keep = feature_filter_mask(Xn, 25, 5)
    assert 25 <= keep.sum() <= 30 and not keep[0]               # all-zero bin dropped
    L = log10_transform(Xn)
    assert np.isfinite(L).all()
    assert np.isfinite(log10_transform(Xn, 'per_bin')).all()
    with pytest.raises(ValueError, match='no signal'):
        tic_normalise(np.vstack([X, np.zeros(40)]))
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


def test_log_offset_global_and_per_bin():
    X = np.array([[0., 4., 0.], [2., 8., 0.], [6., 0., 0.]])
    assert log_offset(X).tolist() == [1.0, 1.0, 1.0]            # half the smallest positive value
    assert log_offset(X, 'per_bin').tolist() == [1.0, 2.0, 1.0]  # per bin; empty bin -> global
    with pytest.raises(ValueError):
        log_offset(X, 'median')


def test_fold_preprocessor_is_fitted_on_training_rows_only():
    rng = np.random.default_rng(3)
    X = rng.lognormal(0, 1, (12, 30)); X[rng.random(X.shape) < 0.05] = 0
    X[:, 0] = 0; X[:, 1] = 2.0                                  # empty and constant bins
    X = np.column_stack([X[:, :2], X[:, 2:] / X[:, 2:].sum(1, keepdims=True)])
    tr, te = np.arange(8), np.arange(8, 12)
    for rule in ('global', 'per_bin'):
        p = FoldPreprocessor(0, 0, 'autoscale', rule).fit(X[tr])
        assert not p.keep_[0] and not p.keep_[1]                # nothing to scale: always dropped
        Ztr, Zte = p.transform(X[tr]), p.transform(X[te])
        assert np.allclose(Ztr.mean(0), 0) and np.allclose(Ztr.std(0, ddof=1), 1)
        assert np.isfinite(Zte).all() and Zte.shape[1] == Ztr.shape[1]
        X2 = X.copy(); X2[te] *= 50                             # change only the held-out rows
        q = FoldPreprocessor(0, 0, 'autoscale', rule).fit(X2[tr])
        assert np.array_equal(q.keep_, p.keep_) and np.array_equal(q.mean_, p.mean_)
        assert np.array_equal(q.offset_, p.offset_) and np.array_equal(q.scale_, p.scale_)
    with pytest.raises(ValueError, match='non-negative'):
        FoldPreprocessor().fit(-X)
    with pytest.raises(ValueError, match='every bin'):
        FoldPreprocessor(0, 0).fit(np.ones((4, 3)))


def test_identical_spectra_are_reported():
    from standard.preprocessing import identical_spectra
    X = np.array([[1.0, 2, 3], [1, 2, 3], [1, 2, 4], [9, 9, 9], [1, 2, 3]])
    assert identical_spectra(X, list('abcde')) == [['a', 'b', 'e']]
    assert identical_spectra(X[2:4], ['c', 'd']) == []
