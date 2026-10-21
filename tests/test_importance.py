"""Tests for algorithm-specific importance (shared/importance.py)."""
import os, sys
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from shared.importance import (fit_importance, top_n, n_ties_at_cutoff, METHODS,
                               FAMILY, lda_structure_correlation)


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(1)
    K, n_per, p = 4, 9, 60
    y = np.repeat([f'c{k}' for k in range(K)], n_per)
    X = rng.normal(size=(K * n_per, p))
    markers = {0: 3, 1: 11, 2: 27, 3: 42}                    # one marker per class
    for k, j in markers.items():
        X[y == f'c{k}', j] += 3.0
    X = (X - X.mean(0)) / X.std(0, ddof=1)
    return X, y, markers


def test_original_configurations_are_reproduced_exactly(data):
    X, y, _ = data
    yi = np.unique(y, return_inverse=True)[1]
    ref = {
        'ridge': np.abs(RidgeClassifier().fit(X, yi).coef_).mean(0),
        'lr': np.abs(LogisticRegression(max_iter=1000, random_state=42).fit(X, yi).coef_).mean(0),
        'svm': np.abs(SVC(kernel='linear', random_state=42).fit(X, yi).coef_).mean(0),
        'rf': RandomForestClassifier(n_estimators=100, random_state=42).fit(X, yi).feature_importances_,
        'gb': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                         random_state=42).fit(X, yi).feature_importances_,
    }
    for m, v in ref.items():
        np.testing.assert_allclose(fit_importance(m, X, y, seed=42)['global'], v, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('method', METHODS)
def test_each_method_recovers_planted_markers(data, method):
    X, y, markers = data
    top = set(top_n(fit_importance(method, X, y, n_seeds=5 if method in ('rf', 'gb') else 1)['global'], 8))
    assert len(top & set(markers.values())) >= 3


def test_signed_output_only_for_coefficient_models(data):
    X, y, _ = data
    assert fit_importance('rf', X, y)['signed'] is None
    assert fit_importance('gb', X, y)['signed'] is None
    s = fit_importance('svm', X, y)
    assert s['signed'].shape[0] == 6 and s['signed_labels'][0] == 'c0 vs c1'      # OvO pairs
    assert fit_importance('svm', X, y, svm_multiclass='ovr')['signed'].shape[0] == 4
    assert fit_importance('lda', X, y)['signed'].shape == (4, X.shape[1])
    assert set(FAMILY[m] for m in METHODS) == {'L', 'T', 'P'}


def test_seed_averaging_returns_mean_of_single_seeds(data):
    X, y, _ = data
    r = fit_importance('rf', X, y, seed=7, n_seeds=3)
    singles = [fit_importance('rf', X, y, seed=s)['global'] for s in (7, 8, 9)]
    np.testing.assert_allclose(r['global'], np.mean(singles, 0))


def test_top_n_rules_and_ties():
    imp = np.array([0.0, 0.5, 0.2, 0.5, 0.0, 0.0])
    assert list(top_n(imp, 3)) == [1, 3, 2]                     # stable: ties by index
    assert list(top_n(imp, 3, rule='original')) == list(np.argsort(imp)[::-1][:3])
    assert n_ties_at_cutoff(imp, 5) == 3                        # three zeros tie at rank 5


def test_lda_structure_correlation_bounds(data):
    X, y, markers = data
    s = lda_structure_correlation(X, y)
    assert s.shape == (X.shape[1],) and np.all((s >= 0) & (s <= 1 + 1e-9))
    assert len(set(top_n(s, 8)) & set(markers.values())) >= 3


def test_saved_seed_pool_is_resumable_and_matches_direct_fit(data, tmp_path):
    from validation.importance_analysis import saved_seed_pool, seed_pool_summary, seed_topn_frequency
    X, y, markers = data
    path = tmp_path / 'pool.npy'
    part = saved_seed_pool('rf', X, y, str(path), 4, chunk=2, log=lambda s: None)
    full = saved_seed_pool('rf', X, y, str(path), 6, chunk=2, log=lambda s: None)   # resumes at 4
    np.testing.assert_array_equal(full[:4], part)
    np.testing.assert_allclose(full, fit_importance('rf', X, y, seed=0, n_seeds=6)['per_seed'])
    s = seed_pool_summary(full, ns=(5,))
    assert s.loc[0, 'split_half_seeds'] == 3 and 0 <= s.loc[0, 'jaccard_split_half_averages'] <= 1
    f = seed_topn_frequency(full, np.array(list(markers.values())), n=8)
    assert f.seed_top_frequency.min() >= 0.5
