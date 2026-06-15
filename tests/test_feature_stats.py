"""
Tests for the univariate statistics layer (A3): fold-change, univariate test,
BH-FDR, and — critically — replicate-aware (colony-level) aggregation so technical
replicates are not pseudoreplicated.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from shared.feature_stats import (  # noqa: E402
    univariate_feature_stats, benjamini_hochberg, importance_correlation_matrix,
)


def _two_class_with_replicates(seed=0):
    """3 colonies/class x 3 technical replicates; feature 0 differs by class,
    feature 1 is pure noise. Returns X_log, y, groups."""
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    for cls, base in [('A', 0.0), ('B', 2.0)]:
        for colony in range(3):
            colony_shift = rng.normal(0, 0.3)
            for _t in range(3):
                f0 = base + colony_shift + rng.normal(0, 0.05)   # signal
                f1 = rng.normal(0, 1.0)                          # noise
                X.append([f0, f1])
                y.append(cls)
                groups.append(f'{cls}::colony{colony}')
    return np.array(X), np.array(y), np.array(groups)


def test_aggregation_reduces_units_to_colonies():
    X, y, groups = _two_class_with_replicates()
    res = univariate_feature_stats(X, y, log_transform='none', groups=groups,
                                   aggregate_within_group=True)
    assert res['aggregated'] is True
    assert res['n_units_per_class'] == {'A': 3, 'B': 3}      # colonies, not 9 spectra
    assert 'colony-aggregated' in res['test_name']


def test_per_spectrum_uses_all_samples():
    X, y, groups = _two_class_with_replicates()
    res = univariate_feature_stats(X, y, log_transform='none',
                                   aggregate_within_group=False)
    assert res['aggregated'] is False
    assert res['n_units_per_class'] == {'A': 9, 'B': 9}
    assert 'per-spectrum' in res['test_name']


def test_pseudoreplication_inflates_significance():
    # The per-spectrum p-value for the real signal should be smaller (more
    # 'significant') than the honest colony-level p-value — the inflation A3 fixes.
    X, y, groups = _two_class_with_replicates()
    agg = univariate_feature_stats(X, y, log_transform='none', groups=groups,
                                   aggregate_within_group=True)
    raw = univariate_feature_stats(X, y, log_transform='none',
                                   aggregate_within_group=False)
    assert raw['p_value'][0] < agg['p_value'][0]


def test_columns_present_and_bounded():
    X, y, groups = _two_class_with_replicates()
    res = univariate_feature_stats(X, y, log_transform='none', groups=groups)
    for key in ('p_value', 'q_value', 'fold_change', 'log2_fold_change'):
        assert res[key].shape == (2,)
    assert np.all((res['q_value'] >= 0) & (res['q_value'] <= 1))


def test_multiclass_uses_max_pairwise_fold_change():
    # 3 classes, 2 colonies each, constant within class: means 1, 2, 8 on feature 0.
    X, y, groups = [], [], []
    for cls, val in [('A', 1.0), ('B', 2.0), ('C', 8.0)]:
        for colony in range(2):
            for _t in range(3):
                X.append([val]); y.append(cls); groups.append(f'{cls}::c{colony}')
    X = np.array(X, dtype=float)
    res = univariate_feature_stats(X, np.array(y), log_transform='none',
                                   groups=np.array(groups))
    assert 'ANOVA' in res['test_name']
    assert np.isclose(res['fold_change'][0], 8.0)   # brightest/dimmest = 8/1


def test_wilcoxon_path_selected():
    X, y, groups = _two_class_with_replicates()
    res = univariate_feature_stats(X, y, log_transform='none', groups=groups,
                                   test='wilcoxon')
    assert 'Wilcoxon' in res['test_name']


def test_aggregate_rejects_multiclass_group():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array(['A', 'A', 'B'])
    groups = np.array(['g1', 'g1', 'g1'])    # one group spanning two classes
    with pytest.raises(ValueError):
        univariate_feature_stats(X, y, log_transform='none', groups=groups)


def test_benjamini_hochberg_basic():
    p = np.array([0.001, 0.01, 0.5, 1.0])
    q = benjamini_hochberg(p)
    assert np.all(q >= p)                     # q >= p elementwise
    assert np.all((q >= 0) & (q <= 1))


# --------------------------------------------------------------------------- #
# B1: method x method Spearman importance-correlation matrix                     #
# --------------------------------------------------------------------------- #
def test_importance_corr_matrix_shape_and_diagonal():
    rng = np.random.default_rng(0)
    imp = {m: rng.random(40) for m in ['rf', 'svm', 'gb', 'lr', 'ridge', 'vip']}
    M = importance_correlation_matrix(imp)
    assert M.shape == (6, 6)
    assert list(M.index) == list(M.columns) == list(imp)
    assert np.allclose(np.diag(M.values), 1.0)


def test_importance_corr_matrix_symmetric():
    rng = np.random.default_rng(1)
    imp = {'a': rng.random(50), 'b': rng.random(50), 'c': rng.random(50)}
    M = importance_correlation_matrix(imp).values
    assert np.allclose(M, M.T, equal_nan=True)


def test_importance_corr_matrix_detects_redundancy():
    # A monotone transform of a vector must be Spearman-correlated 1.0 with it
    # (this is the redundancy B1 is meant to expose between collinear models).
    base = np.linspace(1, 5, 60)
    imp = {'x': base, 'y': base ** 3 + 2}   # strictly increasing transform
    M = importance_correlation_matrix(imp)
    assert np.isclose(M.loc['x', 'y'], 1.0)
