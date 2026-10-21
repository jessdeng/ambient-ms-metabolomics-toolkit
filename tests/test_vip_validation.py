"""VIP validation: component-1 reference and multi-component PLS2 VIP.

Expected values in tests/fixtures/vip_synthetic_expected.csv were produced by
ropls 1.42.0 (opls, scaleC='center', orthoI=0) and mixOmics 6.34.0
(plsda(scale=FALSE) + vip) on tests/fixtures/vip_synthetic_X.csv; the two R
implementations agree to < 3e-7.
"""
import os, sys
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from standard.pipeline import (compute_vip, compute_vip_1comp, grouped_q2_profile,
                               VALIDATED_PLS_TOL, VALIDATED_PLS_MAX_ITER)
TIGHT = dict(tol=VALIDATED_PLS_TOL, max_iter=VALIDATED_PLS_MAX_ITER)

FIX = os.path.join(HERE, 'fixtures')


@pytest.fixture(scope='module')
def synthetic():
    d = pd.read_csv(os.path.join(FIX, 'vip_synthetic_X.csv'))
    exp = pd.read_csv(os.path.join(FIX, 'vip_synthetic_expected.csv'))
    return d.drop(columns='cls').values, d['cls'].values, exp


@pytest.mark.parametrize('A', [1, 2, 3])
def test_vip_matches_ropls_and_mixomics(synthetic, A):
    X, y, exp = synthetic
    v = compute_vip(X, y, n_components=A, **TIGHT)
    np.testing.assert_allclose(v, exp[f'ropls_A{A}'].values, atol=1e-5)
    np.testing.assert_allclose(v, exp[f'mixomics_A{A}'].values, atol=1e-5)
    # Default solver settings (all original results) agree to ~1e-3 and give the
    # same top ranking.
    v0 = compute_vip(X, y, n_components=A)
    np.testing.assert_allclose(v0, exp[f'ropls_A{A}'].values, atol=2e-3)
    assert set(np.argsort(-v0)[:10]) == set(np.argsort(-exp[f'ropls_A{A}'].values)[:10])


def test_component1_vip_is_normalised_first_weight(synthetic):
    # Reference analysis: VIP_j = sqrt(p) |w1_j| / ||w1||, w1 = dominant left
    # singular vector of Xc^T Yc (independent of any Y-loading weighting).
    X, y, _ = synthetic
    Y = pd.get_dummies(y).values.astype(float)
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    w1 = np.linalg.svd(Xc.T @ Yc, full_matrices=False)[0][:, 0]
    expected = np.sqrt(X.shape[1]) * np.abs(w1) / np.linalg.norm(w1)
    np.testing.assert_allclose(compute_vip_1comp(X, y, **TIGHT), expected, atol=1e-6)


@pytest.mark.parametrize('A', [1, 2, 3])
def test_vip_mean_square_is_one(synthetic, A):
    X, y, _ = synthetic
    assert np.isclose(np.mean(compute_vip(X, y, n_components=A) ** 2), 1.0)


class _SpyScaler(BaseEstimator, TransformerMixin):
    """Records the row indices (via a sentinel column) seen during fit."""
    seen = []
    def fit(self, X, y=None):
        _SpyScaler.seen.append(set(np.round(X[:, -1]).astype(int).tolist()))
        self.mu_ = X[:, :-1].mean(0); return self
    def transform(self, X):
        return X[:, :-1] - self.mu_


def test_grouped_q2_profile_refits_preprocessing_on_training_colonies_only(synthetic):
    X, y, _ = synthetic
    groups = np.repeat(np.arange(16), 2)                    # 2 "technical" rows per colony
    Xs = np.column_stack([X, np.arange(len(y))])            # sentinel: row index
    _SpyScaler.seen = []
    res = grouped_q2_profile(Xs, y, groups, [('spy', _SpyScaler())], max_components=3,
                             n_splits=2, n_repeats=1, random_state=0)
    assert len(_SpyScaler.seen) == 2
    for seen in _SpyScaler.seen:                            # no colony split across fit/test
        cols = {groups[i] for i in seen}
        assert all(set(np.where(groups == g)[0]) <= seen for g in cols)
        assert len(seen) < len(y)
    assert res['q2'].shape == (3,) and 1 <= res['best_n_components'] <= 3


def test_prefix_component_prediction_equals_refit(synthetic):
    X, y, _ = synthetic
    Y = pd.get_dummies(y).values.astype(float)
    big = PLSRegression(3, scale=False).fit(X[:24], Y[:24])
    for a in (1, 2, 3):
        ref = PLSRegression(a, scale=False).fit(X[:24], Y[:24]).predict(X[24:])
        pre = (X[24:] - X[:24].mean(0)) @ big.x_rotations_[:, :a] @ big.y_loadings_[:, :a].T + Y[:24].mean(0)
        np.testing.assert_allclose(pre, ref, atol=1e-10)


def test_grouped_q2_accepts_leave_one_group_out_splitter():
    import numpy as np
    from sklearn.model_selection import LeaveOneGroupOut
    from standard.pipeline import grouped_q2_profile
    from shared.classifier_comparison import make_preprocessor
    rng = np.random.default_rng(0)
    y = np.repeat(['a', 'b', 'c'], 6); g = np.repeat([f'k{i}' for i in range(9)], 2)
    X = rng.lognormal(size=(18, 40)); X[y == 'a', :3] *= 4
    steps = make_preprocessor(normalization='tic', log_transform='log10', scaling='autoscale',
                              variance_percentile=0, abundance_percentile=0, prevalence_threshold=0,
                              snr_floor_enabled=False)
    r = grouped_q2_profile(X, y, g, steps, max_components=3, splitter=LeaveOneGroupOut())
    assert r['q2_repeats'].shape == (1, 3) and 1 <= r['best_n_components'] <= 3
