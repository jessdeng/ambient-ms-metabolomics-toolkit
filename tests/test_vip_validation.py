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

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from standard.pipeline import (compute_vip, compute_vip_1comp, VALIDATED_PLS_TOL,
                               VALIDATED_PLS_MAX_ITER)
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
