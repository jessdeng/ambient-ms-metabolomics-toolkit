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


def test_component1_vip_closed_form_matches_nipals_and_r(synthetic):
    # compute_vip_1comp is VIP_j = sqrt(p) |w1_j|, w1 = leading left singular
    # vector of Xc^T Yc. It must equal the iterative solution at the validated
    # settings and the two R implementations.
    X, y, exp = synthetic
    v = compute_vip_1comp(X, y)
    np.testing.assert_allclose(v, compute_vip(X, y, n_components=1, **TIGHT), atol=1e-6)
    np.testing.assert_allclose(v, exp['ropls_A1'].values, atol=1e-6)
    np.testing.assert_allclose(v, exp['mixomics_A1'].values, atol=1e-6)
    assert np.isclose(np.mean(v ** 2), 1.0)
    np.testing.assert_allclose(compute_vip_1comp(X[::-1], y[::-1]), v, atol=1e-12)   # row order
    with pytest.raises(ValueError, match='two classes'):
        compute_vip_1comp(X, np.zeros(len(y)))


@pytest.mark.parametrize('A', [1, 2, 3])
def test_vip_mean_square_is_one(synthetic, A):
    X, y, _ = synthetic
    assert np.isclose(np.mean(compute_vip(X, y, n_components=A) ** 2), 1.0)
