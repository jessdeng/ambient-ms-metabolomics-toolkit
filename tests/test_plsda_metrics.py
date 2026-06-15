"""
Tests for B3: PLS-DA reporting polish — multi-component VIP and the combined
R2Y + Q2 evaluation with permutation p-values for both metrics.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from standard.pipeline import (  # noqa: E402
    compute_vip, compute_vip_1comp, compute_plsda_r2y, evaluate_plsda_q2,
)


def _dataset(seed=0, n_per=9):
    """3-class separable-ish data with replicate groups (3 colonies x 3 reps)."""
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    centers = {'A': 0.0, 'B': 3.0, 'C': 6.0}
    for cls, c in centers.items():
        for colony in range(3):
            shift = rng.normal(0, 0.3)
            for _t in range(3):
                row = [c + shift + rng.normal(0, 0.2)] + list(rng.normal(0, 1, size=8))
                X.append(row); y.append(cls); groups.append(f'{cls}::col{colony}')
    return np.array(X), np.array(y), np.array(groups)


def test_compute_vip_1comp_matches_general_one_component():
    X, y, _ = _dataset()
    assert np.allclose(compute_vip_1comp(X, y), compute_vip(X, y, n_components=1))


def test_compute_vip_multicomponent_runs_and_shapes():
    X, y, _ = _dataset()
    v1 = compute_vip(X, y, n_components=1)
    v3 = compute_vip(X, y, n_components=3)
    assert v1.shape == v3.shape == (X.shape[1],)
    # Aggregating more components generally changes the ranking values.
    assert not np.allclose(v1, v3)


def test_compute_vip_clamps_excess_components():
    X, y, _ = _dataset()
    # Asking for absurdly many components must clamp, not raise.
    v = compute_vip(X, y, n_components=999)
    assert v.shape == (X.shape[1],)
    assert np.all(np.isfinite(v))


def test_r2y_is_fraction_and_high_for_separable_data():
    X, y, _ = _dataset()
    r2y = compute_plsda_r2y(X, y, n_components=4)
    assert -1.0 <= r2y <= 1.0
    assert r2y > 0.3            # separable signal -> appreciable apparent fit


def test_evaluate_returns_both_metrics_with_pvalues():
    X, y, groups = _dataset()
    res = evaluate_plsda_q2(X, y, n_components=4, groups=groups, n_splits=3,
                            n_perm=30, random_state=42)
    for key in ('r2y', 'q2', 'r2y_null', 'q2_null', 'r2y_p', 'q2_p'):
        assert key in res
    assert res['r2y_null'].shape == (30,) and res['q2_null'].shape == (30,)
    for p in (res['r2y_p'], res['q2_p']):
        assert 0.0 < p <= 1.0
    # Apparent fit should not be below cross-validated predictivity.
    assert res['r2y'] >= res['q2'] - 1e-6


def test_evaluate_signal_beats_null():
    # Real labels should give a Q2 above the bulk of the permuted null.
    X, y, groups = _dataset()
    res = evaluate_plsda_q2(X, y, n_components=4, groups=groups, n_splits=3,
                            n_perm=50, random_state=0)
    assert res['q2'] > np.nanmedian(res['q2_null'])
