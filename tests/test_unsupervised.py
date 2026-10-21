"""Tests for label-free structure analysis (validated/unsupervised.py)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from validation.unsupervised import (pca, nested_structure, nested_variance_components,
                                    outlier_statistics, nearest_neighbour_consistency,
                                    tucker_congruence, colony_means)


def nested_design(s_cond, s_col, s_tech, rng, n_cond=5, n_col=3, n_tech=3):
    c = np.repeat(np.arange(n_cond), n_col * n_tech)
    g = np.repeat(np.arange(n_cond * n_col), n_tech)
    v = (rng.normal(0, s_cond, n_cond)[c] + rng.normal(0, s_col, n_cond * n_col)[g]
         + rng.normal(0, s_tech, len(c)))
    return v, c.astype(str), g.astype(str)


def test_variance_components_recover_planted_values():
    rng = np.random.default_rng(0)
    est = np.array([[nested_variance_components(*nested_design(2.0, 1.0, 0.5, rng))[k]
                     for k in ('s2_cond', 's2_col', 's2_tech')] for _ in range(3000)])
    # method-of-moments estimators are unbiased before truncation at 0; truncation
    # adds small positive bias for the smallest components
    np.testing.assert_allclose(est.mean(0), [4.0, 1.0, 0.25], rtol=0.08)


def test_balanced_flag_and_icc_range():
    rng = np.random.default_rng(1)
    v, c, g = nested_design(1, 1, 1, rng)
    r = nested_variance_components(v, c, g)
    assert r['balanced'] and 0 <= r['icc_colony'] <= 1 and r['df'] == (4, 10, 30)
    r2 = nested_variance_components(v[:-1], c[:-1], g[:-1])
    assert not r2['balanced']


def test_pca_basic_properties():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(45, 30)) @ np.diag(np.linspace(3, 0.1, 30))
    p = pca(X, 5)
    np.testing.assert_allclose(np.linalg.norm(p['loadings'], axis=0), 1)
    assert np.all(np.diff(p['evr']) <= 1e-12) and p['evr'].sum() <= 1
    assert abs(np.corrcoef(p['scores'][:, 0], p['scores'][:, 1])[0, 1]) < 1e-8


def test_colony_structure_shared_by_D45_and_D15():
    rng = np.random.default_rng(3)
    p = 40
    load = np.zeros(p); load[:5] = 1
    _, c, g = nested_design(1, 1, 1, rng)
    col_eff = rng.normal(0, 3, 15)[g.astype(int)]                     # strong colony axis
    X = np.outer(col_eff, load) + rng.normal(0, 0.3, (45, p))
    tab, p45, p15, _ = nested_structure(X, g, k=3)
    assert tab.loc[0, 'tucker_congruence'] > 0.99 and tab.loc[0, 'colony_score_r'] > 0.99
    assert tucker_congruence(load, p45['loadings'][:, 0]) > 0.99


def test_outlier_limits_have_roughly_nominal_coverage_on_iid_data():
    rng = np.random.default_rng(4)
    hits_t2 = hits_q = 0; n = 0
    for _ in range(200):
        X = rng.normal(size=(45, 20)) * np.linspace(2, 0.5, 20)
        o = outlier_statistics(X, k=3)
        hits_t2 += (o['T2'] > o['T2_lim95']).sum(); hits_q += (o['Q'] > o['Q_lim95']).sum(); n += 45
    assert hits_t2 / n < 0.08 and hits_q / n < 0.12


def test_nearest_neighbour_detects_colony_clustering():
    rng = np.random.default_rng(5)
    _, c, g = nested_design(1, 1, 1, rng)
    X = rng.normal(0, 5, (15, 10))[g.astype(int)] + rng.normal(0, 0.2, (45, 10))
    r = nearest_neighbour_consistency(X, c, g)
    assert r['nn_same_colony'] == 1.0 and abs(r['nn_same_colony_chance'] - 2 / 44) < 1e-12
    Xm, ids = colony_means(X, g)
    assert Xm.shape == (15, 10) and len(ids) == 15
