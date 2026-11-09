"""PCA and the condition-plate / colony / technical variance partition."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.pca import pca, nested_variance_components, run_pca


def _design(rng, effect=2.0):
    y = np.repeat(list('ABCDE'), 9); g = np.repeat([f'k{i}' for i in range(15)], 3)
    X = rng.normal(size=(45, 40))
    X[:, 0] += np.repeat(np.arange(5) * effect, 9)
    X += np.repeat(rng.normal(scale=0.3, size=(15, 40)), 3, axis=0)
    return X, y, g


def test_pca_matches_svd_and_is_sign_stable():
    X, _, _ = _design(np.random.default_rng(0))
    r = pca(X, k=3)
    ev = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
    assert np.allclose(r['evr'], ev[:3] / ev.sum())
    assert np.all(r['loadings'][np.abs(r['loadings']).argmax(0), np.arange(3)] > 0)


def test_variance_partition_recovers_condition_dominance():
    rng = np.random.default_rng(1)
    y = np.repeat(list('ABCDE'), 9); g = np.repeat([f'k{i}' for i in range(15)], 3)
    vc = nested_variance_components(np.repeat(np.arange(5) * 3.0, 9) + rng.normal(scale=0.2, size=45), y, g)
    assert vc['frac_cond'] > 0.9 and vc['balanced']
    assert abs(vc['frac_cond'] + vc['frac_col'] + vc['frac_tech'] - 1) < 1e-9


def test_run_pca_writes_outputs(tmp_path):
    X, y, g = _design(np.random.default_rng(2))
    ev = run_pca(X, y, g, np.arange(40) + 100.0, [f's{i}' for i in range(45)], str(tmp_path), 't', 't', 4)
    for f in ('pca_explained_variance_t.csv', 'pca_scores_t.csv', 'pca_top_loadings_t.csv', 'pca_t.png'):
        assert (tmp_path / f).exists()
    assert ev.loc[0, 'frac_condition_plate'] > 0.5
