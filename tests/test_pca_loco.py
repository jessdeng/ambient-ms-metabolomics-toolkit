"""Tests for the v2 core additions: PCA step and leave-one-colony-out evaluation."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from shared.pca import pca, nested_variance_components, run_pca


def _design(rng, effect=2.0):
    y = np.repeat(list('ABCDE'), 9); g = np.repeat([f'k{i}' for i in range(15)], 3)
    X = rng.normal(size=(45, 40))
    X[:, 0] += np.repeat(np.arange(5) * effect, 9)        # condition structure in feature 0
    X += np.repeat(rng.normal(scale=0.3, size=(15, 40)), 3, axis=0)   # colony effect
    return X, y, g


def test_pca_matches_svd_and_is_sign_stable():
    rng = np.random.default_rng(0); X, _, _ = _design(rng)
    r = pca(X, k=3)
    assert np.allclose(r['evr'].sum() <= 1, True)
    ev = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
    assert np.allclose(r['evr'], ev[:3] / ev.sum())
    assert np.all(r['loadings'][np.abs(r['loadings']).argmax(0), np.arange(3)] > 0)


def test_variance_partition_recovers_condition_dominance():
    rng = np.random.default_rng(1)
    y = np.repeat(list('ABCDE'), 9); g = np.repeat([f'k{i}' for i in range(15)], 3)
    v = np.repeat(np.arange(5) * 3.0, 9) + rng.normal(scale=0.2, size=45)
    vc = nested_variance_components(v, y, g)
    assert vc['frac_cond'] > 0.9 and vc['balanced']
    assert abs(vc['frac_cond'] + vc['frac_col'] + vc['frac_tech'] - 1) < 1e-9


def test_run_pca_writes_outputs(tmp_path):
    rng = np.random.default_rng(2); X, y, g = _design(rng)
    ev = run_pca(X, y, g, np.arange(40) + 100.0, [f's{i}' for i in range(45)], str(tmp_path), 't', 't', 4)
    for f in ('pca_explained_variance_t.csv', 'pca_scores_t.csv', 'pca_top_loadings_t.csv', 'pca_t.png'):
        assert (tmp_path / f).exists()
    assert ev.loc[0, 'frac_condition_plate'] > 0.5


def test_loco_holds_out_whole_colonies():
    from shared.loco_cv import loco_table
    from shared.classifier_comparison_standard import make_preprocessor
    rng = np.random.default_rng(3)
    X = np.abs(rng.normal(size=(45, 60))) * 100
    y = np.repeat(list('ABCDE'), 9); g = np.repeat([f'k{i}' for i in range(15)], 3)
    X[:, :5] += np.repeat(np.arange(5) * 200.0, 9)[:, None]
    prep = make_preprocessor(normalization='tic', log_transform='log10', scaling='autoscale',
                             variance_percentile=0, abundance_percentile=0, prevalence_threshold=0.0)
    tab, conf = loco_table(X, y, g, prep, methods=['ridge', 'lda'])
    assert set(tab.classifier) == {'Ridge', 'LDA'} and (tab.n_colonies == 15).all()
    assert all(c.values.sum() == 45 for c in conf.values())
