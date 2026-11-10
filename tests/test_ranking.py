"""Colony-level prioritisation: aggregation, screen, voters, consistency, partition."""
import os, sys
import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.ranking import (colony_means, univariate_screen, colony_consistency, variance_partition,
                            technical_cv, rank_features, concordance, benjamini_hochberg, top_set,
                            selected_by, VOTERS, METHODS)
from shared.pca import nested_variance_components

COND = np.repeat(list('ABCDE'), 9)
COL = np.array([f'{c}::{k}' for c in 'ABCDE' for k in range(3) for _ in range(3)])


def _data(seed=0, p=300, effect=np.log10(4)):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.2, (45, p))
    X += np.repeat(rng.normal(0, 0.1, (15, p)), 3, axis=0)        # colony effect
    X[COND == 'E', :5] += effect                                   # higher in E, bins 0-4
    return X


def test_colony_means_and_screen():
    X = _data()
    Xc, ids, cc = colony_means(X, COL, COND)
    assert Xc.shape == (15, 300) and ids[0] == 'A::0' and cc.tolist() == list(np.repeat(list('ABCDE'), 3))
    np.testing.assert_allclose(Xc[0], X[:3].mean(0))
    s = univariate_screen(Xc, cc)
    ref = stats.f_oneway(*[Xc[cc == c] for c in 'ABCDE'], axis=0).statistic
    np.testing.assert_allclose(s.screen_F, ref)
    assert (s.screen_eta2.between(0, 1)).all() and (s.top_condition[:5] == 'E').all()
    assert s.log2_fc_top_vs_bottom[:5].mean() == pytest.approx(2.0, abs=0.35)


def test_voters_find_the_planted_bins_and_sensitivity_does_not_vote():
    X = _data(1, effect=1.0)                                       # 10x
    Xc, _, cc = colony_means(X, COL, COND)
    Xs = (Xc - Xc.mean(0)) / Xc.std(0, ddof=1)
    table, info = rank_features(Xs, Xc, cc, top_n=10, min_rankers=2, rf_seeds=4, seed=0)
    assert table.loc[:4, 'candidate'].all()
    assert set(VOTERS) == {'univariate', 'vip'}
    assert table['n_rankers'].max() == 2                           # ridge / rf never vote
    assert 0 <= info['rf_seed_agreement'] <= 1


def test_method_comparison_counts_votes_and_families():
    X = _data(1, effect=1.0)
    Xc, _, cc = colony_means(X, COL, COND)
    Xs = (Xc - Xc.mean(0)) / Xc.std(0, ddof=1)
    table, info = rank_features(Xs, Xc, cc, top_n=10, rf_seeds=2, seed=0)
    assert list(info['scores']) == ['univariate', 'vip', 'ridge', 'lr', 'svm', 'rf', 'gb']
    cols = [f'in_{m}_top' for m in METHODS]
    assert (table['n_methods'] == table[cols].sum(axis=1)).all() and table['n_methods'].max() <= 6
    assert all(table[c].sum() == 10 for c in cols)                 # top-N of every method
    assert (table['consensus'] == (table['agg_n_families_topk'] >= 2)).all()
    assert table.loc[:4, 'n_methods'].min() >= 1 and table.loc[:4, 'n_methods'].mean() >= 2
    assert (table.loc[:4, 'top_condition_mean'] == 'E').all()
    assert table.loc[:4, 'mean_margin'].min() > 5                  # planted 10x over the rest
    strict, _ = rank_features(Xs, Xc, cc, top_n=10, rf_seeds=2, seed=0,
                              consensus_rule='worst_family_rank', max_worst_rank=10)
    assert (strict['consensus'] == (strict['agg_worst_family_rank'] <= 10)).all()
    by = selected_by(table)
    assert set(by) <= {'', 'screen + VIP', 'method consensus', 'screen + VIP and method consensus'}
    assert ((by != '') == (table['candidate'] | table['consensus'])).all()


def test_colony_consistency():
    cc = np.array(['A', 'A', 'B', 'B', 'C', 'C'])
    Xc = np.array([[3.0, 1.0, 1.0],      # bin 0: A above all; bin 1: B below all; bin 2: overlap
                   [2.5, 1.2, 2.0],
                   [1.0, 0.2, 2.5],
                   [0.5, 0.4, 0.9],
                   [1.2, 1.1, 1.5],
                   [0.8, 1.3, 1.1]])
    out = colony_consistency(Xc, cc, np.array(['A', 'A', 'B']), np.array(['B', 'B', 'B']))
    assert out.colony_gap_log10.tolist() == pytest.approx([1.3, 0.6, -1.1])
    assert out.colony_pattern.tolist() == ['elevated in A', 'depleted in B', 'none']
    assert out.colonies_consistent.tolist() == [True, True, False]


def test_variance_partition_matches_scalar_version():
    X = _data(2, p=20)
    vp = variance_partition(X, COND, COL)
    for j in (0, 7, 19):
        ref = nested_variance_components(X[:, j], COND, COL)
        assert vp.loc[j, 'frac_condition_plate'] == pytest.approx(ref['frac_cond'])
        assert vp.loc[j, 'frac_technical'] == pytest.approx(ref['frac_tech'])
    assert vp.loc[0, 'frac_condition_plate'] > 0.5


def test_technical_cv():
    X = np.array([[1.0], [3.0], [10.0], [10.0]])
    assert technical_cv(X, np.array(['a', 'a', 'b', 'b']))[0] == pytest.approx(100 * np.sqrt(2) / 2 / 2, abs=0.01)


def test_concordance_and_helpers():
    s = {'a': np.arange(100.0), 'b': np.arange(100.0), 'c': -np.arange(100.0)}
    t = concordance(s, 10).set_index(['ranker_a', 'ranker_b'])
    assert t.loc[('a', 'b'), 'jaccard'] == 1.0 and t.loc[('a', 'c'), 'jaccard'] == 0.0
    assert 0 < t.chance.iloc[0] < 0.1
    assert top_set(np.array([1, 3, 3, 0]), 1) == {1}               # tie -> lower index
    p = np.array([0.01, 0.04, 0.03, 0.5])
    np.testing.assert_allclose(benjamini_hochberg(p), [0.04, 0.16 / 3, 0.16 / 3, 0.5])
