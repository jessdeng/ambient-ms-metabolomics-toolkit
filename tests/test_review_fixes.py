"""Regression tests for the code-review fixes: uncentred fold-change input,
class-stratified colony resampling, and family-aware rank aggregation."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from shared.feature_stats import stratified_colony_resample, univariate_feature_stats
from shared.rank_aggregation import FAMILY, aggregate_ranks, concordance_table, rbo, to_ranks
from standard.preprocessing import preprocess

K, C, T = 5, 3, 3
Y = np.repeat([f'c{k}' for k in range(K)], C * T)
G = np.array([f'c{k}::{c}' for k in range(K) for c in range(C) for _ in range(T)])


# -- fold change must come from the uncentred transformed stage ---------------------
@pytest.mark.parametrize('transform', ['log10', 'glog', 'sqrt'])
def test_fold_change_uses_uncentred_stage(transform):
    rng = np.random.default_rng(0)
    X = rng.lognormal(3, 0.3, (len(Y), 30)) * 1000
    X[Y == 'c4', 0] *= 4.0
    Xt = preprocess(X, 'tic', transform, 'none', return_stages=True)['transformed']
    s = univariate_feature_stats(Xt, Y, log_transform=transform, groups=G)
    assert s['fold_change'][0] > 2.5                       # planted ~4x, positive
    assert np.all(s['fold_change'][1:] > 0)                # no sign flips from centring


def test_scaling_none_is_centred_so_it_must_not_feed_fold_change():
    X = np.random.default_rng(1).lognormal(3, 0.3, (len(Y), 10)) * 1000
    Xc = preprocess(X, 'tic', 'glog', 'none')
    assert np.allclose(Xc.mean(0), 0.0, atol=1e-9)         # documents the pitfall


# -- stratified colony resampling ----------------------------------------------------
def test_resample_keeps_every_class_and_colony_unit():
    rng = np.random.default_rng(0)
    for _ in range(500):
        idx = stratified_colony_resample(Y, G, rng, replace=True)
        assert set(Y[idx]) == set(Y)
        assert len(idx) == len(Y)                          # class sizes preserved
        for c in set(Y):
            assert len(set(G[idx][Y[idx] == c])) >= 2
        # whole colonies only: every drawn colony contributes all its technical reps
        _, counts = np.unique(G[idx], return_counts=True)
        assert set(counts) <= {T, 2 * T, 3 * T}


def test_resample_subsample_has_no_duplicates():
    idx = stratified_colony_resample(Y, G, np.random.default_rng(3), replace=False)
    assert len(idx) == len(set(idx)) == K * 2 * T
    assert len(set(G[idx])) == K * 2


def test_resample_rejects_thin_class():
    with pytest.raises(ValueError):
        stratified_colony_resample(np.array(['a', 'a', 'b']), np.array(['a::1', 'a::1', 'b::1']),
                                   np.random.default_rng(0))


# -- family-aware rank aggregation ----------------------------------------------------
def test_to_ranks_ties_and_nan():
    r = to_ranks([0.0, 3.0, 0.0, np.nan, 1.0])
    assert r[1] == 1 and r[4] == 2 and r[0] == r[2] == 3.5 and r[3] == 5


def test_linear_block_counts_as_one_family():
    p = 200
    rng = np.random.default_rng(0)
    base = rng.random(p)
    imps = {m: base + 1e-6 * rng.random(p) for m in ('ridge', 'lr', 'svm')}   # identical
    imps['rf'] = rng.random(p)
    imps['gb'] = rng.random(p)
    imps['vip'] = rng.random(p)
    top_linear = np.argsort(-base)[:5]
    # linear-only top features are NOT consensus: another family ranks them arbitrarily
    agg = aggregate_ranks(imps, top_k=10)
    assert (agg.loc[top_linear, 'agg_n_families_topk'] <= 2).all()
    assert (agg.agg_worst_family_rank.loc[top_linear] > 10).mean() >= 0.6


def test_feature_good_in_all_families_wins():
    p = 300
    rng = np.random.default_rng(1)
    imps = {m: rng.random(p) for m in FAMILY}
    for m in imps:
        imps[m][42] = 10.0
    agg = aggregate_ranks(imps, top_k=5)
    assert agg.agg_worst_family_rank.idxmin() == 42
    assert agg.loc[42, 'agg_n_families_topk'] == 3
    assert agg.loc[42, 'agg_rra_score'] == agg.agg_rra_score.min()


def test_aggregate_respects_enabled_subset_and_validates_lengths():
    rng = np.random.default_rng(2)
    agg = aggregate_ranks({'rf': rng.random(50), 'ridge': rng.random(50)})
    assert {'agg_rank_tree', 'agg_rank_linear'} <= set(agg.columns)
    assert 'agg_rank_pls' not in agg.columns
    with pytest.raises(ValueError):
        aggregate_ranks({'rf': np.ones(5), 'lr': np.ones(6)})


def test_concordance_identical_vs_random():
    rng = np.random.default_rng(3)
    v = rng.random(1000)
    t = concordance_table({'a': v, 'b': v.copy(), 'c': rng.random(1000)}, ks=(50,))
    ab = t[(t.method_a == 'a') & (t.method_b == 'b')].iloc[0]
    ac = t[(t.method_a == 'a') & (t.method_b == 'c')].iloc[0]
    assert ab['jaccard@50'] == 1.0 and ab['rbo@100'] > 0.99 * (1 - 0.9 ** 100)
    assert ac['jaccard@50'] < 0.1 and ac['chance@50'] < 0.05
    assert rbo([1, 2, 3], [1, 2, 3]) == pytest.approx(1 - 0.9 ** 3)


# -- follow-up: neutral screen columns, consensus rule, subsampling, A*, guards ---------
from shared import feature_stats as fs
from shared.rank_aggregation import consensus_mask, consensus_label
from standard.pipeline import parsimonious_n_components, grouped_q2_profile
from shared.classifier_comparison_standard import (guard_group_count, small_sample_guards,
                                                   feature_importance_analysis)


def test_screen_statistic_matches_scipy_anova():
    from scipy import stats as sps
    rng = np.random.default_rng(0)
    Xl = rng.normal(size=(len(Y), 12))
    Xl[Y == 'c0', 0] += 3
    s = univariate_feature_stats(Xl, Y, groups=G, aggregate_within_group=False)
    ref = sps.f_oneway(*[Xl[Y == c] for c in np.unique(Y)], axis=0)
    np.testing.assert_allclose(s['statistic'], ref.statistic)


def test_bootstrap_uses_subsampling_without_replacement(monkeypatch):
    seen = []
    real = fs.stratified_colony_resample

    def spy(*a, **k):
        seen.append(k.get('replace'))
        return real(*a, **k)

    monkeypatch.setattr(fs, 'stratified_colony_resample', spy)
    X = np.abs(np.random.default_rng(0).normal(size=(len(Y), 20))) + 1
    fs._one_bootstrap(np.random.SeedSequence(1), X, Y, G, np.unique(G),
                      lambda X, **k: X, lambda X, y: np.ones(X.shape[1]),
                      'none', 'none', 'none', 5, 2, 20, {'ridge'})
    assert seen == [False]


def test_parsimonious_a_star_and_boundary_warning(caplog):
    q2 = np.array([0.226, 0.303, 0.463, 0.404, 0.388, 0.412, 0.463, 0.493])   # shipped F11A
    with caplog.at_level('WARNING', logger='ms_toolkit'):
        assert parsimonious_n_components(q2, tol=0.05) == 3
    assert 'boundary' in caplog.text
    assert parsimonious_n_components(q2, tol=0.0) == 8                         # argmax
    assert parsimonious_n_components([np.nan, np.nan]) == 1


def test_grouped_q2_profile_reports_both_choices():
    from standard.pipeline import grouped_q2_profile
    from shared.classifier_comparison_standard import make_preprocessor
    rng = np.random.default_rng(0)
    Xb = rng.lognormal(3, .5, (len(Y), 60)) * 1000
    Xb[:, :5] *= np.where(np.isin(Y, ['c0', 'c1']), 3, 1)[:, None]
    r = grouped_q2_profile(Xb, Y, G, make_preprocessor('tic', 'log10', 'autoscale', 0, 0, 0.0),
                           max_components=4, n_splits=3, n_repeats=1)
    assert 1 <= r['parsimonious_n_components'] <= r['best_n_components'] <= 4


def test_guard_key_is_per_class_by_default():
    assert guard_group_count(Y, G) == C                        # 3 colonies in the smallest class
    assert guard_group_count(Y, G, 'total') == K * C           # legacy: 15
    assert any(small_sample_guards(guard_group_count(Y, G)).values())      # guards on
    g3 = small_sample_guards(3)
    assert g3['rf']['max_depth'] >= 3 and g3['gb']['max_depth'] >= 2       # never a stump
    assert all(small_sample_guards(n)['rf']['max_depth'] >= 3 for n in (2, 3, 4, 5))
    assert not any(small_sample_guards(guard_group_count(Y, G, 'total')).values())   # legacy: off
    with pytest.raises(ValueError):
        guard_group_count(Y, G, 'nonsense')


def test_consensus_mask_rules():
    import pandas as pd
    df = pd.DataFrame({'agg_worst_family_rank': [10, 80, 40], 'agg_n_families_topk': [3, 1, 2]})
    assert consensus_mask(df, 'worst_family_rank', 50).tolist() == [True, False, True]
    assert consensus_mask(df, 'n_families_topk', min_families=2).tolist() == [True, False, True]
    assert 'Cross-family consensus' in consensus_label()
    with pytest.raises(KeyError):
        consensus_mask(df[['agg_n_families_topk']], 'worst_family_rank')
    with pytest.raises(ValueError):
        consensus_mask(df, 'votes')


def test_feature_tiers_high_mask_overrides_vote_rule():
    import pandas as pd
    from shared.visualization import _feature_tiers
    df = pd.DataFrame({'mz': [100., 200., 300.], 'n_methods': [6, 2, 1]})
    allf, high = _feature_tiers(df, 4, high_mask=[False, False, True])
    assert allf.tolist() == [100., 200., 300.] and high.tolist() == [300.]


def test_feature_importance_outputs_neutral_columns_and_consensus_order(tmp_path):
    rng = np.random.default_rng(5)
    p = 120
    Xs = rng.normal(size=(len(Y), p))
    for k in range(K):                                         # class-specific signal in a few bins
        Xs[Y == f'c{k}', k * 3:(k * 3) + 3] += 3
    Xl = Xs - Xs.min()
    df, counts = feature_importance_analysis(
        Xs, Y, np.arange(100., 100. + p), 't', str(tmp_path), top_n=10, X_norm=Xl,
        log_transform='log10', groups=G, n_biological=guard_group_count(Y, G))
    cols = set(df.columns)
    assert {'screen_F', 'screen_p', 'screen_q', 'screen_test'} <= cols
    assert not ({'p_value', 'q_value_BH', 'univariate_test'} & cols)
    assert {'agg_worst_family_rank', 'agg_n_families_topk', 'agg_rra_score'} <= cols
    assert df['agg_worst_family_rank'].is_monotonic_increasing       # not sorted by screen_q
    ranks = (tmp_path / 'feature_rankings_t.csv')
    assert ranks.exists()
