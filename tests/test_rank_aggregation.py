"""Family-aware rank aggregation and the cross-family consensus rule."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from shared.rank_aggregation import (FAMILY, aggregate_ranks, concordance_table, consensus_label,
                                     consensus_mask, rbo)


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


def test_consensus_rules():
    df = pd.DataFrame({'agg_worst_family_rank': [3, 60, 20], 'agg_n_families_topk': [3, 1, 2]})
    assert consensus_mask(df, 'n_families_topk', min_families=2).tolist() == [True, False, True]
    assert consensus_mask(df, 'worst_family_rank', max_worst_rank=10).tolist() == [True, False, False]
    assert '2 families' in consensus_label('n_families_topk', min_families=2)
    with pytest.raises(ValueError):
        consensus_mask(df, 'something else')
    with pytest.raises(KeyError):
        consensus_mask(df.drop(columns='agg_n_families_topk'), 'n_families_topk')
