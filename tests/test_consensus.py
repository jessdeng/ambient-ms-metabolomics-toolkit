"""Tests for consensus summaries and the shuffled-label benchmark helpers."""
import collections, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from validation.consensus import (vote_table, full_data_statistics, resampled_statistics, shuffle_colony_labels,
                                 exceedance, DESIGN9, SIX)


def test_design9_balanced_per_condition():
    assert len(set(DESIGN9)) == 9
    for j in range(5):
        assert collections.Counter(r[j] for r in DESIGN9) == {0: 3, 1: 3, 2: 3}


def test_vote_and_family_counts():
    p = 20; base = np.arange(p, dtype=float)     # distinct values; features 0/1 lowest unless boosted
    imps = {m: base.copy() for m in SIX}
    for m in ('ridge', 'lr', 'svm', 'rf'):
        imps[m][0] = 100           # feature 0: 4 votes, families L and T
    imps['lda'][1] = 100; imps['gb'][1] = 100; imps['ridge'][1] = 50   # feature 1: 3 votes, 3 families
    votes, fam, _ = vote_table(imps, n=2)
    assert votes[0] == 4 and fam[0] == 2 and votes[1] == 3 and fam[1] == 3
    s = full_data_statistics(imps, n=2)
    assert s['n_votes_ge4'] >= 1 and s['n_three_family'] >= 1


def test_resampled_statistics_identical_subsets_are_fully_stable():
    rng = np.random.default_rng(0); v = rng.normal(size=50)
    sub = [{m: v.copy() for m in SIX} for _ in range(4)]
    out, pi = resampled_statistics(sub, n=5)
    assert all(out[f'pairwise_jaccard_{m}'] == 1 for m in SIX)
    assert out['n_stable_in_ge4_methods'] == 5 and out['n_stable_three_family'] == 5


def test_shuffle_keeps_colonies_and_label_counts():
    y = np.repeat(list('ABCDE'), 9); g = np.repeat([f'k{i}' for i in range(15)], 3)
    ys = shuffle_colony_labels(y, g, np.random.default_rng(1))
    assert collections.Counter(ys) == collections.Counter(y)
    assert all(len(set(ys[g == c])) == 1 for c in set(g))


def test_exceedance_definition():
    e = exceedance(5, [1, 2, 5, 7])
    assert e['exceedance'] == 0.5 and e['quantile'] == (2 + 0.5) / 4
