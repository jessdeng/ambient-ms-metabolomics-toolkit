"""Tests for colony-removal stability (validated/stability.py)."""
import collections, itertools, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from validation.stability import subset_design, subset_indices, selection_frequency


def design_labels():
    y = np.repeat(['A', 'B', 'C', 'D', 'E'], 9)
    g = np.array([f'{c}::{k}' for c in 'ABCDE' for k in ('k1', 'k2', 'k3') for _ in range(3)])
    return y, g


def test_full_design_has_243_unique_subsets():
    d = subset_design(design='full'); assert len(d) == 243 and len(set(d)) == 243


def test_oa27_is_strength_two_and_a_subset_of_full():
    oa = subset_design(design='oa27'); full = set(subset_design(design='full'))
    assert len(set(oa)) == 27 and set(oa) <= full
    for i, j in itertools.combinations(range(5), 2):
        assert collections.Counter((r[i], r[j]) for r in oa) == collections.Counter({(a, b): 3 for a in range(3) for b in range(3)})


def test_subset_drops_exactly_one_colony_per_condition():
    y, g = design_labels()
    idx, dropped = subset_indices(y, g, (0, 1, 2, 0, 1))
    assert len(idx) == 30 and len(dropped) == 5
    assert sorted(d.split('::')[0] for d in dropped) == list('ABCDE')
    assert not set(g[idx]) & set(dropped)


def test_selection_frequency_counts_filtered_features_as_unselected(tmp_path):
    p = 6
    f1, f2 = tmp_path / 'a.npz', tmp_path / 'b.npz'
    np.savez(f1, m=np.array([5, 4, 3, 2, 1, 0], float))
    np.savez(f2, m=np.array([np.nan, 4, 5, 2, 1, 0], float))      # feature 0 filtered out
    pi = selection_frequency([str(f1), str(f2)], methods=('m',), ns=(2,))[('m', 2)]
    np.testing.assert_allclose(pi, [0.5, 1.0, 0.5, 0, 0, 0])


def test_technical_draws_keep_every_colony_and_are_distinct():
    from validation.stability import technical_draws, technical_positions, colony_mean_data
    y, g = design_labels()
    assert list(technical_positions(g)[:6]) == [0, 1, 2, 0, 1, 2]
    draws = technical_draws(g, 2, 50, seed=1)
    assert len({tuple(d) for d in draws}) == 50
    for d in draws:
        assert len(d) == 30 and all((g[d] == c).sum() == 2 for c in set(g))
    one = technical_draws(g, 1, 10, seed=2)
    assert all(len(d) == 15 and len(set(g[d])) == 15 for d in one)
    X = np.arange(45 * 2, dtype=float).reshape(45, 2)
    Xm, ym, cm = colony_mean_data(X, y, g)
    assert Xm.shape == (15, 2) and np.allclose(Xm[0], X[:3].mean(0)) and ym[0] == 'A'
