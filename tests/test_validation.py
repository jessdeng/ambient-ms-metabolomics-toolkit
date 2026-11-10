"""Validation helpers: balanced colony folds, colony-level scoring, permutation test."""
import os
import sys

import numpy as np
import pytest
from sklearn.linear_model import RidgeClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared import validation as va

COND = np.repeat(list('ABCDE'), 9)
COL = np.array([f'{c}::{k}' for c in 'ABCDE' for k in range(3) for _ in range(3)])


def _tic(seed=0, p=60, signal=True):
    """45 spectra (5 conditions x 3 colonies x 3 sprays); every condition has
    its own four elevated bins when ``signal`` is True."""
    rng = np.random.default_rng(seed)
    X = rng.lognormal(3, 0.3, (45, p))
    if signal:
        for i, c in enumerate('ABCDE'):
            X[COND == c, 4 * i:4 * i + 4] *= 6
    return X / X.sum(1, keepdims=True)


def _colonies_per_condition(cond, col, rows):
    return tuple(len(set(col[rows][cond[rows] == c])) for c in np.unique(cond))


def test_every_test_fold_holds_one_whole_colony_per_condition():
    cv = va.RepeatedColonyKFold(n_splits=3, n_repeats=4, random_state=0)
    splits = list(cv.split(None, COND, COL))
    assert len(splits) == cv.get_n_splits() == 12
    for tr, te in splits:
        assert not set(COL[tr]) & set(COL[te])
        assert _colonies_per_condition(COND, COL, te) == (1, 1, 1, 1, 1)
        assert sorted(np.r_[tr, te]) == list(range(45))
    for r in range(4):                                     # each repeat is a partition
        tested = np.concatenate([te for _, te in splits[3 * r:3 * r + 3]])
        assert sorted(tested) == list(range(45))


def test_splits_are_reproducible_and_repeats_differ():
    a = [te.tolist() for _, te in va.RepeatedColonyKFold(3, 5, 7).split(None, COND, COL)]
    b = [te.tolist() for _, te in va.RepeatedColonyKFold(3, 5, 7).split(None, COND, COL)]
    assert a == b
    assert len({tuple(te) for te in a}) > 3                # not the same three folds every repeat


def test_unbalanced_design_spreads_the_extra_colony():
    # condition A has four colonies with 1-3 spectra each
    cond = np.r_[np.repeat('A', 9), np.repeat('B', 9), np.repeat('C', 9)]
    col = np.r_[['A::0'] * 3, ['A::1'] * 3, ['A::2'] * 2, ['A::3'],
                np.repeat(['B::0', 'B::1', 'B::2'], 3), np.repeat(['C::0', 'C::1', 'C::2'], 3)]
    for tr, te in va.RepeatedColonyKFold(3, 6, 1).split(None, cond, col):
        n_a, n_b, n_c = _colonies_per_condition(cond, col, te)
        assert n_a in (1, 2) and n_b == n_c == 1
        assert not set(col[tr]) & set(col[te])


def test_colony_with_two_conditions_is_rejected():
    cond = COND.copy()
    cond[0] = 'B'                                          # one spectrum of colony A::0 relabelled
    with pytest.raises(ValueError, match='more than one condition'):
        list(va.RepeatedColonyKFold().split(None, cond, COL))


def test_permuted_labels_keep_colonies_intact():
    rng = np.random.default_rng(0)
    seen_change = False
    for _ in range(20):
        yp = va.permute_colony_labels(COND, COL, rng)
        assert all(len(set(yp[COL == g])) == 1 for g in np.unique(COL))
        assert sorted(yp) == sorted(COND)                  # three colonies per condition kept
        seen_change |= not np.array_equal(yp, COND)
    assert seen_change


def test_colony_vote_takes_the_majority():
    vote = va.colony_vote(np.array(['A', 'A', 'B', 'B', 'B', 'A']), np.array(['x'] * 3 + ['y'] * 3))
    assert vote == {'x': 'A', 'y': 'B'}


def test_grouped_cv_scores_planted_signal_per_repeat():
    res = va.grouped_cv(RidgeClassifier(), _tic(), COND, COL, va.RepeatedColonyKFold(3, 3, 0))
    assert set(res) == {'accuracy', 'balanced_accuracy', 'colony_accuracy'}
    assert all(v.shape == (3,) for v in res.values())
    assert res['accuracy'].mean() > 0.9 and res['colony_accuracy'].mean() > 0.9
    np.testing.assert_allclose(res['accuracy'], res['balanced_accuracy'])   # balanced design


def test_grouped_cv_preprocessing_ignores_held_out_rows(monkeypatch):
    calls = []
    real = va._fold_matrices
    def spy(X, tr, te, *a):
        calls.append((tr.copy(), te.copy()))
        return real(X, tr, te, *a)
    monkeypatch.setattr(va, '_fold_matrices', spy)
    va.grouped_cv(RidgeClassifier(), _tic(), COND, COL, va.RepeatedColonyKFold(3, 2, 0))
    assert len(calls) == 6
    assert all(not set(COL[tr]) & set(COL[te]) for tr, te in calls)


def test_permutation_test_detects_signal_and_is_reproducible():
    cv = va.RepeatedColonyKFold(3, 1, 0)
    obs, null, p = va.colony_permutation_test(RidgeClassifier(), _tic(), COND, COL, cv,
                                              n_permutations=19, seed=3)
    assert obs > 0.9 and null.shape == (19,)
    assert null.max() < obs and p == pytest.approx(1 / 20)   # smallest attainable p
    _, null2, _ = va.colony_permutation_test(RidgeClassifier(), _tic(), COND, COL, cv,
                                             n_permutations=19, seed=3)
    np.testing.assert_array_equal(null, null2)


def test_permutation_test_without_signal_is_not_significant():
    cv = va.RepeatedColonyKFold(3, 1, 0)
    obs, null, p = va.colony_permutation_test(RidgeClassifier(), _tic(signal=False), COND, COL, cv,
                                              n_permutations=19, seed=3)
    assert obs < 0.6 and p > 0.05


def test_plsda_classifier_separates_planted_classes():
    X = np.log10(_tic())
    X = (X - X.mean(0)) / X.std(0, ddof=1)
    m = va.PLSDAClassifier(n_components=4).fit(X, COND)
    assert list(m.classes_) == list('ABCDE')
    assert (m.predict(X) == COND).mean() > 0.9
    two = COND[:18]                                        # two classes: A and B
    assert (va.PLSDAClassifier(n_components=1).fit(X[:18], two).predict(X[:18]) == two).mean() > 0.9
    res = va.grouped_cv(va.PLSDAClassifier(n_components=4), _tic(), COND, COL,
                        va.RepeatedColonyKFold(3, 2, 0))
    assert res['accuracy'].mean() > 0.8
