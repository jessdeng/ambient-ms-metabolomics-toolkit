"""Classifier comparison: colony-grouped folds, training-only preprocessing, outputs."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared import classifiers as cl

COND = np.repeat(list('ABC'), 9)
COL = np.array([f'{c}::{k}' for c in 'ABC' for k in range(3) for _ in range(3)])


def _tic(seed=0, p=60):
    rng = np.random.default_rng(seed)
    X = rng.lognormal(3, 0.3, (27, p))
    X[COND == 'C', :4] *= 6
    return X / X.sum(1, keepdims=True)


def test_folds_hold_out_whole_colonies_and_summary_is_complete(monkeypatch):
    seen = []
    real = cl._fold_matrices
    def spy(X, tr, te, *a):
        seen.append((set(COL[tr]), set(COL[te])))
        return real(X, tr, te, *a)
    monkeypatch.setattr(cl, '_fold_matrices', spy)
    folds, summary = cl.compare_classifiers(_tic(), COND, COL, n_splits=5, n_repeats=2, seed=0)
    assert len(seen) == 6                                   # 3 folds (3 colonies/condition) x 2 repeats
    assert all(not (a & b) for a, b in seen)
    assert list(summary.model) == list(cl.MODELS)
    assert (summary.n_folds == 6).all() and summary.cv.iloc[0] == '3-fold colony-grouped x 2'
    assert summary.test_accuracy_mean.between(0, 1).all()
    assert summary.loc[summary.model == 'Ridge', 'test_accuracy_mean'].iloc[0] > 0.6   # planted signal


def test_preprocessing_uses_training_rows_only():
    X = _tic(1)
    tr, te = np.arange(18), np.arange(18, 27)
    a_tr, a_te = cl._fold_matrices(X, tr, te, 'autoscale', 25, 5)
    X2 = X.copy(); X2[te] *= 50                             # change only the held-out rows
    b_tr, _ = cl._fold_matrices(X2, tr, te, 'autoscale', 25, 5)
    np.testing.assert_allclose(a_tr, b_tr)                  # training matrix unaffected
    assert np.allclose(a_tr.mean(0), 0) and a_te.shape[1] == a_tr.shape[1]


def test_plot_writes_file(tmp_path):
    folds, _ = cl.compare_classifiers(_tic(2), COND, COL, n_repeats=1, seed=0)
    cl.plot_accuracy_comparison(folds, 'test', str(tmp_path / 'c.png'), chance=1 / 3)
    assert (tmp_path / 'c.png').exists()
