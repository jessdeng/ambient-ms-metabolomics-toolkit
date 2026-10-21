"""Tests for classification validation (validated/classification.py)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from validation.classification import original_estimators, loco_evaluate
from shared.classifier_comparison import make_preprocessor


def _data(signal, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat(list('ABC'), 9); g = np.repeat([f'k{i}' for i in range(9)], 3)
    X = rng.lognormal(size=(27, 30))
    X[y == 'A', :3] *= signal
    return X, y, g


def test_six_original_estimators_present():
    assert set(original_estimators()) == {'ridge', 'lr', 'svm', 'lda', 'rf', 'gb'}


def test_loco_separable_vs_null():
    steps = make_preprocessor(normalization='tic', log_transform='log10', scaling='autoscale',
                              variance_percentile=0, abundance_percentile=0, prevalence_threshold=0, snr_floor_enabled=False)
    X, y, g = _data(20.0)
    r = loco_evaluate(original_estimators()['lr'], X, y, g, steps)
    assert r['accuracy'] > 0.6 and r['confusion'].sum() == 27
    X0, y0, g0 = _data(1.0)
    r0 = loco_evaluate(original_estimators()['lr'], X0, y0, g0, steps)
    assert r0['accuracy'] < r['accuracy']
