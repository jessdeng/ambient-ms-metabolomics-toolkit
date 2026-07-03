"""
Unit tests for src/shared/quality_metrics.py.

The reproducibility metrics are validated against closed-form values computed by
hand on tiny synthetic designs, and the benchmarking metrics against a synthetic
label vector with a known confusion structure. Matplotlib uses the non-interactive
Agg backend so the confusion-matrix figure test writes a real file without a
display.
"""

import os
import sys

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

# Make ``import src.shared...`` work when tests are run from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.shared import quality_metrics as qm


# ---------------------------------------------------------------------------------
# Technical CV
# ---------------------------------------------------------------------------------

def test_technical_cv_known_value_single_unit():
    # One biological unit, three technical replicates, one feature.
    # values [10, 12, 14]: mean=12, sample SD (ddof=1)=2 -> CV = 100*2/12 = 16.6667%
    X = np.array([[10.0], [12.0], [14.0]])
    groups = np.array(['A', 'A', 'A'])
    out = qm.technical_cv(X, groups)
    assert out['n_units_used'] == 1
    np.testing.assert_allclose(out['per_feature'], [100.0 * 2.0 / 12.0], rtol=1e-12)


def test_technical_cv_pools_across_units_by_median():
    # Two units. Feature CVs: unit A = 16.6667%, unit B constructed to be 25%.
    # B values [8,10,12] -> mean 10, SD 2 -> CV 20%. Use median of {16.67, 20} = 18.33.
    X = np.array([[10.0], [12.0], [14.0],     # A
                  [8.0], [10.0], [12.0]])     # B
    groups = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
    out = qm.technical_cv(X, groups, aggregate='median')
    cv_a = 100.0 * 2.0 / 12.0
    cv_b = 100.0 * 2.0 / 10.0
    np.testing.assert_allclose(out['per_feature'], [np.median([cv_a, cv_b])], rtol=1e-12)


def test_technical_cv_zero_mean_feature_is_nan():
    # Feature that is all zeros -> mean 0 -> CV undefined (NaN), not inf.
    X = np.array([[0.0, 10.0], [0.0, 12.0], [0.0, 14.0]])
    groups = np.array(['A', 'A', 'A'])
    out = qm.technical_cv(X, groups)
    assert np.isnan(out['per_feature'][0])
    assert np.isfinite(out['per_feature'][1])


def test_technical_cv_singleton_units_excluded():
    # Two singleton units (no replication) + one real triplicate.
    X = np.array([[10.0], [12.0], [14.0],   # A (triplicate)
                  [5.0],                     # B (singleton)
                  [99.0]])                   # C (singleton)
    groups = np.array(['A', 'A', 'A', 'B', 'C'])
    out = qm.technical_cv(X, groups)
    assert out['n_units_used'] == 1
    np.testing.assert_allclose(out['per_feature'], [100.0 * 2.0 / 12.0], rtol=1e-12)


def test_technical_cv_raises_without_replication():
    X = np.array([[1.0], [2.0]])
    groups = np.array(['A', 'B'])   # no unit has >= 2 reps
    with pytest.raises(ValueError):
        qm.technical_cv(X, groups)


def test_cv_rejects_negative_input():
    # Log-transformed / autoscaled data has negatives -> must raise.
    X = np.array([[-1.0], [0.5], [2.0]])
    groups = np.array(['A', 'A', 'A'])
    with pytest.raises(ValueError):
        qm.technical_cv(X, groups)


# ---------------------------------------------------------------------------------
# Biological CV
# ---------------------------------------------------------------------------------

def test_biological_cv_collapses_reps_then_cv():
    # Two colonies, duplicate tech reps each. Colony means: A=10, B=20.
    # mean of means = 15, SD (ddof=1) = 7.0710678 -> CV = 47.1405%.
    X = np.array([[9.0], [11.0],     # colony A -> mean 10
                  [19.0], [21.0]])   # colony B -> mean 20
    bio = np.array(['A', 'A', 'B', 'B'])
    out = qm.biological_cv(X, bio)
    expected = 100.0 * np.std([10.0, 20.0], ddof=1) / 15.0
    np.testing.assert_allclose(out['per_feature'], [expected], rtol=1e-12)
    assert out['n_units'] == 2


def test_biological_cv_within_class_pooling():
    # Two classes, two colonies each; CV computed within class then median-pooled.
    X = np.array([[10.0], [20.0],    # class P: colonies means 10, 20
                  [30.0], [60.0]])   # class Q: colonies means 30, 60
    bio = np.array(['A', 'B', 'C', 'D'])
    cls = np.array(['P', 'P', 'Q', 'Q'])
    out = qm.biological_cv(X, bio, class_labels=cls)
    cv_p = 100.0 * np.std([10.0, 20.0], ddof=1) / 15.0
    cv_q = 100.0 * np.std([30.0, 60.0], ddof=1) / 45.0
    np.testing.assert_allclose(out['per_feature'], [np.median([cv_p, cv_q])], rtol=1e-12)


# ---------------------------------------------------------------------------------
# D-ratio
# ---------------------------------------------------------------------------------

def test_dispersion_ratio_known_value():
    # Two colonies, duplicate tech reps.
    # Within-unit SDs: A over [9,11] = 1.41421 (var 2); B over [18,22] = 2.82843 (var 8).
    # sigma_tech = sqrt(mean(var)) = sqrt((2+8)/2) = sqrt(5) = 2.2360680.
    # Unit means: A=10, B=20 -> sigma_total = SD([10,20], ddof=1) = 7.0710678.
    # D-ratio = 100 * 2.2360680 / 7.0710678 = 31.6228%.
    X = np.array([[9.0], [11.0],     # A
                  [18.0], [22.0]])   # B
    groups = np.array(['A', 'A', 'B', 'B'])
    out = qm.dispersion_ratio(X, groups)
    sigma_tech = np.sqrt((2.0 + 8.0) / 2.0)
    sigma_total = np.std([10.0, 20.0], ddof=1)
    np.testing.assert_allclose(out['per_feature'],
                               [100.0 * sigma_tech / sigma_total], rtol=1e-9)


def test_qc_report_summary_keys_and_thresholds():
    rng = np.random.default_rng(0)
    # 4 colonies x 3 tech reps, 6 features; low-noise -> most features pass.
    base = rng.uniform(50, 200, size=6)
    rows, bio, cls = [], [], []
    for ci, colony in enumerate(['A', 'B', 'C', 'D']):
        colony_mean = base * rng.uniform(0.9, 1.1, size=6)
        for _ in range(3):
            rows.append(colony_mean * rng.uniform(0.98, 1.02, size=6))
            bio.append(colony)
            cls.append('P' if ci < 2 else 'Q')
    X = np.array(rows)
    mz = np.linspace(100, 200, 6)
    per_feature, summary = qm.qc_report(X, mz, np.array(bio), np.array(cls))
    for key in ('median_technical_cv_pct', 'frac_cv_under_30',
                'median_dratio_pct', 'n_pass_both'):
        assert key in summary
    assert per_feature['mz'].shape == (6,)
    assert 0.0 <= summary['frac_cv_under_30'] <= 1.0
    # Tight replicates -> low technical CV.
    assert summary['median_technical_cv_pct'] < 10.0


# ---------------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------------

def test_classification_metrics_perfect():
    y = np.array([0, 0, 1, 1, 2, 2])
    out = qm.classification_metrics(y, y, class_names=['a', 'b', 'c'])
    assert out['macro_f1'] == pytest.approx(1.0)
    assert out['accuracy'] == pytest.approx(1.0)
    assert np.array_equal(out['confusion'], np.eye(3) * 2)
    assert out['per_class']['a']['recall'] == pytest.approx(1.0)


def test_classification_metrics_known_confusion():
    # 3 classes, one 'b' misread as 'c'.
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 2, 2, 2])
    out = qm.classification_metrics(y_true, y_pred, class_names=['a', 'b', 'c'])
    cm = out['confusion']
    assert cm[1, 2] == 1        # one true-b predicted c
    assert cm[1, 1] == 1        # one true-b correct
    assert out['per_class']['b']['recall'] == pytest.approx(0.5)
    assert out['accuracy'] == pytest.approx(5 / 6)


def test_grouped_oof_predictions_leakfree_shape():
    # End-to-end small grouped design through the real Pipeline path.
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(1)
    n_per_group, n_feat = 3, 8
    X, y, groups = [], [], []
    gid = 0
    for cls in range(3):
        for _ in range(3):                        # 3 colonies per class
            centre = rng.normal(cls * 4, 0.3, size=n_feat)
            for _ in range(n_per_group):          # 3 tech reps per colony
                X.append(centre + rng.normal(0, 0.1, size=n_feat))
                y.append(cls)
                groups.append(gid)
            gid += 1
    X = np.array(X); y = np.array(y); groups = np.array(groups)
    prep = [('scale', StandardScaler())]
    y_pred = qm.grouped_oof_predictions(
        LogisticRegression(max_iter=500), X, y, groups, prep,
        n_splits=3, random_state=0)
    assert y_pred.shape == y.shape
    # Well-separated clusters -> should classify most held-out samples correctly.
    assert (y_pred == y).mean() > 0.7


def test_plot_confusion_matrix_writes_file(tmp_path):
    cm = np.array([[3, 0, 0], [0, 2, 1], [0, 0, 3]])
    out = tmp_path / 'cm.png'
    qm.plot_confusion_matrix(cm, ['Amber', 'Green', 'White'], str(out),
                             title='Test', normalize=True)
    assert out.exists() and out.stat().st_size > 0
