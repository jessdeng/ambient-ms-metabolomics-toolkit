"""
tests/test_pipeline_integration.py
==================================
Regression scaffold locking down the Step-2 leaf-module refactor:

  1. QC scale contract — preprocess(return_stages=True) exposes a strictly linear,
     normalised matrix; the chemometric QC (technical %CV, D-ratio) accepts that
     matrix and rejects any log-transformed / autoscaled matrix with a ValueError.
  2. Preprocessing/normalisation drift — the full-data primitives and the per-fold
     CV transformers apply identical mathematics.
  3. Leakage & grouping — _require_grouped is the only evaluation path (it raises
     without groups/prep_steps), grouped folds never split a colony, and grouped
     CV runs to completion on realistic MS data (high dynamic range, exact zeros,
     degenerate features) without emitting non-finite/NaN RuntimeWarnings.

Synthetic fixtures mimic ambient-MS profiles: log-normal abundances with a wide
dynamic range, sparse (zero) detections, class-discriminative peaks, tight
technical-replicate noise, and deliberately degenerate columns.

Run from the repository root:
    pytest tests/test_pipeline_integration.py -v
"""

import os
import sys
import warnings

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, 'src')
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

from standard import preprocessing as pp
from src.shared import quality_metrics as qm
from src.shared import classifier_comparison_standard as ccs


# ---------------------------------------------------------------------------------
# Synthetic ambient-MS fixture
# ---------------------------------------------------------------------------------

def make_ms_dataset(n_classes=3, n_colonies=4, n_tech=3, n_features=400, seed=0,
                    add_degenerate=True):
    """
    Realistic ambient-MS intensity matrix with a nested colony/technical design.

    Structure
    ---------
    * ``n_classes`` conditions, each with ``n_colonies`` biological replicates
      (colonies), each measured in ``n_tech`` technical replicates.
    * Colony-level true abundances are log-normal (wide dynamic range: baseline
      ~e^3, heavy upper tail), with a distinct subset of class-discriminative
      peaks elevated 5–30x per condition.
    * Low-abundance features are sparsely zeroed (undetected), and technical
      replicates add tight multiplicative noise with occasional dropouts, so
      within-colony %CV is small and finite for detected features.
    * When ``add_degenerate`` is True, an all-zero column and a constant column
      are injected to stress the zero/NaN guards.

    Returns
    -------
    X : ndarray (n_samples, n_features)   linear, non-negative intensities.
    y : ndarray (n_samples,)              class label per sample.
    groups : ndarray (n_samples,)         colony id per sample (CV group).
    mz : ndarray (n_features,)            m/z axis.
    """
    rng = np.random.default_rng(seed)
    mz = np.linspace(100.0, 1000.0, n_features)
    class_names = [f'Cond{c}' for c in range(n_classes)]
    signature = {c: rng.choice(n_features, size=20, replace=False)
                 for c in range(n_classes)}

    rows, labels, colony_ids = [], [], []
    colony_counter = 0
    for c in range(n_classes):
        for _ in range(n_colonies):
            colony_counter += 1
            colony_name = f'colony{colony_counter:02d}'
            base = rng.lognormal(mean=3.0, sigma=1.8, size=n_features)
            base[signature[c]] *= rng.uniform(5.0, 30.0, size=signature[c].size)
            faint = base < np.percentile(base, 30)
            drop = faint & (rng.random(n_features) < 0.7)
            base = base.copy()
            base[drop] = 0.0
            for _t in range(n_tech):
                tech = base * rng.lognormal(0.0, 0.08, size=n_features)
                tech[(tech > 0) & (rng.random(n_features) < 0.02)] = 0.0
                rows.append(tech)
                labels.append(class_names[c])
                colony_ids.append(colony_name)

    X = np.asarray(rows, dtype=float)
    if add_degenerate:
        X[:, 0] = 0.0     # all-zero feature (never detected)
        X[:, 1] = 7.0     # constant feature (zero variance)
    return X, np.asarray(labels), np.asarray(colony_ids), mz


@pytest.fixture(scope='module')
def ms_data():
    return make_ms_dataset()


# ---------------------------------------------------------------------------------
# 1. QC scale contract
# ---------------------------------------------------------------------------------

class TestQCScaleContract:
    """The QC metrics must run on the linear normalised matrix and reject any
    transformed/scaled matrix."""

    @pytest.mark.parametrize('normalization', ['tic', 'median', 'pqn', 'quantile'])
    def test_return_stages_normalized_is_linear(self, ms_data, normalization):
        X, _, _, _ = ms_data
        stages = pp.preprocess(X, normalization=normalization,
                               log_transform='glog', scaling='autoscale',
                               return_stages=True)
        norm = stages['normalized']
        assert norm.shape == X.shape
        assert np.isfinite(norm).all()
        assert norm.min() >= -1e-9, 'normalised stage must be non-negative (linear)'
        # The normalised stage must equal the dedicated linear accessor exactly.
        np.testing.assert_allclose(norm, pp.normalize_only(X, normalization),
                                   rtol=0, atol=0)

    def test_default_return_matches_scaled_stage(self, ms_data):
        X, _, _, _ = ms_data
        stages = pp.preprocess(X, normalization='tic', log_transform='glog',
                               scaling='autoscale', return_stages=True)
        default = pp.preprocess(X, normalization='tic', log_transform='glog',
                                scaling='autoscale')
        np.testing.assert_allclose(default, stages['scaled'], rtol=0, atol=0)

    def test_technical_cv_accepts_linear_normalized(self, ms_data):
        X, _, groups, _ = ms_data
        linear = pp.normalize_only(X, 'tic')
        out = qm.technical_cv(linear, groups)
        tcv = out['per_feature']
        assert out['n_units_used'] == len(np.unique(groups))
        # Detected features yield a finite, physically-plausible %CV.
        assert np.isfinite(tcv).mean() > 0.5
        assert np.nanmedian(tcv) >= 0.0

    def test_dispersion_ratio_accepts_linear_normalized(self, ms_data):
        X, y, groups, _ = ms_data
        linear = pp.normalize_only(X, 'tic')
        out = qm.dispersion_ratio(linear, groups, class_labels=y)
        assert np.isfinite(out['per_feature']).any()

    def test_technical_cv_rejects_model_matrix(self, ms_data):
        # The matrix actually passed around the pipeline (glog + autoscale) carries
        # negatives from mean-centring; feeding it to the QC engine is the concrete
        # leakage/misuse vector and must raise.
        X, _, groups, _ = ms_data
        model_matrix = pp.preprocess(X, normalization='tic',
                                     log_transform='glog', scaling='autoscale')
        assert model_matrix.min() < -1e-9
        with pytest.raises(ValueError, match='linear'):
            qm.technical_cv(model_matrix, groups)

    def test_technical_cv_rejects_log_transformed(self, ms_data):
        # A log10-transformed (unscaled) matrix has negatives wherever normalised
        # intensity < 1, so the strictly-linear contract rejects it too.
        X, _, groups, _ = ms_data
        stages = pp.preprocess(X, normalization='tic', log_transform='log10',
                               scaling='none', return_stages=True)
        transformed = stages['transformed']
        assert transformed.min() < -1e-9
        with pytest.raises(ValueError, match='linear'):
            qm.technical_cv(transformed, groups)

    def test_dispersion_ratio_rejects_scaled_matrix(self, ms_data):
        X, y, groups, _ = ms_data
        model_matrix = pp.preprocess(X, normalization='tic',
                                     log_transform='glog', scaling='autoscale')
        with pytest.raises(ValueError, match='linear'):
            qm.dispersion_ratio(model_matrix, groups, class_labels=y)


# ---------------------------------------------------------------------------------
# 2. Preprocessing / normalisation drift
# ---------------------------------------------------------------------------------

class TestNoPreprocessingDrift:
    """The full-data preprocess() and the per-fold CV transformer chain must apply
    byte-identical mathematics (single source of truth)."""

    METHODS = [
        (norm, tf, sc)
        for norm in ('tic', 'median', 'pqn', 'quantile')
        for tf in ('glog', 'log10', 'log2', 'sqrt', 'none')
        for sc in ('autoscale', 'pareto', 'range', 'vast', 'level', 'none')
    ]

    @pytest.mark.parametrize('normalization,log_transform,scaling', METHODS)
    def test_global_equals_per_fold(self, ms_data, normalization, log_transform,
                                    scaling):
        X, _, _, _ = ms_data
        reference = pp.preprocess(X.copy(), normalization=normalization,
                                  log_transform=log_transform, scaling=scaling)
        steps = [('normalize', ccs.Normalizer(normalization)),
                 ('logtrans', ccs.LogTransform(log_transform)),
                 ('scale', ccs.Scaler(scaling))]
        got = Pipeline([(n, clone(t)) for n, t in steps]).fit_transform(X.copy())
        assert np.nanmax(np.abs(reference - got)) < 1e-9

    def test_filter_masks_match_transformers(self, ms_data):
        X, _, _, _ = ms_data
        vf = ccs.VarianceFilter(25).fit(X)
        af = ccs.AbundanceFilter(5).fit(X)
        assert np.array_equal(vf.keep_, pp.variance_keep_mask(X, 25))
        assert np.array_equal(af.keep_, pp.abundance_keep_mask(X, 5))


# ---------------------------------------------------------------------------------
# 3. Leakage & grouping
# ---------------------------------------------------------------------------------

class TestLeakageAndGrouping:
    """_require_grouped blocks any ungrouped/pre-preprocessed path and grouped CV
    runs cleanly on degenerate MS data."""

    def _prep(self):
        return ccs.make_preprocessor(
            normalization='tic', log_transform='glog', scaling='autoscale',
            variance_percentile=25, abundance_percentile=5,
            prevalence_threshold=0.0, snr_floor_enabled=False)

    def test_require_grouped_rejects_missing_groups(self, ms_data):
        X, y, _, _ = ms_data
        with pytest.raises(ValueError, match='requires both groups'):
            ccs._require_grouped(RandomForestClassifier(), X, y, 3,
                                 None, self._prep())

    def test_require_grouped_rejects_missing_prep(self, ms_data):
        X, y, groups, _ = ms_data
        with pytest.raises(ValueError, match='requires both groups'):
            ccs._require_grouped(RandomForestClassifier(), X, y, 3, groups, None)

    def test_legacy_paths_removed(self):
        for mod in (ccs, __import__('src.shared.classifier_comparison',
                                    fromlist=['x'])):
            assert not hasattr(mod, '_run_cv_legacy')
            assert not hasattr(mod, '_dispatch')

    def test_classifier_entrypoint_requires_grouping(self, ms_data):
        # The public classifier entrypoints must not silently run ungrouped CV.
        X, y, _, _ = ms_data
        with pytest.raises(ValueError, match='requires both groups'):
            ccs.RandomForest(X, y, n_splits=3, groups=None, prep_steps=None)

    def test_grouped_folds_never_split_a_colony(self, ms_data):
        X, y, groups, _ = ms_data
        yi = LabelEncoder().fit_transform(y)
        k = ccs.auto_n_splits(y, groups, desired=5)
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
        for train_idx, test_idx in sgkf.split(X, yi, groups):
            train_colonies = set(groups[train_idx])
            test_colonies = set(groups[test_idx])
            assert train_colonies.isdisjoint(test_colonies)

    def test_grouped_cv_runs_clean_no_nonfinite_warnings(self, ms_data):
        X, y, groups, _ = ms_data
        prep = self._prep()
        n_splits = ccs.auto_n_splits(y, groups, desired=5)
        n_bio = len(np.unique(groups))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            metrics = ccs.RandomForest(
                X, y, n_splits=n_splits, groups=groups, prep_steps=prep,
                n_repeats=3, return_metrics=True, n_biological=n_bio)
        for key in ('test_accuracy', 'train_accuracy', 'test_balanced_accuracy'):
            assert np.isfinite(metrics[key]).all()
            assert metrics[key].size == n_splits * 3
        offenders = [
            str(w.message) for w in caught
            if issubclass(w.category, RuntimeWarning)
            and any(tok in str(w.message).lower()
                    for tok in ('invalid value', 'divide by zero', 'overflow',
                                'nan', 'all-nan'))
        ]
        assert not offenders, f'degenerate-feature RuntimeWarnings: {offenders}'

    def test_grouped_oof_benchmark_is_leak_free_and_finite(self, ms_data):
        X, y, groups, _ = ms_data
        prep = self._prep()
        n_splits = ccs.auto_n_splits(y, groups, desired=5)
        yi = LabelEncoder().fit_transform(y)
        y_pred = qm.grouped_oof_predictions(
            RandomForestClassifier(n_estimators=50, random_state=0),
            X, yi, groups, prep, n_splits, random_state=0)
        assert y_pred.shape == yi.shape
        result = qm.classification_metrics(yi, y_pred,
                                           class_names=list(np.unique(y)))
        assert 0.0 <= result['macro_f1'] <= 1.0
        assert np.isfinite(result['confusion']).all()
        assert result['confusion'].sum() == len(yi)
