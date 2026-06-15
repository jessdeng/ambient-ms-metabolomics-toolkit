"""
Test B2: grouped (held-out, colony-aware) permutation importance.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from shared.classifier_comparison_standard import (  # noqa: E402
    grouped_permutation_importance, make_preprocessor,
)


def _synthetic(seed=0):
    """2 classes x 3 colonies x 3 technical replicates. Feature 0 is class-
    informative, features 1-5 are noise. Returns X_binned, y, groups, mz."""
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    for cls, base in [('A', 0.0), ('B', 5.0)]:
        for colony in range(3):
            shift = rng.normal(0, 0.2)
            for _t in range(3):
                row = [base + shift + rng.normal(0, 0.1)] + list(rng.normal(10, 1, size=5))
                X.append(row); y.append(cls); groups.append(f'{cls}::col{colony}')
    mz = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
    return np.array(X), np.array(y), np.array(groups), mz


def _prep():
    # Minimal, stable preprocessor: no feature-dropping filters so column count
    # (and m/z alignment) is preserved across folds.
    return make_preprocessor(
        normalization='none', log_transform='none', scaling='autoscale',
        variance_percentile=0, abundance_percentile=0, prevalence_threshold=0.0,
        snr_floor_enabled=False,
    )


def test_perm_importance_schema_and_alignment():
    X, y, groups, mz = _synthetic()
    df = grouped_permutation_importance(X, y, groups, _prep(), mz,
                                        models=('rf', 'gb'), n_splits=3,
                                        n_perm_repeats=3, random_state=0)
    assert len(df) == X.shape[1]
    assert list(df['mz']) == list(mz)
    for col in ['rf_perm_importance_mean', 'rf_perm_importance_std',
                'gb_perm_importance_mean', 'gb_perm_importance_std']:
        assert col in df.columns


def test_perm_importance_ranks_informative_feature_top():
    X, y, groups, mz = _synthetic()
    df = grouped_permutation_importance(X, y, groups, _prep(), mz,
                                        models=('rf',), n_splits=3,
                                        n_perm_repeats=5, random_state=0)
    # Feature 0 (m/z 100) is the only class-informative feature -> highest
    # held-out permutation importance.
    top_mz = df.loc[df['rf_perm_importance_mean'].idxmax(), 'mz']
    assert top_mz == 100.0


def test_perm_importance_single_model_subset():
    X, y, groups, mz = _synthetic()
    df = grouped_permutation_importance(X, y, groups, _prep(), mz,
                                        models=('gb',), n_splits=2,
                                        n_perm_repeats=2, random_state=1)
    assert 'gb_perm_importance_mean' in df.columns
    assert 'rf_perm_importance_mean' not in df.columns
