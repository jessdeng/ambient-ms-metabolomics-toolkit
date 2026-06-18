"""
Classifier Comparison for Mass Spectrometry Metabolomics Data — R-comparable path
==================================================================================
Trains and evaluates six supervised classifiers and reports cross-validated
train/test accuracy, plus an ensemble feature-importance overlap.

This module mirrors classifier_comparison_standard.py in all methodology,
differing only in the import source for compute_vip_1comp (r_comparable.pipeline)
and in the per-pipeline default choices exposed to callers.

CORRECTED VERSION — addresses two methodological issues:

  (1) Pseudoreplication. StratifiedGroupKFold with colony-level groups prevents
      technical replicates from straddling the train/test boundary.

  (2) Preprocessing leakage. All filtering, normalisation, transformation, and
      scaling are now fit inside each CV fold via an sklearn Pipeline.

Classifiers: Random Forest, SVM (linear), Gradient Boosting,
             Logistic Regression, LDA, Ridge.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from src.shared.plot_style import apply_style, pub_savefig

apply_style()

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_validate
from sklearn.metrics import accuracy_score

from r_comparable.pipeline import compute_vip_1comp
# Single source of truth for the small-sample capacity guards (shared with the
# standard pipeline so both branches constrain models identically).
from shared.classifier_comparison_standard import small_sample_guards

try:
    import config as _config
    _SEED = _config.RANDOM_SEED
except Exception:
    _SEED = 42


# -- Preprocessing transformers ------------------------------------------------
# Each transformer learns its parameters in fit() (training fold only) and
# applies them in transform(). Fitting on the full dataset reproduces
# r_comparable/preprocessing.py preprocess() exactly (verified: max abs diff 0.0).


class SNRFloor(BaseEstimator, TransformerMixin):
    """Remove features that never exceed the per-sample noise floor in any group.

    Mirrors standard.preprocessing.filter_snr_floor.  Label-aware — must live
    INSIDE the CV pipeline and be fit on the training fold only.

    Parameters
    ----------
    snr_threshold  : float  — min SNR to count a sample as detected (default 3)
    noise_quantile : float  — percentile of each row used to identify noise
                     region for the MAD estimate (default 60)
    min_fraction   : float  — min fraction of group samples that must exceed
                     the SNR threshold for the feature to be retained (default 0.5)
    enabled        : bool   — set False to skip entirely (pass-through)
    """
    def __init__(self, snr_threshold=3, noise_quantile=60, min_fraction=0.5,
                 enabled=True):
        self.snr_threshold  = snr_threshold
        self.noise_quantile = noise_quantile
        self.min_fraction   = min_fraction
        self.enabled        = enabled

    def fit(self, X, y=None):
        if not self.enabled or y is None or self.snr_threshold <= 0:
            self.keep_ = np.ones(X.shape[1], dtype=bool)
            return self
        y = np.asarray(y)
        n_samples, n_features = X.shape

        sigma = np.empty(n_samples)
        for i in range(n_samples):
            row      = X[i]
            thr      = np.percentile(row, self.noise_quantile)
            low_vals = row[row <= thr]
            if len(low_vals) >= 2:
                med = np.median(low_vals)
                mad = np.median(np.abs(low_vals - med))
                sigma[i] = 1.4826 * mad if mad > 0 else max(abs(med), 1e-10)
            else:
                sigma[i] = max(float(row.min()), 1e-10)

        SNR  = X / sigma[:, np.newaxis]
        keep = np.zeros(n_features, dtype=bool)
        for g in np.unique(y):
            passes = (SNR[y == g] >= self.snr_threshold).mean(axis=0) >= self.min_fraction
            keep  |= passes
        self.keep_ = keep if keep.any() else np.ones(n_features, dtype=bool)
        return self

    def transform(self, X):
        return X[:, self.keep_]


class PrevalenceFilter(BaseEstimator, TransformerMixin):
    """Remove features not genuinely detected in >= threshold of samples in any class."""
    def __init__(self, threshold=0.5, min_intensity=0.0):
        self.threshold     = threshold
        self.min_intensity = min_intensity

    def fit(self, X, y=None):
        if self.threshold <= 0 or y is None:
            self.keep_ = np.ones(X.shape[1], dtype=bool)
            return self
        y        = np.asarray(y)
        detected = X > self.min_intensity
        keep     = np.zeros(X.shape[1], dtype=bool)
        for c in np.unique(y):
            keep |= detected[y == c].mean(axis=0) >= self.threshold
        self.keep_ = keep if keep.any() else np.ones(X.shape[1], dtype=bool)
        return self

    def transform(self, X):
        return X[:, self.keep_]


class VarianceFilter(BaseEstimator, TransformerMixin):
    """Remove features with low relative standard deviation (RSD)."""
    def __init__(self, percentile=25):
        self.percentile = percentile

    def fit(self, X, y=None):
        if self.percentile <= 0:
            self.keep_ = np.ones(X.shape[1], dtype=bool)
            return self
        mean = X.mean(axis=0).copy()
        mean[mean == 0] = 1e-12
        rsd  = X.std(axis=0) / mean
        self.keep_ = rsd > np.percentile(rsd, self.percentile)
        return self

    def transform(self, X):
        return X[:, self.keep_]


class AbundanceFilter(BaseEstimator, TransformerMixin):
    """Remove features with low mean intensity."""
    def __init__(self, percentile=5):
        self.percentile = percentile

    def fit(self, X, y=None):
        if self.percentile <= 0:
            self.keep_ = np.ones(X.shape[1], dtype=bool)
            return self
        m = X.mean(axis=0)
        self.keep_ = m > np.percentile(m, self.percentile)
        return self

    def transform(self, X):
        return X[:, self.keep_]


class Normalizer(BaseEstimator, TransformerMixin):
    """Sample normalisation: 'tic', 'median', 'pqn', 'quantile', 'none'."""
    def __init__(self, method='tic'):
        self.method = method

    def fit(self, X, y=None):
        if self.method == 'tic':
            self.const_ = np.median(X.sum(axis=1, keepdims=True))
        elif self.method == 'median':
            self.const_ = np.median(np.median(X, axis=1, keepdims=True))
        elif self.method == 'pqn':
            rs  = X.sum(axis=1, keepdims=True); rs[rs == 0] = 1
            ref = np.median(X / rs, axis=0); ref[ref == 0] = 1
            self.ref_ = ref
        elif self.method == 'quantile':
            self.row_means_ = np.sort(X, axis=1).mean(axis=0)
        elif self.method == 'none':
            pass
        else:
            raise ValueError(f"Unknown normalization: '{self.method}'")
        return self

    def transform(self, X):
        X = X.astype(float)
        if self.method == 'tic':
            rs = X.sum(axis=1, keepdims=True); rs[rs == 0] = 1
            return X / rs * self.const_
        if self.method == 'median':
            rm = np.median(X, axis=1, keepdims=True); rm[rm == 0] = 1
            return X / rm * self.const_
        if self.method == 'pqn':
            rs = X.sum(axis=1, keepdims=True); rs[rs == 0] = 1
            Xt = X / rs
            q  = Xt / self.ref_
            d  = np.median(q, axis=1, keepdims=True); d[d == 0] = 1
            return Xt / d
        if self.method == 'quantile':
            ranks = np.argsort(np.argsort(X, axis=1), axis=1)
            return self.row_means_[ranks]
        return X


class LogTransform(BaseEstimator, TransformerMixin):
    """Transformation: 'glog', 'log10', 'log2', 'sqrt', 'none'.

    'glog' = arcsinh(X / lambda_) where lambda_ = 5th percentile of positive
    values in the TRAINING fold, fitted in fit() and reused in transform().
    This matches preprocess(log_transform='glog') when fit() and preprocess()
    see the same data (verified: max abs diff 0.0).
    """
    def __init__(self, method='log10'):
        self.method = method

    def fit(self, X, y=None):
        if self.method in ('log10', 'log2'):
            mp = X[X > 0].min() if (X > 0).any() else 1e-6
            self.half_ = mp / 2
        elif self.method == 'glog':
            pos = X[X > 0]
            lam = np.percentile(pos, 5) if len(pos) else 1.0
            self.lambda_ = max(float(lam), 1e-10)
        return self

    def transform(self, X):
        if self.method == 'glog':
            return np.arcsinh(X / self.lambda_)
        if self.method == 'log10':
            return np.log10(X + self.half_)
        if self.method == 'log2':
            return np.log2(X + self.half_)
        if self.method == 'sqrt':
            return np.sqrt(X)
        if self.method == 'none':
            return X
        raise ValueError(f"Unknown log_transform: '{self.method}'")


class Scaler(BaseEstimator, TransformerMixin):
    """Scaling: 'autoscale', 'pareto', 'range', 'vast', 'level', 'none'."""
    def __init__(self, method='autoscale'):
        self.method = method

    def fit(self, X, y=None):
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0, ddof=1)
        self.std_[self.std_ == 0] = 1
        if self.method == 'range':
            self.range_ = X.max(axis=0) - X.min(axis=0)
            self.range_[self.range_ == 0] = 1
        return self

    def transform(self, X):
        m = self.mean_
        if self.method == 'autoscale':
            return (X - m) / self.std_
        if self.method == 'pareto':
            return (X - m) / np.sqrt(self.std_)
        if self.method == 'range':
            return (X - m) / self.range_
        if self.method == 'vast':
            return ((X - m) / self.std_) * (m / (self.std_ + 1e-10))
        if self.method == 'level':
            lvl = np.abs(m); lvl[lvl == 0] = 1
            return (X - m) / lvl
        if self.method == 'none':
            return X - m
        raise ValueError(f"Unknown scaling: '{self.method}'")


# -- Preprocessor helper -------------------------------------------------------
# CV sample grouping lives in shared/grouping.py (directory-topology based) and
# is imported directly by the pipeline modules — it is intentionally not
# re-exported here.


def make_preprocessor(normalization='tic', log_transform='log10',
                      scaling='autoscale', variance_percentile=25,
                      abundance_percentile=5, prevalence_threshold=0.0,
                      prevalence_min_intensity=0.0,
                      snr_floor_enabled=False, snr_threshold=3,
                      noise_quantile=60, min_fraction_in_group=0.5):
    """
    Return the ordered list of (name, transformer) steps that reproduce the
    full preprocessing chain. Order:
    SNR floor -> prevalence filter -> variance filter -> abundance filter ->
    normalise -> transform -> scale.

    prevalence_threshold=0.0 by default (disabled) — the r_comparable pipeline
    does not apply a prevalence filter to match MetaboAnalyst behaviour.
    snr_floor_enabled=False by default — floor is off for r_comparable.
    """
    return [
        ('snrfloor',   SNRFloor(snr_threshold, noise_quantile,
                                min_fraction_in_group, snr_floor_enabled)),
        ('prevalence', PrevalenceFilter(prevalence_threshold,
                                        prevalence_min_intensity)),
        ('variance',   VarianceFilter(variance_percentile)),
        ('abundance',  AbundanceFilter(abundance_percentile)),
        ('normalize',  Normalizer(normalization)),
        ('logtrans',   LogTransform(log_transform)),
        ('scale',      Scaler(scaling)),
    ]


def auto_n_splits(y_labels, groups, desired=5):
    """
    Largest fold count compatible with the grouping. Mirrors
    classifier_comparison_standard.auto_n_splits.
    """
    y_labels = np.asarray(y_labels)
    groups   = np.asarray(groups)
    classes  = np.unique(y_labels)
    per_class = {c: len(set(groups[y_labels == c])) for c in classes}
    deficient = {c: n for c, n in per_class.items() if n < 2}
    if deficient:
        raise ValueError(
            "Grouped CV requires >= 2 biological groups per class; these "
            f"classes have fewer: {deficient}."
        )
    return int(max(2, min(desired, min(per_class.values()))))


# -- Cross-validation runners -------------------------------------------------

def _encode(y_labels):
    return LabelEncoder().fit_transform(y_labels)


def _run_grouped_cv(estimator, X_binned, y, groups, prep_steps, n_splits,
                    n_repeats=1, return_metrics=False):
    """Leak-free, group-aware CV with optional repeats and balanced accuracy.

    Returns (test_accs, train_accs) by default; with return_metrics=True returns
    a dict with pooled 'test_accuracy', 'train_accuracy', and
    'test_balanced_accuracy' arrays. Repeats re-shuffle the splitter (seed
    `_SEED + r`) and pool all fold scores to reduce small-design variance.
    """
    scoring = {'acc': 'accuracy', 'bal': 'balanced_accuracy'}
    test_acc, train_acc, test_bal = [], [], []
    for r in range(max(1, n_repeats)):
        steps = [(name, clone(t)) for name, t in prep_steps] + [('clf', clone(estimator))]
        pipe  = Pipeline(steps)
        sgkf  = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                     random_state=_SEED + r)
        with warnings.catch_warnings():
            # Small grouped folds in many-class designs can leave a class out of
            # a test fold; balanced_accuracy then warns "y_pred contains classes
            # not in y_true". The metric is still computed correctly, so the
            # cosmetic warning is suppressed here.
            warnings.filterwarnings(
                'ignore', message='.*y_pred contains classes not in y_true.*')
            res = cross_validate(pipe, X_binned, y, groups=groups, cv=sgkf,
                                 scoring=scoring, return_train_score=True)
        test_acc.append(res['test_acc'])
        train_acc.append(res['train_acc'])
        test_bal.append(res['test_bal'])
    test_acc  = np.concatenate(test_acc)
    train_acc = np.concatenate(train_acc)
    test_bal  = np.concatenate(test_bal)
    if return_metrics:
        return {'test_accuracy': test_acc, 'train_accuracy': train_acc,
                'test_balanced_accuracy': test_bal}
    return test_acc, train_acc


def _run_cv_legacy(model_fn, X, y, n_splits=5, random_state=_SEED):
    """Original ungrouped CV on an already-preprocessed X. Kept for back-compat."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    test_accs, train_accs = [], []
    for train_idx, test_idx in cv.split(X, y):
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        train_accs.append(accuracy_score(y[train_idx], model.predict(X[train_idx])))
        test_accs.append(accuracy_score(y[test_idx],  model.predict(X[test_idx])))
    return np.array(test_accs), np.array(train_accs)


def _dispatch(estimator, legacy_fn, X, y, n_splits, groups, prep_steps,
              n_repeats=1, return_metrics=False):
    if groups is not None and prep_steps is not None:
        return _run_grouped_cv(estimator, X, y, groups, prep_steps, n_splits,
                               n_repeats=n_repeats, return_metrics=return_metrics)
    warnings.warn(
        "Running LEGACY ungrouped CV on pre-preprocessed X. This reintroduces "
        "pseudoreplication and preprocessing leakage. Pass groups= and "
        "prep_steps= (with the binned matrix as X) for corrected estimates.",
        stacklevel=2)
    return _run_cv_legacy(legacy_fn, X, y, n_splits)


# -- Individual classifiers ---------------------------------------------------
# New signature: pass the BINNED matrix as X, plus groups= and prep_steps=.
# `random_forest` (lowercase) is the canonical name; `RandomForest` is kept
# as an alias for backward compatibility.

def random_forest(X, y_labels, n_splits=3, groups=None, prep_steps=None,
                  random_state=_SEED, n_repeats=1, return_metrics=False,
                  n_biological=None):
    y  = _encode(y_labels)
    g  = small_sample_guards(n_biological)['rf']
    mk = lambda: RandomForestClassifier(n_estimators=100,
                                        random_state=random_state, **g)
    return _dispatch(mk(), mk, X, y, n_splits, groups, prep_steps,
                     n_repeats=n_repeats, return_metrics=return_metrics)

# Backward-compatible alias
RandomForest = random_forest


def svm_classify(X, y_labels, n_splits=3, groups=None, prep_steps=None,
                 random_state=_SEED, n_repeats=1, return_metrics=False,
                 n_biological=None):
    y  = _encode(y_labels)
    g  = small_sample_guards(n_biological)['svm']
    mk = lambda: SVC(kernel='linear', random_state=random_state, **g)
    return _dispatch(mk(), mk, X, y, n_splits, groups, prep_steps,
                     n_repeats=n_repeats, return_metrics=return_metrics)


def gradient_boosting(X, y_labels, n_splits=3, groups=None, prep_steps=None,
                      random_state=_SEED, n_repeats=1, return_metrics=False,
                      n_biological=None):
    y    = _encode(y_labels)
    g    = small_sample_guards(n_biological)['gb']
    base = dict(n_estimators=100, learning_rate=0.1, max_depth=3)
    base.update(g)
    mk   = lambda: GradientBoostingClassifier(random_state=random_state, **base)
    return _dispatch(mk(), mk, X, y, n_splits, groups, prep_steps,
                     n_repeats=n_repeats, return_metrics=return_metrics)


def logistic_regression(X, y_labels, n_splits=3, groups=None, prep_steps=None,
                        random_state=_SEED, n_repeats=1, return_metrics=False,
                        n_biological=None):
    y  = _encode(y_labels)
    g  = small_sample_guards(n_biological)['logreg']
    mk = lambda: LogisticRegression(max_iter=1000, random_state=random_state, **g)
    return _dispatch(mk(), mk, X, y, n_splits, groups, prep_steps,
                     n_repeats=n_repeats, return_metrics=return_metrics)


def lda_classify(X, y_labels, n_splits=3, groups=None, prep_steps=None,
                 random_state=_SEED, n_repeats=1, return_metrics=False,
                 n_biological=None):
    y  = _encode(y_labels)
    mk = lambda: LinearDiscriminantAnalysis()
    return _dispatch(mk(), mk, X, y, n_splits, groups, prep_steps,
                     n_repeats=n_repeats, return_metrics=return_metrics)


def ridge_classify(X, y_labels, n_splits=3, groups=None, prep_steps=None,
                   random_state=_SEED, n_repeats=1, return_metrics=False,
                   n_biological=None):
    y  = _encode(y_labels)
    g  = small_sample_guards(n_biological)['ridge']
    mk = lambda: RidgeClassifier(**g)
    return _dispatch(mk(), mk, X, y, n_splits, groups, prep_steps,
                     n_repeats=n_repeats, return_metrics=return_metrics)


# -- Plotting -----------------------------------------------------------------

def plot_accuracy_comparison(results, experiment_name, out_path, chance=None):
    """
    results: dict {model_name: (test_accs, train_accs)}
    chance : optional float; draws the chance line at 1/n_classes.
    """
    names = list(results.keys())
    n     = len(names)
    palette = sns.color_palette('colorblind', n_colors=n)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(max(10, n * 1.4), 9),
        gridspec_kw={'height_ratios': [2, 1]},
        dpi=300,
    )

    for i, (name, (test_accs, train_accs)) in enumerate(results.items()):
        color  = palette[i]
        ax_top.bar(i, test_accs.mean(), color=color, alpha=0.5, width=0.6, zorder=1)
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(test_accs))
        ax_top.scatter(np.full(len(test_accs), i) + jitter, test_accs,
                       color=color, edgecolors='black', linewidths=0.5, s=50, zorder=2)
        ax_top.hlines(test_accs.mean(), i - 0.3, i + 0.3,
                      colors='black', linewidths=1.5, zorder=3)

    ax_top.set_xticks(range(n))
    ax_top.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax_top.set_ylabel('Test Accuracy (grouped CV)')
    ax_top.set_ylim(0, 1.05)
    ax_top.set_title(f'Classifier Comparison -- {experiment_name}')
    if chance is not None:
        ax_top.axhline(chance, color='grey', linestyle='--', linewidth=0.8, alpha=0.6,
                       label=f'Chance ({chance:.3f})')
        ax_top.legend(fontsize=8)

    x     = np.arange(n)
    width = 0.35
    train_means = [results[name][1].mean() for name in names]
    test_means  = [results[name][0].mean() for name in names]
    cb = sns.color_palette('colorblind')
    ax_bot.bar(x - width / 2, train_means, width, label='Train', color=cb[0], alpha=0.7)
    ax_bot.bar(x + width / 2, test_means,  width, label='Test',  color=cb[1], alpha=0.7)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax_bot.set_ylabel('Mean Accuracy')
    ax_bot.set_ylim(0, 1.05)
    ax_bot.set_title('Train vs Test Accuracy (overfitting check)')
    ax_bot.legend(fontsize=8)

    plt.tight_layout()
    pub_savefig(out_path)


# -- Feature importance overlap (DESCRIPTIVE — full-data fit is correct here) --

def feature_importance_analysis(X, y_labels, mz, safe_name, out_dir,
                                top_n=50, X_norm=None, log_transform='log10',
                                X_filt_raw=None, groups=None,
                                normalization='tic', scaling='autoscale',
                                univariate_test='auto',
                                compute_stability=False, n_boot=200,
                                compute_overlap_null=False, n_overlap_perm=200,
                                random_state=_SEED):
    """
    Final ensemble feature ranking. Fits RF, SVM, GB, LR, Ridge and PLS-DA VIP
    on the FULL preprocessed dataset and reports features appearing in the top
    `top_n` of >= 2 of the 6 methods.

    This is the reported candidate list, not a generalisation estimate, so it is
    fit on all data by design. Univariate fold-change/p/BH-q, colony-bootstrap
    selection stability, and a label-permutation overlap null are added as extra
    columns (see _attach_feature_statistics).
    """
    le      = LabelEncoder()
    y       = le.fit_transform(y_labels)
    classes = le.classes_

    rf = RandomForestClassifier(n_estimators=100, random_state=_SEED); rf.fit(X, y)
    rf_imp = rf.feature_importances_

    svm = SVC(kernel='linear', random_state=_SEED); svm.fit(X, y)
    svm_imp = (np.abs(svm.coef_).mean(axis=0) if svm.coef_.ndim > 1
               else np.abs(svm.coef_).ravel())

    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                    max_depth=3, random_state=_SEED); gb.fit(X, y)
    gb_imp = gb.feature_importances_

    lr = LogisticRegression(max_iter=1000, random_state=_SEED); lr.fit(X, y)
    lr_imp = (np.abs(lr.coef_).mean(axis=0) if lr.coef_.ndim > 1
              else np.abs(lr.coef_).ravel())

    ridge = RidgeClassifier(); ridge.fit(X, y)
    ridge_imp = (np.abs(ridge.coef_).mean(axis=0) if ridge.coef_.ndim > 1
                 else np.abs(ridge.coef_).ravel())

    vip_imp = compute_vip_1comp(X, y_labels)

    # Consensus uses only the ENABLED methods (USE_* flags), so n_methods and the
    # observed overlap share a denominator with the permutation null. All six
    # importance columns are still written below for reference.
    from src.shared.feature_stats import enabled_methods_from_config
    try:
        import config as _cfg_fi
    except Exception:
        _cfg_fi = None
    _enabled = enabled_methods_from_config(_cfg_fi)
    _imp_by_method = {'rf': rf_imp, 'svm': svm_imp, 'gb': gb_imp,
                      'lr': lr_imp, 'ridge': ridge_imp, 'vip': vip_imp}

    # -- B1: method x method Spearman importance-correlation matrix ---------------
    # Saved next to the candidate table for a supplementary redundancy heat map.
    from src.shared.feature_stats import importance_correlation_matrix
    _active = [m for m in ['rf', 'svm', 'gb', 'lr', 'ridge', 'vip'] if m in _enabled]
    _corr_df = importance_correlation_matrix(
        {m: _imp_by_method[m] for m in _active}, method='spearman')
    _corr_path = os.path.join(out_dir, f'model_importance_correlation_{safe_name}.csv')
    _corr_df.to_csv(_corr_path, encoding='utf-8')
    print(f"  Saved model-importance Spearman matrix -> {_corr_path}")

    _tops_by_method = {m: set(np.argsort(im)[::-1][:top_n])
                       for m, im in _imp_by_method.items()}
    tops = [_tops_by_method[m] for m in ['rf', 'svm', 'gb', 'lr', 'ridge', 'vip']]
    counts = Counter(idx for m in _enabled for idx in _tops_by_method[m])

    overlap_2plus = {idx for idx, c in counts.items() if c >= 2}
    overlap_3plus = {idx for idx, c in counts.items() if c >= 3}
    print(f"  Features in at least 2 of 6 methods: {len(overlap_2plus)}")
    print(f"  Features in at least 3 of 6 methods: {len(overlap_3plus)}")

    overlap_list = sorted(overlap_2plus)
    method_names = ['rf', 'svm', 'gb', 'lr', 'ridge', 'vip']
    membership   = {name: np.array([idx in top for idx in overlap_list], dtype=bool)
                    for name, top in zip(method_names, tops)}

    overlap_df = pd.DataFrame({
        'mz':               mz[overlap_list],
        'rf_importance':    rf_imp[overlap_list],
        'svm_importance':   svm_imp[overlap_list],
        'gb_importance':    gb_imp[overlap_list],
        'lr_importance':    lr_imp[overlap_list],
        'ridge_importance': ridge_imp[overlap_list],
        'vip_score':        vip_imp[overlap_list],
        'n_methods':        [counts[idx] for idx in overlap_list],
        'in_rf_top':        membership['rf'],
        'in_svm_top':       membership['svm'],
        'in_gb_top':        membership['gb'],
        'in_lr_top':        membership['lr'],
        'in_ridge_top':     membership['ridge'],
        'in_vip_top':       membership['vip'],
    })

    print(f"  Adding per-group attribution: mean intensity + Ridge one-vs-rest")
    if ridge.coef_.ndim == 1:
        ridge_signed = np.vstack([-ridge.coef_, ridge.coef_])
    elif ridge.coef_.shape[0] == 1:
        ridge_signed = np.vstack([-ridge.coef_[0], ridge.coef_[0]])
    else:
        ridge_signed = ridge.coef_

    X_for_means = X_norm if X_norm is not None else X
    if X_norm is None:
        print("  WARNING: X_norm not provided -- mean_<group> columns will be on "
              "the scaled axis, not raw log-intensity")

    for j, group in enumerate(classes):
        mask = (y_labels == group)
        means_full = X_for_means[mask].mean(axis=0)
        overlap_df[f'mean_{group}']  = means_full[overlap_list]
        overlap_df[f'ridge_{group}'] = ridge_signed[j, overlap_list]

    mean_cols  = [f'mean_{g}'  for g in classes]
    ridge_cols = [f'ridge_{g}' for g in classes]

    mean_arr = overlap_df[mean_cols].values
    overlap_df['top_condition_mean'] = [classes[i] for i in np.argmax(mean_arr, axis=1)]

    sorted_means = np.sort(mean_arr, axis=1)[:, ::-1]
    top1 = sorted_means[:, 0]
    top2 = sorted_means[:, 1] if sorted_means.shape[1] > 1 else top1

    if log_transform == 'log10':
        overlap_df['mean_margin'] = 10.0 ** (top1 - top2)
    elif log_transform == 'log2':
        overlap_df['mean_margin'] = 2.0 ** (top1 - top2)
    elif log_transform == 'sqrt':
        lin1 = top1 ** 2; lin2 = top2 ** 2
        overlap_df['mean_margin'] = np.where(lin2 > 0, lin1 / lin2, np.nan)
    elif log_transform == 'glog':
        s1 = np.sinh(top1); s2 = np.sinh(top2)
        overlap_df['mean_margin'] = np.where(np.abs(s2) > 1e-9, s1 / s2, np.nan)
    else:
        overlap_df['mean_margin'] = np.where(top2 > 0, top1 / top2, np.nan)

    ridge_arr = overlap_df[ridge_cols].values
    overlap_df['top_condition_ridge'] = [classes[i] for i in np.argmax(ridge_arr, axis=1)]
    n_pos = (ridge_arr > 0).sum(axis=1)
    n_neg = (ridge_arr < 0).sum(axis=1)
    direction = []
    for p, nn in zip(n_pos, n_neg):
        if p == 1 and nn >= 1:    direction.append('elevated')
        elif nn == 1 and p >= 1:  direction.append('suppressed')
        else:                     direction.append('mixed')
    overlap_df['ridge_direction'] = direction

    # -- Statistical-rigor columns (shared helper) --------------------------------
    from src.shared.classifier_comparison_standard import _attach_feature_statistics
    _attach_feature_statistics(
        overlap_df, overlap_list, counts, X, X_norm, y_labels, mz,
        top_n=top_n, log_transform=log_transform, univariate_test=univariate_test,
        X_filt_raw=X_filt_raw, groups=groups, normalization=normalization,
        scaling=scaling, compute_stability=compute_stability, n_boot=n_boot,
        compute_overlap_null=compute_overlap_null, n_overlap_perm=n_overlap_perm,
        random_state=random_state,
    )

    if 'q_value_BH' in overlap_df.columns:
        overlap_df = overlap_df.sort_values(['q_value_BH', 'n_methods'],
                                            ascending=[True, False])
    else:
        overlap_df = overlap_df.sort_values('n_methods', ascending=False)
    csv_path   = os.path.join(out_dir, f'feature_overlap_{safe_name}.csv')
    overlap_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"  Saved -> {csv_path}")
    return overlap_df, counts
