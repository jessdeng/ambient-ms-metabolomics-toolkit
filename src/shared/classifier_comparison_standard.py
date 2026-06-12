"""
Classifier Comparison for Mass Spectrometry Metabolomics Data (Standard pipeline)
==================================================================================
Trains and evaluates 6 supervised classifiers using leave-one-biological-replicate-out
cross-validation (StratifiedGroupKFold), with all preprocessing fit inside each fold
to eliminate data leakage.

Methodological design
---------------------
(1) Pseudoreplication fix.  Technical replicates (T1/T2/T3) of the same colony are
    NOT independent observations. Cross-validation uses StratifiedGroupKFold with the
    colony (condition × well) as the grouping unit, so a colony's replicates are always
    on the same side of every fold boundary. This gives a leave-one-biological-replicate-
    out estimate of generalisation rather than an inflated within-replicate estimate.

(2) Preprocessing leakage fix.  Variance/abundance filtering, normalisation,
    log-transformation and scaling are encapsulated in an sklearn Pipeline fitted
    exclusively on the training fold inside each CV iteration. Test-fold data never
    influences feature selection or scaling parameters.

Note on feature_importance_analysis(): the ensemble feature ranking is a descriptive
model fit on all data (the final candidate list), not a generalisation estimate, so
full-data preprocessing is intentional and correct there.

Classifiers used for accuracy evaluation:
    Random Forest, SVM (linear), Gradient Boosting, Logistic Regression, LDA, Ridge

Classifiers used for ensemble feature importance (n_methods count):
    Random Forest, SVM, Gradient Boosting, Logistic Regression, Ridge, PLS-DA VIP
    (LDA excluded — no comparable feature-level importance measure)

Usage:
    python -m standard.run_analysis
"""

import os
import sys
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from collections import Counter

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_validate
from sklearn.metrics import accuracy_score

from standard.pipeline import compute_vip_1comp


# ── Preprocessing transformers (train-fit; mirror preprocessing.preprocess) ────
# Each transformer learns its parameters on .fit() (training fold only) and
# applies them on .transform(). Fitting any of these on the full dataset
# reproduces standard/preprocessing.py exactly (verified: max abs diff 0.0).

class VarianceFilter(BaseEstimator, TransformerMixin):
    """Remove features with low relative standard deviation (RSD)."""
    def __init__(self, percentile=25):
        self.percentile = percentile
    def fit(self, X, y=None):
        if self.percentile <= 0:
            self.keep_ = np.ones(X.shape[1], dtype=bool)
            return self
        mean = X.mean(axis=0).copy()
        mean[mean == 0] = 1e-12          # guard; train fold may zero a feature
        rsd = X.std(axis=0) / mean
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
            rs = X.sum(axis=1, keepdims=True); rs[rs == 0] = 1
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
            q = Xt / self.ref_
            d = np.median(q, axis=1, keepdims=True); d[d == 0] = 1
            return Xt / d
        if self.method == 'quantile':
            ranks = np.argsort(np.argsort(X, axis=1), axis=1)
            return self.row_means_[ranks]
        return X


class LogTransform(BaseEstimator, TransformerMixin):
    """Transformation: 'log10', 'log2', 'sqrt', 'none'."""
    def __init__(self, method='log10'):
        self.method = method
    def fit(self, X, y=None):
        mp = X[X > 0].min() if (X > 0).any() else 1e-6
        self.half_ = mp / 2
        return self
    def transform(self, X):
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
        self.std_ = X.std(axis=0, ddof=1)
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


# ── Grouping + preprocessor helpers ────────────────────────────────────────────

def make_groups(y_labels, names):
    """
    Build a cross-validation group label per sample so that technical
    replicates of one colony share a group and never split across folds.

    group = '<condition>::<well>'  e.g. 'ConditionA::W1'

    Parsed from the filename token immediately before T<n> (e.g. W1T2 -> W1).
    Falls back to the filename if the pattern is absent (then each file is its
    own group, which simply disables grouping for that sample).
    """
    groups = []
    for lab, nm in zip(y_labels, names):
        m = re.search(r'([A-Za-z]\d+)[Tt]\d+\.(?:csv|txt)$', str(nm))
        groups.append(f"{lab}::{m.group(1)}" if m else f"{lab}::{nm}")
    return np.array(groups)


def make_preprocessor(normalization='tic', log_transform='log10',
                      scaling='autoscale', variance_percentile=25,
                      abundance_percentile=5):
    """
    Return the ordered list of (name, transformer) steps that reproduce the
    original preprocessing chain. Order matches standard/run_analysis.py:
    variance filter -> abundance filter -> normalise -> transform -> scale.
    """
    return [
        ('variance',  VarianceFilter(variance_percentile)),
        ('abundance', AbundanceFilter(abundance_percentile)),
        ('normalize', Normalizer(normalization)),
        ('logtrans',  LogTransform(log_transform)),
        ('scale',     Scaler(scaling)),
    ]


def auto_n_splits(y_labels, groups, desired=5):
    """
    Largest fold count compatible with the grouping. With g biological
    replicates per class, StratifiedGroupKFold needs n_splits <= g; with
    3 replicates this returns 3 (leave-one-replicate-out).
    """
    y_labels = np.asarray(y_labels)
    per_class = [len(set(groups[y_labels == c])) for c in np.unique(y_labels)]
    return int(max(2, min(desired, min(per_class))))


# ── Cross-validation runners ───────────────────────────────────────────────────

def _encode(y_labels):
    return LabelEncoder().fit_transform(y_labels)


def _run_grouped_cv(estimator, X_binned, y, groups, prep_steps, n_splits):
    """
    Leak-free, group-aware CV. `X_binned` is the binned matrix BEFORE any
    filtering/normalisation/scaling; the preprocessor is cloned and fit inside
    each fold. Returns (test_accs, train_accs).
    """
    steps = [(name, clone(t)) for name, t in prep_steps] + [('clf', clone(estimator))]
    pipe = Pipeline(steps)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED)
    res = cross_validate(pipe, X_binned, y, groups=groups, cv=sgkf,
                         scoring='accuracy', return_train_score=True)
    return res['test_score'], res['train_score']


def _run_cv_legacy(model_fn, X, y, n_splits=5, random_state=config.RANDOM_SEED):
    """Original ungrouped CV on an already-preprocessed X. Leaky — kept only
    for backward compatibility with callers that have not been updated."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    test_accs, train_accs = [], []
    for train_idx, test_idx in cv.split(X, y):
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        train_accs.append(accuracy_score(y[train_idx], model.predict(X[train_idx])))
        test_accs.append(accuracy_score(y[test_idx], model.predict(X[test_idx])))
    return np.array(test_accs), np.array(train_accs)


def _dispatch(estimator, legacy_fn, X, y, n_splits, groups, prep_steps):
    """Use corrected grouped/leak-free CV when groups+prep_steps are supplied;
    otherwise fall back to the legacy path with a warning."""
    if groups is not None and prep_steps is not None:
        return _run_grouped_cv(estimator, X, y, groups, prep_steps, n_splits)
    warnings.warn(
        "Running LEGACY ungrouped CV on pre-preprocessed X. This reintroduces "
        "pseudoreplication and preprocessing leakage. Pass groups= and "
        "prep_steps= (with the binned matrix as X) for corrected estimates.",
        stacklevel=2)
    return _run_cv_legacy(legacy_fn, X, y, n_splits)


# ── Individual classifiers ──────────────────────────────────────────────────────
# New signature: pass the BINNED matrix as X, plus groups= and prep_steps=.

def random_forest(X, y_labels, n_splits=3, groups=None, prep_steps=None, random_state=config.RANDOM_SEED):
    y = _encode(y_labels)
    est = RandomForestClassifier(n_estimators=100, random_state=random_state)
    return _dispatch(est, lambda: RandomForestClassifier(n_estimators=100, random_state=random_state),
                     X, y, n_splits, groups, prep_steps)


def svm_classify(X, y_labels, n_splits=3, groups=None, prep_steps=None, random_state=config.RANDOM_SEED):
    y = _encode(y_labels)
    est = SVC(kernel='linear', random_state=random_state)
    return _dispatch(est, lambda: SVC(kernel='linear', random_state=random_state),
                     X, y, n_splits, groups, prep_steps)


def gradient_boosting(X, y_labels, n_splits=3, groups=None, prep_steps=None, random_state=config.RANDOM_SEED):
    y = _encode(y_labels)
    est = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                     max_depth=3, random_state=random_state)
    return _dispatch(est, lambda: GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                     max_depth=3, random_state=random_state),
                     X, y, n_splits, groups, prep_steps)


def logistic_regression(X, y_labels, n_splits=3, groups=None, prep_steps=None, random_state=config.RANDOM_SEED):
    y = _encode(y_labels)
    est = LogisticRegression(max_iter=1000, random_state=random_state)
    return _dispatch(est, lambda: LogisticRegression(max_iter=1000, random_state=random_state),
                     X, y, n_splits, groups, prep_steps)


def lda_classify(X, y_labels, n_splits=3, groups=None, prep_steps=None, random_state=config.RANDOM_SEED):
    y = _encode(y_labels)
    est = LinearDiscriminantAnalysis()
    return _dispatch(est, lambda: LinearDiscriminantAnalysis(),
                     X, y, n_splits, groups, prep_steps)


def ridge_classify(X, y_labels, n_splits=3, groups=None, prep_steps=None, random_state=config.RANDOM_SEED):
    y = _encode(y_labels)
    est = RidgeClassifier()
    return _dispatch(est, lambda: RidgeClassifier(),
                     X, y, n_splits, groups, prep_steps)


# ── Plotting ────────────────────────────────────────────────────────────────────

def plot_accuracy_comparison(results, experiment_name, out_path, chance=None):
    """
    results: dict {model_name: (test_accs, train_accs)}
    chance : optional float; draws the chance line at 1/n_classes (pass
             1/len(classes)) instead of the old hard-coded 0.5.
    """
    names = list(results.keys())
    n = len(names)
    palette = sns.color_palette('colorblind', n_colors=n)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(max(10, n * 1.4), 10),
        gridspec_kw={'height_ratios': [2, 1]}
    )

    for i, (name, (test_accs, train_accs)) in enumerate(results.items()):
        color = palette[i]
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
    ax_top.set_title(f'Classifier Comparison — {experiment_name}')
    if chance is not None:
        ax_top.axhline(chance, color='grey', linestyle='--', linewidth=0.8, alpha=0.6,
                       label=f'Chance ({chance:.3f})')
        ax_top.legend(fontsize=8)

    x = np.arange(n)
    width = 0.35
    train_means = [results[name][1].mean() for name in names]
    test_means = [results[name][0].mean() for name in names]
    cb = sns.color_palette('colorblind')
    ax_bot.bar(x - width / 2, train_means, width, label='Train', color=cb[0], alpha=0.7)
    ax_bot.bar(x + width / 2, test_means, width, label='Test', color=cb[1], alpha=0.7)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax_bot.set_ylabel('Mean Accuracy')
    ax_bot.set_ylim(0, 1.05)
    ax_bot.set_title('Train vs Test Accuracy (overfitting check)')
    ax_bot.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {out_path}")


# ── Feature importance overlap (DESCRIPTIVE — full-data fit is correct here) ─────

def feature_importance_analysis(X, y_labels, mz, safe_name, out_dir,
                                top_n=50, X_norm=None, log_transform='log10'):
    """
    Final ensemble feature ranking. Fits RF, SVM, GB, LR, Ridge and PLS-DA VIP
    on the FULL preprocessed dataset and reports features appearing in the top
    `top_n` of >=2 of the 6 methods.

    This is the reported candidate list, not a generalisation estimate, so it is
    fit on all data by design. Unchanged from the original implementation other
    than this note. `X` here is the fully preprocessed matrix (as before).
    """
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    classes = le.classes_

    rf = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_SEED); rf.fit(X, y)
    rf_imp = rf.feature_importances_

    svm = SVC(kernel='linear', random_state=config.RANDOM_SEED); svm.fit(X, y)
    svm_imp = np.abs(svm.coef_).mean(axis=0) if svm.coef_.ndim > 1 else np.abs(svm.coef_).ravel()

    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                    max_depth=3, random_state=config.RANDOM_SEED); gb.fit(X, y)
    gb_imp = gb.feature_importances_

    lr = LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED); lr.fit(X, y)
    lr_imp = np.abs(lr.coef_).mean(axis=0) if lr.coef_.ndim > 1 else np.abs(lr.coef_).ravel()

    ridge = RidgeClassifier(); ridge.fit(X, y)
    ridge_imp = np.abs(ridge.coef_).mean(axis=0) if ridge.coef_.ndim > 1 else np.abs(ridge.coef_).ravel()

    vip_imp = compute_vip_1comp(X, y_labels)

    tops = [set(np.argsort(imp)[::-1][:top_n])
            for imp in [rf_imp, svm_imp, gb_imp, lr_imp, ridge_imp, vip_imp]]
    counts = Counter(idx for top in tops for idx in top)

    overlap_2plus = {idx for idx, c in counts.items() if c >= 2}
    overlap_3plus = {idx for idx, c in counts.items() if c >= 3}
    print(f"  Features in at least 2 of 6 methods: {len(overlap_2plus)}")
    print(f"  Features in at least 3 of 6 methods: {len(overlap_3plus)}")

    overlap_list = sorted(overlap_2plus)
    method_names = ['rf', 'svm', 'gb', 'lr', 'ridge', 'vip']
    membership = {name: np.array([idx in top for idx in overlap_list], dtype=bool)
                  for name, top in zip(method_names, tops)}

    overlap_df = pd.DataFrame({
        'mz': mz[overlap_list],
        'rf_importance': rf_imp[overlap_list],
        'svm_importance': svm_imp[overlap_list],
        'gb_importance': gb_imp[overlap_list],
        'lr_importance': lr_imp[overlap_list],
        'ridge_importance': ridge_imp[overlap_list],
        'vip_score': vip_imp[overlap_list],
        'n_methods': [counts[idx] for idx in overlap_list],
        'in_rf_top': membership['rf'], 'in_svm_top': membership['svm'],
        'in_gb_top': membership['gb'], 'in_lr_top': membership['lr'],
        'in_ridge_top': membership['ridge'], 'in_vip_top': membership['vip'],
    })

    print(f"  Adding per-group attribution: mean intensity + Ridge one-vs-rest")
    if ridge.coef_.shape[0] == 1 and len(classes) == 2:
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
        overlap_df[f'mean_{group}'] = means_full[overlap_list]
        overlap_df[f'ridge_{group}'] = ridge_signed[j, overlap_list]

    mean_cols = [f'mean_{g}' for g in classes]
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
    else:
        overlap_df['mean_margin'] = np.where(top2 > 0, top1 / top2, np.nan)

    ridge_arr = overlap_df[ridge_cols].values
    overlap_df['top_condition_ridge'] = [classes[i] for i in np.argmax(ridge_arr, axis=1)]
    n_pos = (ridge_arr > 0).sum(axis=1); n_neg = (ridge_arr < 0).sum(axis=1)
    direction = []
    for p, nn in zip(n_pos, n_neg):
        if p == 1 and nn >= 1:   direction.append('elevated')
        elif nn == 1 and p >= 1: direction.append('suppressed')
        else:                    direction.append('mixed')
    overlap_df['ridge_direction'] = direction

    overlap_df = overlap_df.sort_values('n_methods', ascending=False)
    csv_path = os.path.join(out_dir, f'feature_overlap_{safe_name}.csv')
    overlap_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"  Saved -> {csv_path}")
    return overlap_df, counts
