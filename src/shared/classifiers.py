"""
Classifier comparison: how well does each model tell the conditions apart?

Six models with fixed, default settings are evaluated with folds that hold out
whole colonies and keep every condition in every test fold (RepeatedColonyKFold
in shared/validation.py), repeated with different splits. Inside
every fold the label-free filters, the log offset and the scaling are fitted on
the training spectra only; TIC normalisation is per spectrum and needs no fit.

Read the results as condition-plate discriminability: each condition was grown
on one plate, so accuracy cannot separate a light effect from a plate effect.
With ~2,700 bins and ~30 training spectra most models fit the training data
perfectly, so the train-vs-test gap shows how much is lost on unseen colonies.
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.svm import SVC

from standard.preprocessing import FoldPreprocessor
from shared.plot_style import apply_style, pub_savefig

apply_style()

MODELS = {
    'Random Forest': lambda s: RandomForestClassifier(n_estimators=100, random_state=s),
    'SVM': lambda s: SVC(kernel='linear', random_state=s),
    'Gradient Boosting': lambda s: GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                              max_depth=3, random_state=s),
    'Logistic Regression': lambda s: LogisticRegression(max_iter=1000, random_state=s),
    'LDA': lambda s: LinearDiscriminantAnalysis(),
    'Ridge': lambda s: RidgeClassifier(),
}


def _fold_matrices(X_tic, tr, te, scaling, variance_percentile, abundance_percentile,
                   log_offset='global'):
    """Filters, log10 offset and scaling fitted on the training rows only."""
    prep = FoldPreprocessor(variance_percentile, abundance_percentile, scaling,
                            log_offset).fit(X_tic[tr])
    return prep.transform(X_tic[tr]), prep.transform(X_tic[te])


def compare_classifiers(X_tic, condition, colony_id, n_splits=5, n_repeats=10, seed=42,
                        scaling='autoscale', variance_percentile=25, abundance_percentile=5,
                        progress=None, log_offset='global'):
    """Colony-grouped, repeated CV of the six models on TIC-normalised spectra.

    ``n_splits`` is reduced to the smallest number of colonies in any condition,
    so every test fold holds at least one whole colony of every condition.
    ``progress``, if given, is called as progress(repeat, n_repeats) after each
    repeat. Returns (per_fold DataFrame, summary DataFrame).
    """
    from shared.validation import RepeatedColonyKFold    # validation imports this module

    y, g = np.asarray(condition), np.asarray(colony_id)
    min_col = min(len(set(g[y == c])) for c in np.unique(y))
    k = int(max(2, min(n_splits, min_col)))
    cv = RepeatedColonyKFold(n_splits=k, n_repeats=n_repeats, random_state=seed)
    rows = []
    for i, (tr, te) in enumerate(cv.split(X_tic, y, g)):
        r, f = divmod(i, k)
        assert not set(g[tr]) & set(g[te]), "a colony is in both training and test"
        Xtr, Xte = _fold_matrices(X_tic, tr, te, scaling, variance_percentile, abundance_percentile,
                                  log_offset)
        for name, make in MODELS.items():
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Variables are collinear')  # LDA, p >> n
                m = make(seed).fit(Xtr, y[tr])
            pred = m.predict(Xte)
            rows.append(dict(model=name, repeat=r, fold=f,
                             test_accuracy=accuracy_score(y[te], pred),
                             test_balanced_accuracy=balanced_accuracy_score(y[te], pred),
                             train_accuracy=accuracy_score(y[tr], m.predict(Xtr)),
                             test_colonies=len(set(g[te]))))
        if progress is not None and f == k - 1:
            progress(r + 1, n_repeats)
    folds = pd.DataFrame(rows)
    summary = (folds.groupby('model', sort=False)
               .agg(test_accuracy_mean=('test_accuracy', 'mean'), test_accuracy_sd=('test_accuracy', 'std'),
                    balanced_accuracy_mean=('test_balanced_accuracy', 'mean'),
                    train_accuracy_mean=('train_accuracy', 'mean'), n_folds=('test_accuracy', 'size'))
               .reset_index())
    summary['cv'] = f'{k}-fold colony-grouped x {n_repeats}'
    summary['chance_level'] = 1.0 / len(np.unique(y))
    summary['n_spectra'], summary['n_colonies'] = len(y), len(set(g))
    return folds, summary


def plot_accuracy_comparison(folds, experiment_name, out_path, chance=None):
    """Top: test accuracy per fold (dots) and mean (bar + line) for each model.
    Bottom: mean train vs mean test accuracy."""
    names = list(dict.fromkeys(folds.model))
    n = len(names)
    palette = sns.color_palette('colorblind', n_colors=n)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(max(10, n * 1.4), 9),
                                         gridspec_kw={'height_ratios': [2, 1]}, dpi=300)
    rng = np.random.default_rng(42)
    for i, name in enumerate(names):
        test = folds.loc[folds.model == name, 'test_accuracy'].to_numpy()
        ax_top.bar(i, test.mean(), color=palette[i], alpha=0.5, width=0.6, zorder=1)
        ax_top.scatter(i + rng.uniform(-0.15, 0.15, test.size), test, color=palette[i],
                       edgecolors='black', linewidths=0.5, s=50, zorder=2)
        ax_top.hlines(test.mean(), i - 0.3, i + 0.3, colors='black', linewidths=1.5, zorder=3)
    ax_top.set_xticks(range(n)); ax_top.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax_top.set_ylabel('Test Accuracy (grouped CV)'); ax_top.set_ylim(0, 1.05)
    ax_top.set_title(f'Classifier Comparison -- {experiment_name}')
    if chance is not None:
        ax_top.axhline(chance, color='grey', linestyle='--', linewidth=0.8, alpha=0.6,
                       label=f'Chance ({chance:.3f})')
        ax_top.legend(fontsize=8)

    x, w = np.arange(n), 0.35
    means = folds.groupby('model', sort=False)[['train_accuracy', 'test_accuracy']].mean().loc[names]
    cb = sns.color_palette('colorblind')
    ax_bot.bar(x - w / 2, means.train_accuracy, w, label='Train', color=cb[0], alpha=0.7)
    ax_bot.bar(x + w / 2, means.test_accuracy, w, label='Test', color=cb[1], alpha=0.7)
    ax_bot.set_xticks(x); ax_bot.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax_bot.set_ylabel('Mean Accuracy'); ax_bot.set_ylim(0, 1.05)
    ax_bot.set_title('Train vs Test Accuracy (overfitting check)')
    ax_bot.legend(fontsize=8)
    plt.tight_layout()
    pub_savefig(out_path)
