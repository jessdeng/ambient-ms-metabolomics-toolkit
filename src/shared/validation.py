"""
Validation helpers: balanced colony folds, colony-level accuracy, permutation test.

These complement the classifier comparison (shared/classifiers.py) and are not
run by default.

RepeatedColonyKFold      Folds that hold out whole colonies and keep every
                         condition in every test fold. StratifiedGroupKFold with
                         shuffling does not guarantee the second property: with
                         three colonies per condition a test fold can hold two
                         colonies of one condition and none of another.
grouped_cv               Out-of-fold predictions for one model, scored per repeat
                         on spectra (accuracy, balanced accuracy) and on colonies
                         (majority vote of a colony's spectra). Filters, log
                         offset and scaling are refitted on the training colonies
                         in every fold.
colony_permutation_test  Could the cross-validated accuracy arise with no link
                         between spectra and condition? Labels are shuffled among
                         colonies (a colony's spectra keep one label) and the
                         whole procedure is repeated for every permutation.
PLSDAClassifier          PLS-DA as a classifier, so the model behind VIP can be
                         scored in the same way as the others.

Condition and plate are aliased, so a small p-value shows that the
condition-plates can be told apart; it is not a test of a light effect.
"""

import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted

from shared.classifiers import _fold_matrices
from standard.pipeline import VALIDATED_PLS_MAX_ITER, VALIDATED_PLS_TOL


class RepeatedColonyKFold:
    """K-fold over colonies, blocked by condition and repeated.

    Within every condition the colonies are shuffled and dealt across the folds,
    so each test fold holds floor(n/k) or ceil(n/k) whole colonies of every
    condition (n = colonies of that condition) and no colony is ever split
    between training and test. Splits depend only on ``random_state``, the
    labels and the colony ids.
    """

    def __init__(self, n_splits=3, n_repeats=10, random_state=42):
        if n_splits < 2 or n_repeats < 1:
            raise ValueError("n_splits must be >= 2 and n_repeats >= 1")
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * self.n_repeats

    def split(self, X, y, groups):
        """Yield (train, test) row indices: ``n_splits`` folds for each repeat."""
        y, groups = np.asarray(y), np.asarray(groups)
        ids, first, inverse = np.unique(groups, return_index=True, return_inverse=True)
        cond = y[first]
        mixed = [g for g in ids if len(set(y[groups == g])) != 1]
        if mixed:
            raise ValueError(f"colonies with more than one condition label: {mixed[:3]}")
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_repeats):
            fold = np.empty(len(ids), dtype=int)
            for c in np.unique(cond):
                members = rng.permutation(np.flatnonzero(cond == c))
                start = rng.integers(self.n_splits)
                fold[members] = (np.arange(members.size) + start) % self.n_splits
            for k in range(self.n_splits):
                test = fold[inverse] == k
                yield np.flatnonzero(~test), np.flatnonzero(test)


class PLSDAClassifier(ClassifierMixin, BaseEstimator):
    """PLS2 regression on one-hot classes; predicts the class with the largest
    fitted response. Uses the NIPALS settings validated for the VIP calculation."""

    def __init__(self, n_components=2, tol=VALIDATED_PLS_TOL, max_iter=VALIDATED_PLS_MAX_ITER):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.binarizer_ = LabelBinarizer().fit(y)
        self.classes_ = self.binarizer_.classes_
        Y = self.binarizer_.transform(y).astype(float)
        if Y.shape[1] == 1:                                   # two classes: one column
            Y = np.hstack([1 - Y, Y])
        a = int(max(1, min(self.n_components, X.shape[0] - 1, X.shape[1])))
        self.pls_ = PLSRegression(n_components=a, scale=False, tol=self.tol,
                                  max_iter=self.max_iter).fit(X, Y)
        return self

    def predict(self, X):
        check_is_fitted(self, 'pls_')
        return self.classes_[np.argmax(self.pls_.predict(np.asarray(X, float)), axis=1)]


def colony_vote(y_pred, colony_id):
    """Majority label of each colony's spectra. Ties go to the label that sorts first."""
    y_pred, colony_id = np.asarray(y_pred), np.asarray(colony_id)
    out = {}
    for g in np.unique(colony_id):
        labels, counts = np.unique(y_pred[colony_id == g], return_counts=True)
        out[g] = labels[np.argmax(counts)]
    return out


def grouped_cv(estimator, X_tic, condition, colony_id, cv=None, scaling='autoscale',
               variance_percentile=25, abundance_percentile=5):
    """Colony-grouped cross-validation of one model on TIC-normalised spectra.

    ``estimator`` is an unfitted scikit-learn classifier (cloned for every fold);
    ``cv`` defaults to RepeatedColonyKFold(3, 10, 42). Returns a dict of arrays
    with one value per repeat: 'accuracy' and 'balanced_accuracy' over spectra,
    and 'colony_accuracy' over colonies (majority vote).
    """
    X_tic = np.asarray(X_tic, float)
    y, g = np.asarray(condition), np.asarray(colony_id)
    cv = RepeatedColonyKFold() if cv is None else cv
    truth = {c: y[g == c][0] for c in np.unique(g)}
    splits = list(cv.split(X_tic, y, g))
    k = cv.n_splits
    acc, bal, col = [], [], []
    for r in range(len(splits) // k):
        pred = np.empty(len(y), dtype=object)
        seen = np.zeros(len(y), dtype=int)
        for tr, te in splits[r * k:(r + 1) * k]:
            if set(g[tr]) & set(g[te]):
                raise ValueError("a colony is in both training and test")
            Xtr, Xte = _fold_matrices(X_tic, tr, te, scaling, variance_percentile,
                                      abundance_percentile)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Variables are collinear')  # LDA, p >> n
                pred[te] = clone(estimator).fit(Xtr, y[tr]).predict(Xte)
            seen[te] += 1
        if not (seen == 1).all():
            raise ValueError("every spectrum must be in exactly one test fold per repeat")
        pred = pred.astype(y.dtype)
        vote = colony_vote(pred, g)
        acc.append(accuracy_score(y, pred))
        bal.append(balanced_accuracy_score(y, pred))
        col.append(float(np.mean([vote[c] == truth[c] for c in truth])))
    return {'accuracy': np.array(acc), 'balanced_accuracy': np.array(bal),
            'colony_accuracy': np.array(col)}


def permute_colony_labels(condition, colony_id, rng):
    """Shuffle the condition labels among colonies; a colony's spectra keep one
    label and the number of colonies per condition is unchanged."""
    _, first, inverse = np.unique(np.asarray(colony_id), return_index=True, return_inverse=True)
    return rng.permutation(np.asarray(condition)[first])[inverse]


def colony_permutation_test(estimator, X_tic, condition, colony_id, cv=None,
                            n_permutations=1000, seed=42, n_jobs=1, **cv_kwargs):
    """Permutation test of the mean cross-validated accuracy.

    For each permutation the condition labels are shuffled among colonies and
    the whole procedure (fold assignment, training-only preprocessing, model
    fit) is repeated. Returns (observed, null, p) with
    p = (number of null values >= observed + 1) / (n_permutations + 1), so the
    smallest attainable p is 1 / (n_permutations + 1). The null distribution is
    reproducible from ``seed`` whatever ``n_jobs`` is.
    """
    y, g = np.asarray(condition), np.asarray(colony_id)
    cv = RepeatedColonyKFold() if cv is None else cv
    observed = float(grouped_cv(estimator, X_tic, y, g, cv, **cv_kwargs)['accuracy'].mean())
    children = np.random.SeedSequence(seed).spawn(n_permutations)

    def one(child):
        y_perm = permute_colony_labels(y, g, np.random.default_rng(child))
        return float(grouped_cv(estimator, X_tic, y_perm, g, cv, **cv_kwargs)['accuracy'].mean())

    null = np.array(Parallel(n_jobs=n_jobs)(delayed(one)(c) for c in children))
    p = (np.sum(null >= observed) + 1) / (n_permutations + 1)
    return observed, null, float(p)
