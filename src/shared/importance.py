"""Algorithm-specific feature importance for the six complementary classifiers.

Each method keeps its NATIVE importance definition. Values are never compared
across methods; only ranks / top-N memberships are.

| key   | model (original configuration)                        | native quantity                     | signed, class-specific?        |
|-------|--------------------------------------------------------|-------------------------------------|--------------------------------|
| ridge | RidgeClassifier(alpha=1)                               | |coef| (one-vs-rest least squares)  | yes: coef per class            |
| lr    | LogisticRegression(max_iter=1000), multinomial L2, C=1 | |coef|                              | yes: coef per class            |
| svm   | SVC(kernel='linear', C=1), one-vs-one                  | |w| over the K(K-1)/2 pair machines | yes: w per class PAIR          |
| lda   | LinearDiscriminantAnalysis(solver='lsqr',              | |discriminant coefficient|          | yes: coef per class            |
|       |   shrinkage='auto')  [new; no LDA ranking existed]     |                                     |                                |
| rf    | RandomForestClassifier(n_estimators=100)               | mean decrease in impurity (Gini)    | no (global, unsigned)          |
| gb    | GradientBoostingClassifier(100, lr=0.1, depth=3)       | total split gain (Friedman MSE)     | no (global, unsigned)          |

Global importance for coefficient models = mean |coef| across class (or pair)
vectors, exactly as in the original feature_importance_analysis(); an L2
aggregation is available for sensitivity only. Tree importances are averaged over
`n_seeds` random seeds (the original single-seed value is seed 42); direction for
tree-selected features must come from separate univariate statistics.
"""
import warnings

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

METHODS = ('ridge', 'lr', 'svm', 'lda', 'rf', 'gb')
FAMILY = {'ridge': 'L', 'lr': 'L', 'svm': 'L', 'rf': 'T', 'gb': 'T',
          'lda': 'P', 'vip_ref': 'P', 'vip_multi': 'P'}
FAMILY_NAMES = {'L': 'penalised linear (Ridge, LR, SVM)', 'T': 'tree ensembles (RF, GB)',
                'P': 'projection / discriminant (LDA, PLS-DA VIP)'}
IMPORTANCE_TYPE = {'ridge': 'coefficient', 'lr': 'coefficient', 'svm': 'coefficient',
                   'lda': 'coefficient', 'rf': 'impurity (MDI)', 'gb': 'split gain'}


def _aggregate(coef, how):
    coef = np.atleast_2d(coef)
    if how == 'mean_abs':
        return np.abs(coef).mean(axis=0)
    if how == 'l2':
        return np.sqrt((coef ** 2).sum(axis=0))
    raise ValueError(how)


def fit_importance(method, X, y, seed=42, n_seeds=1, aggregation='mean_abs',
                   svm_multiclass='ovo', ridge_alpha=1.0, lr_C=1.0, svm_C=1.0):
    """Fit one method and return a dict:
    'global'  : (p,) native global importance (ranking quantity),
    'signed'  : (m, p) native signed coefficients or None (trees),
    'signed_labels' : class / pair labels for the rows of 'signed',
    'type'    : importance type string, 'n_nonzero' : # features with importance > 0.
    """
    le = LabelEncoder()
    yi = le.fit_transform(y)
    cls = le.classes_
    signed, labels = None, None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if method == 'ridge':
            m = RidgeClassifier(alpha=ridge_alpha).fit(X, yi)
            signed = np.atleast_2d(m.coef_); labels = list(cls) if signed.shape[0] > 1 else [cls[1]]
        elif method == 'lr':
            m = LogisticRegression(max_iter=1000, C=lr_C, random_state=seed).fit(X, yi)
            signed = np.atleast_2d(m.coef_); labels = list(cls) if signed.shape[0] > 1 else [cls[1]]
        elif method == 'svm':
            if svm_multiclass == 'ovo':
                m = SVC(kernel='linear', C=svm_C, random_state=seed).fit(X, yi)
                signed = np.atleast_2d(m.coef_)
                k = len(cls)
                labels = [f'{cls[i]} vs {cls[j]}' for i in range(k) for j in range(i + 1, k)]
            elif svm_multiclass == 'ovr':
                m = OneVsRestClassifier(SVC(kernel='linear', C=svm_C, random_state=seed)).fit(X, yi)
                signed = np.vstack([e.coef_.ravel() for e in m.estimators_]); labels = list(cls)
            else:
                raise ValueError(svm_multiclass)
        elif method == 'lda':
            m = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X, yi)
            signed = np.atleast_2d(m.coef_); labels = list(cls) if signed.shape[0] > 1 else [cls[1]]
        elif method in ('rf', 'gb'):
            imps = []
            for s in range(seed, seed + n_seeds):
                if method == 'rf':
                    mm = RandomForestClassifier(n_estimators=100, random_state=s)
                else:
                    mm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                    max_depth=3, random_state=s)
                imps.append(mm.fit(X, yi).feature_importances_)
            g = np.mean(imps, axis=0)
            return {'global': g, 'signed': None, 'signed_labels': None,
                    'type': IMPORTANCE_TYPE[method], 'n_nonzero': int((g > 0).sum()),
                    'per_seed': np.vstack(imps)}
        else:
            raise ValueError(method)
    g = _aggregate(signed, aggregation)
    return {'global': g, 'signed': signed, 'signed_labels': labels,
            'type': IMPORTANCE_TYPE[method], 'n_nonzero': int((g > 0).sum())}


def lda_structure_correlation(X, y):
    """Correlation-robust LDA check: max_k |corr(x_j, decision_function_k)|."""
    yi = LabelEncoder().fit_transform(y)
    m = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X, yi)
    D = m.decision_function(X)
    Xc = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    Dc = (D - D.mean(0)) / np.where(D.std(0) == 0, 1, D.std(0))
    return np.abs(Xc.T @ Dc / X.shape[0]).max(axis=1)


def top_n(imp, n, rule='stable'):
    """Indices of the n highest-importance features.

    rule='stable'   : ties broken by feature index (ascending m/z); used by the
                      validated workflow.
    rule='original' : np.argsort(imp)[::-1][:n], exactly as the original toolkit
                      (ties broken by descending index).
    """
    imp = np.asarray(imp)
    if rule == 'original':
        return np.argsort(imp)[::-1][:n]
    return np.lexsort((np.arange(imp.size), -imp))[:n]


def n_ties_at_cutoff(imp, n):
    """Number of features tied with the n-th largest value (top-N not unique if > 1)."""
    imp = np.asarray(imp)
    v = np.sort(imp)[::-1][n - 1]
    return int(np.sum(imp == v))
