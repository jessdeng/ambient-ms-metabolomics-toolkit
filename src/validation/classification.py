"""Classification as supporting evidence (plan step 11; validity gate, SI).

The six original classifier configurations (shared.classifier_comparison_standard)
are evaluated with leak-free, colony-grouped cross-validation: the full
preprocessing chain is refitted inside every training fold. Primary scheme:
leave-one-colony-out with pooled out-of-fold predictions (accuracy, balanced
accuracy, macro-F1, confusion matrix). The toolkit's 3-fold StratifiedGroupKFold
is reported for continuity with the manuscript's Table 1.

Accuracy here measures discrimination of condition-plate groups; it is not
evidence of a light effect. Shuffled-label benchmarks give exceedance
proportions, not p-values.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from shared.classifier_comparison_standard import small_sample_guards, _run_grouped_cv

SEED = 42


def original_estimators(n_biological=None, seed=SEED):
    """The six classifiers exactly as configured in classifier_comparison_standard."""
    g = small_sample_guards(n_biological)
    gb = dict(n_estimators=100, learning_rate=0.1, max_depth=3); gb.update(g['gb'])
    return {'ridge': RidgeClassifier(**g['ridge']),
            'lr': LogisticRegression(max_iter=1000, random_state=seed, **g['logreg']),
            'svm': SVC(kernel='linear', random_state=seed, **g['svm']),
            'lda': LinearDiscriminantAnalysis(),
            'rf': RandomForestClassifier(n_estimators=100, random_state=seed, **g['rf']),
            'gb': GradientBoostingClassifier(random_state=seed, **gb)}


def loco_evaluate(estimator, X_binned, y_labels, groups, prep_steps):
    le = LabelEncoder(); y = le.fit_transform(y_labels)
    pipe = Pipeline([(n, clone(t)) for n, t in prep_steps] + [('clf', clone(estimator))])
    pred = cross_val_predict(pipe, X_binned, y, groups=np.asarray(groups), cv=LeaveOneGroupOut())
    return dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
                macro_f1=f1_score(y, pred, average='macro'),
                confusion=confusion_matrix(y, pred, labels=np.arange(len(le.classes_))), classes=list(le.classes_))


def evaluate_all(X_binned, y_labels, groups, prep_steps, methods=None, kfold_repeats=5):
    est = original_estimators()
    rows, conf = [], {}
    for m in methods or est:
        r = loco_evaluate(est[m], X_binned, y_labels, groups, prep_steps)
        k = _run_grouped_cv(est[m], X_binned, LabelEncoder().fit_transform(y_labels), np.asarray(groups), prep_steps,
                            n_splits=3, n_repeats=kfold_repeats, return_metrics=True)
        rows.append(dict(method=m, loco_accuracy=r['accuracy'], loco_balanced_accuracy=r['balanced_accuracy'],
                         loco_macro_f1=r['macro_f1'], kfold3_accuracy_mean=float(np.mean(k['test_accuracy'])),
                         kfold3_accuracy_sd=float(np.std(k['test_accuracy'])), kfold3_repeats=kfold_repeats))
        conf[m] = pd.DataFrame(r['confusion'], index=r['classes'], columns=r['classes'])
    return pd.DataFrame(rows), conf
