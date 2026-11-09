"""Leave-one-colony-out (LOCO) classification -- supplementary to the 3-fold
colony-grouped CV used for the principal accuracy table.

Every colony (biological replicate) is held out once with all of its technical
replicates; the complete preprocessing chain is refitted on the remaining
colonies, and the predictions for all held-out spectra are pooled before
computing accuracy, balanced accuracy, macro-F1 and the confusion matrix.

Accuracy measures how well condition-plate groups are discriminated; because
each condition was grown on one plate/enclosure, it is not by itself evidence of
a light effect.
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

from shared.classifier_comparison_standard import small_sample_guards

DISPLAY = {'rf': 'Random Forest', 'svm': 'SVM', 'gb': 'Gradient Boosting',
           'lr': 'Logistic Regression', 'lda': 'LDA', 'ridge': 'Ridge'}


def original_estimators(n_biological=None, seed=42):
    """The six classifiers exactly as configured in classifier_comparison_standard."""
    g = small_sample_guards(n_biological)
    gb = dict(n_estimators=100, learning_rate=0.1, max_depth=3); gb.update(g['gb'])
    return {'rf': RandomForestClassifier(n_estimators=100, random_state=seed, **g['rf']),
            'svm': SVC(kernel='linear', random_state=seed, **g['svm']),
            'gb': GradientBoostingClassifier(random_state=seed, **gb),
            'lr': LogisticRegression(max_iter=1000, random_state=seed, **g['logreg']),
            'lda': LinearDiscriminantAnalysis(),
            'ridge': RidgeClassifier(**g['ridge'])}


def loco_evaluate(estimator, X_binned, y_labels, groups, prep_steps):
    le = LabelEncoder(); y = le.fit_transform(y_labels)
    pipe = Pipeline([(n, clone(t)) for n, t in prep_steps] + [('clf', clone(estimator))])
    pred = cross_val_predict(pipe, X_binned, y, groups=np.asarray(groups), cv=LeaveOneGroupOut())
    return dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
                macro_f1=f1_score(y, pred, average='macro'),
                confusion=confusion_matrix(y, pred, labels=np.arange(len(le.classes_))),
                classes=list(le.classes_))


def loco_table(X_binned, y_labels, groups, prep_steps, methods=None, n_biological=None, seed=42):
    """Returns (DataFrame one row per classifier, dict name -> confusion DataFrame)."""
    est = original_estimators(n_biological, seed)
    rows, conf = [], {}
    for m in methods or est:
        r = loco_evaluate(est[m], X_binned, y_labels, groups, prep_steps)
        rows.append(dict(classifier=DISPLAY[m], loco_accuracy=r['accuracy'],
                         loco_balanced_accuracy=r['balanced_accuracy'], loco_macro_f1=r['macro_f1'],
                         n_colonies=len(set(np.asarray(groups)))))
        conf[DISPLAY[m]] = pd.DataFrame(r['confusion'], index=r['classes'], columns=r['classes'])
    return pd.DataFrame(rows), conf
