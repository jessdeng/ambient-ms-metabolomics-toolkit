"""
src/shared/quality_metrics.py
=============================
Reproducibility and benchmarking metrics for the ambient-MS pipeline.

Two families of metric live here, both computed with explicit, auditable
formulae so they can be quoted directly in a Methods section:

1. Analytical reproducibility (chemometric QC)
   - technical_cv     : per-feature technical coefficient of variation (%CV)
                        across technical replicates of the same biological unit.
   - biological_cv    : per-feature biological %CV across biological units
                        (e.g. colonies) within a class.
   - dispersion_ratio : the Broadhurst D-ratio, sigma_technical / sigma_total,
                        the recommended feature-quality filter for untargeted
                        metabolomics.
   - qc_report        : rolls the three up into per-feature and summary tables.

2. Classification benchmarking (multi-class, small-n designs)
   - grouped_oof_predictions : leak-free pooled out-of-fold predictions using the
                        SAME per-fold preprocessing Pipeline and StratifiedGroup
                        KFold as the accuracy path, so the reported confusion
                        structure is generated without preprocessing leakage.
   - classification_metrics  : macro-F1, per-class recall/precision and the
                        pooled confusion matrix from those predictions.
   - plot_confusion_matrix   : Analyst-standard confusion-matrix figure.

Scale contract (important)
--------------------------
Coefficient of variation is only meaningful on a *linear* intensity scale where
the ratio SD/mean is dimensionless and interpretable. It MUST NOT be computed on
log-transformed or autoscaled data (a z-scored feature has mean ~0, so SD/mean
explodes). The QC functions here therefore expect the NORMALISED-BUT-UNTRANSFORMED
intensity matrix -- i.e. after TIC/median/PQN normalisation but *before* glog/log
and before autoscaling. Passing transformed data is a methodological error and the
functions raise if the input contains negative values (a signature of a log or
mean-centre step having already been applied).

References
----------
Broadhurst D. et al. (2018) "Guidelines and considerations for the use of system
    suitability and quality control samples in mass spectrometry assays applied
    in untargeted clinical metabolomic studies." Metabolomics 14:72.
    doi:10.1007/s11306-018-1367-3   (D-ratio; %CV acceptance thresholds)
Dunn W.B. et al. (2011) Nat. Protoc. 6:1060.  (QC-RSD reproducibility reporting)
"""

from __future__ import annotations

import contextlib
import warnings

import numpy as np


# -- Acceptance thresholds (exposed as constants for reproducibility) -------------
# Feature-level QC thresholds widely used in untargeted metabolomics. A feature is
# considered analytically reproducible when its technical %CV is at or below
# CV_ACCEPT_PCT and its D-ratio is at or below DRATIO_ACCEPT_PCT.
CV_ACCEPT_PCT      = 30.0   # <=30% technical CV: standard untargeted-MS threshold
CV_STRICT_PCT      = 20.0   # <=20%: stricter (targeted-assay) threshold
DRATIO_ACCEPT_PCT  = 50.0   # <=50% D-ratio: technical spread < biological spread


@contextlib.contextmanager
def _suppress_allnan_warnings():
    """Silence the expected all-NaN-slice warnings from nanmedian/nanmean.

    Features with no usable replicate pool to NaN by design, so the RuntimeWarning
    numpy raises for an all-NaN slice is semantically correct noise, not an error.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'.*(Mean|All-NaN).*', RuntimeWarning)
        yield


# ---------------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------------

def _validate_linear(X):
    """Return X as float64 ndarray, raising if it is not on a linear scale."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_samples, n_features); got shape {X.shape}.")
    finite = X[np.isfinite(X)]
    if finite.size and finite.min() < -1e-9:
        raise ValueError(
            "quality_metrics received negative intensities. Coefficient of "
            "variation must be computed on linear-scale (normalised, "
            "untransformed) intensities, not on log-transformed or autoscaled "
            "data. Pass the matrix after normalisation but BEFORE glog/log and "
            "scaling.")
    return X


def _group_mean_sd(X, groups, min_reps=2):
    """
    Per-group mean and (ddof=1) SD for every feature.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)   linear-scale intensities.
    groups : array-like (n_samples,)      group id per sample.
    min_reps : int                        groups with fewer than this many samples
                                          are excluded from the SD estimate (a
                                          single replicate carries no dispersion
                                          information); their mean is still
                                          returned for between-group statistics.

    Returns
    -------
    unit_ids   : ndarray (n_units,)                 ordered unique group ids.
    means      : ndarray (n_units, n_features)      per-group feature means.
    sds        : ndarray (n_units, n_features)      per-group SD; rows for groups
                                                    with < min_reps samples are NaN.
    n_per_unit : ndarray (n_units,)                 replicate count per group.
    """
    groups = np.asarray(groups)
    if groups.shape[0] != X.shape[0]:
        raise ValueError("groups length must match the number of rows in X.")
    unit_ids = np.unique(groups)
    n_features = X.shape[1]
    means = np.empty((unit_ids.size, n_features), dtype=float)
    sds = np.full((unit_ids.size, n_features), np.nan, dtype=float)
    n_per_unit = np.empty(unit_ids.size, dtype=int)
    for k, uid in enumerate(unit_ids):
        rows = X[groups == uid]
        n_per_unit[k] = rows.shape[0]
        means[k] = rows.mean(axis=0)
        # SD needs >= 2 observations for ddof=1; means are always returned so
        # singleton units still contribute to between-unit statistics.
        if rows.shape[0] >= max(2, min_reps):
            sds[k] = rows.std(axis=0, ddof=1)
    return unit_ids, means, sds, n_per_unit


def _cv_percent(sd, mean):
    """Percentage CV with an explicit zero/negative-mean guard (returns NaN there)."""
    sd = np.asarray(sd, dtype=float)
    mean = np.asarray(mean, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        cv = 100.0 * sd / mean
    cv[~np.isfinite(cv)] = np.nan
    cv[mean <= 0] = np.nan
    return cv


# ---------------------------------------------------------------------------------
# 1. Analytical reproducibility
# ---------------------------------------------------------------------------------

def technical_cv(X, tech_groups, aggregate='median', min_reps=2):
    """
    Per-feature technical coefficient of variation (%CV) across technical replicates.

    For every biological unit (colony) that has at least ``min_reps`` technical
    replicates, the within-unit %CV is computed for each feature as
    ``100 * SD / mean`` (SD with ddof=1). Per-unit CVs are then pooled across units
    into one value per feature by ``aggregate`` ('median', robust default, or
    'mean'). Median pooling is preferred because %CV is right-skewed.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)   normalised, UNTRANSFORMED intensities.
    tech_groups : array-like (n_samples,) biological-unit id per sample; samples
                                          sharing an id are technical replicates.
    aggregate : {'median', 'mean'}        how to pool per-unit CVs across units.
    min_reps : int                        minimum technical replicates for a unit
                                          to contribute (default 2).

    Returns
    -------
    dict with keys
        per_feature : ndarray (n_features,)  pooled technical %CV per feature (NaN
                                             where no unit had a positive mean).
        per_unit    : ndarray (n_units, n_features) within-unit %CV.
        unit_ids    : ndarray (n_units,)
        n_units_used: int                    units with >= min_reps replicates.
    """
    if aggregate not in ('median', 'mean'):
        raise ValueError("aggregate must be 'median' or 'mean'.")
    X = _validate_linear(X)
    unit_ids, means, sds, n_per_unit = _group_mean_sd(X, tech_groups, min_reps=min_reps)

    per_unit_cv = _cv_percent(sds, means)                 # (n_units, n_features)
    usable = n_per_unit >= min_reps
    if not usable.any():
        raise ValueError(
            f"No biological unit has >= {min_reps} technical replicates; "
            "technical CV is undefined. Check tech_groups.")

    block = per_unit_cv[usable]
    with _suppress_allnan_warnings():
        pooled = (np.nanmedian(block, axis=0) if aggregate == 'median'
                  else np.nanmean(block, axis=0))
    return {
        'per_feature': pooled,
        'per_unit': per_unit_cv,
        'unit_ids': unit_ids,
        'n_units_used': int(usable.sum()),
    }


def biological_cv(X, bio_groups, class_labels=None, aggregate='median', min_units=2):
    """
    Per-feature biological %CV across biological units (colonies).

    Technical replicates are first collapsed to per-unit means, then the %CV is
    taken across those unit means. When ``class_labels`` is given the CV is computed
    within each class and pooled across classes by ``aggregate`` -- this removes the
    between-condition biological signal so the metric reflects genuine within-group
    biological dispersion rather than the treatment effect.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)   normalised, UNTRANSFORMED intensities.
    bio_groups : array-like (n_samples,)  biological-unit id per sample.
    class_labels : array-like or None     condition label per sample. If provided,
                                          CV is computed within-class then pooled.
    aggregate : {'median', 'mean'}        pooling across classes.
    min_units : int                       minimum units required to define a CV.

    Returns
    -------
    dict with keys
        per_feature : ndarray (n_features,) biological %CV per feature.
        n_units     : int                   number of biological units.
    """
    X = _validate_linear(X)
    bio_groups = np.asarray(bio_groups)
    unit_ids, unit_means, _, _ = _group_mean_sd(X, bio_groups, min_reps=1)

    if class_labels is None:
        if unit_means.shape[0] < min_units:
            raise ValueError(f"Need >= {min_units} biological units; "
                             f"got {unit_means.shape[0]}.")
        sd = unit_means.std(axis=0, ddof=1)
        cv = _cv_percent(sd, unit_means.mean(axis=0))
        return {'per_feature': cv, 'n_units': int(unit_ids.size)}

    # Within-class CV, then pool across classes.
    class_labels = np.asarray(class_labels)
    # Map each biological unit to its (unique) class label.
    unit_class = np.array([class_labels[bio_groups == uid][0] for uid in unit_ids])
    per_class_cv = []
    for cls in np.unique(unit_class):
        block = unit_means[unit_class == cls]
        if block.shape[0] < min_units:
            continue
        sd = block.std(axis=0, ddof=1)
        per_class_cv.append(_cv_percent(sd, block.mean(axis=0)))
    if not per_class_cv:
        raise ValueError(f"No class has >= {min_units} biological units.")
    stack = np.vstack(per_class_cv)
    with _suppress_allnan_warnings():
        pooled = (np.nanmedian(stack, axis=0) if aggregate == 'median'
                  else np.nanmean(stack, axis=0))
    return {'per_feature': pooled, 'n_units': int(unit_ids.size)}


def dispersion_ratio(X, tech_groups, class_labels=None):
    """
    Per-feature Broadhurst D-ratio (%): technical SD / total SD.

    D-ratio_j = 100 * sigma_technical,j / sigma_total,j, where

        sigma_technical,j = sqrt( mean over units of within-unit variance_j )
                            (pooled analytical/technical standard deviation)
        sigma_total,j     = SD across all per-unit means_j
                            (biological + technical spread of the study)

    A feature whose analytical noise is small relative to the study's biological
    variation has a low D-ratio and is trustworthy; D-ratio -> 100% means the
    feature is dominated by measurement noise. Broadhurst et al. (2018) recommend
    retaining features with D-ratio <= 50%.

    When ``class_labels`` is supplied, sigma_total is computed *within class* and
    pooled (root-mean-square across classes) so the treatment effect does not
    inflate the biological term.

    Returns
    -------
    dict with keys
        per_feature : ndarray (n_features,)  D-ratio (%) per feature.
        sigma_tech  : ndarray (n_features,)  pooled technical SD.
        sigma_total : ndarray (n_features,)  total SD.
    """
    X = _validate_linear(X)
    tech_groups = np.asarray(tech_groups)
    unit_ids, unit_means, unit_sds, n_per_unit = _group_mean_sd(
        X, tech_groups, min_reps=2)

    usable = n_per_unit >= 2
    if not usable.any():
        raise ValueError("No unit has >= 2 technical replicates; D-ratio undefined.")
    # Pooled technical SD = RMS of within-unit SDs over units with replication.
    with _suppress_allnan_warnings():
        var_tech = np.nanmean(unit_sds[usable] ** 2, axis=0)
    sigma_tech = np.sqrt(var_tech)

    # Total SD across unit means (optionally within-class then pooled).
    if class_labels is None:
        sigma_total = unit_means.std(axis=0, ddof=1)
    else:
        class_labels = np.asarray(class_labels)
        unit_class = np.array([class_labels[tech_groups == uid][0] for uid in unit_ids])
        var_terms = []
        for cls in np.unique(unit_class):
            block = unit_means[unit_class == cls]
            if block.shape[0] >= 2:
                var_terms.append(block.var(axis=0, ddof=1))
        if not var_terms:
            raise ValueError("No class has >= 2 biological units for the total-SD term.")
        sigma_total = np.sqrt(np.mean(np.vstack(var_terms), axis=0))

    with np.errstate(divide='ignore', invalid='ignore'):
        dratio = 100.0 * sigma_tech / sigma_total
    dratio[~np.isfinite(dratio)] = np.nan
    return {'per_feature': dratio, 'sigma_tech': sigma_tech, 'sigma_total': sigma_total}


def qc_report(X, mz, tech_groups, class_labels=None):
    """
    Per-feature QC table plus a study-level summary.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)   normalised, UNTRANSFORMED intensities.
    mz : array-like (n_features,)         m/z label per feature.
    tech_groups : array-like (n_samples,) biological-unit id per sample.
    class_labels : array-like or None     condition label per sample.

    Returns
    -------
    per_feature : dict of ndarrays keyed 'mz', 'technical_cv_pct',
                  'biological_cv_pct', 'dratio_pct', 'pass_cv', 'pass_dratio'.
    summary : dict of study-level floats/ints (median technical CV, fraction of
              features under each acceptance threshold, feature counts).
    """
    mz = np.asarray(mz, dtype=float)
    tcv = technical_cv(X, tech_groups)['per_feature']
    bcv = biological_cv(X, tech_groups, class_labels)['per_feature']
    drt = dispersion_ratio(X, tech_groups, class_labels)['per_feature']

    pass_cv = tcv <= CV_ACCEPT_PCT
    pass_dr = drt <= DRATIO_ACCEPT_PCT

    per_feature = {
        'mz': mz,
        'technical_cv_pct': tcv,
        'biological_cv_pct': bcv,
        'dratio_pct': drt,
        'pass_cv': pass_cv,
        'pass_dratio': pass_dr,
    }
    n = mz.size
    with _suppress_allnan_warnings():
        summary = {
            'n_features': int(n),
            'median_technical_cv_pct': float(np.nanmedian(tcv)),
            'median_biological_cv_pct': float(np.nanmedian(bcv)),
            'median_dratio_pct': float(np.nanmedian(drt)),
            'frac_cv_under_20': float(np.nanmean(tcv <= CV_STRICT_PCT)),
            'frac_cv_under_30': float(np.nanmean(tcv <= CV_ACCEPT_PCT)),
            'frac_dratio_under_50': float(np.nanmean(drt <= DRATIO_ACCEPT_PCT)),
            'n_pass_both': int(np.nansum(pass_cv & pass_dr)),
        }
    return per_feature, summary


# ---------------------------------------------------------------------------------
# 2. Classification benchmarking (leak-free pooled OOF predictions)
# ---------------------------------------------------------------------------------

def grouped_oof_predictions(estimator, X_binned, y, groups, prep_steps,
                            n_splits, random_state=0):
    """
    Pooled out-of-fold predictions from grouped, leak-free CV.

    Builds the identical per-fold ``Pipeline(prep_steps + [clf])`` used by the
    accuracy path and evaluates it with ``StratifiedGroupKFold`` via
    ``sklearn.model_selection.cross_val_predict``, so every test prediction comes
    from a model whose preprocessing was fit on the training fold only. The
    returned vector is aligned with ``y`` (one prediction per sample).

    Parameters mirror ``classifier_comparison._run_grouped_cv``.

    Returns
    -------
    y_pred : ndarray (n_samples,)   out-of-fold predicted label per sample.
    """
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

    steps = [(name, clone(t)) for name, t in prep_steps] + [('clf', clone(estimator))]
    pipe = Pipeline(steps)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                random_state=random_state)
    return cross_val_predict(pipe, X_binned, y, groups=groups, cv=sgkf)


def classification_metrics(y_true, y_pred, class_names=None):
    """
    Macro-F1, per-class recall/precision and the pooled confusion matrix.

    Parameters
    ----------
    y_true, y_pred : array-like (n_samples,)   integer-encoded or string labels.
    class_names : list or None                 display names; inferred if None.

    Returns
    -------
    dict with keys
        macro_f1, weighted_f1, accuracy, balanced_accuracy : floats
        per_class : dict name -> {'recall', 'precision', 'f1', 'support'}
        confusion : ndarray (n_classes, n_classes)  rows = true, cols = predicted.
        labels    : list of class names in confusion-matrix order.
    """
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                  confusion_matrix, f1_score,
                                  precision_recall_fscore_support)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    names = ([str(class_names[i]) for i in range(len(labels))]
             if class_names is not None else [str(l) for l in labels])

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {
        names[i]: {'recall': float(rec[i]), 'precision': float(prec[i]),
                   'f1': float(f1[i]), 'support': int(sup[i])}
        for i in range(len(labels))
    }
    return {
        'macro_f1': float(f1_score(y_true, y_pred, labels=labels,
                                   average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, labels=labels,
                                      average='weighted', zero_division=0)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'per_class': per_class,
        'confusion': cm,
        'labels': names,
    }


def plot_confusion_matrix(cm, labels, out_path, title=None, normalize=True):
    """
    Save an Analyst-standard confusion-matrix figure.

    Cells show the row-normalised rate (fraction of each true class predicted as
    each class) with the raw count underneath, colour-mapped on a colour-blind-safe
    sequential map. Axes are fully labelled ('Predicted condition' / 'True
    condition'); the figure is written at 300 DPI via ``pub_savefig``.

    Parameters
    ----------
    cm : ndarray (n_classes, n_classes)   rows = true, cols = predicted (counts).
    labels : list of str                  class names in matrix order.
    out_path : str                        output path (.png/.pdf/.svg).
    title : str or None
    normalize : bool                      annotate row-normalised rates (default).
    """
    import matplotlib.pyplot as plt
    from src.shared.plot_style import apply_style, pub_savefig

    apply_style()
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm / row_sums

    n = len(labels)
    fig, ax = plt.subplots(figsize=(0.7 * n + 2.2, 0.7 * n + 2.0))
    im = ax.imshow(cm_norm if normalize else cm, cmap='cividis',
                   vmin=0, vmax=1 if normalize else None, aspect='equal')

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted condition')
    ax.set_ylabel('True condition')
    if title:
        ax.set_title(title)

    thresh = (cm_norm.max() if normalize else cm.max()) / 2.0
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j] if normalize else cm[i, j]
            txt = (f"{cm_norm[i, j]:.2f}\n({int(cm[i, j])})" if normalize
                   else f"{int(cm[i, j])}")
            ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                    color='white' if val < thresh else 'black')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Row-normalised rate' if normalize else 'Count')
    pub_savefig(out_path)
