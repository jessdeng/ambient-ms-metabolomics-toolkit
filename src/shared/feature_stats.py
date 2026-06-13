"""
feature_stats.py — univariate significance, FDR, and selection-stability tools
==============================================================================
The statistical-rigor layer added for publication. Three concerns are covered:

1. **Univariate significance + multiple-testing control.**
   `univariate_feature_stats()` computes, per m/z feature, a fold-change and a
   univariate test (Welch's t for 2 groups, one-way ANOVA for >2; Wilcoxon
   rank-sum / Kruskal-Wallis non-parametric alternatives), then applies
   Benjamini-Hochberg FDR to return q-values. This gives every reported
   candidate a p-value and a BH q-value instead of rank-overlap alone.

2. **Out-of-sample selection stability.**
   `bootstrap_selection_frequency()` resamples *colonies* (biological groups)
   with replacement, re-fits the whole ensemble on each bootstrap, and records
   how often each feature lands in the consensus set. A feature selected in 95 %
   of bootstraps is reproducible; one that flips in and out is not. This is the
   multivariate analogue of an FDR for the ensemble ranking.

3. **Null model for the cross-method overlap.**
   `overlap_permutation_null()` permutes the labels, recomputes the ensemble,
   and reports how many features reach "top-N in >= k of M methods" under the
   null. Because four of the six methods are collinear linear discriminants, the
   observed overlap must be compared against this null, not taken at face value.

All randomness is seeded; results are reproducible.
"""

import numpy as np
from scipy import stats
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder

# Canonical ensemble method keys (PLS-DA VIP is keyed 'vip').
_ALL_METHODS = ('rf', 'svm', 'gb', 'lr', 'ridge', 'vip')


# ----------------------------------------------------------------------------- #
# Benjamini-Hochberg FDR                                                          #
# ----------------------------------------------------------------------------- #
def benjamini_hochberg(p_values):
    """Benjamini-Hochberg (1995) FDR-adjusted q-values.

    NaN p-values (e.g. constant features) are treated as 1.0 so they never
    pass. Returns q-values in [0, 1] with the same shape as the input.
    """
    p = np.asarray(p_values, dtype=float).copy()
    nan_mask = ~np.isfinite(p)
    p[nan_mask] = 1.0
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1.0)
    # Enforce monotonicity from the largest p downward.
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n, dtype=float)
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    return q


# ----------------------------------------------------------------------------- #
# Univariate fold-change + test + FDR                                             #
# ----------------------------------------------------------------------------- #
def _to_linear(mu, log_transform):
    """Invert the variance-stabilising transform so group means become linear
    intensities for an interpretable fold-change. lambda_ in glog cancels in a
    ratio, so the sinh inversion is exact up to that constant."""
    if log_transform == 'log10':
        return np.power(10.0, mu)
    if log_transform == 'log2':
        return np.power(2.0, mu)
    if log_transform == 'sqrt':
        return np.square(mu)
    if log_transform == 'glog':
        return np.sinh(mu)
    return mu  # 'none'


def univariate_feature_stats(X_log, y_labels, log_transform='log10', test='auto'):
    """Per-feature fold-change, univariate p-value, and BH-FDR q-value.

    Parameters
    ----------
    X_log : ndarray (n_samples, n_features)
        Normalised + log/glog-transformed but **unscaled** intensities
        (i.e. ``X_norm`` from the pipeline). Running the test on this axis keeps
        the variance-stabilised values comparable across features while leaving
        the fold-change interpretable after inversion.
    y_labels : array-like (n_samples,)
        Class label per sample.
    log_transform : {'log10','log2','sqrt','glog','none'}
        Transform used to build ``X_log`` — needed to invert group means to a
        linear fold-change.
    test : {'auto','ttest','wilcoxon'}
        'auto'    -> Welch's t (2 groups) / one-way ANOVA (>2 groups)
        'ttest'   -> same as auto (parametric)
        'wilcoxon'-> Wilcoxon rank-sum (2 groups) / Kruskal-Wallis (>2 groups)

    Returns
    -------
    dict with arrays of length n_features:
        p_value, q_value, log2_fold_change, fold_change, neg_log10_q,
        top_group, bottom_group  (+ scalar 'test_name')
    """
    X_log = np.asarray(X_log, dtype=float)
    y = np.asarray(y_labels)
    classes = np.unique(y)
    n_features = X_log.shape[1]
    group_idx = [np.where(y == c)[0] for c in classes]
    parametric = test in ('auto', 'ttest')

    # --- group means on the linear axis -> fold change --------------------------
    mu_log = np.vstack([X_log[idx].mean(axis=0) for idx in group_idx])   # (G, F)
    mu_lin = _to_linear(mu_log, log_transform)
    mu_lin = np.where(np.isfinite(mu_lin), mu_lin, np.nan)

    top_arg = np.nanargmax(mu_lin, axis=0)
    bot_arg = np.nanargmin(mu_lin, axis=0)
    top_group = classes[top_arg]
    bottom_group = classes[bot_arg]

    if len(classes) == 2:
        # Signed fold change: class[1] relative to class[0].
        denom = np.where(np.abs(mu_lin[0]) < 1e-30, np.nan, mu_lin[0])
        fold_change = mu_lin[1] / denom
        if parametric:
            res = stats.ttest_ind(X_log[group_idx[0]], X_log[group_idx[1]],
                                  axis=0, equal_var=False)
            p = np.asarray(res.pvalue, dtype=float)
            test_name = "Welch t-test"
        else:
            p = np.full(n_features, np.nan)
            a, b = X_log[group_idx[0]], X_log[group_idx[1]]
            for j in range(n_features):
                try:
                    p[j] = stats.ranksums(a[:, j], b[:, j]).pvalue
                except Exception:
                    p[j] = np.nan
            test_name = "Wilcoxon rank-sum"
    else:
        # >2 groups: unsigned fold change = brightest / dimmest group.
        hi = np.nanmax(mu_lin, axis=0)
        lo = np.where(np.abs(np.nanmin(mu_lin, axis=0)) < 1e-30, np.nan,
                      np.nanmin(mu_lin, axis=0))
        fold_change = hi / lo
        groups_data = [X_log[idx] for idx in group_idx]
        if parametric:
            res = stats.f_oneway(*groups_data, axis=0)
            p = np.asarray(res.pvalue, dtype=float)
            test_name = "one-way ANOVA"
        else:
            p = np.full(n_features, np.nan)
            for j in range(n_features):
                try:
                    p[j] = stats.kruskal(*[g[:, j] for g in groups_data]).pvalue
                except Exception:
                    p[j] = np.nan
            test_name = "Kruskal-Wallis"

    p = np.where(np.isfinite(p), p, 1.0)
    q = benjamini_hochberg(p)
    with np.errstate(divide='ignore', invalid='ignore'):
        log2fc = np.log2(np.abs(fold_change))
        log2fc = np.where(np.isfinite(log2fc), log2fc, np.nan)
        neg_log10_q = -np.log10(np.clip(q, 1e-300, 1.0))

    return {
        'p_value': p,
        'q_value': q,
        'fold_change': fold_change,
        'log2_fold_change': log2fc,
        'neg_log10_q': neg_log10_q,
        'top_group': top_group,
        'bottom_group': bottom_group,
        'test_name': test_name,
    }


# ----------------------------------------------------------------------------- #
# Ensemble helper (shared by stability + null)                                    #
# ----------------------------------------------------------------------------- #
def _ensemble_top_sets(Xp, y_enc, y_labels, top_n, seed, vip_fn, use_methods=None):
    """Return a list of top-`top_n` index sets, one per *enabled* method.

    `use_methods` is an iterable of method keys drawn from
    ('rf','svm','gb','lr','ridge','vip'). Only those methods are fit, so toggling
    a model off in config (e.g. USE_GRADIENT_BOOSTING=False) omits it from the
    stability/permutation ensemble — and, crucially, the consensus denominator
    matches whatever the observed ensemble uses. None means "all six".

    Methods that fail on a degenerate resample (e.g. a class dropped by the
    bootstrap) are silently skipped, so the returned list may be shorter.
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression, RidgeClassifier

    if use_methods is None:
        use_methods = set(_ALL_METHODS)
    else:
        use_methods = set(use_methods)

    importances = []
    if 'rf' in use_methods:
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=seed).fit(Xp, y_enc)
            importances.append(rf.feature_importances_)
        except Exception:
            pass
    if 'svm' in use_methods:
        try:
            svm = SVC(kernel='linear', random_state=seed).fit(Xp, y_enc)
            importances.append(np.abs(svm.coef_).mean(axis=0) if svm.coef_.ndim > 1
                                else np.abs(svm.coef_).ravel())
        except Exception:
            pass
    if 'gb' in use_methods:
        try:
            gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                            max_depth=3, random_state=seed).fit(Xp, y_enc)
            importances.append(gb.feature_importances_)
        except Exception:
            pass
    if 'lr' in use_methods:
        try:
            lr = LogisticRegression(max_iter=1000, random_state=seed).fit(Xp, y_enc)
            importances.append(np.abs(lr.coef_).mean(axis=0) if lr.coef_.ndim > 1
                               else np.abs(lr.coef_).ravel())
        except Exception:
            pass
    if 'ridge' in use_methods:
        try:
            ridge = RidgeClassifier().fit(Xp, y_enc)
            importances.append(np.abs(ridge.coef_).mean(axis=0) if ridge.coef_.ndim > 1
                               else np.abs(ridge.coef_).ravel())
        except Exception:
            pass
    if 'vip' in use_methods and vip_fn is not None:
        try:
            importances.append(np.asarray(vip_fn(Xp, y_labels)))
        except Exception:
            pass

    return [set(np.argsort(imp)[::-1][:top_n]) for imp in importances]


def _consensus_counts(top_sets, n_features):
    """Vote count per feature across the method top-sets."""
    counts = np.zeros(n_features, dtype=int)
    for s in top_sets:
        for idx in s:
            counts[idx] += 1
    return counts


# ----------------------------------------------------------------------------- #
# Group-aware bootstrap selection frequency                                       #
# ----------------------------------------------------------------------------- #
def _one_bootstrap(child_seed, X_filt_raw, y_labels, groups, uniq_groups,
                   preprocess_fn, vip_fn, normalization, log_transform, scaling,
                   top_n, min_methods, n_features, use_methods):
    """Single colony-bootstrap replicate. Returns a boolean (n_features,) mask
    of features that cleared the consensus rule, or None if the resample was
    degenerate. Pure function of `child_seed` -> deterministic & worker-safe."""
    rng = np.random.default_rng(child_seed)
    drawn = rng.choice(uniq_groups, size=uniq_groups.size, replace=True)
    idx = np.concatenate([np.where(groups == g)[0] for g in drawn])
    yb = y_labels[idx]
    if np.unique(yb).size < 2:
        return None
    Xb = preprocess_fn(X_filt_raw[idx].copy(), normalization=normalization,
                       log_transform=log_transform, scaling=scaling)
    yb_enc = LabelEncoder().fit_transform(yb)
    # Independent integer seed for the estimators, drawn from this worker's rng.
    model_seed = int(rng.integers(0, 2**31 - 1))
    top_sets = _ensemble_top_sets(Xb, yb_enc, yb, top_n, model_seed, vip_fn,
                                  use_methods)
    if len(top_sets) < 2:
        return None
    counts = _consensus_counts(top_sets, n_features)
    return counts >= min_methods


def bootstrap_selection_frequency(X_filt_raw, y_labels, groups, preprocess_fn,
                                  vip_fn, normalization='tic',
                                  log_transform='log10', scaling='autoscale',
                                  top_n=50, min_methods=2, n_boot=200, seed=42,
                                  use_methods=None, n_jobs=-1):
    """Fraction of colony-bootstrap resamples in which each feature is selected.

    A feature is 'selected' in a bootstrap if it appears in the top-`top_n` of
    at least `min_methods` of the ENABLED ensemble methods (see `use_methods`) —
    the same consensus rule used for the reported list. Colonies (unique
    `groups`) are resampled WITH replacement so the unit of resampling is the
    biological replicate, never the technical replicate (consistent with the
    grouped CV). Preprocessing is re-fit on each bootstrap (no full-data leak).

    Parallelism & reproducibility
    -----------------------------
    The `n_boot` replicates are independent and run across all cores with
    `joblib.Parallel(n_jobs=n_jobs)`. Each replicate gets its own child stream
    from `np.random.SeedSequence(seed).spawn(n_boot)`, so results are
    statistically independent AND bit-for-bit reproducible regardless of
    `n_jobs` or worker scheduling.

    Parameters
    ----------
    X_filt_raw : ndarray (n_samples, n_features)
        Variance/abundance-filtered RAW (pre-normalisation) intensities whose
        columns are aligned 1:1 with the reported feature matrix.
    preprocess_fn : callable(X, normalization, log_transform, scaling) -> X
    vip_fn : callable(X, y_labels) -> importances  (PLS-DA VIP)
    use_methods : iterable of {'rf','svm','gb','lr','ridge','vip'} or None
        Methods to include; None = all six. Pass the config-derived set so a
        globally disabled model is also dropped here.
    n_jobs : int
        Cores for joblib (-1 = all).

    Returns
    -------
    freq : ndarray (n_features,)   selection frequency in [0, 1]
    n_used : int                   number of bootstraps that were usable
    """
    X_filt_raw = np.asarray(X_filt_raw, dtype=float)
    y_labels = np.asarray(y_labels)
    groups = np.asarray(groups)
    n_features = X_filt_raw.shape[1]
    uniq_groups = np.unique(groups)
    child_seeds = np.random.SeedSequence(seed).spawn(n_boot)

    masks = Parallel(n_jobs=n_jobs)(
        delayed(_one_bootstrap)(
            cs, X_filt_raw, y_labels, groups, uniq_groups, preprocess_fn, vip_fn,
            normalization, log_transform, scaling, top_n, min_methods,
            n_features, use_methods)
        for cs in child_seeds
    )

    hit = np.zeros(n_features, dtype=float)
    n_used = 0
    for m in masks:
        if m is not None:
            hit[m] += 1.0
            n_used += 1

    freq = hit / n_used if n_used else np.zeros(n_features)
    return freq, n_used


# ----------------------------------------------------------------------------- #
# Label-permutation null for the overlap counts                                   #
# ----------------------------------------------------------------------------- #
def _one_permutation(child_seed, X_scaled, y_labels, vip_fn, top_n, k_values,
                     n_features, use_methods):
    """Single label-permutation replicate. Returns (per_k_counts_dict,
    feature_hit_mask) or None if degenerate. Pure function of `child_seed`."""
    rng = np.random.default_rng(child_seed)
    yp = rng.permutation(y_labels)
    yp_enc = LabelEncoder().fit_transform(yp)
    model_seed = int(rng.integers(0, 2**31 - 1))
    top_sets = _ensemble_top_sets(X_scaled, yp_enc, yp, top_n, model_seed, vip_fn,
                                  use_methods)
    if len(top_sets) < 2:
        return None
    counts = _consensus_counts(top_sets, n_features)
    per_k = {k: int((counts >= k).sum()) for k in k_values}
    k_min = min(k_values)
    return per_k, (counts >= k_min)


def overlap_permutation_null(X_scaled, y_labels, vip_fn, top_n=50,
                             k_values=(2, 3, 4, 5, 6), n_perm=200, seed=42,
                             use_methods=None, n_jobs=-1):
    """Null distribution of the cross-method overlap under label permutation.

    Preprocessing is label-independent, so the already-scaled descriptive matrix
    `X_scaled` is reused and only the labels are permuted. For each permutation
    the ENABLED ensemble (see `use_methods`) is recomputed and the number of
    features reaching "top-N in >= k methods" is recorded for every k.

    Parallelism & reproducibility
    -----------------------------
    The `n_perm` permutations run across all cores with
    `joblib.Parallel(n_jobs=n_jobs)`. Each permutation draws from its own child
    stream of `np.random.SeedSequence(seed).spawn(n_perm)`, so the null is
    independent and reproducible irrespective of `n_jobs`.

    Returns
    -------
    dict with, per k:
        null_counts[k]   : ndarray (n_perm,)  overlap size under the null
    plus
        per_feature_null_freq : ndarray (n_features,)
            fraction of permutations in which each feature reached >= min(k)
            methods — a per-feature empirical selection p-value.
    """
    X_scaled = np.asarray(X_scaled, dtype=float)
    y_labels = np.asarray(y_labels)
    n_features = X_scaled.shape[1]
    k_min = min(k_values)
    child_seeds = np.random.SeedSequence(seed).spawn(n_perm)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_one_permutation)(
            cs, X_scaled, y_labels, vip_fn, top_n, k_values, n_features,
            use_methods)
        for cs in child_seeds
    )

    null_counts = {k: np.full(n_perm, np.nan, dtype=float) for k in k_values}
    per_feature_hits = np.zeros(n_features, dtype=float)
    n_used = 0
    for i, r in enumerate(results):
        if r is None:
            continue
        per_k, feat_hit = r
        for k in k_values:
            null_counts[k][i] = per_k[k]
        per_feature_hits[feat_hit] += 1.0
        n_used += 1

    per_feature_null_freq = (per_feature_hits / n_used if n_used
                             else np.zeros(n_features))
    return {'null_counts': null_counts,
            'per_feature_null_freq': per_feature_null_freq,
            'n_used': n_used,
            'k_min': k_min}


def enabled_methods_from_config(cfg):
    """Build the set of enabled ensemble method keys from a config module's
    USE_* flags. PLS-DA VIP ('vip') has no toggle and is always included.
    Returns all six methods if `cfg` is None."""
    if cfg is None:
        return set(_ALL_METHODS)
    s = set()
    if getattr(cfg, 'USE_RANDOM_FOREST', True):       s.add('rf')
    if getattr(cfg, 'USE_SVM', True):                 s.add('svm')
    if getattr(cfg, 'USE_GRADIENT_BOOSTING', True):   s.add('gb')
    if getattr(cfg, 'USE_LOGISTIC_REGRESSION', True): s.add('lr')
    if getattr(cfg, 'USE_RIDGE', True):               s.add('ridge')
    s.add('vip')
    return s


def empirical_p(observed, null_samples):
    """One-sided (>=) empirical p-value with the +1/+1 correction
    (Phipson & Smyth 2010), robust to NaNs in the null."""
    null = np.asarray(null_samples, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0:
        return np.nan
    return (np.sum(null >= observed) + 1.0) / (null.size + 1.0)
