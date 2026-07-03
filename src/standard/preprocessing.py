"""
Preprocessing functions for mass spectrometry data — standard version.

This is a drop-in replacement for preprocessing.py with one difference:
bin_features() labels each bin at its true geometric center (bin_edge + bin_width/2)
rather than applying MetaboAnalyst's -0.05 Da cosmetic offset.

Use this version when MetaboAnalyst validation is not the goal.
Use preprocessing.py when you need output labels to match MetaboAnalyst exactly.
"""

import os
import numpy as np
import pandas as pd


def load_experiment(experiment_dir, tech_rep_pattern=None, validate=True):
    """
    Read all CSVs and TXTs under the experiment folder and derive class + CV-group
    topology with a flexible, **bottom-up relative-depth** rule (see grouping.py).
    Both common layouts are accommodated without forcing a single folder shape.

    Layouts (relative to ``experiment_dir``)
    -----------------------------------------
    * **Nested** ``<...>/<condition>/<biological_replicate>/file`` (≥2 dir levels):
      the file's immediate parent folder is the biological replicate (CV group);
      everything above it is joined to form the class label. Files in that folder
      are technical replicates and stay grouped.
    * **Flat** ``<condition>/file`` (exactly 1 dir level): the folder is the class;
      the biological replicate is recovered from the filename prefix by stripping a
      trailing technical-replicate token (default ``T<number>``, e.g. ``A1T3``), so
      ``S5_Green_A1T{1,2,3}.csv`` collapse to replicate ``S5_Green_A1``.

    Class labels and CV groups are produced by ``grouping.parse_sample`` /
    ``grouping.make_groups`` so the two are always consistent. With ``validate=True``
    (default) a strict pre-flight check runs after parsing and RAISES (listing the
    offending folders) if any class has fewer than two biological groups — no silent
    warn-and-continue.

    Parameters
    ----------
    experiment_dir : str
    tech_rep_pattern : None | str | compiled regex
        Trailing technical-replicate token for the flat-layout prefix rule.
    validate : bool
        Run the ≥2-groups-per-class pre-flight check and raise on failure.

    Returns
    -------
    X      : ndarray (n_samples, n_features)
    labels : ndarray (n_samples,)   — class label per sample (topology-derived)
    names  : list[str]              — each file's path RELATIVE to
             ``experiment_dir`` in POSIX form. Carries the sample+class topology
             so make_groups() rebuilds the identical leak-free CV groups.
    mz     : ndarray (n_features,)  — the common m/z axis
    """
    # Local import keeps grouping a single source of truth for the topology rule
    # and avoids any import-order coupling at module load time.
    from shared.grouping import parse_sample, make_groups, check_batch_confound

    experiment_dir = os.path.abspath(experiment_dir)
    samples, labels, names = [], [], []
    raw_mz_list = []

    # 1. Discover every data file anywhere under the experiment dir (any depth),
    #    walking deterministically and skipping hidden dirs (.git, .ipynb_checkpoints).
    discovered = []
    for dirpath, dirnames, filenames in os.walk(experiment_dir):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith('.'))
        for fname in sorted(filenames):
            if fname.lower().endswith(('.csv', '.txt')) and not fname.startswith('.'):
                file_path = os.path.join(dirpath, fname)
                rel = os.path.relpath(file_path, experiment_dir).replace(os.sep, '/')
                discovered.append((rel, file_path))
    discovered.sort(key=lambda t: t[0])

    if not discovered:
        raise ValueError(
            f"No .csv/.txt data files found anywhere under {experiment_dir!r}."
        )

    for rel, file_path in discovered:
        # Topology -> class label (raises for a bare filename with no condition dir).
        class_label, _group_id, _source = parse_sample(rel, tech_rep_pattern)

        fname = os.path.basename(file_path)
        sep = '\t' if fname.lower().endswith('.txt') else ','
        df = pd.read_csv(file_path, sep=sep, encoding='utf-8-sig')
        df.columns = df.columns.str.lstrip('﻿').str.strip()
        col_map = {c.lower(): c for c in df.columns}
        mz_col  = col_map.get('mz')  or col_map.get('mass/charge')
        int_col = col_map.get('int') or col_map.get('intensity')

        if mz_col is None or int_col is None:
            print(f"  [warning] Skipping {rel}: could not find m/z or "
                  f"intensity columns. Found: {list(df.columns)}")
            continue

        raw_mz_list.append(df[mz_col].values)
        samples.append(df[int_col].values)
        labels.append(class_label.strip())
        names.append(rel)

    if not raw_mz_list:
        raise ValueError(
            "No readable spectra: every file was missing m/z / intensity columns."
        )

    # Fail-fast pre-flight: rebuild the groups now and validate (raises loudly if
    # any class has < 2 biological groups), so problems surface here, not deep in
    # StratifiedGroupKFold. Same call run_analysis makes later -> identical groups.
    if validate:
        _groups = make_groups(np.array(labels), names,
                              tech_rep_pattern=tech_rep_pattern, validate=True)
        # B4: non-fatal advisory — flag single-replicate dominance / thin
        # replication that grouped CV cannot detect on its own.
        check_batch_confound(np.array(labels), _groups, names=names)

    # 2. Build a common m/z axis covering the overlap range across all files.
    mz_min = max(mz.min() for mz in raw_mz_list)
    mz_max = min(mz.max() for mz in raw_mz_list)
    common_mz = np.linspace(mz_min, mz_max, num=5000)

    # Interpolate every sample onto the common axis.
    interpolated = []
    for mz_i, int_i in zip(raw_mz_list, samples):
        interp_int = np.interp(common_mz, mz_i, int_i)
        interpolated.append(interp_int)

    return np.array(interpolated, dtype=float), np.array(labels), names, common_mz

def filter_mass_range(X, mz, mz_min=100, mz_max=1000):
    keep = (mz >= mz_min) & (mz <= mz_max)
    return X[:, keep], mz[keep]

def bin_features(X, mz, bin_width=0.5):
    """
    Bin m/z features into fixed-width bins by summing intensities.
    Each bin is labeled with the mean of the actual m/z values that fell into it —
    i.e. where the signal genuinely was, not an arithmetic midpoint.

    This differs from preprocessing.py, which labels bins at a fixed arithmetic
    center shifted by -0.05 Da to match MetaboAnalyst's output convention.
    That label has no physical meaning. This version gives you the real m/z.
    """
    bin_edges = np.arange(mz.min(), mz.max() + bin_width, bin_width)

    n_bins = len(bin_edges) - 1
    X_binned  = np.zeros((X.shape[0], n_bins))
    bin_labels = np.zeros(n_bins)

    for i, (low, high) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (mz >= low) & (mz < high)
        if mask.any():
            X_binned[:, i]  = X[:, mask].sum(axis=1)
            bin_labels[i]   = mz[mask].mean()   # mean of actual m/z values in this bin
        else:
            bin_labels[i]   = (low + high) / 2  # fallback for empty bins

    non_empty = X_binned.sum(axis=0) > 0
    return X_binned[:, non_empty], bin_labels[non_empty]


def filter_prevalence(X, mz, y_labels, threshold=0.5, min_intensity=0.0):
    """
    Remove features not genuinely detected in at least one condition group.

    A sample is considered to have a 'genuine detection' for a feature when its
    raw intensity is strictly above `min_intensity` (default 0.0). Values at or
    below this are treated as imputed (e.g. half-minimum filled zeros) and do
    NOT count toward the prevalence fraction.

    A feature is retained if, in AT LEAST ONE group, the fraction of samples
    with genuine detection is >= `threshold`.  Features that only survive because
    of imputed values — even across many samples — are dropped.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
        Raw (pre-log, pre-scaling) feature matrix, e.g. the binned intensity
        matrix straight out of bin_features() / filter_mass_range().
    mz : ndarray (n_features,)
        m/z value for each feature column.
    y_labels : array-like (n_samples,)
        Condition/group label per sample. Must be aligned with X rows.
    threshold : float
        Minimum required detection rate within a group (0–1).
        0.5 = at least 50 % of samples in some group must show real signal.
        Raise to 0.8 for stricter filtering when you have enough replicates.
    min_intensity : float
        Intensity values <= this are treated as not detected / imputed.
        Default 0.0 (zero-intensity bins = no signal detected).

    Returns
    -------
    X_filt   : ndarray (n_samples, n_kept_features)
    mz_filt  : ndarray (n_kept_features,)
    """
    y_labels  = np.asarray(y_labels)
    groups    = np.unique(y_labels)
    n_features = X.shape[1]

    # Vectorized detection mask: True wherever raw intensity is genuine signal.
    # Shape (n_samples, n_features) — computed once, reused for every group.
    detected = X > min_intensity

    # keep[j] will be set True as soon as feature j passes in any group.
    keep = np.zeros(n_features, dtype=bool)

    for group in groups:
        group_mask = (y_labels == group)              # boolean row selector

        # --- Group-wise prevalence check (vectorized) ---
        # detected[group_mask] has shape (n_group_samples, n_features).
        # .mean(axis=0) gives the fraction of group samples with real signal
        # for every feature simultaneously — no Python loop over features.
        group_prevalence = detected[group_mask].mean(axis=0)   # (n_features,)

        # Feature passes if prevalence reaches the threshold in THIS group.
        keep |= (group_prevalence >= threshold)

    n_removed = n_features - keep.sum()
    print(f"  Prevalence filter (>={threshold*100:.0f}% genuine detections "
          f"in >= 1 group): removed {n_removed} / {n_features} features, "
          f"{keep.sum()} retained")

    # Per-group breakdown so you can audit which groups are driving retention.
    for group in groups:
        group_mask = (y_labels == group)
        gp = detected[group_mask].mean(axis=0)
        n_pass = int((gp >= threshold).sum())
        print(f"    {group}: {n_pass} features reach >={threshold*100:.0f}% detection")

    return X[:, keep], mz[keep]


def filter_snr_floor(X, mz, y_labels, snr_threshold=3, noise_quantile=60, min_fraction=0.5):
    """
    Remove bins whose signal never exceeds the per-sample baseline noise in any
    condition group.  Applied on RAW binned intensities, BEFORE normalization.

    ROUTING (A2): this is a label/group-aware (supervised) filter and is intended
    as a DESCRIPTIVE-ONLY characterisation step. Apply it to the descriptive copy
    of the data (the matrix used for full-data PLS-DA / VIP / the ensemble feature
    list) — NOT to the matrix that feeds cross-validation. Inside the grouped CV,
    leave it out by default (config.SNR_FLOOR_IN_CV = False); the per-fold
    SNRFloor transformer in make_preprocessor exists only for the opt-in case and,
    when enabled, is fit on the training fold only. Feeding an SNR-floored matrix
    straight into CV would select features using held-out labels (leakage).

    Per-sample robust noise
    -----------------------
    For each sample i, noise is estimated from its own spectrum only (no
    cross-sample information):

        low_bins_i  = X[i, mz_bins] where X[i, b] <= percentile(X[i,:], noise_quantile)
        sigma_i     = 1.4826 * MAD(low_bins_i)

    where 1.4826 scales MAD to be consistent with the Gaussian std-dev.
    If all bins have the same value (degenerate) sigma_i is set to a small
    guard value so SNR still becomes finite.

    Group-aware, discovery-preserving
    ----------------------------------
    Feature j is KEPT if, within at LEAST ONE condition group, the fraction of
    samples where SNR[i,j] = X[i,j]/sigma_i >= snr_threshold is >= min_fraction.
    This mirrors filter_prevalence: a feature present in only one group is
    retained; requiring it to pass in EVERY group would discard condition-
    specific metabolites.

    Parameters
    ----------
    X              : ndarray (n_samples, n_features) — raw binned intensities
    mz             : ndarray (n_features,) — m/z per bin
    y_labels       : array-like (n_samples,) — condition label per sample
    snr_threshold  : float — minimum SNR to count a sample as detected (default 3)
    noise_quantile : float — percentile of each sample's intensities used to
                     identify the noise region (default 60)
    min_fraction   : float — minimum fraction of group samples that must meet
                     snr_threshold for the feature to be kept (default 0.5)

    Returns
    -------
    X_filt  : ndarray (n_samples, n_kept)
    mz_filt : ndarray (n_kept,)
    """
    y_labels   = np.asarray(y_labels)
    groups     = np.unique(y_labels)
    n_samples, n_features = X.shape

    # Per-sample robust noise: sigma_i = 1.4826 * MAD of the low-intensity bins
    sigma = np.empty(n_samples)
    for i in range(n_samples):
        row       = X[i]
        threshold = np.percentile(row, noise_quantile)
        low_vals  = row[row <= threshold]
        if len(low_vals) >= 2:
            med  = np.median(low_vals)
            mad  = np.median(np.abs(low_vals - med))
            if mad > 0:
                sigma[i] = 1.4826 * mad
            else:
                # MAD = 0 means all noise-region values are identical — the noise
                # floor IS that constant level.  Use the median as sigma so that
                # a feature at exactly that level gets SNR = 1.0 (below threshold).
                sigma[i] = max(abs(med), 1e-10)
        else:
            sigma[i] = max(float(row.min()), 1e-10)

    # SNR matrix: shape (n_samples, n_features)
    SNR = X / sigma[:, np.newaxis]

    # Group-aware keep: retain feature j if it passes in >= 1 group
    keep = np.zeros(n_features, dtype=bool)
    group_info = []
    for group in groups:
        g_mask     = (y_labels == group)
        passes     = (SNR[g_mask] >= snr_threshold).mean(axis=0) >= min_fraction
        keep      |= passes
        group_info.append((group, int(passes.sum())))

    n_removed = n_features - keep.sum()
    print(f"  SNR floor (SNR>={snr_threshold}, noise_q={noise_quantile}, "
          f"min_frac={min_fraction}): removed {n_removed} / {n_features} features, "
          f"{keep.sum()} retained")
    for group, n_pass in group_info:
        print(f"    {group}: {n_pass} features meet SNR threshold")

    return X[:, keep], mz[keep]


def filter_low_variance(X, mz, percentile=25):
    """
    Remove features with low relative standard deviation (RSD).
    Matches MetaboAnalyst's 'Interquartile range' filter at 25%.

    The retention rule is delegated to ``variance_keep_mask`` so the full-data
    descriptive path and the per-fold ``VarianceFilter`` transformer apply the
    identical RSD computation and zero-mean guard.
    """
    keep = variance_keep_mask(X, percentile)
    return X[:, keep], mz[keep]


def filter_low_abundance(X, mz, percentile=5):
    """
    Remove features with low mean intensity.
    Uses mean (not median) and 5% cutoff to match MetaboAnalyst.

    The retention rule is delegated to ``abundance_keep_mask`` so the full-data
    descriptive path and the per-fold ``AbundanceFilter`` transformer share one
    implementation.
    """
    keep = abundance_keep_mask(X, percentile)
    return X[:, keep], mz[keep]


# ── Feature-retention masks (single source of truth) ────────────────────────────
# Both the full-data descriptive filters (filter_low_variance / filter_low_abundance)
# and the per-fold CV transformers (VarianceFilter / AbundanceFilter) call these,
# so feature selection is computed identically on both paths.

def variance_keep_mask(X, percentile=25):
    """Boolean keep-mask for the low-relative-variance (RSD) filter.

    Features with an (near-)zero mean would drive RSD to inf/NaN and corrupt the
    percentile threshold, so the denominator is floored and their RSD forced to
    0 (they sort to the bottom and are dropped). ``percentile <= 0`` disables the
    filter (all features kept).
    """
    X = np.asarray(X, dtype=float)
    if percentile <= 0:
        return np.ones(X.shape[1], dtype=bool)
    mean = X.mean(axis=0)
    mean_safe = np.where(np.abs(mean) < 1e-12, 1e-12, mean)
    rsd = X.std(axis=0) / mean_safe
    rsd = np.where(np.abs(mean) < 1e-12, 0.0, rsd)
    rsd = np.where(np.isfinite(rsd), rsd, 0.0)
    return rsd > np.percentile(rsd, percentile)


def abundance_keep_mask(X, percentile=5):
    """Boolean keep-mask for the low-mean-intensity filter.

    ``percentile <= 0`` disables the filter (all features kept).
    """
    X = np.asarray(X, dtype=float)
    if percentile <= 0:
        return np.ones(X.shape[1], dtype=bool)
    mean_intensity = X.mean(axis=0)
    return mean_intensity > np.percentile(mean_intensity, percentile)


# ── Normalisation / transformation / scaling primitives ─────────────────────────
# Canonical fit_*/apply_* pairs. The full-data preprocess() below AND the per-fold
# transformer classes in shared/classifier_comparison*.py both delegate here, so
# the descriptive path and the in-fold CV path apply byte-identical mathematics —
# eliminating the drift risk of two parallel implementations.
#
#   * on the full-data path fit and apply see the same matrix, reproducing the
#     previous monolithic preprocess() exactly;
#   * inside CV, fit runs on the training fold and apply runs on the held-out
#     fold, so no test-fold statistic ever informs a training transform.

def fit_normalization(X, method):
    """Fit sample-normalisation parameters on X. Returns a params dict."""
    X = np.asarray(X, dtype=float)
    if method == 'tic':
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return {'const': float(np.median(row_sums))}
    if method == 'median':
        row_medians = np.median(X, axis=1, keepdims=True)
        row_medians = np.where(row_medians == 0, 1.0, row_medians)
        return {'const': float(np.median(row_medians))}
    if method == 'pqn':
        # Probabilistic Quotient Normalization (Dieterle et al. 2006 Anal Chem 78:4281).
        # Reference spectrum = median of the TIC-normalised training samples.
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        reference = np.median(X / row_sums, axis=0)
        reference = np.where(reference == 0, 1.0, reference)
        return {'reference': reference}
    if method == 'quantile':
        return {'row_means': np.sort(X, axis=1).mean(axis=0)}
    if method == 'none':
        return {}
    raise ValueError(f"Unknown normalization: '{method}'. "
                     f"Choose from: 'tic', 'median', 'pqn', 'quantile', 'none'.")


def apply_normalization(X, method, params):
    """Apply fitted sample normalisation to X."""
    X = np.asarray(X, dtype=float)
    if method == 'tic':
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return X / row_sums * params['const']
    if method == 'median':
        row_medians = np.median(X, axis=1, keepdims=True)
        row_medians = np.where(row_medians == 0, 1.0, row_medians)
        return X / row_medians * params['const']
    if method == 'pqn':
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        X_tic = X / row_sums
        quotients = X_tic / params['reference']
        dilution = np.median(quotients, axis=1, keepdims=True)
        dilution = np.where(dilution == 0, 1.0, dilution)
        return X_tic / dilution
    if method == 'quantile':
        ranks = np.argsort(np.argsort(X, axis=1), axis=1)
        return params['row_means'][ranks]
    if method == 'none':
        return X
    raise ValueError(f"Unknown normalization: '{method}'.")


def fit_transform_params(X, method):
    """Fit transform parameters (glog lambda / log half-min offset) on X."""
    X = np.asarray(X, dtype=float)
    if method in ('log10', 'log2'):
        min_positive = X[X > 0].min() if (X > 0).any() else 1e-6
        return {'half_min': min_positive / 2.0}
    if method == 'glog':
        # lambda_ = 5th percentile of positive values — a robust noise-floor estimate.
        pos_vals = X[X > 0]
        lambda_ = np.percentile(pos_vals, 5) if len(pos_vals) else 1.0
        return {'lambda': max(float(lambda_), 1e-10)}
    if method in ('sqrt', 'none'):
        return {}
    raise ValueError(f"Unknown log_transform: '{method}'. "
                     f"Choose from: 'glog', 'log10', 'log2', 'sqrt', 'none'.")


def apply_transform(X, method, params):
    """Apply the fitted transform to X."""
    X = np.asarray(X, dtype=float)
    if method == 'glog':
        return np.arcsinh(X / params['lambda'])
    if method == 'log10':
        return np.log10(X + params['half_min'])
    if method == 'log2':
        return np.log2(X + params['half_min'])
    if method == 'sqrt':
        return np.sqrt(X)
    if method == 'none':
        return X
    raise ValueError(f"Unknown log_transform: '{method}'.")


def fit_scaling(X, method):
    """Fit per-feature scaling parameters on X."""
    X = np.asarray(X, dtype=float)
    feat_mean = X.mean(axis=0)
    feat_std = X.std(axis=0, ddof=1)
    feat_std = np.where(feat_std == 0, 1.0, feat_std)
    params = {'mean': feat_mean, 'std': feat_std}
    if method == 'range':
        feat_range = X.max(axis=0) - X.min(axis=0)
        params['range'] = np.where(feat_range == 0, 1.0, feat_range)
    elif method not in ('autoscale', 'pareto', 'vast', 'level', 'none'):
        raise ValueError(f"Unknown scaling: '{method}'. Choose from: "
                         f"'autoscale', 'pareto', 'range', 'vast', 'level', 'none'.")
    return params


def apply_scaling(X, method, params):
    """Apply the fitted scaling to X."""
    X = np.asarray(X, dtype=float)
    mean = params['mean']
    std = params['std']
    if method == 'autoscale':
        return (X - mean) / std
    if method == 'pareto':
        return (X - mean) / np.sqrt(std)
    if method == 'range':
        return (X - mean) / params['range']
    if method == 'vast':
        return ((X - mean) / std) * (mean / (std + 1e-10))
    if method == 'level':
        level = np.abs(mean)
        level = np.where(level == 0, 1.0, level)
        return (X - mean) / level
    if method == 'none':
        return X - mean
    raise ValueError(f"Unknown scaling: '{method}'.")


def preprocess(X, normalization='tic', log_transform='log10', scaling='autoscale',
               return_stages=False):
    """
    Normalize, transform, and scale the data.

    The three operations are delegated to the shared fit_*/apply_* primitives, so
    this full-data path applies the identical mathematics to the per-fold
    transformers used inside cross-validation. On this path fit and apply see the
    same matrix (reproducing the previous monolithic implementation exactly).

    Parameters
    ----------
    normalization : str
        Sample normalization method. Options: 'tic', 'median', 'pqn', 'quantile', 'none'.
        See Wu & Li (2016) Sci Rep 6:38881 for a comparison of methods.
    log_transform : str
        Transformation to apply.
        Options: 'glog', 'log10', 'log2', 'sqrt', 'none'.
        'glog' = arcsinh(X / lambda_) where lambda_ = 5th percentile of positive
        values.  Linear near zero (suppresses baseline noise amplification),
        log-like at high intensity, handles exact zeros without a half-min offset.
    scaling : str
        Scaling method. Options: 'autoscale', 'pareto', 'range', 'vast', 'level', 'none'.
        See van den Berg et al. (2006) BMC Genomics 7:142 for guidance.
    return_stages : bool
        If True, return the intermediate matrices instead of only the final scaled
        matrix::

            {'normalized':  linear normalised matrix (BEFORE transform/scaling),
             'transformed': after the log/glog transform,
             'scaled':      final normalised+transformed+scaled matrix}

        The 'normalized' matrix is the linear, non-negative intensity matrix the
        chemometric QC metrics (technical %CV, D-ratio) require; it must NOT be
        derived from the transformed or scaled outputs.

    Returns
    -------
    ndarray
        Final scaled matrix when ``return_stages`` is False (default), otherwise a
        dict of the three stage matrices described above.
    """
    X = np.asarray(X, dtype=float)

    X_normalized = apply_normalization(X, normalization,
                                       fit_normalization(X, normalization))
    X_transformed = apply_transform(X_normalized, log_transform,
                                    fit_transform_params(X_normalized, log_transform))
    X_scaled = apply_scaling(X_transformed, scaling,
                             fit_scaling(X_transformed, scaling))

    if return_stages:
        return {'normalized': X_normalized,
                'transformed': X_transformed,
                'scaled': X_scaled}
    return X_scaled


def normalize_only(X, normalization='tic'):
    """
    Return the LINEAR normalised intensity matrix — normalisation applied, but
    BEFORE any glog/log transform or scaling.

    This is the matrix required by src/shared/quality_metrics.py for technical
    %CV, biological %CV and the Broadhurst D-ratio: coefficient of variation is
    only meaningful on a linear, non-negative intensity scale, so it is isolated
    here rather than derived from the transformed or scaled output. Equivalent to
    ``preprocess(X, normalization, log_transform='none', scaling='none')['normalized']``
    but returned directly.
    """
    X = np.asarray(X, dtype=float)
    return apply_normalization(X, normalization, fit_normalization(X, normalization))
