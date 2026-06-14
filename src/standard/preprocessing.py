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


def load_experiment(experiment_dir):
    """
    Read all CSVs and TXTs under the experiment folder using DIRECTORY TOPOLOGY.
    Filenames are never parsed for metadata.

    Topology rule (filename-agnostic)
    ----------------------------------
    * Any sub-folder that DIRECTLY contains raw data files is one biological
      sample / condition. The folder's own name is the sample id.
    * That folder's PARENT directory name is the class label (``y_labels``).
      If a sample folder sits directly inside ``experiment_dir`` (legacy 2-level
      layout) there is no class level above it, so the folder is used as its own
      class and a warning is printed — grouped CV then has only one biological
      sample per class.
    * The data files inside a sample folder are technical replicates of that
      sample. They are sorted alphabetically and assigned replicate indices
      1, 2, 3 ...  No part of the filename is inspected to decide this.

    All samples are interpolated onto a common m/z axis.

    Returns
    -------
    X      : ndarray (n_samples, n_features)
    labels : ndarray (n_samples,)   — class label per sample (parent folder)
    names  : list[str]              — each file's path RELATIVE to
             ``experiment_dir`` in POSIX form (e.g. 'Amber/colony_A/inj_03.csv').
             This carries the sample+class topology so make_groups() can rebuild
             leak-free CV groups from the directory structure alone.
    mz     : ndarray (n_features,)  — the common m/z axis
    """
    experiment_dir = os.path.abspath(experiment_dir)
    samples, labels, names = [], [], []
    raw_mz_list = []

    # 1. Discover every folder that DIRECTLY contains data files. Each such
    #    folder is one biological sample. Walk deterministically and skip
    #    hidden directories (e.g. .git, .ipynb_checkpoints).
    sample_dirs = []
    for dirpath, dirnames, filenames in os.walk(experiment_dir):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith('.'))
        if os.path.abspath(dirpath) == experiment_dir:
            continue  # the root itself is not a sample folder
        data_files = sorted(
            f for f in filenames
            if f.lower().endswith(('.csv', '.txt')) and not f.startswith('.')
        )
        if data_files:
            sample_dirs.append((dirpath, data_files))
    sample_dirs.sort(key=lambda t: t[0])

    if not sample_dirs:
        raise ValueError(
            f"No .csv/.txt data files found in any sub-folder of {experiment_dir!r}."
        )

    for dirpath, data_files in sample_dirs:
        dirpath_norm = dirpath.rstrip(os.sep)
        sample_id    = os.path.basename(dirpath_norm)            # leaf folder
        parent       = os.path.dirname(dirpath_norm)             # one level up

        if os.path.abspath(parent) == experiment_dir:
            # 2-level layout: no class folder above the sample folder.
            class_label = sample_id
            print(f"  [warning] '{sample_id}' sits directly under the experiment "
                  f"root, so it is treated as BOTH the class and its only "
                  f"biological sample. Grouped cross-validation needs >= 2 "
                  f"samples per class — nest sample folders one level deeper "
                  f"(experiment/<condition>/<sample>/files) for leak-free CV.")
        else:
            class_label = os.path.basename(parent)

        # Files sorted alphabetically => technical replicates 1, 2, 3 ...
        for rep_idx, fname in enumerate(data_files, start=1):
            file_path = os.path.join(dirpath, fname)
            sep = '\t' if fname.lower().endswith('.txt') else ','
            df = pd.read_csv(file_path, sep=sep, encoding='utf-8-sig')
            df.columns = df.columns.str.lstrip('﻿').str.strip()
            col_map = {c.lower(): c for c in df.columns}
            mz_col  = col_map.get('mz')  or col_map.get('mass/charge')
            int_col = col_map.get('int') or col_map.get('intensity')

            if mz_col is None or int_col is None:
                print(f"  [warning] Skipping {fname}: could not find m/z or "
                      f"intensity columns. Found: {list(df.columns)}")
                continue

            raw_mz_list.append(df[mz_col].values)
            samples.append(df[int_col].values)
            labels.append(class_label.strip())
            # Relative POSIX path = the topology carrier consumed by make_groups.
            rel = os.path.relpath(file_path, experiment_dir).replace(os.sep, '/')
            names.append(rel)

    if not raw_mz_list:
        raise ValueError(
            "No readable spectra: every file was missing m/z / intensity columns."
        )

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
    """
    mean = X.mean(axis=0)
    # Zero-guard: features with a (near-)zero mean would make RSD blow up to
    # inf/NaN and corrupt the percentile threshold. Floor the denominator and
    # force RSD=0 for those features so they sort to the bottom and are dropped.
    mean_safe = np.where(np.abs(mean) < 1e-12, 1e-12, mean)
    rsd = X.std(axis=0) / mean_safe
    rsd = np.where(np.abs(mean) < 1e-12, 0.0, rsd)
    rsd = np.where(np.isfinite(rsd), rsd, 0.0)
    threshold = np.percentile(rsd, percentile)
    keep = rsd > threshold
    return X[:, keep], mz[keep]


def filter_low_abundance(X, mz, percentile=5):
    """
    Remove features with low mean intensity.
    Uses mean (not median) and 5% cutoff to match MetaboAnalyst.
    """
    mean_intensity = np.mean(X, axis=0)
    threshold = np.percentile(mean_intensity, percentile)
    keep = mean_intensity > threshold
    return X[:, keep], mz[keep]


def preprocess(X, normalization='tic', log_transform='log10', scaling='autoscale'):
    """
    Normalize, transform, and scale the data.

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
    """

    # -- Normalization -----------------------------------------------------------
    if normalization == 'tic':
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        X = X / row_sums * np.median(row_sums)

    elif normalization == 'median':
        row_medians = np.median(X, axis=1, keepdims=True)
        row_medians[row_medians == 0] = 1
        X = X / row_medians * np.median(row_medians)

    elif normalization == 'pqn':
        # Probabilistic Quotient Normalization
        # Reference: Dieterle et al. (2006) Anal Chem 78:4281
        # 1. TIC-normalize first as a preliminary step
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        X_tic = X / row_sums
        # 2. Reference spectrum = median of all TIC-normalized samples
        reference = np.median(X_tic, axis=0)
        reference[reference == 0] = 1
        # 3. Quotients = each sample divided by reference
        quotients = X_tic / reference
        # 4. Dilution factor = median of quotients per sample
        dilution = np.median(quotients, axis=1, keepdims=True)
        dilution[dilution == 0] = 1
        X = X_tic / dilution

    elif normalization == 'quantile':
        # Quantile normalization: force all samples to same distribution
        ranks = np.argsort(np.argsort(X, axis=1), axis=1)
        sorted_X = np.sort(X, axis=1)
        row_means = sorted_X.mean(axis=0)
        X = row_means[ranks]

    elif normalization == 'none':
        pass

    else:
        raise ValueError(f"Unknown normalization: '{normalization}'. "
                         f"Choose from: 'tic', 'median', 'pqn', 'quantile', 'none'.")

    # -- Transformation ----------------------------------------------------------
    min_positive = X[X > 0].min() if (X > 0).any() else 1e-6
    half_min = min_positive / 2

    if log_transform == 'glog':
        # Generalised-log / asinh transform: arcsinh(x / lambda_)
        # lambda_ = 5th percentile of positive values — a robust noise-floor
        # estimate computed from all data (full-data path; per-fold lambda_ is
        # fitted from the training fold inside the CV pipeline transformers).
        pos_vals  = X[X > 0]
        lambda_   = np.percentile(pos_vals, 5) if len(pos_vals) else 1.0
        lambda_   = max(float(lambda_), 1e-10)
        X = np.arcsinh(X / lambda_)
    elif log_transform == 'log10':
        X = np.log10(X + half_min)
    elif log_transform == 'log2':
        X = np.log2(X + half_min)
    elif log_transform == 'sqrt':
        X = np.sqrt(X)
    elif log_transform == 'none':
        pass
    else:
        raise ValueError(f"Unknown log_transform: '{log_transform}'. "
                         f"Choose from: 'glog', 'log10', 'log2', 'sqrt', 'none'.")

    # -- Scaling -----------------------------------------------------------------
    feat_mean = X.mean(axis=0)
    feat_std  = X.std(axis=0, ddof=1)
    feat_std[feat_std == 0] = 1

    if scaling == 'autoscale':
        X = (X - feat_mean) / feat_std
    elif scaling == 'pareto':
        X = (X - feat_mean) / np.sqrt(feat_std)
    elif scaling == 'range':
        feat_range = X.max(axis=0) - X.min(axis=0)
        feat_range[feat_range == 0] = 1
        X = (X - feat_mean) / feat_range
    elif scaling == 'vast':
        X = ((X - feat_mean) / feat_std) * (feat_mean / (feat_std + 1e-10))
    elif scaling == 'level':
        level = np.abs(feat_mean)
        level[level == 0] = 1
        X = (X - feat_mean) / level
    elif scaling == 'none':
        X = X - feat_mean
    else:
        raise ValueError(f"Unknown scaling: '{scaling}'. "
                         f"Choose from: 'autoscale', 'pareto', 'range', 'vast', 'level', 'none'.")

    return X
