"""Evidence-based grouping of m/z bins into spectral signal groups.

Fixed-width bins can split one unit-resolution peak across a bin edge, so a
single ion may appear as two adjacent, strongly co-varying features. This module
proposes adjacent-bin CANDIDATES and merges a candidate only when the raw
(native-resolution) profiles support a single peak envelope. Correlation and m/z
proximity alone are never treated as proof of a shared chemical origin.

Merge criteria for adjacent bins i, i+1 (window W = [edge_i, edge_i+2)), all of which
must hold (defaults, fixed a priori):
  1. candidate:  Pearson r >= r_min between colony-mean log intensities;
  2. apex:       the mean raw profile has its maximum over W inside W, and that
                 maximum is a local maximum of the extended window W +/- 0.5 Da
                 (the pair is not merely a shoulder of a neighbouring peak);
  3. straddle:   that apex lies within bin_width/2 of the shared bin edge, i.e. the
                 peak genuinely spans the edge (a bin holding only the tail of a
                 neighbouring peak, which may also contain another ion, is NOT merged);
     split:      each bin holds >= min_share of the pair's mean intensity;
  4. alignment:  per-spectrum apex positions within W have SD <= apex_sd_max Da;
  5. shape:      SD of log10(bin_i / bin_i+1) across spectra <= logratio_sd_max.
Candidates failing 2-5 are kept separate and labelled 'co-varying, unresolved'.
Isotopologue (+1.003 Da) and common adduct spacings are FLAGGED only, never merged.

All steps are label-free (colony identities are used, class labels are not).
"""
import numpy as np
import pandas as pd

ADDUCT_SPACINGS = {'Na-H (+21.982)': 21.982, 'K-H (+37.956)': 37.956,
                   'NH4-H (+17.027)': 17.027}
ISOTOPE_SPACING = 1.003


def _log_tic(X):
    X = np.asarray(X, dtype=float)
    s = X.sum(1, keepdims=True)
    s[s == 0] = 1.0
    Xn = X / s * np.median(s)
    pos = Xn[Xn > 0]
    half = pos.min() / 2 if pos.size else 1e-6
    return np.log10(Xn + half)


def colony_means(X, groups):
    groups = np.asarray(groups)
    u = list(dict.fromkeys(groups.tolist()))
    return np.vstack([X[groups == g].mean(0) for g in u]), u


def _bin_edges_of(mzb, mz0, bin_width):
    return mz0 + np.floor((np.asarray(mzb) - mz0) / bin_width + 1e-9) * bin_width


def build_signal_groups(X_binned, mz_binned, raw_X, raw_mz, groups, bin_width=0.5,
                        mz0=None, r_min=0.9, min_share=0.15, apex_sd_max=0.15,
                        logratio_sd_max=0.15):
    """Return (map_df, pair_df).

    map_df : one row per bin: bin_index, mz_bin, signal_group, is_representative,
             signal_mz (apex of the mean raw profile in the group's window) and
             signal_centroid_mz; both nominal labels, not accurate masses.
    pair_df: one row per adjacent candidate pair with all evidence and the decision.
    """
    X_binned = np.asarray(X_binned, float)
    mz_binned = np.asarray(mz_binned, float)
    raw_X = np.asarray(raw_X, float)
    raw_mz = np.asarray(raw_mz, float)
    if mz0 is None:
        mz0 = float(raw_mz.min())
    edges = _bin_edges_of(mz_binned, mz0, bin_width)
    cm, _ = colony_means(_log_tic(X_binned), groups)
    cmc = cm - cm.mean(0)
    norm = np.sqrt((cmc ** 2).sum(0))
    norm[norm == 0] = np.inf
    mean_raw = raw_X.mean(0)
    p = len(mz_binned)
    parent = np.arange(p)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    rows = []
    for i in range(p - 1):
        if not np.isclose(edges[i + 1] - edges[i], bin_width):
            continue
        r = float(cmc[:, i] @ cmc[:, i + 1] / (norm[i] * norm[i + 1]))
        if r < r_min:
            continue
        lo, hi = edges[i], edges[i] + 2 * bin_width
        w = (raw_mz >= lo) & (raw_mz < hi)
        ext = (raw_mz >= lo - bin_width) & (raw_mz < hi + bin_width)
        prof = mean_raw[w]
        apex_mz = float(raw_mz[w][np.argmax(prof)])
        apex_ok = bool(prof.max() >= mean_raw[ext].max() * 0.999 or
                       (np.argmax(prof) not in (0, prof.size - 1)))
        a, b = X_binned[:, i], X_binned[:, i + 1]
        share = float(min(a.mean(), b.mean()) / max(a.mean() + b.mean(), 1e-300))
        apex_each = raw_mz[w][np.argmax(raw_X[:, w], axis=1)]
        apex_sd = float(np.std(apex_each, ddof=1))
        with np.errstate(divide='ignore', invalid='ignore'):
            lr = np.log10(np.where((a > 0) & (b > 0), a / b, np.nan))
        lr_sd = float(np.nanstd(lr, ddof=1)) if np.isfinite(lr).sum() > 2 else np.inf
        shared_edge = edges[i] + bin_width
        straddle = abs(apex_mz - shared_edge) <= bin_width / 2 + 1e-9
        crit = dict(apex=apex_ok, straddle=bool(straddle), split=share >= min_share,
                    alignment=apex_sd <= apex_sd_max, shape=lr_sd <= logratio_sd_max)
        merge = all(crit.values())
        rows.append(dict(bin_i=i, bin_j=i + 1, mz_i=mz_binned[i], mz_j=mz_binned[i + 1],
                         r_colony_means=r, apex_mz=apex_mz, min_share=share,
                         apex_sd_da=apex_sd, logratio_sd=lr_sd,
                         **{f'pass_{k}': v for k, v in crit.items()},
                         decision='merged' if merge else 'co-varying, unresolved'))
        if merge:
            parent[find(i + 1)] = find(i)
    roots = np.array([find(i) for i in range(p)])
    _, gid = np.unique(roots, return_inverse=True)
    mean_bin = X_binned.mean(0)
    map_df = pd.DataFrame({'bin_index': np.arange(p), 'mz_bin': mz_binned,
                           'bin_lo': edges, 'signal_group': gid})
    rep = map_df.assign(m=mean_bin).groupby('signal_group')['m'].idxmax()
    map_df['is_representative'] = map_df.index.isin(rep.values)
    cent, apex = {}, {}
    for g, sub in map_df.groupby('signal_group'):
        lo, hi = sub.bin_lo.min(), sub.bin_lo.max() + bin_width
        w = (raw_mz >= lo) & (raw_mz < hi)
        wt = mean_raw[w]
        cent[g] = float((raw_mz[w] * wt).sum() / wt.sum()) if wt.sum() > 0 else float(sub.mz_bin.mean())
        apex[g] = float(raw_mz[w][np.argmax(wt)]) if wt.size else float(sub.mz_bin.mean())
    # signal_mz: apex of the mean raw profile inside the group window (nominal label;
    # a truncated-window centroid is biased toward the window centre at unit resolution).
    map_df['signal_mz'] = map_df.signal_group.map(apex)
    map_df['signal_centroid_mz'] = map_df.signal_group.map(cent)
    map_df['n_bins_in_group'] = map_df.signal_group.map(map_df.signal_group.value_counts())
    return map_df, pd.DataFrame(rows)


def aggregate_to_groups(X_binned, map_df):
    """Sum raw (linear) bin intensities within each signal group.
    Returns (X_groups, group_ids, signal_mz) ordered by signal group id."""
    gids = np.sort(map_df.signal_group.unique())
    Xg = np.column_stack([X_binned[:, map_df.loc[map_df.signal_group == g, 'bin_index'].values].sum(1)
                          for g in gids])
    smz = map_df.drop_duplicates('signal_group').set_index('signal_group').loc[gids, 'signal_mz'].values
    return Xg, gids, smz


def flag_relations(signal_mz, X_groups, groups, r_min=0.8, tol=0.15, adduct_tol=0.3):
    """Flag candidate isotopologue / adduct relations between signal groups
    (positive colony-mean correlation >= r_min). Flags only: never merged."""
    cm, _ = colony_means(_log_tic(X_groups), groups)
    C = np.corrcoef(cm.T)
    mz = np.asarray(signal_mz)
    inten = np.asarray(X_groups).mean(0)
    out = []
    for i in range(len(mz)):
        d = mz - mz[i]
        for j in np.where((d > ISOTOPE_SPACING - tol) & (d < ISOTOPE_SPACING + tol))[0]:
            if C[i, j] >= r_min and inten[j] < inten[i]:
                out.append(dict(group_i=i, group_j=int(j), mz_i=mz[i], mz_j=mz[j], r=C[i, j],
                                relation='candidate 13C isotopologue (+1.003)'))
        for name, sp in ADDUCT_SPACINGS.items():
            for j in np.where(np.abs(d - sp) <= adduct_tol)[0]:
                if C[i, j] >= r_min:
                    out.append(dict(group_i=i, group_j=int(j), mz_i=mz[i], mz_j=mz[j], r=C[i, j],
                                    relation=f'candidate adduct spacing {name}'))
    return pd.DataFrame(out, columns=['group_i', 'group_j', 'mz_i', 'mz_j', 'r', 'relation'])
