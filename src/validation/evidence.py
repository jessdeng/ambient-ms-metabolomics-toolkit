"""Spectral and replicate evidence for prioritised features (plan step 10).

Every quantity is descriptive and label-agnostic in construction; conditions are
used only to summarise. Condition differences are condition-plate associations
(light and plate/enclosure are aliased). Related-ion checks test nominal-mass
spacings only; at unit resolution they cannot establish identities.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from validation.unsupervised import nested_variance_components

RELATED = {'M+1 (13C isotope spacing)': 1.003, 'M+2': 2.0,
           'M+22 (Na vs H adduct spacing)': 21.982, 'M-22 (candidate as Na adduct)': -21.982,
           'M+38 (K vs H adduct spacing)': 37.956, 'M-17 (candidate as NH4 adduct)': -17.027,
           'M-18 (H2O-loss spacing)': -18.011, '2M-1 ([2M+H]+ dimer)': None}


def feature_index(d, mz, tol=0.06):
    j = int(np.argmin(np.abs(d['mz_binned'] - mz)))
    if abs(d['mz_binned'][j] - mz) > tol:
        raise KeyError(f'no feature within {tol} Da of {mz}')
    return j


def normalised_intensity(d, j, how='tic'):
    """Feature intensity (raw, pre-filter feature matrix) per spectrum."""
    raw = d['X_binned'][:, j].astype(float)
    if how == 'none':
        return raw
    if how == 'tic':
        return raw / d['X_raw'].sum(1)
    if how == 'pqn':
        T = d['X_binned'] / d['X_binned'].sum(1, keepdims=True)
        ref = np.median(T, 0); ok = ref > 0
        return (raw / d['X_binned'].sum(1)) / np.median(T[:, ok] / ref[ok], 1)
    raise ValueError(how)


def signal_bins(d, j):
    gm = d['group_map']
    gid = gm.loc[(gm.signal_mz - d['mz_binned'][j]).abs().idxmin(), 'signal_group']
    return gm[gm.signal_group == gid]


def feature_evidence(d, mz, window=1.6):
    """Replicate structure, raw-profile shape and normalisation sensitivity."""
    y, g = np.asarray(d['y']), np.asarray(d['groups']); j = feature_index(d, mz)
    bins = signal_bins(d, j)
    rel = normalised_intensity(d, j); vc = nested_variance_components(np.log10(rel + 1e-12), y, g)
    cols = list(dict.fromkeys(g)); conds = list(dict.fromkeys(y))
    cv = np.array([rel[g == c].std(ddof=1) / rel[g == c].mean() * 100 if rel[g == c].mean() > 0 else np.nan for c in cols])
    col_cond = np.array([y[g == c][0] for c in cols]); col_mean = np.array([rel[g == c].mean() for c in cols])
    cond_mean = pd.Series({c: col_mean[col_cond == c].mean() for c in conds})
    cond_colony_cv = pd.Series({c: col_mean[col_cond == c].std(ddof=1) / col_mean[col_cond == c].mean() * 100 for c in conds})
    mzr = d['raw_mz']; w = (mzr >= mz - window) & (mzr <= mz + window)
    P = d['X_raw'][:, w] / d['X_raw'].sum(1, keepdims=True)
    lo, hi = bins.bin_lo.min(), bins.bin_lo.max() + 0.5; inb = (mzr[w] >= lo) & (mzr[w] < hi)
    apex = mzr[w][inb][P[:, inb].argmax(1)]
    pat = {h: np.array([normalised_intensity(d, j, h)[g == c].mean() for c in cols]) for h in ('tic', 'pqn', 'none')}
    out = dict(signal_mz=float(d['mz_binned'][j]), bins='; '.join(f'{b:.1f}-{b + 0.5:.1f}' for b in bins.bin_lo),
               n_bins=len(bins), apex_median=float(np.median(apex)), apex_sd=float(apex.std(ddof=1)),
               apex_range=f'{apex.min():.1f}-{apex.max():.1f}', detected_frac=float((d['X_binned'][:, j] > 0).mean()),
               tech_cv_median_pct=float(np.nanmedian(cv)), tech_cv_range_pct=f'{np.nanmin(cv):.0f}-{np.nanmax(cv):.0f}',
               frac_condition_plate=vc['frac_cond'], frac_colony=vc['frac_col'], frac_technical=vc['frac_tech'],
               icc_colony=vc['icc_colony'], highest_condition=cond_mean.idxmax(), lowest_condition=cond_mean.idxmin(),
               max_over_min_condition=float(cond_mean.max() / cond_mean.min()) if cond_mean.min() > 0 else np.inf,
               colony_pattern_spearman_tic_vs_pqn=float(spearmanr(pat['tic'], pat['pqn']).statistic),
               colony_pattern_spearman_tic_vs_raw=float(spearmanr(pat['tic'], pat['none']).statistic))
    out.update({f'mean_rel_{c}': float(cond_mean[c]) for c in conds})
    out.update({f'colony_cv_pct_{c}': float(cond_colony_cv[c]) for c in conds})
    return out, dict(mz=mzr[w], profile=P, y=y, groups=g, rel=rel, bins=(lo, hi))


def colony_mean_log(d):
    g = np.asarray(d['groups']); cols = list(dict.fromkeys(g))
    return np.vstack([np.log10(d['X_binned'][g == c].sum(0) / d['X_raw'][g == c].sum() + 1e-12) for c in cols])


def related_ions(d, mz, tol=0.45):
    """Signals at characteristic nominal-mass spacings: intensity ratio and
    colony-mean correlation with the candidate."""
    j = feature_index(d, mz); L = colony_mean_log(d); rows = []
    for lab, delta in RELATED.items():
        tgt = 2 * mz - 1.007 if delta is None else mz + delta
        k = int(np.argmin(np.abs(d['mz_binned'] - tgt)))
        if abs(d['mz_binned'][k] - tgt) > tol:
            rows.append(dict(relation=lab, expected_mz=tgt, partner_mz=np.nan)); continue
        rows.append(dict(relation=lab, expected_mz=tgt, partner_mz=float(d['mz_binned'][k]),
                         median_intensity_ratio=float(np.median(d['X_binned'][:, k] / np.maximum(d['X_binned'][:, j], 1e-12))),
                         r_colony_means=float(np.corrcoef(L[:, j], L[:, k])[0, 1])))
    return pd.DataFrame(rows)


def correlated_features(d, mz, r_min=0.9, top=5):
    j = feature_index(d, mz); L = colony_mean_log(d)
    Lc = L - L.mean(0); s = np.sqrt((Lc ** 2).sum(0)); s[s == 0] = np.inf
    r = (Lc * Lc[:, [j]]).sum(0) / (s * s[j]); r[j] = 0
    idx = np.argsort(-np.abs(r))[:top]
    return int((np.abs(r) >= r_min).sum()), [(float(d['mz_binned'][k]), float(r[k])) for k in idx]


def default_flanks(d, mz, gap=0.2, width=0.7):
    """Two local-baseline windows (raw 0.1-Da points) just outside the signal
    group's bins: [lo - gap - width, lo - gap] and [hi + gap, hi + gap + width].
    Inspect the raw profile and pass explicit ``flanks`` when a neighbouring
    peak falls inside either window."""
    b = signal_bins(d, feature_index(d, mz)); lo, hi = b.bin_lo.min(), b.bin_lo.max() + 0.5
    return [(lo - gap - width, lo - gap), (hi + gap, hi + gap + width)]


def baseline_corrected_area(d, mz, flanks=None):
    """Per-spectrum gross and baseline-corrected (net) intensity of a signal group.

    Gross = sum of raw sampled points inside the signal-group bins; the local
    baseline is the per-spectrum median of the flank points, multiplied by the
    number of in-bin points. Returns a DataFrame (one row per spectrum) with raw
    and TIC-normalised gross/net values and the baseline share of the gross.
    A net value <= 0 means no resolved peak above the local baseline.
    """
    flanks = flanks or default_flanks(d, mz)
    mzr, X = d['raw_mz'], d['X_raw']
    b = signal_bins(d, feature_index(d, mz)); lo, hi = b.bin_lo.min(), b.bin_lo.max() + 0.5
    inb = (mzr >= lo) & (mzr < hi); fm = np.zeros(len(mzr), bool)
    for a, z in flanks:
        fm |= (mzr >= a) & (mzr <= z)
    base = np.median(X[:, fm], 1) * inb.sum(); gross = X[:, inb].sum(1); net = gross - base; tic = X.sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        share = np.where(gross > 0, base / gross, np.nan)
    return pd.DataFrame(dict(cond=np.asarray(d['y']), colony=np.asarray(d['groups']), gross_raw=gross, net_raw=net,
                             gross_rel=gross / tic, net_rel=net / tic, base_frac=share))
