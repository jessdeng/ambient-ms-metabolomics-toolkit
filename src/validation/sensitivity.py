"""Preprocessing and analytical-choice sensitivity (plan step 9).

One-factor-at-a-time departures from the primary validated setting (native-grid
0.5-Da bins merged into signal groups; TIC normalisation; log10; autoscaling;
variance filter 25th percentile; abundance filter 5th percentile; prevalence
0.5; manifest grouping). Every setting re-runs preprocessing, PLS-DA component
selection (leave-one-colony-out Q2) and all rankings on the full data, with the
same seeds as the shuffled-label benchmark (RF 25, GB 5).

Features are compared across feature definitions on the nominal-mass scale
(edges at integer + 0.6, see standard.preprocessing.bin_features_nominal).
A feature that is absent from a setting's matrix is reported as NOT ELIGIBLE,
which is distinct from being eligible but receiving low importance.
"""
import numpy as np

# name, description, load_isolate kwargs, cfg overrides
SETTINGS = [
    ('primary', 'native 0.5-Da bins -> signal groups; TIC; filters 25/5', {}, {}),
    ('bins_0.5Da', 'native 0.5-Da bins, no signal grouping', {'feature_level': 'bin'}, {}),
    ('nominal_1Da_o0.6', '1-Da nominal-mass bins, edges at integer + 0.6 (pre-specified)', {'feature_level': 'nominal'}, {'NOMINAL_BIN_OFFSET': 0.6}),
    ('nominal_1Da_o0.5', '1-Da nominal-mass bins, edges at integer + 0.5', {'feature_level': 'nominal'}, {'NOMINAL_BIN_OFFSET': 0.5}),
    ('nominal_1Da_o0.7', '1-Da nominal-mass bins, edges at integer + 0.7', {'feature_level': 'nominal'}, {'NOMINAL_BIN_OFFSET': 0.7}),
    ('original_resampled_bins', 'original 5,000-point resampled 0.5-Da bins', {'feature_level': 'bin', 'resample': 'linspace5000'}, {}),
    ('norm_pqn', 'PQN normalisation', {}, {'NORMALIZATION': 'pqn'}),
    ('norm_none', 'no sample normalisation', {}, {'NORMALIZATION': 'none'}),
    ('scaling_pareto', 'Pareto scaling', {}, {'SCALING': 'pareto'}),
    ('no_abundance_filter', 'abundance filter off', {}, {'ABUNDANCE_PERCENTILE': 0}),
    ('abundance_25', 'abundance filter 25th percentile', {}, {'ABUNDANCE_PERCENTILE': 25}),
    ('no_variance_filter', 'variance filter off', {}, {'VARIANCE_PERCENTILE': 0}),
    ('variance_50', 'variance filter 50th percentile', {}, {'VARIANCE_PERCENTILE': 50}),
    ('no_percentile_filters', 'variance and abundance filters off (prevalence only)', {}, {'VARIANCE_PERCENTILE': 0, 'ABUNDANCE_PERCENTILE': 0}),
]
GROUPING_VARIANTS = [  # optional: --grouping-variants
    ('grouping_filename', 'original file-name grouping', {'grouping': 'filename'}, {}),
    ('grouping_exclude_unresolved', 'manifest grouping, files with unresolved identity excluded', {'grouping': 'manifest_exclude'}, {}),
]


def nominal_mass(mz, offset=0.6):
    """Nominal mass of an m/z label on the [n - 1 + offset, n + offset) grid."""
    return np.floor(np.asarray(mz, dtype=float) - offset).astype(int) + 1


# ------------------------------------------------------------------ analysis
METHODS8 = ('vip_ref', 'vip_multi', 'ridge', 'lr', 'svm', 'lda', 'rf', 'gb')


def load_setting(path):
    import numpy as np
    z = np.load(path, allow_pickle=True)
    return z, {m: np.nan_to_num(z[m].astype(float), nan=-np.inf) for m in METHODS8}


def nominal_tops(imps, labels, nominal_level, n=50):
    """Top-n of each method expressed as a set of nominal masses."""
    from shared.importance import top_n
    nm = np.asarray(labels).astype(int) if nominal_level else nominal_mass(labels)
    return {m: set(nm[list(top_n(imps[m], n))]) for m in imps}


def candidate_index(labels, mz, nominal_level, tol=0.3):
    labels = np.asarray(labels)
    if nominal_level:
        k = np.where(labels.astype(int) == nominal_mass([mz])[0])[0]
        return int(k[0]) if len(k) else None
    k = int(np.argmin(np.abs(labels - mz)))
    return k if abs(labels[k] - mz) <= tol else None


def analyse_isolate(result_dir, isolate, candidates, settings=None, grouping_variants=False):
    """Conclusion-level and candidate-level sensitivity tables for one isolate."""
    import os
    import pandas as pd
    from validation.consensus import full_data_statistics, vote_table
    names = settings or [s[0] for s in SETTINGS] + ([s[0] for s in GROUPING_VARIANTS] if grouping_variants else [])
    zp, ip = load_setting(os.path.join(result_dir, f'{isolate}__primary.npz'))
    tp = nominal_tops(ip, zp['mz_binned'], False)
    conc, cand = [], []
    for name in names:
        f = os.path.join(result_dir, f'{isolate}__{name}.npz')
        if not os.path.exists(f):
            continue
        z, imps = load_setting(f); nominal = name.startswith('nominal'); lab = z['mz_binned']
        t = nominal_tops(imps, lab, nominal)
        conc.append(dict(isolate=isolate, setting=name, description=str(z['description']),
                         n_eligible=int(z['eligible'].sum()), a_star=int(z['a_star']),
                         **{f'J_{m}': len(t[m] & tp[m]) / len(t[m] | tp[m]) for m in METHODS8},
                         **full_data_statistics(imps)))
        votes, fam, _ = vote_table(imps)
        ranks = {m: pd.Series(imps[m]).rank(ascending=False, method='min').values for m in METHODS8}
        for c in candidates:
            k = candidate_index(lab, c, nominal)
            row = dict(isolate=isolate, candidate_mz=c, setting=name, feature_label=None if k is None else float(lab[k]))
            if k is None or not z['eligible'][k]:
                row['status'] = 'not eligible'
            else:
                row.update(status='eligible', votes_of6=int(votes[k]), families=int(fam[k]),
                           **{f'rank_{m}': int(ranks[m][k]) for m in METHODS8},
                           methods_top50=','.join(m for m in METHODS8 if ranks[m][k] <= 50))
            cand.append(row)
    return pd.DataFrame(conc), pd.DataFrame(cand)
