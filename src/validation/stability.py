"""Colony-removal sensitivity of feature rankings (plan step 7).

Each subset drops one colony per condition (3^5 = 243 balanced subsets for
5 conditions x 3 colonies). Inside every subset the whole preprocessing chain
(filters, normalisation, log, scaling) is refitted, the PLS-DA component number
A* is re-selected by grouped Q2, and every ranking is recomputed. The
signal-group map (label-free, Step 3) is held fixed.

Selection frequency over subsets is a DESCRIPTIVE measure of sensitivity to
colony removal. Subsets overlap heavily and are not independent experiments, so
no binomial intervals are attached.
"""
import itertools
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from shared.importance import fit_importance, top_n
from standard.pipeline import (compute_vip, grouped_q2_profile, VALIDATED_PLS_TOL,
                               VALIDATED_PLS_MAX_ITER)

METHODS = ('vip_ref', 'vip_multi', 'vip_multi_pars', 'vip_fixedA', 'ridge', 'lr', 'svm', 'lda', 'rf', 'gb')
Q2_TOL = 0.05


def colonies_by_condition(y, groups):
    y, g = np.asarray(y), np.asarray(groups)
    conds = sorted(set(y))
    return conds, {c: list(dict.fromkeys(g[y == c])) for c in conds}


def subset_design(n_cond=5, n_col=3, design='full'):
    """Index of the colony dropped in each condition, one tuple per subset.
    'full': all n_col**n_cond subsets. 'oa27': 27-run orthogonal array (strength 2
    for 5 three-level factors: every pair of conditions sees every pair of
    dropped colonies exactly 3 times)."""
    if design == 'full':
        return list(itertools.product(range(n_col), repeat=n_cond))
    if design == 'oa27':
        assert n_col == 3 and n_cond <= 5
        rows = [(a, b, c, (a + b) % 3, (a + 2 * b + c) % 3) for a in range(3) for b in range(3) for c in range(3)]
        return [r[:n_cond] for r in rows]
    raise ValueError(design)


def subset_indices(y, groups, drop):
    conds, cols = colonies_by_condition(y, groups)
    dropped = {cols[c][k] for c, k in zip(conds, drop)}
    g = np.asarray(groups)
    return np.where(~np.isin(g, list(dropped)))[0], sorted(dropped)


def refit_preprocessing(X_binned, y, prep_steps):
    """Fit the toolkit preprocessing chain on (X_binned, y); return the matrix
    and the indices of retained columns of X_binned."""
    from sklearn.preprocessing import LabelEncoder
    pipe = Pipeline([(n, clone(t)) for n, t in prep_steps])
    yy = LabelEncoder().fit_transform(y)
    X = pipe.fit_transform(X_binned, yy)
    keep = np.arange(X_binned.shape[1])
    for _, step in pipe.steps:
        if hasattr(step, 'keep_'):
            keep = keep[step.keep_]
    assert len(keep) == X.shape[1]
    return X, keep


def rank_subset(X_binned, y, groups, prep_steps, rf_seeds=25, gb_seeds=5, seed=42,
                max_components=8, methods=METHODS, fixed_A=None):
    """All rankings for one subset. Returns dict method -> importance vector in
    the full X_binned column space (NaN = feature removed by refitted filters),
    plus the re-selected A*, the Q2 profile and convergence-warning counts.

    Component selection uses leave-one-colony-out grouped CV (each fold holds
    out all technical replicates of one colony). After one colony per condition
    is removed only two colonies per condition remain; leave-one-colony-out
    keeps the largest possible training set (9 of 10 colonies), whereas a 2-fold
    split would train on a single colony per condition."""
    p = X_binned.shape[1]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        X, keep = refit_preprocessing(X_binned, y, prep_steps)
        q2 = {'q2': np.full(max_components, np.nan), 'best_n_components': 0}
        if {'vip_multi', 'vip_multi_pars'} & set(methods):
            from sklearn.model_selection import LeaveOneGroupOut
            q2 = grouped_q2_profile(X_binned, y, groups, prep_steps, max_components=max_components,
                                    splitter=LeaveOneGroupOut(), random_state=seed)
        a_star = int(q2['best_n_components'])
        q2v = np.asarray(q2['q2'], float)
        a_pars = int(np.where(q2v >= np.nanmax(q2v) - Q2_TOL)[0][0]) + 1 if a_star else 0
        out = {}
        if 'vip_ref' in methods:
            out['vip_ref'] = compute_vip(X, y, 1)
        if 'vip_multi' in methods:
            out['vip_multi'] = compute_vip(X, y, a_star, tol=VALIDATED_PLS_TOL, max_iter=VALIDATED_PLS_MAX_ITER)
        if 'vip_multi_pars' in methods:
            out['vip_multi_pars'] = compute_vip(X, y, a_pars, tol=VALIDATED_PLS_TOL, max_iter=VALIDATED_PLS_MAX_ITER)
        if 'vip_fixedA' in methods and fixed_A:
            out['vip_fixedA'] = compute_vip(X, y, int(fixed_A), tol=VALIDATED_PLS_TOL, max_iter=VALIDATED_PLS_MAX_ITER)
        for m in ('ridge', 'lr', 'svm', 'lda'):
            if m in methods:
                out[m] = fit_importance(m, X, y, seed=seed)['global']
        if 'rf' in methods:
            out['rf'] = fit_importance('rf', X, y, seed=seed, n_seeds=rf_seeds)['global']
        if 'gb' in methods and gb_seeds:
            out['gb'] = fit_importance('gb', X, y, seed=seed, n_seeds=gb_seeds)['global']
    conv = sum('converge' in str(x.message).lower() for x in w)
    full = {}
    for m, v in out.items():
        f = np.full(p, np.nan); f[keep] = v; full[m] = f
    return full, dict(a_star=a_star, a_parsimony=a_pars, q2_max=float(np.nanmax(q2v)) if np.isfinite(q2v).any() else np.nan,
                      q2_profile=q2v, n_features=len(keep), n_convergence_warnings=conv)


def run_subsets(d, cfg_prep_steps, drops, out_dir, tag, log=print, **kw):
    """Resumable loop: one compressed file per subset."""
    os.makedirs(out_dir, exist_ok=True)
    meta = []
    for drop in drops:
        name = os.path.join(out_dir, f'{tag}_drop{"".join(map(str, drop))}.npz')
        if not os.path.exists(name):
            idx, dropped = subset_indices(d['y'], d['groups'], drop)
            imps, info = rank_subset(d['X_binned'][idx], np.asarray(d['y'])[idx],
                                     np.asarray(d['groups'])[idx], cfg_prep_steps, **kw)
            np.savez_compressed(name, **{m: imps[m].astype(np.float32) for m in imps},
                                info=np.array([info['a_star'], info['q2_max'], info['n_features'],
                                               info['n_convergence_warnings'], info['a_parsimony']]),
                                q2_profile=info['q2_profile'],
                                dropped=np.array(dropped))
            log(f'{tag} {drop} A*={info["a_star"]} warn={info["n_convergence_warnings"]}')
        z = np.load(name)
        meta.append(dict(drop=''.join(map(str, drop)), a_star=int(z['info'][0]), a_parsimony=int(z['info'][4]),
                         q2_max=float(z['info'][1]),
                         n_features=int(z['info'][2]), n_convergence_warnings=int(z['info'][3]),
                         dropped=';'.join(z['dropped']),
                         **{f'q2_A{a + 1}': float(v) for a, v in enumerate(z['q2_profile'])}))
    return pd.DataFrame(meta)


def selection_frequency(files, methods=METHODS, ns=(25, 50, 100)):
    """pi[method, N] per feature: fraction of subsets in which the feature is in
    the top-N (features removed by filters count as not selected)."""
    loaded = [np.load(f) for f in files]
    methods = [m for m in methods if m in loaded[0].files]
    p = len(loaded[0][methods[0]])
    out = {}
    for m in methods:
        for n in ns:
            c = np.zeros(p)
            for z in loaded:
                v = np.nan_to_num(z[m].astype(float), nan=-np.inf)
                c[top_n(v, n)] += 1
            out[(m, n)] = c / len(loaded)
    return out


# ---------------------------------------------------------------- analysis
def load_rankings(files, method):
    """Matrix (subsets x features) of importance for one method; NaN = filtered."""
    return np.vstack([np.load(f)[method].astype(float) for f in files])


def design_agreement(pi_a, pi_b):
    """Agreement of two selection-frequency vectors over the features selected in
    either design (pi > 0)."""
    from scipy.stats import spearmanr
    m = (pi_a > 0) | (pi_b > 0)
    return dict(n_features_selected_either=int(m.sum()),
                spearman=float(spearmanr(pi_a[m], pi_b[m]).statistic) if m.sum() > 2 else np.nan,
                max_abs_diff=float(np.abs(pi_a - pi_b).max()), mean_abs_diff_selected=float(np.abs(pi_a - pi_b)[m].mean()),
                n_pi_ge_0_8_a=int((pi_a >= 0.8).sum()), n_pi_ge_0_8_b=int((pi_b >= 0.8).sum()),
                jaccard_pi_ge_0_8=float(len(set(np.where(pi_a >= .8)[0]) & set(np.where(pi_b >= .8)[0])) /
                                        max(1, len(set(np.where(pi_a >= .8)[0]) | set(np.where(pi_b >= .8)[0])))))


def subset_to_reference_jaccard(R, ref, n=50):
    """Top-n Jaccard of each subset ranking (rows of R) with a reference ranking."""
    from validation.agreement import jaccard
    t = set(top_n(ref, n))
    return np.array([jaccard(top_n(np.nan_to_num(r, nan=-np.inf), n), t) for r in R])


def pairwise_subset_jaccard(R, n=50, max_pairs=3000, seed=0):
    """Mean top-n Jaccard over pairs of subsets (random sample of pairs if many)."""
    from validation.agreement import jaccard
    tops = [set(top_n(np.nan_to_num(r, nan=-np.inf), n)) for r in R]
    pairs = list(itertools.combinations(range(len(R)), 2))
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(seed); pairs = [pairs[i] for i in rng.choice(len(pairs), max_pairs, replace=False)]
    return float(np.mean([jaccard(tops[i], tops[j]) for i, j in pairs]))


# ------------------------------------------------- technical-replicate designs
def technical_positions(groups):
    """Position (0, 1, 2) of each spectrum within its colony, in file order."""
    g = np.asarray(groups); pos = np.zeros(len(g), dtype=int); seen = {}
    for i, k in enumerate(g):
        pos[i] = seen.get(k, 0); seen[k] = pos[i] + 1
    return pos


def technical_draws(groups, keep_per_colony, B, seed=0):
    """B distinct draws; each keeps `keep_per_colony` technical replicates of every
    colony (colonies fixed). Returns a list of index arrays."""
    g = np.asarray(groups); cols = list(dict.fromkeys(g)); pos = technical_positions(g)
    n_tech = int(pos.max()) + 1
    combos = list(itertools.combinations(range(n_tech), keep_per_colony))
    rng = np.random.default_rng(seed); seen, out = set(), []
    while len(out) < B:
        pick = tuple(int(rng.integers(len(combos))) for _ in cols)
        if pick in seen:
            continue
        seen.add(pick)
        keep = {c: set(combos[k]) for c, k in zip(cols, pick)}
        out.append(np.array([i for i in range(len(g)) if pos[i] in keep[g[i]]]))
    return out


def colony_mean_data(X_binned, y, groups):
    """D15-mean: technical replicates averaged on the linear (raw) scale, before
    normalisation and log transform."""
    g = np.asarray(groups); y = np.asarray(y); cols = list(dict.fromkeys(g))
    Xm = np.vstack([X_binned[g == c].mean(0) for c in cols])
    return Xm, np.array([y[g == c][0] for c in cols]), np.array(cols)


def run_index_sets(d, prep_steps, index_sets, out_dir, tag, log=print, X_binned=None, y=None, groups=None, **kw):
    """Resumable ranking of arbitrary row subsets (one file per set)."""
    os.makedirs(out_dir, exist_ok=True)
    Xb = d['X_binned'] if X_binned is None else X_binned
    yy = np.asarray(d['y'] if y is None else y); gg = np.asarray(d['groups'] if groups is None else groups)
    meta = []
    for b, idx in enumerate(index_sets):
        name = os.path.join(out_dir, f'{tag}_{b:03d}.npz')
        if not os.path.exists(name):
            imps, info = rank_subset(Xb[idx], yy[idx], gg[idx], prep_steps, **kw)
            np.savez_compressed(name, **{m: imps[m].astype(np.float32) for m in imps},
                                info=np.array([info['a_star'], info['q2_max'], info['n_features'],
                                               info['n_convergence_warnings'], info['a_parsimony']]),
                                q2_profile=info['q2_profile'], rows=np.asarray(idx))
            log(f'{tag} {b} A*={info["a_star"]}')
        z = np.load(name)
        meta.append(dict(set=b, a_star=int(z['info'][0]), a_parsimony=int(z['info'][4]), q2_max=float(z['info'][1]),
                         n_rows=len(z['rows']), n_convergence_warnings=int(z['info'][3])))
    return pd.DataFrame(meta)
