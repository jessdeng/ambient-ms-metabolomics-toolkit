"""
Feature prioritisation at the colony level.

The colony is the experimental unit, so technical replicates (repeated sprays)
are averaged on the log scale first and every ranking uses one row per colony.
Two rankers vote, each answering a different question:

  univariate F  does this bin's colony-mean level differ between conditions,
                relative to colony-to-colony variation?          (one bin at a time)
  PLS-DA VIP    does it load on the leading direction separating the conditions?
                (component 1, comparable with MetaboAnalyst)     (multivariate projection)

A candidate is in the top-N of at least MIN_RANKERS of them (default: both).

Two model-based rankings are reported as sensitivity columns, not voters, because
with three colonies per condition and thousands of bins their top-N lists are not
determined by the data:
  Ridge          the list is set by the penalty: on S9 (15 colonies, 2,659 bins) its
                 top-50 shares 0 of 50 with both voters for every alpha from 1 to 1e4
                 and only approaches VIP as alpha -> infinity;
  random forest  the list changes with the random seed (seed-to-seed agreement is
                 reported in every run).

Condition and plate are aliased, so screen_p / screen_q describe condition-plate
separation among colonies; they are not tests of a treatment effect.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import hypergeom
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier

from standard.pipeline import compute_vip_1comp

VOTERS = ('univariate', 'vip')
SENSITIVITY = ('ridge', 'rf')


def colony_means(X, colony_id, condition):
    """Average the rows of each colony. Returns (X_colony, colony_ids, condition_of_colony)."""
    df = pd.DataFrame(X)
    ids = list(dict.fromkeys(colony_id))
    Xc = df.groupby(np.asarray(colony_id), sort=False).mean().loc[ids].to_numpy()
    cond_c = np.array([condition[np.asarray(colony_id) == i][0] for i in ids])
    return Xc, np.array(ids), cond_c


def top_set(score, n):
    """Indices of the n highest scores; ties broken by feature index (ascending m/z)."""
    score = np.nan_to_num(np.asarray(score, float), nan=-np.inf)
    return set(np.lexsort((np.arange(score.size), -score))[:n].tolist())


def benjamini_hochberg(p):
    p = np.where(np.isfinite(p), p, 1.0)
    order = np.argsort(p)
    ranked = p[order] * p.size / np.arange(1, p.size + 1)
    q = np.empty_like(p)
    q[order] = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    return q


def univariate_screen(Xc_log, cond_c):
    """One-way ANOVA on colony means (log10) per bin, plus the effect it describes.

    Columns: screen_F, screen_eta2 (share of colony-mean variance between
    conditions), screen_p, screen_q (BH), top/bottom condition by mean, and
    log2_fc_top_vs_bottom (ratio of geometric means), mean_log10_<condition>.
    """
    conds = np.unique(cond_c)
    groups = [Xc_log[cond_c == c] for c in conds]
    with np.errstate(divide='ignore', invalid='ignore'):
        res = stats.f_oneway(*groups, axis=0)
    F = np.nan_to_num(res.statistic, nan=0.0)
    grand = Xc_log.mean(0)
    ss_between = sum(len(g) * (g.mean(0) - grand) ** 2 for g in groups)
    ss_total = ((Xc_log - grand) ** 2).sum(0)
    means = np.vstack([g.mean(0) for g in groups])                     # (n_cond, p)
    out = pd.DataFrame({
        'screen_F': F,
        'screen_eta2': np.divide(ss_between, ss_total, out=np.zeros_like(grand), where=ss_total > 0),
        'screen_p': np.nan_to_num(res.pvalue, nan=1.0),
    })
    out['screen_q'] = benjamini_hochberg(out['screen_p'].to_numpy())
    out['top_condition'] = conds[means.argmax(0)]
    out['bottom_condition'] = conds[means.argmin(0)]
    out['log2_fc_top_vs_bottom'] = (means.max(0) - means.min(0)) * np.log2(10)
    for c, m in zip(conds, means):
        out[f'mean_log10_{c}'] = m
    return out


def colony_consistency(Xc_log, cond_c, top_condition, bottom_condition):
    """Does one condition stand apart from all the others, colony by colony?

    elevated gap: lowest colony of the top condition minus the highest colony of
                  every other condition;
    depleted gap: lowest colony of every other condition minus the highest colony
                  of the bottom condition.
    colony_gap_log10 is the larger of the two (log10 units); > 0 means every
    colony agrees, and ``colony_pattern`` names the condition that stands apart.
    """
    p = Xc_log.shape[1]
    gap, pattern = np.empty(p), np.empty(p, dtype=object)
    for j in range(p):
        x = Xc_log[:, j]
        top, bot = cond_c == top_condition[j], cond_c == bottom_condition[j]
        up = x[top].min() - x[~top].max()
        down = x[~bot].min() - x[bot].max()
        gap[j] = max(up, down)
        pattern[j] = (f'elevated in {top_condition[j]}' if up >= down else
                      f'depleted in {bottom_condition[j]}') if gap[j] > 0 else 'none'
    return pd.DataFrame({'colony_gap_log10': gap, 'colony_pattern': pattern,
                         'colonies_consistent': gap > 0})


def variance_partition(X_log, condition, colony_id):
    """Per bin: share of variance between condition-plates, between colonies
    within a condition, and between technical replicates (nested random-effects
    ANOVA, method of moments; exact for balanced designs)."""
    X = np.asarray(X_log, float)
    cond, col = np.asarray(condition), np.asarray(colony_id)
    n = X.shape[0]
    df = pd.DataFrame(X)
    col_mean = df.groupby(col).transform('mean').to_numpy()
    cond_mean = df.groupby(cond).transform('mean').to_numpy()
    gm = X.mean(0)
    n_cond, n_col = np.unique(cond).size, np.unique(col).size
    ms_cond = ((cond_mean - gm) ** 2).sum(0) / (n_cond - 1)
    ms_col = ((col_mean - cond_mean) ** 2).sum(0) / (n_col - n_cond)
    ms_tech = ((X - col_mean) ** 2).sum(0) / (n - n_col)
    s2_tech = ms_tech
    s2_col = np.maximum(0, (ms_col - ms_tech) / (n / n_col))
    s2_cond = np.maximum(0, (ms_cond - ms_col) / (n / n_cond))
    tot = s2_cond + s2_col + s2_tech
    frac = lambda s: np.divide(s, tot, out=np.full_like(tot, np.nan), where=tot > 0)
    return pd.DataFrame({'frac_condition_plate': frac(s2_cond), 'frac_colony': frac(s2_col),
                         'frac_technical': frac(s2_tech)})


def technical_cv(X_lin, colony_id):
    """Median over colonies of the technical %CV (linear, TIC-normalised scale)."""
    df = pd.DataFrame(np.asarray(X_lin, float))
    g = df.groupby(np.asarray(colony_id))
    with np.errstate(divide='ignore', invalid='ignore'):
        cv = 100 * g.std(ddof=1) / g.mean()
    return cv.median(axis=0, skipna=True).to_numpy()


def rank_features(Xc_scaled, Xc_log, cond_c, top_n=50, min_rankers=2, rf_seeds=25, seed=42):
    """Scores, ranks and top-N membership for the two voters and the two
    sensitivity rankings. Returns (table, info) where info holds the scores and
    the random-forest seed-to-seed agreement."""
    screen = univariate_screen(Xc_log, cond_c)
    ridge = RidgeClassifier(alpha=1.0).fit(Xc_scaled, cond_c)
    scores = {
        'univariate': screen['screen_F'].to_numpy(),
        'vip': compute_vip_1comp(Xc_scaled, cond_c),
        'ridge': np.abs(np.atleast_2d(ridge.coef_)).mean(0),
    }
    per_seed = [RandomForestClassifier(n_estimators=500, random_state=seed + s)
                .fit(Xc_scaled, cond_c).feature_importances_ for s in range(rf_seeds)]
    scores['rf'] = np.mean(per_seed, axis=0)
    half = max(1, rf_seeds // 2)
    rf_agreement = jaccard(top_set(np.mean(per_seed[:half], 0), top_n),
                           top_set(np.mean(per_seed[half:], 0), top_n)) if rf_seeds > 1 else np.nan

    table = pd.DataFrame(index=range(Xc_log.shape[1]))
    for m, s in scores.items():
        table[f'{m}_score'] = s
        table[f'{m}_rank'] = pd.Series(-s).rank(method='min').astype(int).to_numpy()
        table[f'in_{m}_top'] = np.isin(np.arange(s.size), list(top_set(s, top_n)))
    table['n_rankers'] = table[[f'in_{m}_top' for m in VOTERS]].sum(axis=1)
    table['mean_voter_rank'] = table[[f'{m}_rank' for m in VOTERS]].mean(axis=1)
    table['candidate'] = table['n_rankers'] >= min_rankers
    table = pd.concat([table, screen], axis=1)
    return table, {'rf_seed_agreement': rf_agreement, 'scores': scores}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else np.nan


def concordance(scores, top_n):
    """Pairwise top-N Jaccard between rankers, with the value expected by chance."""
    names = list(scores)
    p = len(next(iter(scores.values())))
    k = np.arange(top_n + 1)
    chance = float((hypergeom(p, top_n, top_n).pmf(k) * k / (2 * top_n - k)).sum())
    tops = {m: top_set(scores[m], top_n) for m in names}
    rows = [dict(ranker_a=a, ranker_b=b, jaccard=jaccard(tops[a], tops[b]), chance=chance,
                 shared=len(tops[a] & tops[b]))
            for i, a in enumerate(names) for b in names[i + 1:]]
    return pd.DataFrame(rows)
