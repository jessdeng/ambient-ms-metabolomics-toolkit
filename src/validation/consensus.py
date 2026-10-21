"""Consensus summaries and the exploratory shuffled-label benchmark (plan step 8).

Consensus is reported in several forms side by side, none of which is treated as
proof of importance:
  * the original six-vote count (top-50 in >= 4 of 6 rankings; historical rule),
  * model-family support (linear L = Ridge/LR/SVM, tree T = RF/GB, LDA),
  * selection stability under colony removal (per-method selection frequency).

Shuffled-label benchmark (EXPLORATORY): condition labels are shuffled among
colonies (three colonies per label, technical replicates kept together) and the
identical ranking, resampling and consensus pipeline is re-run. Because each
condition was a single plate/enclosure, colonies are not exchangeable across
plates; the benchmark therefore describes how much agreement the pipeline
produces from the data's correlation structure when labels carry no condition
information. Results are exceedance proportions, never p-values.
"""
import itertools

import numpy as np

from shared.importance import top_n

SIX = ('ridge', 'lr', 'svm', 'lda', 'rf', 'gb')
FAMILIES = {'L': ('ridge', 'lr', 'svm'), 'T': ('rf', 'gb'), 'LDA': ('lda',)}

# 9 drop-one-colony-per-condition subsets: rows of the 27-run orthogonal array in
# which every colony of every condition is dropped exactly 3 times.
DESIGN9 = ((0, 0, 0, 0, 0), (0, 1, 2, 1, 1), (0, 2, 1, 2, 2), (1, 0, 1, 1, 2), (1, 1, 0, 2, 0),
           (1, 2, 2, 0, 1), (2, 0, 2, 2, 1), (2, 1, 1, 0, 2), (2, 2, 0, 1, 0))


def _top(v, n):
    return set(top_n(np.nan_to_num(np.asarray(v, float), nan=-np.inf), n))


def _jac(a, b):
    return len(a & b) / len(a | b) if (a | b) else np.nan


def vote_table(imps, n=50, methods=SIX):
    """Per-feature six-vote count and number of supporting families (0-3)."""
    p = len(next(iter(imps.values())))
    tops = {m: _top(imps[m], n) for m in methods}
    votes = np.zeros(p, int)
    for m in methods:
        votes[list(tops[m])] += 1
    fam = np.zeros(p, int)
    for ms in FAMILIES.values():
        sel = set().union(*(tops[m] for m in ms if m in tops))
        fam[list(sel)] += 1
    return votes, fam, tops


def full_data_statistics(imps, n=50):
    votes, fam, tops = vote_table(imps, n)
    fam_of = {m: f for f, ms in FAMILIES.items() for m in ms}
    pairs = list(itertools.combinations(SIX, 2))
    cross = [_jac(tops[a], tops[b]) for a, b in pairs if fam_of[a] != fam_of[b]]
    within_L = [_jac(tops[a], tops[b]) for a, b in itertools.combinations(FAMILIES['L'], 2)]
    return {'n_votes_ge4': int((votes >= 4).sum()), 'n_votes_ge5': int((votes >= 5).sum()),
            'n_three_family': int((fam == 3).sum()), 'n_two_family': int((fam >= 2).sum()),
            'mean_cross_family_jaccard': float(np.mean(cross)), 'mean_within_linear_jaccard': float(np.mean(within_L)),
            'rf_gb_jaccard': _jac(tops['rf'], tops['gb'])}


def resampled_statistics(subset_imps, n=50, pi_star=0.5):
    """subset_imps: list (one per subset) of dicts method -> importance vector."""
    S = len(subset_imps); p = len(subset_imps[0][SIX[0]])
    out, pi = {}, {}
    for m in SIX:
        tops = [_top(s[m], n) for s in subset_imps]
        cnt = np.zeros(p)
        for t in tops:
            cnt[list(t)] += 1
        pi[m] = cnt / S
        out[f'pairwise_jaccard_{m}'] = float(np.mean([_jac(a, b) for a, b in itertools.combinations(tops, 2)]))
    stable = np.vstack([pi[m] >= pi_star for m in SIX])
    fam_stable = np.vstack([np.any([pi[m] >= pi_star for m in ms], axis=0) for ms in FAMILIES.values()])
    v = np.array([vote_table(s, n)[0] for s in subset_imps])
    f = np.array([vote_table(s, n)[1] for s in subset_imps])
    out.update({'n_stable_in_ge4_methods': int((stable.sum(0) >= 4).sum()),
                'n_stable_three_family': int((fam_stable.sum(0) == 3).sum()),
                'n_votes_ge4_in_half_subsets': int(((v >= 4).mean(0) >= 0.5).sum()),
                'n_three_family_in_half_subsets': int(((f == 3).mean(0) >= 0.5).sum())})
    return out, pi


def shuffle_colony_labels(y, groups, rng):
    """Shuffle condition labels among colonies (technical replicates stay together;
    label counts per condition are preserved)."""
    y, g = np.asarray(y), np.asarray(groups)
    cols = list(dict.fromkeys(g)); lab = np.array([y[g == c][0] for c in cols])
    new = dict(zip(cols, rng.permutation(lab)))
    return np.array([new[c] for c in g])


def exceedance(observed, reference):
    """Share of shuffled-label values >= observed (descriptive; not a p-value) and
    the mid-rank quantile of the observed value within the reference."""
    r = np.asarray(reference, float)
    return {'exceedance': float((r >= observed).mean()),
            'quantile': float(((r < observed).sum() + 0.5 * (r == observed).sum()) / len(r)),
            'reference_median': float(np.median(r)), 'reference_p95': float(np.quantile(r, 0.95))}
