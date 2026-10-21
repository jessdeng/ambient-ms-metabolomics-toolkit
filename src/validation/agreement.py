"""Rank-based agreement between feature-importance rankings (plan step 6).

All comparisons use ranks only. The VIP > 1 threshold is analysed separately
and is never equated with a top-N list.
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

from shared.importance import top_n


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else np.nan


def expected_jaccard_random(n, p):
    """Expected Jaccard of two independent random top-n sets drawn from p features
    (hypergeometric overlap; computed exactly)."""
    from scipy.stats import hypergeom
    k = np.arange(0, n + 1)
    pk = hypergeom(p, n, n).pmf(k)
    return float((pk * k / (2 * n - k)).sum())


def pairwise_agreement(imps, ns=(25, 50, 100)):
    """imps: dict method -> importance vector (higher = more important)."""
    p = len(next(iter(imps.values())))
    rows = []
    for a, b in itertools.combinations(imps, 2):
        r = dict(method_a=a, method_b=b, spearman=float(spearmanr(imps[a], imps[b]).statistic))
        for n in ns:
            r[f'jaccard_top{n}'] = jaccard(top_n(imps[a], n), top_n(imps[b], n))
            r[f'random_expectation_top{n}'] = expected_jaccard_random(n, p)
        rows.append(r)
    return pd.DataFrame(rows)


def vip_threshold_analysis(vip, imps, ns=(25, 50, 100), threshold=1.0):
    """Retention rate of VIP > threshold and the fraction of each method's top-N
    inside that set (random expectation = retention rate)."""
    keep = set(np.where(vip > threshold)[0])
    rate = len(keep) / len(vip)
    rows = []
    for m, v in imps.items():
        r = dict(method=m, n_vip_gt_threshold=len(keep), retention_rate=rate)
        for n in ns:
            r[f'frac_top{n}_inside_vip_gt_threshold'] = len(set(top_n(v, n)) & keep) / n
        rows.append(r)
    return pd.DataFrame(rows)


def family_structure(pair_df, family, metric='spearman', k=None):
    """Compare a priori families with the empirical redundancy matrix:
    mean within- vs between-family agreement, and the adjusted Rand index of an
    average-linkage clustering (distance = 1 - metric) cut at the number of
    a priori families."""
    methods = list(dict.fromkeys(list(pair_df.method_a) + list(pair_df.method_b)))
    M = pd.DataFrame(np.eye(len(methods)), index=methods, columns=methods)
    for _, r in pair_df.iterrows():
        M.loc[r.method_a, r.method_b] = M.loc[r.method_b, r.method_a] = r[metric]
    fam = np.array([family[m] for m in methods])
    within = [M.iloc[i, j] for i, j in itertools.combinations(range(len(methods)), 2) if fam[i] == fam[j]]
    between = [M.iloc[i, j] for i, j in itertools.combinations(range(len(methods)), 2) if fam[i] != fam[j]]
    D = np.clip(1 - M.values, 0, None); np.fill_diagonal(D, 0)
    k = k or len(set(fam))
    lab = fcluster(linkage(squareform(D, checks=False), 'average'), k, 'maxclust')
    return dict(mean_within_family=float(np.mean(within)), mean_between_family=float(np.mean(between)),
                ari_empirical_vs_a_priori=float(adjusted_rand_score(fam, lab)),
                empirical_clusters={m: int(l) for m, l in zip(methods, lab)}), M
