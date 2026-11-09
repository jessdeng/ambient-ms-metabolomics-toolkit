"""Family-aware rank aggregation and top-weighted concordance of importance rankings.

The six importance methods are not six independent voters: Ridge, LR and SVM are
closely related linear solutions, RF and GB are both tree ensembles, and the PLS-DA
VIP is a projection method. A raw "top-N in >= k of 6 methods" count therefore lets
the linear block cast three of the six votes. Here

  * every method is converted to ranks (average ranks for ties, NaN = worst);
  * methods are averaged WITHIN a family, so each family casts one vote;
  * the consensus is the WORST family rank (a feature must rank well in every
    family; minimax) together with the mean family rank and a robust-rank-
    aggregation score (Kolde et al. 2012, beta-distribution rho score).

``rra_score`` is a ranking score, not a p-value: the learners share X and y and,
with light confounded with plate, no null here supports a significance claim.
"""
import numpy as np
import pandas as pd
from scipy.stats import beta, hypergeom

# Method key -> family (same keys as feature_stats._ALL_METHODS).
FAMILY = {'ridge': 'linear', 'lr': 'linear', 'svm': 'linear',
          'rf': 'tree', 'gb': 'tree', 'vip': 'pls'}


def to_ranks(imp):
    """Rank 1 = most important. Ties get the average rank (GB leaves >90% of bins at
    exactly 0 importance); NaN (feature removed by a filter) ranks last."""
    return pd.Series(np.asarray(imp, dtype=float)).rank(
        ascending=False, method='average', na_option='bottom').to_numpy()


def aggregate_ranks(imps, family=FAMILY, top_k=50):
    """Aggregate importance vectors into family-level ranks.

    Parameters
    ----------
    imps : dict {method_key: (p,) importance, higher = more important}
    family : dict method_key -> family name
    top_k : int  cut-off used for ``n_families_topk``

    Returns
    -------
    DataFrame (one row per feature, same order as the input) with
    ``agg_rank_<family>``, ``agg_worst_family_rank``, ``agg_mean_family_rank``,
    ``agg_rra_score`` and ``agg_n_families_topk``.
    """
    if not imps:
        raise ValueError("aggregate_ranks needs at least one importance vector")
    lengths = {len(np.asarray(v)) for v in imps.values()}
    if len(lengths) != 1:
        raise ValueError(f"importance vectors differ in length: {sorted(lengths)}")
    p = lengths.pop()
    ranks = {m: to_ranks(v) for m, v in imps.items()}
    fams = list(dict.fromkeys(family[m] for m in imps))
    fam_rank = {}
    for f in fams:
        mean_r = np.mean([ranks[m] for m in imps if family[m] == f], axis=0)
        fam_rank[f] = pd.Series(mean_r).rank(method='average').to_numpy()
    F = np.column_stack([fam_rank[f] for f in fams])            # (p, n_families)
    m = F.shape[1]
    U = np.sort(F / p, axis=1)                                  # normalised, ascending
    k = np.arange(1, m + 1)
    rho = beta.cdf(U, k, m - k + 1).min(axis=1)
    out = pd.DataFrame({f'agg_rank_{f}': fam_rank[f] for f in fams})
    out['agg_worst_family_rank'] = F.max(axis=1)
    out['agg_mean_family_rank'] = F.mean(axis=1)
    out['agg_rra_score'] = np.minimum(1.0, rho * m)
    out['agg_n_families_topk'] = (F <= top_k).sum(axis=1)
    return out


def rbo(a, b, p=0.9):
    """Truncated rank-biased overlap (lower bound; Webber et al. 2010)."""
    k = min(len(a), len(b))
    s = sum(p ** (d - 1) * len(set(a[:d]) & set(b[:d])) / d for d in range(1, k + 1))
    return (1 - p) * s


def concordance_table(imps, ks=(25, 50, 100)):
    """Pairwise top-k Jaccard with its hypergeometric chance level, top-weighted RBO
    and full-list Spearman (a bulk-dominated secondary summary).

    Do not compute Spearman on the union of two top lists: conditioning on
    'high in A or high in B' induces a spurious negative correlation."""
    p = len(next(iter(imps.values())))
    R = {m: to_ranks(v) for m, v in imps.items()}
    order = {m: np.lexsort((np.arange(p), r)) for m, r in R.items()}   # ties by index
    names = list(imps)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            row = {'method_a': a, 'method_b': b}
            for k in ks:
                A, B = set(order[a][:k]), set(order[b][:k])
                kk = np.arange(k + 1)
                row[f'jaccard@{k}'] = len(A & B) / len(A | B)
                row[f'chance@{k}'] = float(
                    (hypergeom(p, k, k).pmf(kk) * kk / (2 * k - kk)).sum())
            row['rbo@100'] = rbo(order[a][:100].tolist(), order[b][:100].tolist())
            row['spearman_all'] = float(
                pd.Series(R[a]).corr(pd.Series(R[b]), method='spearman'))
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Headline "cross-family consensus" rule (shared by the pipeline and figures)
# ---------------------------------------------------------------------------
CONSENSUS_NAME = 'Cross-family consensus'


def consensus_mask(df, rule='n_families_topk', max_worst_rank=50, min_families=2):
    """Boolean Series: which rows of a ranking/candidate table meet the rule.

    rule='worst_family_rank': agg_worst_family_rank <= max_worst_rank
    rule='n_families_topk'  : agg_n_families_topk   >= min_families
    """
    if rule == 'worst_family_rank':
        col = 'agg_worst_family_rank'
    elif rule == 'n_families_topk':
        col = 'agg_n_families_topk'
    else:
        raise ValueError(f"unknown consensus rule {rule!r}; use 'worst_family_rank' "
                         "or 'n_families_topk'")
    if col not in df.columns:
        raise KeyError(f"column {col!r} missing: re-run run_analysis.py to regenerate "
                       "the tables with the family-aware consensus columns")
    if rule == 'worst_family_rank':
        return df[col] <= max_worst_rank
    return df[col] >= min_families


def consensus_label(rule='n_families_topk', max_worst_rank=50, min_families=2):
    """Plot/legend text for the rule, e.g. 'Cross-family consensus (worst family rank <= 50)'."""
    if rule == 'worst_family_rank':
        return f'{CONSENSUS_NAME} (worst family rank ≤ {max_worst_rank})'
    return f'{CONSENSUS_NAME} (top-ranked in ≥ {min_families} families)'


def consensus_from_config(cfg):
    """(rule, max_worst_rank, min_families) from a config module, with defaults."""
    return (getattr(cfg, 'CONSENSUS_RULE', 'n_families_topk'),
            int(getattr(cfg, 'CONSENSUS_MAX_WORST_FAMILY_RANK', 50)),
            int(getattr(cfg, 'CONSENSUS_MIN_FAMILIES_TOPK', 2)))


def config_consensus_mask(df, cfg):
    return consensus_mask(df, *consensus_from_config(cfg))


def config_consensus_label(cfg):
    return consensus_label(*consensus_from_config(cfg))
