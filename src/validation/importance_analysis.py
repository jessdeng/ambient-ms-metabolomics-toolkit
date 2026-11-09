"""Step 4: per-algorithm rankings, definition sensitivity and seed variability."""
import itertools

import numpy as np
import pandas as pd

from shared.importance import (fit_importance, top_n, n_ties_at_cutoff,
                               lda_structure_correlation, METHODS, FAMILY)
from standard.pipeline import compute_vip, VALIDATED_PLS_TOL, VALIDATED_PLS_MAX_ITER

TIGHT = dict(tol=VALIDATED_PLS_TOL, max_iter=VALIDATED_PLS_MAX_ITER)


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else np.nan


def primary_rankings(X, y, a_star, n_seeds=25, seed=42):
    """Validated primary importance vectors (global, native definitions)."""
    imp = {m: fit_importance(m, X, y, seed=seed, n_seeds=n_seeds if m in ('rf', 'gb') else 1)
           for m in METHODS}
    out = {m: r['global'] for m, r in imp.items()}
    out['vip_ref'] = compute_vip(X, y, 1)                        # component-1 reference (original)
    out['vip_multi'] = compute_vip(X, y, a_star, **TIGHT)        # validated multi-component
    meta = {m: dict(type=imp[m]['type'], n_nonzero=imp[m]['n_nonzero'],
                    ties_at_50=n_ties_at_cutoff(imp[m]['global'], 50)) for m in METHODS}
    return out, imp, meta


def original_rankings(X, y, seed=42):
    """Original toolkit ensemble: 5 classifiers at seed 42 + component-1 VIP."""
    out = {m: fit_importance(m, X, y, seed=seed)['global'] for m in ('rf', 'svm', 'gb', 'lr', 'ridge')}
    out['vip'] = compute_vip(X, y, 1)
    return out


def original_vote_counts(rank_dict, n=50):
    c = np.zeros(len(next(iter(rank_dict.values()))), dtype=int)
    for v in rank_dict.values():
        c[top_n(v, n, rule='original')] += 1
    return c


def definition_sensitivity(X, y, X_pareto, primary, seed=42, ns=(25, 50, 100)):
    """Top-N Jaccard of each documented variant against the primary ranking."""
    variants = {
        ('svm', 'one-vs-rest (vs original one-vs-one)'): fit_importance('svm', X, y, svm_multiclass='ovr')['global'],
        ('ridge', 'L2 aggregation'): fit_importance('ridge', X, y, aggregation='l2')['global'],
        ('lr', 'L2 aggregation'): fit_importance('lr', X, y, aggregation='l2')['global'],
        ('svm', 'L2 aggregation'): fit_importance('svm', X, y, aggregation='l2')['global'],
        ('lda', 'L2 aggregation'): fit_importance('lda', X, y, aggregation='l2')['global'],
        ('ridge', 'alpha = 10'): fit_importance('ridge', X, y, ridge_alpha=10.0)['global'],
        ('ridge', 'alpha = 100'): fit_importance('ridge', X, y, ridge_alpha=100.0)['global'],
        ('lr', 'C = 0.1'): fit_importance('lr', X, y, lr_C=0.1)['global'],
        ('lr', 'C = 0.01'): fit_importance('lr', X, y, lr_C=0.01)['global'],
        ('svm', 'C = 0.1'): fit_importance('svm', X, y, svm_C=0.1)['global'],
        ('lda', 'structure correlation'): lda_structure_correlation(X, y),
        ('rf', 'single seed 42 (original)'): fit_importance('rf', X, y, seed=seed)['global'],
        ('gb', 'single seed 42 (original)'): fit_importance('gb', X, y, seed=seed)['global'],
    }
    for m in METHODS:
        variants[(m, 'Pareto scaling')] = fit_importance(m, X_pareto, y, n_seeds=5 if m in ('rf', 'gb') else 1)['global']
    rows = []
    for (m, var), v in variants.items():
        r = dict(method=m, family=FAMILY[m], variant=var)
        for n in ns:
            r[f'jaccard_top{n}_vs_primary'] = jaccard(top_n(v, n), top_n(primary[m], n))
        rows.append(r)
    return pd.DataFrame(rows)


def seed_study(X, y, n_ref=25, ks=(1, 2, 5, 10, 25), n_pairs_seeds=10, n=50):
    """Tree models: (i) pairwise top-N Jaccard between single seeds; (ii) agreement of a
    k-seed average with an independent n_ref-seed reference average."""
    rows = []
    for m in ('rf', 'gb'):
        ref = fit_importance(m, X, y, seed=1000, n_seeds=n_ref)
        pool = fit_importance(m, X, y, seed=0, n_seeds=max(max(ks), n_pairs_seeds))['per_seed']
        tops = [set(top_n(pool[s], n)) for s in range(n_pairs_seeds)]
        pj = [jaccard(a, b) for a, b in itertools.combinations(tops, 2)]
        rows.append(dict(method=m, quantity=f'pairwise single-seed top{n} Jaccard (mean)', k=1, value=float(np.mean(pj))))
        rows.append(dict(method=m, quantity=f'pairwise single-seed top{n} Jaccard (min)', k=1, value=float(np.min(pj))))
        for k in ks:
            avg = pool[:k].mean(0)
            rows.append(dict(method=m, quantity=f'{k}-seed average vs independent {n_ref}-seed reference, top{n} Jaccard',
                             k=k, value=jaccard(top_n(avg, n), top_n(ref['global'], n))))
    return pd.DataFrame(rows)


def saved_seed_pool(method, X, y, path, n_seeds, seed0=0, chunk=5, log=print):
    """Fit `n_seeds` single-seed tree models and save the per-seed importance
    matrix after every `chunk` seeds (resumable; nothing is lost on interruption)."""
    import os
    done = np.load(path) if os.path.exists(path) else np.empty((0, X.shape[1]))
    while len(done) < n_seeds:
        k = min(chunk, n_seeds - len(done))
        new = fit_importance(method, X, y, seed=seed0 + len(done), n_seeds=k)['per_seed']
        done = np.vstack([done, new]); np.save(path, done)
        log(f'  {method}: {len(done)}/{n_seeds} seeds saved')
    return done


def seed_pool_summary(pool, ns=(25, 50, 100)):
    """Split-half agreement of independent seed averages, and pairwise single-seed
    agreement, from one saved pool."""
    h = len(pool) // 2
    rows = []
    for n in ns:
        a, b = top_n(pool[:h].mean(0), n), top_n(pool[h:2 * h].mean(0), n)
        singles = [set(top_n(p, n)) for p in pool[:min(len(pool), 20)]]
        pj = [jaccard(s, t) for s, t in itertools.combinations(singles, 2)]
        rows.append(dict(n=n, n_seeds=len(pool), split_half_seeds=h,
                         jaccard_split_half_averages=jaccard(a, b),
                         jaccard_single_seed_pairs_mean=float(np.mean(pj)),
                         jaccard_single_seed_pairs_min=float(np.min(pj))))
    return pd.DataFrame(rows)


def seed_topn_frequency(pool, idx, n=50):
    """For features `idx`: fraction of single seeds placing the feature in the
    top-n, and median / 10th-90th percentile single-seed rank."""
    ranks = (-pool).argsort(1).argsort(1) + 1
    r = ranks[:, idx]
    return pd.DataFrame(dict(feature_index=idx, seed_top_frequency=(r <= n).mean(0),
                             median_rank=np.median(r, 0), rank_p10=np.percentile(r, 10, axis=0),
                             rank_p90=np.percentile(r, 90, axis=0)))
