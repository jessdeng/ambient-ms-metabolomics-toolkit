"""Step 2: two PLS-DA VIP analyses kept distinct.

(A) reference   : component-1 VIP (original toolkit setting; MetaboAnalystR
                  'Comp. 1' equivalent). Also recomputed with tight NIPALS
                  tolerance to show that solver precision does not change it.
(B) validated   : multi-component VIP (Wold PLS2 definition) at A* chosen by
                  leakage-free colony-grouped Q^2 (grouped_q2_profile).
"""
import numpy as np
import pandas as pd

from standard.pipeline import (compute_vip, grouped_q2_profile,
                               VALIDATED_PLS_TOL, VALIDATED_PLS_MAX_ITER)

TIGHT = dict(tol=VALIDATED_PLS_TOL, max_iter=VALIDATED_PLS_MAX_ITER)


def _rank(v):
    """1 = highest value."""
    return pd.Series(v).rank(ascending=False, method='min').astype(int).values


def topn_jaccard(a, b, n):
    sa, sb = set(np.argsort(-a)[:n]), set(np.argsort(-b)[:n])
    return len(sa & sb) / len(sa | sb)


def run_vip_reference(d, cfg, focus_mz=()):
    y, X = d['y'], d['X']
    q = grouped_q2_profile(d['X_binned'], y, d['groups'], d['prep_steps'],
                           max_components=cfg['PLS_MAX_COMPONENTS'],
                           n_splits=cfg['Q2_N_SPLITS'], n_repeats=cfg['Q2_N_REPEATS'],
                           random_state=cfg['RANDOM_SEED'])
    a_star = q['best_n_components']
    vips = {'VIP_A1_original': compute_vip(X, y, 1)}            # exact original call
    for a in range(1, cfg['PLS_MAX_COMPONENTS'] + 1):
        vips[f'VIP_A{a}'] = compute_vip(X, y, a, **TIGHT)
    vips['VIP_Astar'] = vips[f'VIP_A{a_star}']
    feat = pd.DataFrame({'mz': d['mz']})
    for k, v in vips.items():
        feat[k] = v
        feat[f'rank_{k}'] = _rank(v)

    q2 = pd.DataFrame({'n_components': q['components'], 'Q2_mean': q['q2'],
                       'Q2_min_repeat': q['q2_repeats'].min(0),
                       'Q2_max_repeat': q['q2_repeats'].max(0)})
    rows = []
    for k in ['VIP_A1_original', 'VIP_A1'] + [f'VIP_A{a}' for a in range(2, cfg['PLS_MAX_COMPONENTS'] + 1)]:
        r = dict(definition=k, n_vip_gt1=int((vips[k] > 1).sum()),
                 frac_vip_gt1=float((vips[k] > 1).mean()))
        for n in cfg['TOP_N']:
            r[f'jaccard_top{n}_vs_A1_original'] = topn_jaccard(vips[k], vips['VIP_A1_original'], n)
            r[f'jaccard_top{n}_vs_Astar'] = topn_jaccard(vips[k], vips['VIP_Astar'], n)
        rows.append(r)
    summary = pd.DataFrame(rows)
    focus = []
    for m in focus_mz:
        j = int(np.argmin(np.abs(d['mz'] - m)))
        focus.append(dict(mz_reported=m, mz_bin=round(float(d['mz'][j]), 2),
                          VIP_A1_original=vips['VIP_A1_original'][j],
                          rank_A1_original=int(feat[f'rank_VIP_A1_original'][j]),
                          VIP_Astar=vips['VIP_Astar'][j], rank_Astar=int(feat['rank_VIP_Astar'][j]),
                          **{f'rank_A{a}': int(feat[f'rank_VIP_A{a}'][j]) for a in range(1, cfg['PLS_MAX_COMPONENTS'] + 1)}))
    return dict(a_star=a_star, q2=q2, features=feat, summary=summary,
                focus=pd.DataFrame(focus))
