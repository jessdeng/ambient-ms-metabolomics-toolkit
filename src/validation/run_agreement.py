"""Cross-model agreement driver (plan step 6).

    PYTHONPATH=src:. python -m validation.run_agreement

Signal level uses the primary importance tables (step4_importance). Native-bin
importances are computed once and cached; multi-component VIP is always
recomputed at the A* selected on the same feature level
(q2_profile_validated_features.csv).
"""
import contextlib, io, itertools, json, os, warnings
import numpy as np, pandas as pd
warnings.simplefilter('ignore')
from validation.data import load_config, load_isolate
from validation.importance_analysis import primary_rankings
from validation.agreement import pairwise_agreement, vip_threshold_analysis, family_structure, jaccard
from shared.importance import top_n
from standard.pipeline import compute_vip, VALIDATED_PLS_TOL, VALIDATED_PLS_MAX_ITER

FAM = dict(ridge='L', lr='L', svm='L', rf='T', gb='T', lda='P', vip_ref='P', vip_multi='P')
METH = ['vip_ref', 'vip_multi', 'ridge', 'lr', 'svm', 'lda', 'rf', 'gb']
S4, OUT = 'results/validation/step4_importance', 'results/validation/step6_agreement'
NS = (25, 50, 100)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Cross-model agreement (requires the importance outputs).')
    ap.add_argument('--isolates', nargs='+', required=True, help='experiment folder names under data/')
    a = ap.parse_args(argv)
    cfg = load_config('configs/validation.json'); os.makedirs(OUT, exist_ok=True)
    AS = pd.read_csv(f'{S4}/q2_profile_validated_features.csv').set_index(['level', 'isolate']).A_star
    pairs, vipt, fams, floors = [], [], [], []
    for level in ('signal', 'bin'):
        for iso in a.isolates:
            if level == 'signal':
                imp = pd.read_csv(f'{S4}/importance_primary_{iso}.csv')
                imps = {m: imp[f'imp_{m}'].values for m in METH}
                pool = {m: np.load(f'{S4}/seed_pools/{iso}_{m}.npy') for m in ('rf', 'gb')}
                for m in ('rf', 'gb'):
                    ref = pool[m][:25].mean(0)
                    for n in NS:
                        r = dict(isolate=iso, level=level, method=m, n=n,
                                 floor_primary25_vs_pool25=jaccard(top_n(imps[m], n), top_n(ref, n)))
                        if m == 'rf':
                            bl = [pool[m][i:i + 25].mean(0) for i in range(0, 100, 25)]
                            r['floor_rf_disjoint25_blocks_mean'] = float(np.mean(
                                [jaccard(top_n(a, n), top_n(b, n)) for a, b in itertools.combinations(bl, 2)]))
                        r['single_seed_pairs_mean'] = float(np.mean(
                            [jaccard(top_n(a, n), top_n(b, n)) for a, b in itertools.combinations(pool[m][:20], 2)]))
                        floors.append(r)
            else:
                cache = f'{OUT}/importance_native_bins_{iso}.csv'
                with contextlib.redirect_stdout(io.StringIO()):
                    d = load_isolate('data', iso, cfg, resample='native', feature_level='bin')
                if os.path.exists(cache):
                    tab = pd.read_csv(cache); assert np.allclose(tab.bin_mz, d['mz'])
                    imps = {m: tab[f'imp_{m}'].values for m in METH}
                else:
                    imps, _, _ = primary_rankings(d['X'], d['y'], int(AS[(level, iso)]))
                imps['vip_multi'] = compute_vip(d['X'], d['y'], int(AS[(level, iso)]),
                                                tol=VALIDATED_PLS_TOL, max_iter=VALIDATED_PLS_MAX_ITER)
                pd.DataFrame({'bin_mz': d['mz'], **{f'imp_{m}': imps[m] for m in METH}}).to_csv(cache, index=False)
            pa = pairwise_agreement(imps, NS).assign(isolate=iso, level=level)
            pa['family_a'] = pa.method_a.map(FAM); pa['family_b'] = pa.method_b.map(FAM)
            pairs.append(pa)
            for v in ('vip_ref', 'vip_multi'):
                vipt.append(vip_threshold_analysis(imps[v], {m: imps[m] for m in METH if m != v}, NS)
                            .assign(isolate=iso, level=level, vip=v))
            for metric in ('spearman', 'jaccard_top50'):
                s, _ = family_structure(pa, FAM, metric=metric)
                fams.append(dict(isolate=iso, level=level, metric=metric,
                                 **{k: v for k, v in s.items() if k != 'empirical_clusters'},
                                 empirical_clusters=json.dumps(s['empirical_clusters'])))
    pd.concat(pairs).to_csv(f'{OUT}/pairwise_agreement.csv', index=False)
    pd.concat(vipt).to_csv(f'{OUT}/vip_threshold_analysis.csv', index=False)
    pd.DataFrame(fams).to_csv(f'{OUT}/family_structure.csv', index=False)
    pd.DataFrame(floors).to_csv(f'{OUT}/within_method_seed_floor.csv', index=False)


if __name__ == '__main__':
    main()
