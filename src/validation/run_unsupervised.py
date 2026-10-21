"""Label-free PCA structure diagnostic (validated workflow, plan step 4).

    PYTHONPATH=src:. python -m validation.run_unsupervised --data data --out results/validation/step_pca
"""
import argparse, contextlib, io, os, warnings
import numpy as np
import pandas as pd

from validation.data import load_config, load_isolate
from validation.unsupervised import (pca, nested_structure, nested_variance_components,
                                    outlier_statistics, nearest_neighbour_consistency,
                                    colony_clustering, colony_means, tucker_congruence,
                                    supervised_overlap)
from standard.preprocessing import (preprocess, filter_prevalence, filter_low_variance,
                                    filter_low_abundance)

NORMS = ('tic', 'pqn', 'none')
SCALINGS = ('autoscale', 'pareto', 'none')


def broken_stick(p, k):
    return np.array([sum(1 / i for i in range(j, p + 1)) / p for j in range(1, k + 1)])


def per_feature_components(X, cond, groups):
    rows = [nested_variance_components(X[:, j], cond, groups) for j in range(X.shape[1])]
    return pd.DataFrame(rows)[['s2_cond', 's2_col', 's2_tech', 'icc_colony']]


def analyse_isolate(iso, d, cfg, k_out=3, k=10):
    X, y, g = d['X'], d['y'], d['groups']
    logtic = np.log10(d['X_binned'].sum(1))
    res = {}
    # (a) nested D45 vs D15-mean
    tab, p45, p15, ids = nested_structure(X, g, k=5)
    res['nested'] = tab.assign(isolate=iso)
    pfull = pca(X, k)
    bs = broken_stick(min(X.shape) - 1, k)
    # (b, d) explained variance and per-PC variance partition, TIC association
    ev = []
    for j in range(k):
        vc = nested_variance_components(pfull['scores'][:, j], y, g)
        ev.append(dict(isolate=iso, pc=j + 1, evr=pfull['evr'][j], cum_evr=pfull['evr'][:j + 1].sum(),
                       broken_stick=bs[j], r_logTIC=np.corrcoef(pfull['scores'][:, j], logtic)[0, 1],
                       frac_condition_plate=vc['frac_cond'], frac_colony=vc['frac_col'],
                       frac_technical=vc['frac_tech'], icc_colony=vc['icc_colony'], balanced=vc['balanced']))
    res['ev'] = pd.DataFrame(ev)
    # whole-matrix partition (autoscaled features carry equal weight)
    pf = per_feature_components(X, y, g)
    tot = pf[['s2_cond', 's2_col', 's2_tech']].sum()
    res['matrix_partition'] = pd.DataFrame([dict(
        isolate=iso, n_features=X.shape[1],
        frac_condition_plate=tot.s2_cond / tot.sum(), frac_colony=tot.s2_col / tot.sum(),
        frac_technical=tot.s2_tech / tot.sum(), median_feature_icc=pf.icc_colony.median(),
        frac_features_icc_gt_0_5=(pf.icc_colony > 0.5).mean())])
    res['feature_components'] = pf.assign(isolate=iso, signal_mz=d['mz'])
    # (c) outliers on D45, k_out PCs
    o = outlier_statistics(X, k=k_out)
    res['outliers'] = pd.DataFrame(dict(
        isolate=iso, spectrum=d['names'], condition=y, colony=g, T2=o['T2'], Q=o['Q'],
        T2_gt_95=o['T2'] > o['T2_lim95'], T2_gt_99=o['T2'] > o['T2_lim99'],
        Q_gt_95=o['Q'] > o['Q_lim95'], Q_gt_99=o['Q'] > o['Q_lim99'], logTIC=logtic))
    # replicate consistency and (g) colony clustering
    nn = nearest_neighbour_consistency(X, y, g)
    col_cond = np.array([y[np.asarray(g) == i][0] for i in ids])
    cl, Z = colony_clustering(p15['scores'][:, :3], col_cond)
    res['consistency'] = pd.DataFrame([dict(isolate=iso, **nn, **{f'ward_colony_{a}': b for a, b in cl.items()})])
    res['_pca'] = dict(p45=pfull, p15=p15, colony_ids=ids, colony_condition=col_cond, Z=Z, logtic=logtic)
    # (e) normalisation x scaling sensitivity (filters are pre-normalisation, so the
    # feature set is unchanged across the grid)
    rows = []
    ref = pfull['loadings'][:, :3]
    for nm in NORMS:
        for sc in SCALINGS:
            Xv = preprocess(d['X_filt_raw'], normalization=nm, log_transform=cfg['LOG_TRANSFORM'], scaling=sc)
            rows.append(_grid_row(iso, nm, sc, 'per-condition prevalence (primary)', Xv, y, g, logtic, ref, None))
    # label-free prevalence variant (prevalence over all 45 spectra)
    Xb, mzb = d['X_binned'], d['mz_binned']
    one = np.array(['all'] * len(y))
    Xl, mzl = filter_prevalence(Xb, mzb, one, threshold=cfg['PREVALENCE_THRESHOLD'])
    if cfg['VARIANCE_PERCENTILE'] > 0:
        Xl, mzl = filter_low_variance(Xl, mzl, percentile=cfg['VARIANCE_PERCENTILE'])
    if cfg['ABUNDANCE_PERCENTILE'] > 0:
        Xl, mzl = filter_low_abundance(Xl, mzl, percentile=cfg['ABUNDANCE_PERCENTILE'])
    Xv = preprocess(Xl, normalization=cfg['NORMALIZATION'], log_transform=cfg['LOG_TRANSFORM'], scaling=cfg['SCALING'])
    rows.append(_grid_row(iso, cfg['NORMALIZATION'], cfg['SCALING'], 'all-spectra prevalence (label-free)',
                          Xv, y, g, logtic, ref, (mzl, d['mz'])))
    res['norm_grid'] = pd.DataFrame(rows)
    return res


def _grid_row(iso, nm, sc, filt, Xv, y, g, logtic, ref, mz_pair):
    pv = pca(Xv, 5)
    L = pv['loadings'][:, :3]
    if mz_pair is not None:                                  # align on shared features
        mzv, mzr = mz_pair
        common, iv, ir = np.intersect1d(np.round(mzv, 3), np.round(mzr, 3), return_indices=True)
        L, R = L[iv], ref[ir]
        nfeat = f'{Xv.shape[1]} ({len(common)} shared)'
    else:
        R, nfeat = ref, str(Xv.shape[1])
    r = dict(isolate=iso, normalisation=nm, scaling=sc, prevalence_filter=filt, n_features=nfeat)
    for j in range(3):
        vc = nested_variance_components(pv['scores'][:, j], y, g)
        r[f'evr_pc{j+1}'] = pv['evr'][j]
        r[f'abs_r_logTIC_pc{j+1}'] = abs(np.corrcoef(pv['scores'][:, j], logtic)[0, 1])
        r[f'frac_condition_plate_pc{j+1}'] = vc['frac_cond']
        r[f'frac_colony_pc{j+1}'] = vc['frac_col']
        r[f'best_congruence_pc{j+1}_vs_primary'] = max(tucker_congruence(L[:, j], R[:, i]) for i in range(3))
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data'); ap.add_argument('--out', default='results/validation/step_pca')
    ap.add_argument('--config', default='configs/validation.json')
    ap.add_argument('--importance-dir', default='results/validation/step4_importance')
    ap.add_argument('--isolates', nargs='+', required=True, help='experiment folder names under data/')
    a = ap.parse_args(argv)
    warnings.simplefilter('ignore')
    cfg = load_config(a.config); os.makedirs(a.out, exist_ok=True)
    allres = {}
    for iso in a.isolates:
        with contextlib.redirect_stdout(io.StringIO()):
            d = load_isolate(a.data, iso, cfg, resample='native', feature_level='signal')
        allres[iso] = (analyse_isolate(iso, d, cfg), d)
    keys = ['nested', 'ev', 'matrix_partition', 'outliers', 'consistency', 'norm_grid', 'feature_components']
    names = ['pca_nested_congruence', 'pca_explained_variance', 'pca_matrix_variance_partition',
             'pca_outliers', 'pca_replicate_consistency', 'pca_normalisation_sensitivity',
             'feature_variance_components']
    for k, n in zip(keys, names):
        pd.concat([r[0][k] for r in allres.values()]).to_csv(os.path.join(a.out, n + '.csv'), index=False)
    # (f) descriptive overlap with supervised rankings, if step-5 importance tables exist
    rows = []
    for iso, (res, d) in allres.items():
        f = os.path.join(a.importance_dir, f'importance_primary_{iso}.csv')
        if not os.path.exists(f):
            continue
        imp = pd.read_csv(f)
        assert np.allclose(imp.signal_mz.values, d['mz']), 'importance table does not match PCA features'
        for m in [c[4:] for c in imp.columns if c.startswith('imp_')]:
            rows.append(dict(isolate=iso, method=m, **supervised_overlap(res['_pca']['p45']['loadings'], imp[f'imp_{m}'].values)))
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(a.out, 'pca_supervised_overlap.csv'), index=False)
    return allres


if __name__ == '__main__':
    main()
