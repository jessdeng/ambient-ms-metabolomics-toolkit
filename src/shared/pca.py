"""Unsupervised exploratory structure analysis (PCA) for the standard pipeline.

PCA is fitted WITHOUT class labels on the same preprocessed (normalised,
transformed, scaled) descriptive matrix that the supervised models use.
Condition and colony labels are used only afterwards, to describe the fitted
structure:

* explained variance per principal component (scree);
* score plots (spectra and colony means) coloured by condition;
* for every PC, a nested variance partition of the scores into
  condition-plate / colony-within-condition (biological) / technical-replicate
  components (method-of-moments, nested random-effects ANOVA);
* the m/z features with the largest absolute loadings per PC.

Interpretation limits: PCA loadings are not discriminatory importance, and
because each condition was grown on a single plate/enclosure, a condition-wise
separation along a PC is a condition-plate association, not a treatment effect.
"""
import os

import numpy as np
import pandas as pd


def pca(X, k=10):
    """Column-centred PCA by SVD. Returns scores (n x k), loadings (p x k, unit
    norm), explained-variance ratio (k) and all eigenvalues. The sign of every
    component is fixed (largest-|loading| entry positive) for reproducible plots."""
    X = np.asarray(X, float)
    Xc = X - X.mean(0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    sgn = np.sign(Vt[np.arange(len(Vt)), np.abs(Vt).argmax(1)])
    sgn[sgn == 0] = 1
    U, Vt = U * sgn, Vt * sgn[:, None]
    ev = s ** 2 / (len(X) - 1)
    k = min(k, len(s))
    return dict(scores=U[:, :k] * s[:k], loadings=Vt[:k].T, evr=ev[:k] / ev.sum(),
                eigenvalues=ev, mean=X.mean(0))


def colony_means(X, groups):
    g = np.asarray(groups)
    ids = list(dict.fromkeys(g))
    return np.vstack([np.asarray(X)[g == i].mean(0) for i in ids]), np.array(ids)


def nested_variance_components(v, condition, groups):
    """Nested random-effects ANOVA of one variable (e.g. a PC score):
    condition-plate / colony-within-condition / technical replicate, with
    method-of-moments estimators. Exact for balanced designs (e.g. 5 x 3 x 3);
    for unbalanced designs average cell sizes are used and ``balanced`` is False."""
    v = np.asarray(v, float); c = np.asarray(condition); g = np.asarray(groups)
    conds = list(dict.fromkeys(c)); cols = list(dict.fromkeys(g))
    gm = v.mean()
    ss_cond = sum((c == a).sum() * (v[c == a].mean() - gm) ** 2 for a in conds)
    ss_col = sum((g == b).sum() * (v[g == b].mean() - v[c == c[g == b][0]].mean()) ** 2 for b in cols)
    ss_tech = sum(((v[g == b] - v[g == b].mean()) ** 2).sum() for b in cols)
    df_cond, df_col, df_tech = len(conds) - 1, len(cols) - len(conds), len(v) - len(cols)
    if min(df_cond, df_col, df_tech) <= 0:
        return dict(s2_cond=np.nan, s2_col=np.nan, s2_tech=np.nan, frac_cond=np.nan, frac_col=np.nan,
                    frac_tech=np.nan, icc_colony=np.nan, balanced=False, df=(df_cond, df_col, df_tech))
    ms_cond, ms_col, ms_tech = ss_cond / df_cond, ss_col / df_col, ss_tech / df_tech
    n_t = len(v) / len(cols)
    n_c = len(v) / len(conds)
    s2_tech = ms_tech
    s2_col = max(0.0, (ms_col - ms_tech) / n_t)
    s2_cond = max(0.0, (ms_cond - ms_col) / n_c)
    tot = s2_cond + s2_col + s2_tech
    sizes = pd.Series(g).value_counts()
    return dict(s2_cond=s2_cond, s2_col=s2_col, s2_tech=s2_tech,
                frac_cond=s2_cond / tot if tot else np.nan, frac_col=s2_col / tot if tot else np.nan,
                frac_tech=s2_tech / tot if tot else np.nan,
                icc_colony=s2_col / (s2_col + s2_tech) if (s2_col + s2_tech) else np.nan,
                balanced=bool(sizes.nunique() == 1), df=(df_cond, df_col, df_tech))


def run_pca(X_scaled, y_labels, groups, mz, sample_names, out_dir, safe_name,
            experiment_name='', n_components=5, n_top_loadings=15):
    """Fit PCA on the descriptive matrix and write tables + one summary figure.

    Files written to ``out_dir``:
      pca_explained_variance_<exp>.csv   PC, explained-variance ratio, cumulative,
                                         variance partition of the scores
      pca_scores_<exp>.csv               per-spectrum scores with condition/colony
      pca_top_loadings_<exp>.csv         largest |loading| m/z per PC
      pca_<exp>.png                      scree + PC1/PC2 + PC1/PC3 score plots
    Returns the explained-variance DataFrame.
    """
    import matplotlib.pyplot as plt
    from shared.plot_style import pub_savefig

    y = np.asarray(y_labels); g = np.asarray(groups); mz = np.asarray(mz, float)
    k = int(min(n_components, X_scaled.shape[0] - 1, X_scaled.shape[1]))
    res = pca(X_scaled, k=max(k, 2))
    S, L, evr = res['scores'], res['loadings'], res['evr']
    rows = []
    for i in range(S.shape[1]):
        vc = nested_variance_components(S[:, i], y, g)
        rows.append(dict(pc=f'PC{i + 1}', explained_variance_ratio=float(evr[i]),
                         cumulative=float(evr[:i + 1].sum()),
                         frac_condition_plate=vc['frac_cond'], frac_colony=vc['frac_col'],
                         frac_technical=vc['frac_tech'], balanced_design=vc['balanced']))
    ev_df = pd.DataFrame(rows)
    ev_df.to_csv(os.path.join(out_dir, f'pca_explained_variance_{safe_name}.csv'), index=False)
    sc = pd.DataFrame(S, columns=[f'PC{i + 1}' for i in range(S.shape[1])])
    sc.insert(0, 'colony', g); sc.insert(0, 'condition', y); sc.insert(0, 'sample', list(sample_names))
    sc.to_csv(os.path.join(out_dir, f'pca_scores_{safe_name}.csv'), index=False)
    top = []
    for i in range(L.shape[1]):
        idx = np.argsort(np.abs(L[:, i]))[::-1][:n_top_loadings]
        for r, j in enumerate(idx, 1):
            top.append(dict(pc=f'PC{i + 1}', rank=r, mz=float(mz[j]), loading=float(L[j, i])))
    pd.DataFrame(top).to_csv(os.path.join(out_dir, f'pca_top_loadings_{safe_name}.csv'), index=False)

    conds = list(dict.fromkeys(y))
    cmap = plt.get_cmap('tab10')
    colour = {c: cmap(i % 10) for i, c in enumerate(conds)}
    cm, ids = colony_means(S, g)
    cm_cond = np.array([y[g == i][0] for i in ids])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    ax = axes[0]
    allev = res['eigenvalues'] / res['eigenvalues'].sum()
    n_show = min(10, len(allev))
    ax.bar(np.arange(1, n_show + 1), allev[:n_show] * 100, color='0.6')
    ax.set_xlabel('Principal component'); ax.set_ylabel('Explained variance (%)')
    ax.set_xticks(np.arange(1, n_show + 1)); ax.set_title('Scree', loc='left')
    for ax, (a, b) in zip(axes[1:], [(0, 1), (0, 2) if S.shape[1] > 2 else (0, 1)]):
        for c in conds:
            m = y == c
            ax.scatter(S[m, a], S[m, b], s=14, color=colour[c], alpha=0.45, edgecolor='none')
            mc = cm_cond == c
            ax.scatter(cm[mc, a], cm[mc, b], s=46, color=colour[c], edgecolor='k', lw=0.6, label=str(c))
        ax.axhline(0, color='0.85', lw=0.6, zorder=0); ax.axvline(0, color='0.85', lw=0.6, zorder=0)
        ax.set_xlabel(f'PC{a + 1} ({evr[a] * 100:.1f}%)'); ax.set_ylabel(f'PC{b + 1} ({evr[b] * 100:.1f}%)')
    axes[1].set_title('Scores: spectra (small), colony means (large)', loc='left', fontsize=9)
    axes[2].legend(frameon=False, fontsize=7, loc='best')
    fig.suptitle(f'PCA (unsupervised; labels shown after fitting) -- {experiment_name}', fontsize=10)
    fig.tight_layout()
    pub_savefig(os.path.join(out_dir, f'pca_{safe_name}.png'))
    return ev_df
