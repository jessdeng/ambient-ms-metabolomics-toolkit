"""Label-free exploratory structure analysis (PCA) for the validated workflow.

Labels (condition, colony) are used only AFTER fitting, to describe the fitted
structure. PCA is not part of the supervised consensus and loadings are not
discriminatory importance. Under the current design condition and plate/enclosure
are aliased, so any condition-wise grouping is a condition-plate association.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import subspace_angles
from scipy.cluster.hierarchy import linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score


def pca(X, k=10):
    """Column-centred PCA by SVD. Returns scores (n x k), loadings (p x k, unit
    norm), explained-variance ratio (k), and all eigenvalues."""
    Xc = X - X.mean(0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    # sign convention: largest-|loading| entry positive, for reproducible plots
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
    return np.vstack([X[g == i].mean(0) for i in ids]), np.array(ids)


def tucker_congruence(a, b):
    """|phi| between two loading vectors (sign-invariant)."""
    return float(abs(a @ b) / np.sqrt((a @ a) * (b @ b)))


def nested_structure(X, groups, k=5):
    """Compare PCA on all spectra (D45) with PCA on colony means (D15-mean).
    Both share D45's preprocessing/scaling; D15 is re-centred only."""
    p45 = pca(X, k)
    Xm, ids = colony_means(X, groups)
    p15 = pca(Xm, k)
    g = np.asarray(groups)
    rows = []
    for j in range(k):
        # colony-mean of D45 scores vs D15 scores on the same PC
        s45m = np.array([p45['scores'][g == i, j].mean() for i in ids])
        rows.append(dict(pc=j + 1, evr_D45=p45['evr'][j], evr_D15mean=p15['evr'][j],
                         tucker_congruence=tucker_congruence(p45['loadings'][:, j], p15['loadings'][:, j]),
                         colony_score_r=float(abs(np.corrcoef(s45m, p15['scores'][:, j])[0, 1]))))
    out = pd.DataFrame(rows)
    for kk in (2, 3, 5):
        if kk <= k:
            ang = np.degrees(subspace_angles(p45['loadings'][:, :kk], p15['loadings'][:, :kk]))
            out.loc[out.pc == kk, f'max_principal_angle_deg_first{kk}'] = float(ang.max())
    return out, p45, p15, ids


def nested_variance_components(v, condition, groups):
    """Nested random-effects ANOVA (condition-plate / colony-in-condition /
    technical) with method-of-moments estimators. Exact for balanced designs
    (5 x 3 x 3); for unbalanced data the average cell sizes are used and the
    result is flagged."""
    v = np.asarray(v, float); c = np.asarray(condition); g = np.asarray(groups)
    conds = list(dict.fromkeys(c)); cols = list(dict.fromkeys(g))
    gm = v.mean()
    ss_cond = sum((c == a).sum() * (v[c == a].mean() - gm) ** 2 for a in conds)
    ss_col = sum((g == b).sum() * (v[g == b].mean() - v[c == c[g == b][0]].mean()) ** 2 for b in cols)
    ss_tech = sum(((v[g == b] - v[g == b].mean()) ** 2).sum() for b in cols)
    df_cond, df_col, df_tech = len(conds) - 1, len(cols) - len(conds), len(v) - len(cols)
    ms_cond, ms_col, ms_tech = ss_cond / df_cond, ss_col / df_col, ss_tech / df_tech
    n_t = len(v) / len(cols)                      # technical reps per colony
    n_c = len(v) / len(conds)                     # spectra per condition
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


def outlier_statistics(X, k=3, alpha=(0.95, 0.99)):
    """Hotelling T2 and Q (SPE) on a k-PC model of X with parametric limits
    (F-based T2; Jackson-Mudholkar Q). Spectra are nested within colonies, so the
    limits are descriptive flags, not tests."""
    n = len(X)
    p = pca(X, k)
    lam = p['eigenvalues']
    t2 = ((p['scores'] / np.sqrt(lam[:k])) ** 2).sum(1)
    Xc = X - p['mean']
    q = ((Xc - p['scores'] @ p['loadings'].T) ** 2).sum(1)
    res = lam[k:]
    th1, th2, th3 = res.sum(), (res ** 2).sum(), (res ** 3).sum()
    h0 = 1 - 2 * th1 * th3 / (3 * th2 ** 2)
    out = dict(T2=t2, Q=q)
    for a in alpha:
        out[f'T2_lim{int(a*100)}'] = k * (n - 1) * (n + 1) / (n * (n - k)) * stats.f.ppf(a, k, n - k)
        ca = stats.norm.ppf(a)
        out[f'Q_lim{int(a*100)}'] = th1 * (ca * np.sqrt(2 * th2 * h0 ** 2) / th1 + 1
                                          + th2 * h0 * (h0 - 1) / th1 ** 2) ** (1 / h0)
    return out


def nearest_neighbour_consistency(X, condition, groups):
    """Fraction of spectra whose nearest neighbour (Euclidean, full feature space)
    is from the same colony / same condition, with the chance expectation."""
    c = np.asarray(condition); g = np.asarray(groups)
    D = np.linalg.norm(X[:, None] - X[None], axis=2); np.fill_diagonal(D, np.inf)
    nn = D.argmin(1)
    n = len(X)
    exp_col = np.mean([((g == g[i]).sum() - 1) / (n - 1) for i in range(n)])
    exp_cond = np.mean([((c == c[i]).sum() - 1) / (n - 1) for i in range(n)])
    return dict(nn_same_colony=float((g[nn] == g).mean()), nn_same_colony_chance=float(exp_col),
                nn_same_condition=float((c[nn] == c).mean()), nn_same_condition_chance=float(exp_cond))


def colony_clustering(scores_colony, colony_condition, n_clusters=None):
    """Ward clustering of colony-mean PC scores: cophenetic correlation and
    adjusted Rand index of the n_conditions-cluster cut against condition."""
    d = pdist(scores_colony)
    Z = linkage(scores_colony, 'ward')
    k = n_clusters or len(set(colony_condition))
    lab = fcluster(Z, k, 'maxclust')
    return dict(cophenetic_r=float(cophenet(Z, d)[0]),
                ari_vs_condition=float(adjusted_rand_score(colony_condition, lab))), Z


def loading_energy_in_set(loadings, idx):
    """Share of a unit-norm loading vector's squared mass in feature set idx."""
    return float((loadings[idx] ** 2).sum())


def supervised_overlap(loadings, importance, n=50, n_pcs=3):
    """DESCRIPTIVE context only: how a supervised ranking relates to the dominant
    unsupervised axes. Not validation - dominant PCs can reflect TIC or
    condition-plate structure rather than discriminatory chemistry."""
    from scipy.stats import spearmanr
    from shared.importance import top_n
    t = top_n(importance, n)
    out = dict(n_features=loadings.shape[0], random_expectation_energy=n / loadings.shape[0])
    for k in range(n_pcs):
        a = np.abs(loadings[:, k])
        out[f'pc{k+1}_loading_energy_in_top{n}'] = float((loadings[t, k] ** 2).sum())
        out[f'pc{k+1}_spearman_absloading_vs_importance'] = float(spearmanr(a, importance).statistic)
        out[f'pc{k+1}_top{n}_overlap_with_top{n}_absloading'] = len(set(t) & set(top_n(a, n)))
    return out
