"""
PLS-DA Variable Importance in Projection (VIP).

VIP_j = sqrt( p * sum_a (w_aj/||w_a||)^2 SS_a / sum_a SS_a ),  SS_a = ||t_a||^2 ||q_a||^2

With one component this is MetaboAnalyst's "Comp. 1" VIP (validated against
MetaboAnalyst: max difference 5e-5 across 2,707 features). The multi-component
form matches ropls and mixOmics (tests/test_vip_validation.py).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from shared.plot_style import apply_style, pub_savefig

apply_style()

# Tight NIPALS settings (VIP then matches the exact R solutions to < 1e-6).
VALIDATED_PLS_TOL = 1e-14
VALIDATED_PLS_MAX_ITER = 100000


def compute_vip(X, y_labels, n_components=1, tol=None, max_iter=None):
    """VIP over the first ``n_components`` PLS-DA latent variables (one-hot Y)."""
    y = LabelEncoder().fit_transform(y_labels)
    Y = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
    A = int(max(1, min(n_components, X.shape[1], X.shape[0] - 1)))
    kw = {k: v for k, v in (('tol', tol), ('max_iter', max_iter)) if v is not None}
    pls = PLSRegression(n_components=A, scale=False, **kw).fit(X, Y)
    T, W, Q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    SS = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)
    W_norm = W / np.sqrt(np.sum(W ** 2, axis=0))
    return np.sqrt(X.shape[1] * (W_norm ** 2 @ SS) / SS.sum())


def compute_vip_1comp(X, y_labels, tol=None, max_iter=None):
    """Component-1 VIP (MetaboAnalyst 'Comp. 1')."""
    return compute_vip(X, y_labels, n_components=1, tol=tol, max_iter=max_iter)


def plot_vip(vip, mz, X_means, groups, n_top, experiment_name, out_path):
    """Top-VIP dot plot with a row-scaled heatmap of the condition means.

    X_means: (n_conditions, n_features) condition means of the colony means;
    groups: the condition names in the same order."""
    top = np.argsort(vip)[::-1][:n_top][::-1]
    heat = X_means[:, top].T
    lo, hi = heat.min(1, keepdims=True), heat.max(1, keepdims=True)
    heat = (heat - lo) / np.where(hi - lo == 0, 1, hi - lo)

    fig = plt.figure(figsize=(12, 8), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1.2], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    x0 = vip[top].min() * 0.95
    for i, j in enumerate(top):
        ax.plot([x0, vip[j]], [i, i], color='grey', linewidth=0.5)
        ax.plot(vip[j], i, 'o', color='#555555', markersize=7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"{mz[j]:.1f}" for j in top], fontsize=9)
    ax.set_xlabel('Component-1 VIP (colony means)')
    ax.set_xlim(x0, vip[top].max() * 1.05)
    ax.set_ylim(-0.5, len(top) - 0.5)

    axh = fig.add_subplot(gs[1])
    axh.imshow(heat, cmap=plt.cm.RdBu_r, aspect='auto', vmin=0, vmax=1)
    axh.set_xticks(range(len(groups)))
    axh.set_xticklabels(groups, rotation=45, ha='right', fontsize=8)
    axh.set_yticks([])
    axh.set_ylim(-0.5, len(top) - 0.5)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=axh, shrink=0.4, aspect=15, pad=0.02)
    cbar.set_ticks([0, 1]); cbar.set_ticklabels(['Low', 'High'])
    fig.suptitle(f'Top {len(top)} VIP features -- {experiment_name}', fontsize=13)
    pub_savefig(out_path)
