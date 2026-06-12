"""
standard/pipeline.py — PLS-DA, VIP Scores, and Plots
======================================================
PLS-DA is fit as a descriptive model on the full preprocessed dataset.
VIP scores always use 1 component regardless of N_PLSDA_COMPONENTS, as
1-component VIP is the standard metric for feature selection in metabolomics.

Usage:
    python -m standard.run_analysis
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder, label_binarize


# ── PLS-DA ─────────────────────────────────────────────────────────────────────

def fit_plsda(X, y_labels, n_components=8):
    """
    Fit PLS-DA using scikit-learn PLSRegression with a one-hot-encoded
    response matrix Y.

    Returns
    -------
    pls     : fitted PLSRegression model
    T       : X scores (n_samples × n_components)
    y       : integer-encoded labels
    Y       : one-hot response matrix
    classes : ndarray of class names
    """
    le = LabelEncoder()
    y  = le.fit_transform(y_labels)
    classes = le.classes_

    Y = label_binarize(y, classes=np.arange(len(classes)))
    if Y.shape[1] == 1:           # binary case: sklearn returns (n, 1)
        Y = np.hstack([1 - Y, Y])

    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(X, Y)
    T = pls.transform(X)

    return pls, T, y, Y, classes


# ── VIP scores ─────────────────────────────────────────────────────────────────

def compute_vip_1comp(X, y_labels):
    """
    Compute VIP (Variable Importance in Projection) scores using 1 PLS component.

    VIP_j = sqrt( p * sum_a(W_ja^2 * SSY_a) / SSY_total )

    where p = number of features, W = X weights, SSY_a = sum-of-squares of Y
    explained by component a.

    Returns ndarray of shape (n_features,).
    """
    le = LabelEncoder()
    y  = le.fit_transform(y_labels)
    classes = le.classes_

    Y = label_binarize(y, classes=np.arange(len(classes)))
    if Y.shape[1] == 1:
        Y = np.hstack([1 - Y, Y])

    pls = PLSRegression(n_components=1, scale=False)
    pls.fit(X, Y)

    # X weights (normalised): shape (n_features, 1)
    W = pls.x_weights_
    W_norm = W / np.linalg.norm(W, axis=0, keepdims=True)

    # SSY explained by component 1
    T  = pls.transform(X)                  # (n_samples, 1)
    Q  = pls.y_loadings_                   # (n_responses, 1)
    Y_hat = T @ Q.T
    SSY   = np.sum(Y_hat ** 2)

    p = X.shape[1]
    vip = np.sqrt(p * (W_norm[:, 0] ** 2) * SSY / SSY)
    # Simplifies to sqrt(p) * |W_norm| — standard single-component VIP
    vip = np.sqrt(p) * np.abs(W_norm[:, 0])

    return vip


# ── Plots ───────────────────────────────────────────────────────────────────────

def plot_scores_3d(T, pls, y_labels, classes, experiment_name, out_path):
    """
    Interactive 3D PLS-DA scores plot (components 1, 2, 3) saved as HTML.
    """
    palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]

    fig = go.Figure()
    le  = LabelEncoder().fit(y_labels)

    n_comp = T.shape[1]
    x_col  = T[:, 0]
    y_col  = T[:, 1] if n_comp > 1 else np.zeros(len(T))
    z_col  = T[:, 2] if n_comp > 2 else np.zeros(len(T))

    for i, cls in enumerate(classes):
        mask = y_labels == cls
        fig.add_trace(go.Scatter3d(
            x=x_col[mask], y=y_col[mask], z=z_col[mask],
            mode='markers',
            marker=dict(size=6, color=palette[i % len(palette)], opacity=0.85),
            name=cls,
        ))

    var_exp = []
    if hasattr(pls, 'x_scores_'):
        ss_total = np.sum(pls.x_scores_ ** 2, axis=0)
        ss_total[ss_total == 0] = 1
        var_exp = (ss_total / ss_total.sum() * 100).tolist()

    def _ax_label(i):
        if i < len(var_exp):
            return f"PC{i+1} ({var_exp[i]:.1f}%)"
        return f"PC{i+1}"

    fig.update_layout(
        title=f"PLS-DA Scores — {experiment_name}",
        scene=dict(
            xaxis_title=_ax_label(0),
            yaxis_title=_ax_label(1),
            zaxis_title=_ax_label(2),
        ),
        legend_title="Group",
        template="plotly_white",
    )

    fig.write_html(out_path)
    print(f"  Saved → {out_path}")


def plot_vip(vip, mz, X_filt_raw, y_labels, n_top, experiment_name, out_path):
    """
    Dot plot of top VIP features with a per-group intensity heatmap.
    """
    le      = LabelEncoder().fit(y_labels)
    classes = le.classes_

    top_idx = np.argsort(vip)[::-1][:n_top]
    top_mz  = mz[top_idx]
    top_vip = vip[top_idx]

    # Per-group mean intensities for heatmap (log10+1 of raw)
    with np.errstate(divide='ignore', invalid='ignore'):
        X_log = np.log10(X_filt_raw + 1)
    group_means = np.array([
        X_log[y_labels == cls, :][:, top_idx].mean(axis=0)
        for cls in classes
    ])  # shape (n_classes, n_top)

    fig, (ax_dot, ax_heat) = plt.subplots(
        1, 2, figsize=(14, max(4, n_top * 0.35 + 2)),
        gridspec_kw={'width_ratios': [1, len(classes)]}
    )

    palette = sns.color_palette('colorblind', n_colors=len(classes))
    colors  = [palette[le.transform([cls])[0]] for cls in classes]

    # Dot plot
    y_pos = np.arange(n_top)[::-1]
    ax_dot.scatter(top_vip, y_pos, color='steelblue', s=40, zorder=2)
    ax_dot.axvline(1.0, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
    ax_dot.set_yticks(y_pos)
    ax_dot.set_yticklabels([f"m/z {v:.2f}" for v in top_mz], fontsize=8)
    ax_dot.set_xlabel('VIP score')
    ax_dot.set_title(f'Top {n_top} VIP features\n{experiment_name}', fontsize=9)
    ax_dot.invert_yaxis()

    # Heatmap
    vmin = group_means.min()
    vmax = group_means.max()
    im   = ax_heat.imshow(group_means.T, cmap='RdYlBu_r', aspect='auto',
                          vmin=vmin, vmax=vmax)
    ax_heat.set_xticks(range(len(classes)))
    ax_heat.set_xticklabels(classes, rotation=30, ha='right', fontsize=8)
    ax_heat.set_yticks([])
    ax_heat.set_title('Mean log10(intensity+1)\nper group', fontsize=9)
    plt.colorbar(im, ax=ax_heat, shrink=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")
