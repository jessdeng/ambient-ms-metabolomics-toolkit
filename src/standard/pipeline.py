"""
Standard Pipeline — no MetaboAnalyst offset
============================================
Identical to metaboanalyst_pipeline.py except it imports from preprocessing_standard,
which labels bins at their true geometric centers (no -0.05 Da offset).

Use this version when MetaboAnalyst replication is not the goal.
Use metaboanalyst_pipeline.py when you need output to match MetaboAnalyst exactly.

Usage:
    python run_analysis_standard.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.graph_objects as go

from src.shared.plot_style import apply_style, pub_savefig

apply_style()

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from standard.preprocessing import load_experiment, bin_features, filter_low_variance, filter_low_abundance, preprocess
from shared.grouping import permute_labels_by_group

# -- Configuration ---------------------------------------------------------------
N_COMPONENTS = 8   # number of PLS-DA components for scores plot and cross-validation
N_TOP_VIP   = 30   # how many top VIP features to show in the bar chart
# --------------------------------------------------------------------------------
def fit_plsda(X, y_labels, n_components):
    """
    Fit PLS-DA with one-hot encoded Y.
    Used for scores plot and cross-validation (8 components).
    """
    le  = LabelEncoder()
    y   = le.fit_transform(y_labels)
    Y   = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))

    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(X, Y)
    T = pls.transform(X)

    return pls, T, y, Y, le.classes_


def compute_vip(X, y_labels, n_components=1):
    """
    VIP scores aggregated across the first ``n_components`` PLS-DA latent variables.

    The standard VIP formula sums each feature's squared, length-normalised weight
    over the retained components, weighted by the Y-variance each component
    explains (SS_a = ||t_a||^2 * ||q_a||^2):

        VIP_j = sqrt( p * sum_a (w_aj/||w_a||)^2 * SS_a / sum_a SS_a )

    Component asymmetry (intentional — NOT a bug)
    ---------------------------------------------
    ``n_components=1`` (the default) reproduces MetaboAnalyst's component-1 VIP
    exactly, which is why the toolkit defaults to it for cross-tool parity even
    though the 3-D scores plot is drawn with ``config.N_PLSDA_COMPONENTS`` (8)
    latent variables. The two serve different purposes: the scores plot visualises
    the leading multivariate structure, while the 1-LV VIP is the MetaboAnalyst-
    comparable feature ranking. Set ``config.PLSDA_VIP_NUM_COMPONENTS`` (passed
    here) > 1 to aggregate VIP over more components and match the plot dimensions;
    the value used is logged by run_analysis so an independent researcher does not
    mistake the 1-vs-8 asymmetry for an error.
    """
    le = LabelEncoder()
    y  = le.fit_transform(y_labels)
    Y  = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))

    # Clamp to a feasible rank (can't exceed features or n_samples-1).
    A = int(max(1, min(n_components, X.shape[1], X.shape[0] - 1)))
    pls = PLSRegression(n_components=A, scale=False)
    pls.fit(X, Y)

    T = pls.x_scores_       # (n_samples, A)
    W = pls.x_weights_      # (n_features, A)
    Q = pls.y_loadings_     # (n_groups,  A)

    n_features = X.shape[1]
    SS     = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)   # (A,)
    W_norm = W / np.sqrt(np.sum(W ** 2, axis=0))
    vip    = np.sqrt(n_features * (W_norm ** 2 @ SS) / SS.sum())

    return vip


def compute_vip_1comp(X, y_labels):
    """1-component VIP — exact MetaboAnalyst component-1 parity.

    Thin wrapper around ``compute_vip(..., n_components=1)``. This is the canonical
    VIP used inside the ensemble consensus / bootstrap / permutation null so that
    layer stays MetaboAnalyst-comparable regardless of the plotting components.
    """
    return compute_vip(X, y_labels, n_components=1)


def compute_plsda_q2(X, y_labels, n_components, groups=None, n_splits=5,
                     random_state=42):
    """Cross-validated Q^2 for PLS-DA: Q^2 = 1 - PRESS/TSS on the one-hot Y.

    PRESS is the held-out prediction error summed over folds; TSS uses each
    training fold's Y mean as the no-information predictor. If `groups` is given,
    folds respect the biological colony (StratifiedGroupKFold) so technical
    replicates never straddle the split; otherwise StratifiedKFold is used.
    Falls back to ungrouped folds if a grouped split is infeasible (e.g. under a
    permuted labelling).
    """
    le = LabelEncoder()
    y  = le.fit_transform(y_labels)
    Y  = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
    n_comp = int(max(1, min(n_components, X.shape[1], X.shape[0] - 2)))

    def _splits():
        if groups is not None:
            k = int(max(2, min(n_splits, np.unique(groups).size)))
            try:
                return list(StratifiedGroupKFold(
                    n_splits=k, shuffle=True,
                    random_state=random_state).split(X, y, groups))
            except Exception:
                pass
        k = int(max(2, min(n_splits, np.min(np.bincount(y)))))
        return list(StratifiedKFold(n_splits=k, shuffle=True,
                                    random_state=random_state).split(X, y))

    press = 0.0
    tss   = 0.0
    for tr, te in _splits():
        pls = PLSRegression(n_components=min(n_comp, len(tr) - 1), scale=False)
        pls.fit(X[tr], Y[tr])
        Y_hat = pls.predict(X[te])
        press += np.sum((Y[te] - Y_hat) ** 2)
        tss   += np.sum((Y[te] - Y[tr].mean(axis=0)) ** 2)
    return 1.0 - press / tss if tss > 0 else np.nan


def compute_plsda_r2y(X, y_labels, n_components):
    """Apparent (goodness-of-fit) cumulative R^2Y for PLS-DA.

    R^2Y = 1 - SS(Y - Y_hat) / SS(Y - mean(Y)) for the model fit on ALL data, on
    the one-hot response — the fraction of Y variance the components explain. It is
    the in-sample complement to the cross-validated Q^2: a large R^2Y with a small
    Q^2 signals overfitting, so the two are reported together.
    """
    le = LabelEncoder()
    y  = le.fit_transform(y_labels)
    Y  = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
    A  = int(max(1, min(n_components, X.shape[1], X.shape[0] - 1)))
    pls = PLSRegression(n_components=A, scale=False)
    pls.fit(X, Y)
    Y_hat = pls.predict(X)
    ss_res = np.sum((Y - Y_hat) ** 2)
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def adaptive_n_components(n_classes, n_groups, requested, binary_cap=2):
    """Context-aware PLS-DA latent-variable count.

    Caps the *requested* component count to what the design can actually support,
    so the model cannot trivially drive R^2Y -> 1.0:

      * never more than ``n_independent_biological_groups - 1`` (a PLS model with
        as many components as groups reconstructs the response exactly);
      * for a binary problem, never more than ``binary_cap`` (default 2) — a
        2-class separation needs ~1 latent variable; extra LVs only overfit.

    Always returns at least 1. Use this in place of a hard-coded N_PLSDA_COMPONENTS.
    """
    feasible = max(1, int(n_groups) - 1)
    cap = min(int(requested), feasible)
    if int(n_classes) <= 2:
        cap = min(cap, int(binary_cap))
    return int(max(1, cap))


def optimize_plsda_components(X, y_labels, groups=None, max_components=None,
                              n_splits=5, random_state=42):
    """Pick the LV count in ``1..max_components`` that maximises grouped Q^2.

    A principled alternative to a fixed component count: every candidate is scored
    with the SAME leave-one-biological-group-out Q^2 used elsewhere, so the choice
    reflects out-of-sample predictivity rather than in-sample fit. The upper bound
    defaults to ``n_groups - 1``. Returns ``(best_n_components, best_q2)`` and
    falls back to 1 component if no model achieves a finite Q^2.
    """
    n_groups = (len(set(np.asarray(groups).tolist())) if groups is not None
                else int(np.unique(y_labels).size))
    hi = max_components if max_components is not None else max(1, n_groups - 1)
    hi = int(max(1, min(hi, X.shape[1], X.shape[0] - 2)))
    best_a, best_q2 = 1, -np.inf
    for a in range(1, hi + 1):
        q2 = compute_plsda_q2(X, y_labels, a, groups=groups, n_splits=n_splits,
                              random_state=random_state)
        if np.isfinite(q2) and q2 > best_q2:
            best_q2, best_a = q2, a
    return best_a, (best_q2 if np.isfinite(best_q2) else np.nan)


def evaluate_plsda_q2(X, y_labels, n_components, groups=None, n_splits=5,
                      n_perm=200, random_state=42):
    """PLS-DA model quality: observed R^2Y and Q^2 + a permutation null for BOTH.

    Returns a dict::

        {'r2y', 'q2',                      # observed apparent fit & CV predictivity
         'r2y_null', 'q2_null',            # (n_perm,) permuted statistics
         'r2y_p', 'q2_p',                  # empirical +1/+1 p-values (Phipson &
                                           #   Smyth 2010): (#{null>=obs}+1)/(n+1)
         'n_perm_valid'}

    Both R^2Y (in-sample fit) and Q^2 (cross-validated predictivity) are tracked
    so referees see goodness-of-fit and generalisation side by side, each with its
    own calibrated permutation p-value. When ``groups`` is given, labels are
    permuted at the GROUP (colony) level via ``permute_labels_by_group`` so
    technical replicates are never scrambled independently and the grouped Q^2
    folds stay well-defined under the null; otherwise a plain sample-level shuffle
    is used.
    """
    r2y_obs = compute_plsda_r2y(X, y_labels, n_components)
    q2_obs  = compute_plsda_q2(X, y_labels, n_components, groups=groups,
                               n_splits=n_splits, random_state=random_state)
    rng  = np.random.default_rng(random_state)
    r2y_null = np.empty(n_perm, dtype=float)
    q2_null  = np.empty(n_perm, dtype=float)
    y_arr = np.asarray(y_labels)
    for i in range(n_perm):
        if groups is not None:
            yp = permute_labels_by_group(y_arr, groups, rng)
        else:
            yp = rng.permutation(y_arr)
        r2y_null[i] = compute_plsda_r2y(X, yp, n_components)
        q2_null[i]  = compute_plsda_q2(X, yp, n_components, groups=groups,
                                       n_splits=n_splits,
                                       random_state=random_state + i + 1)

    def _emp_p(obs, null):
        valid = null[np.isfinite(null)]
        return (((np.sum(valid >= obs) + 1.0) / (valid.size + 1.0))
                if valid.size else np.nan)

    return {
        'r2y': r2y_obs, 'q2': q2_obs,
        'r2y_null': r2y_null, 'q2_null': q2_null,
        'r2y_p': _emp_p(r2y_obs, r2y_null),
        'q2_p':  _emp_p(q2_obs, q2_null),
        'n_perm_valid': int(np.isfinite(q2_null).sum()),
    }


def plot_scores_3d(T, pls, y_labels, classes, experiment_name, out_path):
    T_all = pls.x_scores_
    Q = pls.y_loadings_
    SS = np.sum(T_all ** 2, axis=0) * np.sum(Q ** 2, axis=0)
    pct_cov = SS / SS.sum() * 100

    # Adaptive component capping can leave fewer than 3 latent variables (a binary
    # design is capped at <=2). The 3-D scatter indexes components 1-3, so pad any
    # missing axes with zeros and a 0% label instead of raising an IndexError.
    if T.shape[1] < 3:
        pad = 3 - T.shape[1]
        T = np.hstack([T, np.zeros((T.shape[0], pad))])
        pct_cov = np.concatenate([pct_cov, np.zeros(pad)])

    fig = go.Figure()
    palette = sns.color_palette('colorblind', n_colors=len(classes))

    for i, group in enumerate(classes):
        mask = y_labels == group
        r, g, b = [int(c * 255) for c in palette[i][:3]]
        fig.add_trace(go.Scatter3d(
            x=T[mask, 0],
            y=T[mask, 1],
            z=T[mask, 2],
            mode='markers',
            name=group,
            marker=dict(size=5, color=f'rgb({r},{g},{b})')
        ))

    fig.update_layout(
        title=f'PLS-DA Scores -- {experiment_name}',
        scene=dict(
            xaxis_title=f'Component 1 ({pct_cov[0]:.1f}%)',
            yaxis_title=f'Component 2 ({pct_cov[1]:.1f}%)',
            zaxis_title=f'Component 3 ({pct_cov[2]:.1f}%)'
        )
    )
    fig.write_html(out_path)
    print(f"  Saved -> {out_path}")

def plot_vip(vip, mz, X, y_labels, n_top, experiment_name, out_path):
    top_idx = np.argsort(vip)[::-1][:n_top]
    top_mz = mz[top_idx]
    top_vip = vip[top_idx]

    top_idx = top_idx[::-1]
    top_mz = top_mz[::-1]
    top_vip = top_vip[::-1]

    groups = sorted(np.unique(y_labels))
    n_groups = len(groups)

    heatmap_data = np.zeros((n_top, n_groups))
    for i, idx in enumerate(top_idx):
        for j, group in enumerate(groups):
            mask = y_labels == group
            heatmap_data[i, j] = X[mask, idx].mean()

    row_mins = heatmap_data.min(axis=1, keepdims=True)
    row_maxs = heatmap_data.max(axis=1, keepdims=True)
    row_range = row_maxs - row_mins
    row_range[row_range == 0] = 1
    heatmap_norm = (heatmap_data - row_mins) / row_range

    fig = plt.figure(figsize=(12, 8), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1.2], wspace=0.05)

    x_min = top_vip.min() * 0.95

    ax_dot = fig.add_subplot(gs[0])
    for i in range(n_top):
        ax_dot.plot([1.7, top_vip[i]], [i, i], color='grey', linewidth=0.5)
        ax_dot.plot(top_vip[i], i, 'o', color='#555555', markersize=7)

    ax_dot.set_yticks(range(n_top))
    ax_dot.set_yticklabels([f"{v:.1f}" for v in top_mz], fontsize=9)
    ax_dot.set_xlabel('VIP Scores')
    ax_dot.set_xlim(top_vip.min() * 0.95, top_vip.max() * 1.05)

    ax_heat = fig.add_subplot(gs[1])
    cmap = plt.cm.RdBu_r
    ax_heat.imshow(heatmap_norm, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    for i in range(n_top + 1):
        ax_heat.axhline(i - 0.5, color='white', linewidth=1)
    for j in range(n_groups + 1):
        ax_heat.axvline(j - 0.5, color='white', linewidth=1)

    ax_heat.set_xticks(range(n_groups))
    ax_heat.set_xticklabels(groups, rotation=45, ha='right', fontsize=8)
    ax_heat.set_yticks([])

    ax_dot.set_ylim(-0.5, n_top - 0.5)
    ax_heat.set_ylim(-0.5, n_top - 0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_heat, shrink=0.4, aspect=15, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'])

    fig.suptitle(f'Top {n_top} VIP Features -- {experiment_name}', fontsize=13)
    pub_savefig(out_path)
