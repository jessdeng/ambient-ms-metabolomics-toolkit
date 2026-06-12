"""
corroboration_figure.py
=======================
Single publication-ready PNG that defends the multi-method ensemble approach.

  Top panel  -- UpSet plot of method intersections among ensemble features
               (n_methods >= 2). Bars show the SIZE of each intersection;
               the dot matrix below shows WHICH methods are in each.
               Together this answers "do the methods agree?" and "what
               does VIP miss that the others catch (and vice versa)?".

  Bottom panel -- Top N most corroborated m/z, ranked by n_methods.
                 Each row is one m/z. Bar length encodes n_methods
                 (out of 6); bar colour encodes top_condition_mean
                 (which group has the highest mean intensity for that
                 m/z); hatched bars flag ambiguous calls (low mean_margin).
                 Numeric annotation at the end of each bar reads "n/6".

Reads from feature_overlap_<experiment>.csv. Requires the per-method
membership columns (in_rf_top, in_svm_top, ...) and the per-group
attribution columns (top_condition_mean, mean_margin, ...).

Usage (from repo root):
    python -m src.scripts.corroboration_figure

Configure PIPELINE and the thresholds below.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, _ROOT)  # config.py lives here
sys.path.insert(0, _SRC)   # src/ packages take priority
import config

# -- Configuration -------------------------------------------------------------
PIPELINE          = 'standard'   # 'standard' or 'r_comparable'
TOP_N_FEATURES    = 30           # how many m/z to list in the bottom panel
AMBIG_MARGIN      = 1.5          # mean_margin below this -> hatched bar
MAX_INTERSECTIONS = 20           # cap on UpSet rows
# ------------------------------------------------------------------------------

BASE_DIR   = _ROOT
OUTPUT_DIR = os.path.join(BASE_DIR, 'results', PIPELINE)

METHOD_NAMES    = ['rf', 'svm', 'gb', 'lr', 'ridge', 'vip']
METHOD_LABELS   = ['RF', 'SVM', 'GB', 'LR', 'Ridge', 'VIP']
METHOD_COLUMNS  = [f'in_{m}_top' for m in METHOD_NAMES]
N_METHODS_TOTAL = len(METHOD_NAMES)


def _load_overlap():
    experiment = config.EXPERIMENT.strip()
    safe_name  = experiment.replace(' ', '_').replace(':', '')
    path = os.path.join(OUTPUT_DIR, f'feature_overlap_{safe_name}.csv')
    if not os.path.exists(path):
        return None, experiment, safe_name
    return pd.read_csv(path), experiment, safe_name


# -- UpSet panel ---------------------------------------------------------------

def _intersection_signature(row):
    return tuple(METHOD_NAMES[i] for i, c in enumerate(METHOD_COLUMNS) if row[c])


def _draw_upset(fig, gs, df, color='#2c3e50'):
    sigs   = df.apply(_intersection_signature, axis=1)
    counts = sigs.value_counts().head(MAX_INTERSECTIONS)
    intersections = list(counts.index)
    sizes    = counts.values
    n_inter  = len(intersections)

    inner = gs.subgridspec(
        nrows=2, ncols=1,
        height_ratios=[2.0, 1.6],
        hspace=0.05,
    )

    ax_bar = fig.add_subplot(inner[0, 0])
    x = np.arange(n_inter)
    ax_bar.bar(x, sizes, color=color, width=0.7)
    for xi, s in zip(x, sizes):
        ax_bar.text(xi, s, str(s), ha='center', va='bottom', fontsize=8)
    ax_bar.set_ylabel('Features in\nintersection', fontsize=9)
    ax_bar.set_xticks([])
    ax_bar.set_xlim(-0.5, n_inter - 0.5)
    ax_bar.set_ylim(0, max(sizes) * 1.18)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.set_title(
        'Method intersections -- corroborated features (n_methods >= 2)',
        fontsize=11, loc='left'
    )

    ax_dot = fig.add_subplot(inner[1, 0], sharex=ax_bar)
    for col_i, sig in enumerate(intersections):
        active_y = []
        for row_i, m in enumerate(METHOD_NAMES):
            in_set = m in sig
            y_pos  = N_METHODS_TOTAL - 1 - row_i
            ax_dot.scatter(col_i, y_pos, s=70,
                           color=color if in_set else '#dddddd',
                           zorder=2)
            if in_set:
                active_y.append(y_pos)
        if len(active_y) >= 2:
            ax_dot.plot([col_i] * len(active_y), active_y,
                        color=color, linewidth=1.5, zorder=1)

    ax_dot.set_xlim(-0.5, n_inter - 0.5)
    ax_dot.set_ylim(-0.5, N_METHODS_TOTAL - 0.5)
    ax_dot.set_yticks(range(N_METHODS_TOTAL))
    ax_dot.set_yticklabels(METHOD_LABELS[::-1], fontsize=9)
    ax_dot.set_xticks([])
    for spine in ('top', 'right', 'bottom'):
        ax_dot.spines[spine].set_visible(False)
    ax_dot.spines['left'].set_visible(False)
    ax_dot.tick_params(left=False)


# -- Ranked m/z bar chart ------------------------------------------------------

def _draw_ranked_bars(fig, gs, df, group_palette):
    sort_cols = ['n_methods']
    if 'vip_score' in df.columns:
        sort_cols.append('vip_score')
    top = df.sort_values(sort_cols, ascending=False).head(TOP_N_FEATURES).copy()
    top = top.iloc[::-1].reset_index(drop=True)

    ax = fig.add_subplot(gs)
    y  = np.arange(len(top))

    for i, row in top.iterrows():
        n      = int(row['n_methods'])
        group  = row['top_condition_mean']
        margin = row['mean_margin']
        ambig  = pd.isna(margin) or margin < AMBIG_MARGIN

        color = group_palette.get(group, '#888888')
        hatch = '///' if ambig else None
        edge  = '#444444' if ambig else color

        ax.barh(i, n, color=color, edgecolor=edge,
                hatch=hatch, alpha=0.85 if not ambig else 0.6,
                height=0.72)
        ax.text(n + 0.08, i, f'{n}/{N_METHODS_TOTAL}',
                va='center', fontsize=8.5, color='#333333')

    ax.set_yticks(y)
    ax.set_yticklabels([f"{row['mz']:.3f}" for _, row in top.iterrows()],
                       fontsize=8.5, family='monospace')
    ax.set_xlim(0, N_METHODS_TOTAL + 0.7)
    ax.set_xlabel('Methods agreeing  (n_methods)', fontsize=10)
    ax.set_ylabel('m/z', fontsize=10)
    ax.set_xticks(range(0, N_METHODS_TOTAL + 1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='x', alpha=0.25, linestyle=':')
    ax.set_title(f'Top {len(top)} m/z by ensemble corroboration',
                 fontsize=11, loc='left')

    handles = [mpatches.Patch(facecolor=color, edgecolor=color, label=g)
               for g, color in group_palette.items()]
    handles.append(mpatches.Patch(facecolor='white', edgecolor='#444444',
                                  hatch='///',
                                  label=f'Ambiguous (margin < {AMBIG_MARGIN}x)'))
    ax.legend(handles=handles, loc='lower right', fontsize=8,
              framealpha=0.9, title='Top condition', title_fontsize=8.5)


# -- Main ----------------------------------------------------------------------

def main():
    if not os.path.isdir(OUTPUT_DIR):
        print(f"Output folder not found: {OUTPUT_DIR}")
        return

    df, experiment, safe_name = _load_overlap()
    if df is None:
        print(f"feature_overlap CSV not found in {OUTPUT_DIR}")
        print("Run run_analysis.py first.")
        return

    needed  = METHOD_COLUMNS + ['mz', 'n_methods', 'mean_margin', 'top_condition_mean']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"CSV missing required columns: {missing}")
        print("Re-run run_analysis.py to regenerate the CSV with the new schema.")
        return

    print(f"Loaded {len(df)} ensemble features from feature_overlap_{safe_name}.csv")

    import seaborn as sns
    groups      = sorted(df['top_condition_mean'].dropna().unique().tolist())
    palette_rgb = sns.color_palette('colorblind', n_colors=len(groups))
    group_palette = {g: palette_rgb[i] for i, g in enumerate(groups)}

    fig = plt.figure(figsize=(12, 12))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1, 1.4], hspace=0.18)

    _draw_upset(fig, gs[0], df)
    _draw_ranked_bars(fig, gs[1], df, group_palette)

    fig.suptitle(f'Ensemble corroboration -- {experiment}',
                 fontsize=13, y=0.998)

    out_path = os.path.join(OUTPUT_DIR, f'corroboration_{safe_name}.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved -> {out_path}")


if __name__ == '__main__':
    main()
