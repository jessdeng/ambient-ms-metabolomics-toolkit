"""
vip_vs_nmethods.py
==================
Plots the ensemble vote count (n_methods) vs component-1 PLS-DA VIP score for all
feature overlap candidates across experiments.

The vote count is descriptive only (Ridge, LR and SVM are closely related, so
votes are not independent). Features meeting the family-aware "Cross-family
consensus" rule (config.CONSENSUS_RULE) are ringed and labelled with their m/z;
those below the conventional VIP > 1.0 threshold are the ones the consensus
catches that component-1 PLS-DA alone would miss.

Usage (from repo root):
    python -m src.scripts.vip_vs_nmethods

Reads feature_overlap_*.csv files from your output folder.
Configure OUTPUT_DIR below to match which pipeline you ran.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns



# Add repo root to path
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, _ROOT)  # config.py lives here
sys.path.insert(0, _SRC)   # src/ packages take priority
from shared.plot_style import apply_style, pub_savefig   # after the path setup
apply_style()
import config
from shared.rank_aggregation import (config_consensus_mask, config_consensus_label,
                                     CONSENSUS_NAME)

# ── Configuration ─────────────────────────────────────────────────────────────
# Change to 'r_comparable' if you ran the R-comparable pipeline
PIPELINE    = 'standard'
OUTPUT_DIR  = os.path.join(_ROOT, 'results', PIPELINE)
# ──────────────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all feature overlap CSVs in the output folder
    # v2: outputs live in results/<pipeline>/<experiment>/; the flat layout is still read.
    csv_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, '*', 'feature_overlap_*.csv'))
                       + glob.glob(os.path.join(OUTPUT_DIR, 'feature_overlap_*.csv')))
    wanted = [a for a in sys.argv[1:] if not a.startswith('-')]
    if wanted:   # optional: experiment names to plot, in this order
        by_name = {os.path.basename(f)[len('feature_overlap_'):-4]: f for f in csv_files}
        missing = [w for w in wanted if w not in by_name]
        if missing:
            raise SystemExit(f"No feature_overlap csv for {missing} under {OUTPUT_DIR}")
        csv_files = [by_name[w] for w in wanted]

    if not csv_files:
        print(f"No feature_overlap_*.csv files found in {OUTPUT_DIR}")
        print("Run run_analysis.py first to generate them.")
        return

    # Build experiment name from filename
    experiments = {}
    for f in csv_files:
        name = os.path.basename(f).replace('feature_overlap_', '').replace('.csv', '').replace('_', ' ')
        experiments[name] = f

    n = len(experiments)
    ncols = 2
    nrows = (n + 1) // 2

    palette = sns.color_palette('colorblind', n_colors=n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.5 * nrows), dpi=300)
    axes = np.atleast_1d(axes).flatten()    # also valid for a single experiment (1 x 2 grid)

    for ax, (name, fpath), color in zip(axes, experiments.items(), palette):
        df = pd.read_csv(fpath)

        rng    = np.random.default_rng(42)
        jitter = rng.uniform(-0.12, 0.12, size=len(df))
        # n = 6 (all six rankings) was previously unmapped -> NaN size -> point dropped.
        # consensus-only candidates can have 0-1 votes: give them the smallest marker
        sizes  = df['n_methods'].map({1: 20, 2: 40, 3: 80, 4: 140, 5: 220, 6: 320}).fillna(10)

        ax.scatter(
            df['n_methods'] + jitter,
            df['vip_score'],
            s=sizes, color=color, alpha=0.6,
            edgecolors='white', linewidths=0.4, zorder=2
        )

        ax.axhline(1.0, color='#888888', linestyle='--',
                   linewidth=0.9, alpha=0.7, zorder=1)

        # Ring + label the cross-family-consensus features
        cons = config_consensus_mask(df, config)
        ax.scatter(
            (df['n_methods'] + jitter)[cons], df['vip_score'][cons],
            s=sizes[cons] + 60, facecolors='none', edgecolors='black',
            linewidths=1.2, zorder=3
        )
        for (_, row), j in zip(df[cons].iterrows(), jitter[cons.values]):
            ax.annotate(
                f"{row['mz']:.1f}",
                xy=(row['n_methods'] + j, row['vip_score']),
                xytext=(6, 3), textcoords='offset points',
                fontsize=7, color='#333333', fontweight='bold'
            )

        ax.set_title(name, fontsize=10, fontweight='bold', color='#1a1a1a')
        ax.set_xlabel('Vote count (not independent)', fontsize=8)
        ax.set_ylabel('Component-1 VIP', fontsize=8)
        ax.set_xticks(range(int(df['n_methods'].min()), 7))
        ax.tick_params(labelsize=8)
        ax.set_xlim(df['n_methods'].min() - 0.5, 6.5)

    # n counts (after ylim is set)
    for ax, (name, fpath) in zip(axes, experiments.items()):
        df   = pd.read_csv(fpath)
        for n_m in [0, 1, 2, 3, 4, 5, 6]:
            count = len(df[df['n_methods'] == n_m])
            if count > 0:
                ax.text(n_m, ax.get_ylim()[0] + 0.02, f'n={count}',
                        ha='center', fontsize=7, color='#666666', style='italic')

    # Hide unused axes
    for ax in axes[len(experiments):]:
        ax.set_visible(False)

    # Legend
    legend_handles = [
        plt.scatter([], [], s=40,  color='grey', alpha=0.6, label='n = 2 methods'),
        plt.scatter([], [], s=80,  color='grey', alpha=0.6, label='n = 3 methods'),
        plt.scatter([], [], s=140, color='grey', alpha=0.6, label='n = 4 methods'),
        plt.scatter([], [], s=220, color='grey', alpha=0.6, label='n = 5 methods'),
        plt.scatter([], [], s=320, color='grey', alpha=0.6, label='n = 6 methods'),
        plt.scatter([], [], s=140, facecolors='none', edgecolors='black', linewidths=1.2,
                    label=config_consensus_label(config)),
        plt.Line2D([0], [0], color='#888888', linestyle='--',
                   linewidth=0.9, label='VIP = 1.0 threshold'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(f'Vote count vs. component-1 PLS-DA VIP ({CONSENSUS_NAME} ringed)',
                 fontsize=12, fontweight='bold', y=1.01)

    plt.tight_layout(rect=[0, 0.12, 1, 1])      # leave room for the legend below the axes
    out_path = os.path.join(OUTPUT_DIR, 'vip_vs_nmethods.png')
    pub_savefig(out_path)


if __name__ == '__main__':
    main()
