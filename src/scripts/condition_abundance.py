"""
condition_abundance.py
======================
Plots a side-by-side heatmap of mean intensity and Ridge one-vs-rest
coefficients for the cross-family-consensus features (config.CONSENSUS_RULE:
agg_worst_family_rank <= CONSENSUS_MAX_WORST_FAMILY_RANK by default, or
agg_n_families_topk >= CONSENSUS_MIN_FAMILIES_TOPK).

As of the per-group attribution update, the underlying numbers (mean_<group>,
ridge_<group>) are written directly to feature_overlap_<experiment>.csv by
run_analysis.py. This script no longer recomputes them — it just reads the
CSV and produces the heatmap figure.

Usage (from the repository folder; run run_analysis.py first):
    python src/scripts/condition_abundance.py            # EXPERIMENT from config.json
    python src/scripts/condition_abundance.py my_experiment   # or name the experiment
The figure is saved next to the results, e.g. results/standard/my_experiment/.

Configure PIPELINE below; the consensus rule is set in config.py.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add repo root to path
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, _ROOT)  # config.py lives here
sys.path.insert(0, _SRC)   # src/ packages take priority
import config

from shared.plot_style import apply_style, pub_savefig   # after the path setup
from scripts._common import experiment_and_folder, report_missing
from shared.rank_aggregation import config_consensus_mask, config_consensus_label

apply_style()

# ── Configuration ─────────────────────────────────────────────────────────────
PIPELINE      = 'standard'    # 'standard' or 'r_comparable'
CONSENSUS_TEXT = config_consensus_label(config)    # rule set in config.py
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR   = _ROOT
OUTPUT_DIR = os.path.join(BASE_DIR, 'results', PIPELINE)


def main():
    experiment, safe_name, folder = experiment_and_folder(OUTPUT_DIR, config)
    if folder is None:
        report_missing(OUTPUT_DIR, experiment, safe_name, 'condition_abundance')
        return
    overlap_path = os.path.join(folder, f'feature_overlap_{safe_name}.csv')

    df = pd.read_csv(overlap_path)

    mean_cols  = [c for c in df.columns
                  if c.startswith('mean_') and c != 'mean_margin']
    ridge_cols_all = [c for c in df.columns
                      if c.startswith('ridge_') and c != 'ridge_importance']

    if not mean_cols or not ridge_cols_all:
        print("CSV does not contain per-group attribution columns.")
        print("Re-run run_analysis.py to regenerate the CSV with the new schema.")
        return

    # Strip prefixes to get group names — order matches both heatmaps
    groups = [c.replace('mean_', '') for c in mean_cols]
    expected_ridge_cols = [f'ridge_{g}' for g in groups]
    missing = [c for c in expected_ridge_cols if c not in df.columns]
    if missing:
        print(f"Missing expected Ridge columns: {missing}")
        return
    ridge_cols = expected_ridge_cols

    try:
        mask = config_consensus_mask(df, config)
    except KeyError as e:
        print(e.args[0])
        return
    candidates = df[mask].copy()
    if candidates.empty:
        print(f"No features meet the rule: {CONSENSUS_TEXT}.")
        print(f"  best agg_worst_family_rank in the table = {df['agg_worst_family_rank'].min():.0f}; "
              f"{int((df['agg_n_families_topk'] >= 2).sum())} feature(s) are top-ranked in >= 2 families. "
              "Relax CONSENSUS_* in config.py to inspect more.")
        return

    print(f"\n{CONSENSUS_TEXT}:")
    summary_cols = ['mz', 'agg_worst_family_rank', 'agg_n_families_topk', 'n_methods',
                    'vip_score', 'top_condition_mean', 'top_condition_ridge',
                    'ridge_direction']
    summary_cols = [c for c in summary_cols if c in candidates.columns]
    print(candidates[summary_cols].to_string(index=False))

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_feat = len(candidates)
    fig, axes = plt.subplots(
        1, 2,
        figsize=(max(12, len(groups) * 2.5), max(3, n_feat * 0.8 + 2)),
        dpi=300,
    )

    ylabels = [f"m/z {row['mz']:.2f}  (families in top-{config.TOP_N_FEATURES}={int(row['agg_n_families_topk'])}, "
               f"VIP={row['vip_score']:.2f})"
               for _, row in candidates.iterrows()]

    # Left: row-normalised mean abundance
    abundance      = candidates[mean_cols].values
    row_min        = abundance.min(axis=1, keepdims=True)
    row_max        = abundance.max(axis=1, keepdims=True)
    row_rng        = np.where(row_max - row_min == 0, 1, row_max - row_min)
    abundance_norm = (abundance - row_min) / row_rng

    ax1 = axes[0]
    ax1.imshow(abundance_norm, cmap='RdBu_r', aspect='auto', vmin=0, vmax=1)
    for i in range(n_feat + 1):
        ax1.axhline(i - 0.5, color='white', linewidth=1)
    for j in range(len(groups) + 1):
        ax1.axvline(j - 0.5, color='white', linewidth=1)
    ax1.set_xticks(range(len(groups)))
    ax1.set_xticklabels(groups, rotation=35, ha='right', fontsize=9)
    ax1.set_yticks(range(n_feat))
    ax1.set_yticklabels(ylabels, fontsize=9)
    ax1.set_title('Mean Abundance per Condition\n(row-scaled)', fontsize=10)
    sm1 = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(0, 1))
    sm1.set_array([])
    cb1 = fig.colorbar(sm1, ax=ax1, shrink=0.4, aspect=12, pad=0.02)
    cb1.set_ticks([0, 1])
    cb1.set_ticklabels(['Low', 'High'])

    # Right: Ridge coefficients (signed, symmetric scale)
    ridge_vals = candidates[ridge_cols].values
    vmax       = np.abs(ridge_vals).max()

    ax2 = axes[1]
    ax2.imshow(ridge_vals, cmap='coolwarm', aspect='auto', vmin=-vmax, vmax=vmax)
    for i in range(n_feat + 1):
        ax2.axhline(i - 0.5, color='white', linewidth=1)
    for j in range(len(groups) + 1):
        ax2.axvline(j - 0.5, color='white', linewidth=1)
    ax2.set_xticks(range(len(groups)))
    ax2.set_xticklabels(groups, rotation=35, ha='right', fontsize=9)
    ax2.set_yticks(range(n_feat))
    ax2.set_yticklabels([], fontsize=9)
    ax2.set_title('Ridge Coefficient per Condition\n'
                  '(positive = associated with that group)', fontsize=10)
    sm2 = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(-vmax, vmax))
    sm2.set_array([])
    cb2 = fig.colorbar(sm2, ax=ax2, shrink=0.4, aspect=12, pad=0.02)
    cb2.set_ticks([-vmax, 0, vmax])
    cb2.set_ticklabels(['Suppressed', '0', 'Elevated'])

    fig.suptitle(
        f'{experiment}  |  {CONSENSUS_TEXT}',
        fontsize=11, y=1.01
    )

    plt.tight_layout()
    plot_path = os.path.join(folder, f'condition_abundance_{safe_name}.png')
    pub_savefig(plot_path)


if __name__ == '__main__':
    main()
