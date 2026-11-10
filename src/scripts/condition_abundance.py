"""
condition_abundance.py — where each candidate sits, condition by condition and
colony by colony.

Left: condition means (row-scaled). Right: every colony's mean (row-scaled), so
colony-to-colony agreement is visible directly. Rows marked * are consistent
across colonies: one condition's colonies all sit above (or all below) every
colony of the other conditions.

The main analysis draws this figure for every run. To redraw it on its own:

    python src/scripts/condition_abundance.py [experiment]
"""
import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.dirname(_SRC), _SRC]

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from shared.plot_style import apply_style, pub_savefig
from scripts._common import experiment_and_folder

apply_style()


def _row_scale(a):
    lo, hi = a.min(1, keepdims=True), a.max(1, keepdims=True)
    return (a - lo) / np.where(hi - lo == 0, 1, hi - lo)


def figure(experiment, safe, folder):
    """Draw the figure from results/<experiment>/candidates_<experiment>.csv."""
    df = pd.read_csv(os.path.join(folder, f'candidates_{safe}.csv'))
    if df.empty:
        print("  No candidates for this experiment: no condition-abundance figure.")
        return None
    df = df.sort_values('mean_voter_rank')
    mean_cols = [c for c in df.columns if c.startswith('mean_log10_')]
    col_cols = [c for c in df.columns if c.startswith('colony_log10_')]
    conds = [c.replace('mean_log10_', '') for c in mean_cols]
    colony_cond = [c.replace('colony_log10_', '').split('::')[0] for c in col_cols]
    order = sorted(range(len(col_cols)), key=lambda i: (conds.index(colony_cond[i]), col_cols[i]))
    col_cols = [col_cols[i] for i in order]
    colony_cond = [colony_cond[i] for i in order]

    labels = [f"m/z {r.mz:.1f}{f' * {r.colony_pattern}' if r.colonies_consistent else ''}" for r in df.itertuples()]
    n = len(df)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(max(12, len(col_cols) * 0.55 + 6), max(3, n * 0.35 + 2)),
                                 gridspec_kw={'width_ratios': [len(conds), len(col_cols)]}, dpi=300)
    a1.imshow(_row_scale(df[mean_cols].to_numpy()), cmap='RdBu_r', aspect='auto', vmin=0, vmax=1)
    a1.set_xticks(range(len(conds))); a1.set_xticklabels(conds, rotation=35, ha='right', fontsize=8)
    a1.set_yticks(range(n)); a1.set_yticklabels(labels, fontsize=8)
    a1.set_title('Condition means\n(row-scaled)', fontsize=10)

    a2.imshow(_row_scale(df[col_cols].to_numpy()), cmap='RdBu_r', aspect='auto', vmin=0, vmax=1)
    for x in np.flatnonzero(np.r_[False, np.array(colony_cond[1:]) != np.array(colony_cond[:-1])]):
        a2.axvline(x - 0.5, color='white', lw=2)
    a2.set_xticks(range(len(col_cols)))
    a2.set_xticklabels([c.replace('colony_log10_', '').split('::')[-1] for c in col_cols],
                       rotation=90, fontsize=7)
    a2.set_yticks([])
    a2.set_title('Colony means, grouped by condition\n(row-scaled)', fontsize=10)
    fig.suptitle(f'{experiment} | candidates (* = consistent across colonies)', fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(folder, f'condition_abundance_{safe}.png')
    pub_savefig(out_path)
    return out_path


def main():
    experiment, safe, folder = experiment_and_folder(config)
    if folder is not None:
        figure(experiment, safe, folder)


if __name__ == '__main__':
    main()
