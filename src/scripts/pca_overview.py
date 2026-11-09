"""Multi-experiment PCA overview from standard-run outputs (manuscript Figure 2).

    python src/scripts/pca_overview.py experiment_a experiment_b experiment_c experiment_d

Reads results/standard/<exp>/pca_scores_<exp>.csv and pca_explained_variance_<exp>.csv
and writes results/standard/pca_overview.png. Spectra are small points, colony means
large points; labels are used only to colour the (unsupervised) scores.
"""
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'results', 'standard')


def main(exps):
    fig, axes = plt.subplots(1, len(exps), figsize=(3.2 * len(exps), 3.2), squeeze=False)
    for ax, e in zip(axes[0], exps):
        sc = pd.read_csv(os.path.join(OUT, e, f'pca_scores_{e}.csv'))
        ev = pd.read_csv(os.path.join(OUT, e, f'pca_explained_variance_{e}.csv'))
        sc['condition'] = sc.condition.str.replace(r'^\S+ ', '', regex=True).replace({'Amb': 'Amber'})
        conds = sorted(sc.condition.unique())
        cmap = plt.get_cmap('tab10')
        for i, c in enumerate(conds):
            s = sc[sc.condition == c]
            ax.scatter(s.PC1, s.PC2, s=10, color=cmap(i), alpha=0.4, edgecolor='none')
            m = s.groupby('colony')[['PC1', 'PC2']].mean()
            ax.scatter(m.PC1, m.PC2, s=36, color=cmap(i), edgecolor='k', lw=0.5, label=c)
        r1, r2 = ev.iloc[0], ev.iloc[1]
        ax.set_xlabel(f'PC1 ({r1.explained_variance_ratio * 100:.1f}%; cond.-plate {r1.frac_condition_plate:.2f})', fontsize=8)
        ax.set_ylabel(f'PC2 ({r2.explained_variance_ratio * 100:.1f}%; cond.-plate {r2.frac_condition_plate:.2f})', fontsize=8)
        ax.set_title(e, fontsize=10, loc='left'); ax.tick_params(labelsize=7)
    axes[0][-1].legend(frameon=False, fontsize=7, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'pca_overview.png'), dpi=300)
    print('Saved ->', os.path.join(OUT, 'pca_overview.png'))


if __name__ == '__main__':
    main(sys.argv[1:])
