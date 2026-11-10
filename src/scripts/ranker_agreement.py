"""
ranker_agreement.py — how far every pair of rankings agrees.

Left: top-N agreement (Jaccard) between every pair of rankers, with the value
expected by chance. Right: univariate-F rank against component-1 VIP rank for
every bin (log axes); candidates are coloured, filled if consistent across
colonies.

The main analysis draws this figure for every run. To redraw it on its own:

    python src/scripts/ranker_agreement.py [experiment]
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
NAMES = {'univariate': 'Univariate F', 'vip': 'VIP (comp. 1)', 'ridge': 'Ridge',
         'lr': 'Logistic regression', 'svm': 'Linear SVM', 'rf': 'Random forest',
         'gb': 'Gradient boosting'}


def figure(experiment, safe, folder):
    """Draw the figure from the tables in results/<experiment>/."""
    conc = pd.read_csv(os.path.join(folder, f'ranker_concordance_{safe}.csv'))
    ranks = pd.read_csv(os.path.join(folder, f'feature_rankings_{safe}.csv'))
    names = list(dict.fromkeys(list(conc.ranker_a) + list(conc.ranker_b)))
    J = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for r in conc.itertuples():
        J.loc[r.ranker_a, r.ranker_b] = J.loc[r.ranker_b, r.ranker_a] = r.jaccard

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=300,
                                 gridspec_kw={'width_ratios': [1, 1.25]})
    im = a1.imshow(J.to_numpy(), cmap='Greys', vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j:
                a1.text(j, i, f'{J.iat[i, j]:.2f}', ha='center', va='center',
                        fontsize=9 if len(names) <= 4 else 7,
                        color='white' if J.iat[i, j] > 0.5 else 'black')
    a1.set_xticks(range(len(names))); a1.set_yticks(range(len(names)))
    a1.set_xticklabels([NAMES.get(n, n) for n in names], rotation=30, ha='right', fontsize=8)
    a1.set_yticklabels([NAMES.get(n, n) for n in names], fontsize=8)
    a1.set_title(f'Top-{config.TOP_N_FEATURES} agreement (Jaccard)\nchance = {conc.chance.iloc[0]:.3f}',
                 fontsize=10, loc='left')
    fig.colorbar(im, ax=a1, shrink=0.7)

    rest = ranks[~ranks.candidate]
    a2.scatter(rest.univariate_rank, rest.vip_rank, s=4, color='0.75', edgecolor='none',
               label='other bins')
    cand = ranks[ranks.candidate]
    for consistent, face in ((True, '#0072B2'), (False, 'white')):
        c = cand[cand.colonies_consistent == consistent]
        a2.scatter(c.univariate_rank, c.vip_rank, s=26, facecolor=face, edgecolor='#0072B2', lw=0.8,
                   label=f"candidate, {'consistent' if consistent else 'not consistent'} across colonies")
    n = config.TOP_N_FEATURES
    a2.axvline(n + 0.5, color='0.4', ls=':', lw=0.8); a2.axhline(n + 0.5, color='0.4', ls=':', lw=0.8)
    a2.set_xscale('log'); a2.set_yscale('log')
    a2.set_xlabel('Univariate F rank (colony means)'); a2.set_ylabel('Component-1 VIP rank')
    a2.legend(fontsize=7, frameon=False, loc='lower right')
    a2.set_title('Rank against rank (dotted: top-N cut-offs)', fontsize=10, loc='left')
    fig.suptitle(f'Ranker agreement -- {experiment}', fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(folder, f'ranker_agreement_{safe}.png')
    pub_savefig(out_path)
    return out_path


def main():
    experiment, safe, folder = experiment_and_folder(config, prefix='feature_rankings')
    if folder is not None:
        figure(experiment, safe, folder)


if __name__ == '__main__':
    main()
