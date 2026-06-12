"""
shared/visualization.py — Spectrum + Feature Overlay Plot
==========================================================
Plots the mean MS spectrum per group with ensemble high-confidence features
highlighted as vertical tick marks.

Usage:
    from shared.visualization import plot_spectrum_with_features
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def plot_spectrum_with_features(X_binned, mz_binned, y_labels, overlap_df,
                                experiment_name, out_path,
                                min_n_methods=2):
    """
    Plot mean m/z spectrum for each group (line) with vertical ticks marking
    ensemble feature candidates.

    Parameters
    ----------
    X_binned       : ndarray (n_samples, n_bins)  — binned intensity matrix
                     (before normalization/scaling, used for visual display)
    mz_binned      : ndarray (n_bins,)            — m/z value per bin
    y_labels       : ndarray of str               — group label per sample
    overlap_df     : DataFrame with columns 'mz' and 'n_methods'
    experiment_name: str                          — plot title
    out_path       : str                          — output file path
    min_n_methods  : int                          — minimum n_methods to mark
    """
    from sklearn.preprocessing import LabelEncoder

    le      = LabelEncoder().fit(y_labels)
    classes = le.classes_
    palette = sns.color_palette('colorblind', n_colors=len(classes))

    fig, ax = plt.subplots(figsize=(14, 5))

    # Per-group mean spectrum (log10 + 1 for display)
    with np.errstate(divide='ignore', invalid='ignore'):
        X_disp = np.log10(X_binned + 1)

    for i, cls in enumerate(classes):
        mask = y_labels == cls
        mean_spec = X_disp[mask].mean(axis=0)
        ax.plot(mz_binned, mean_spec, color=palette[i], linewidth=0.8,
                alpha=0.85, label=cls)

    # Feature ticks
    tick_colors = {2: '#AAAAAA', 3: '#888888', 4: '#FF8C00', 5: '#CC0000', 6: '#990099'}
    y_max = ax.get_ylim()[1]

    if overlap_df is not None and len(overlap_df) > 0:
        feat = overlap_df[overlap_df['n_methods'] >= min_n_methods]
        for _, row in feat.iterrows():
            n = int(row['n_methods'])
            col = tick_colors.get(n, '#333333')
            ax.axvline(row['mz'], color=col, linewidth=1.2, alpha=0.7, zorder=3)

        # Tick legend
        present_n = sorted(feat['n_methods'].unique())
        tick_handles = [
            plt.Line2D([0], [0], color=tick_colors.get(n, '#333333'),
                       linewidth=1.5, label=f'n={n} methods')
            for n in present_n
        ]
        legend2 = ax.legend(handles=tick_handles, loc='upper right',
                            fontsize=8, title='Feature confidence',
                            framealpha=0.9)
        ax.add_artist(legend2)

    ax.legend(loc='upper left', fontsize=8, framealpha=0.9, title='Group')
    ax.set_xlabel('m/z (Da)', fontsize=10)
    ax.set_ylabel('log₁₀(intensity + 1)', fontsize=10)
    ax.set_title(f'Mean MS Spectrum with Ensemble Features — {experiment_name}',
                 fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")
