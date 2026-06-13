import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import seaborn as sns

from src.shared.plot_style import apply_style, pub_savefig

apply_style()


# ---------------------------------------------------------------------------
# Provenance inspection helper
# ---------------------------------------------------------------------------

def _inspect_array(name, arr, context=""):
    """Print a diagnostic summary of an array immediately before it is plotted."""
    tag = f"[PROVENANCE] {name}"
    if context:
        tag += f" ({context})"

    flat = arr.ravel()
    n_total  = flat.size
    n_finite = int(np.isfinite(flat).sum())
    n_zeros  = int((flat == 0).sum())
    n_neg    = int((flat < 0).sum())
    val_min  = float(np.nanmin(flat))
    val_max  = float(np.nanmax(flat))
    val_mean = float(np.nanmean(flat))
    val_std  = float(np.nanstd(flat))

    looks_log_transformed = (val_min < 0) or (val_max < 50 and val_min >= -20)
    looks_mean_centred    = abs(val_mean) < 0.05 * (abs(val_max) + 1e-9)
    looks_unit_variance   = abs(val_std - 1.0) < 0.15

    unique_small = flat[(flat > 0) & (flat < 1.0)]
    halfmin_warning = False
    if len(unique_small) > 0:
        mode_candidate  = float(np.median(unique_small))
        n_at_mode       = int(np.sum(np.abs(flat - mode_candidate) < 1e-12))
        halfmin_warning = (n_at_mode / max(n_total, 1)) > 0.05

    print(f"\n{'='*60}")
    print(f"{tag}")
    print(f"  shape         : {arr.shape}  (dtype={arr.dtype})")
    print(f"  finite values : {n_finite}/{n_total}")
    print(f"  zeros         : {n_zeros} ({100*n_zeros/max(n_total,1):.1f}%)")
    print(f"  negative vals : {n_neg} ({100*n_neg/max(n_total,1):.1f}%)")
    print(f"  min / max     : {val_min:.6g} / {val_max:.6g}")
    print(f"  mean / std    : {val_mean:.6g} / {val_std:.6g}")
    print(f"  log-transformed?  {'YES' if looks_log_transformed else 'no'}")
    print(f"  mean-centred?     {'YES' if looks_mean_centred else 'no'}")
    print(f"  unit-variance?    {'YES' if looks_unit_variance else 'no'}")
    if halfmin_warning:
        print(f"  half-min imputed? YES — many values near {mode_candidate:.4g}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Publication-quality dual-panel spectrum plot
# ---------------------------------------------------------------------------

def plot_spectrum_with_features(
    X_binned, mz, y_labels, overlap_df, experiment_name, out_path,
    min_n_methods=1,
):
    """
    Publication-quality dual-panel mass spectrum plot.

    For each condition group (alphabetical order):

      TOP PANEL  — linear y-axis, full m/z range
        Grey mean spectrum ± 1 SD fill (group colour).
        Feature positions shown as short rug ticks at the top of the panel:
          · all ensemble features → thin, 50 % opacity
          · high-confidence tier  → bold, 90 % opacity

      BOTTOM PANEL  — log₁₀ y-axis, same m/z range
        Same raw data.  Feature positions marked as dots ON the spectrum
        trace so their absolute intensity context is preserved:
          · all ensemble features → '|' tick at bin intensity, 40 % opacity
          · high-confidence tier  → filled circle, 90 % opacity, white edge

    Design decisions
    ----------------
    · No full-height vertical lines — they coalesce into an opaque bar when
      50+ features are present.
    · Rug ticks (top of linear panel) and dot-on-trace (log panel) convey
      position without obscuring the spectrum.
    · Scientific notation y-axis (×10ⁿ offset label, standard MS style).
    · 300 DPI, Arial/DejaVu Sans, Nature/Science minimalist style.
    """

    # ── Provenance check ────────────────────────────────────────────────────
    _inspect_array("X_binned (as received)", X_binned,
                   context="raw_data — should NOT be log-transformed or scaled")

    # ── Feature tiers ────────────────────────────────────────────────────────
    if 'n_methods' in overlap_df.columns:
        feat_df   = overlap_df[overlap_df['n_methods'] >= min_n_methods].copy()
        # high-confidence = top quartile of n_methods (or explicit ≥ 4 if >3 possible)
        hi_thresh = max(int(np.percentile(feat_df['n_methods'], 75)), min_n_methods + 1)
        feat_high = feat_df[feat_df['n_methods'] >= hi_thresh]['mz'].values
        feat_all  = feat_df['mz'].values
    else:
        feat_all  = overlap_df['mz'].values
        feat_high = feat_all

    # mz alignment check
    gap = np.array([np.abs(mz - m).min() for m in feat_high]) if feat_high.size else np.array([0.0])
    print(f"[PROVENANCE] mz alignment — max gap = {gap.max():.6f} Da  "
          f"({'OK' if gap.max() < 0.001 else 'WARNING'})")

    # ── Global layout settings ───────────────────────────────────────────────
    sns.set_style('ticks')
    plt.rcParams.update({
        'font.family':      ['Arial', 'DejaVu Sans'],
        'font.size':        10,
        'axes.labelsize':   10,
        'axes.titlesize':   11,
        'xtick.labelsize':  9,
        'ytick.labelsize':  9,
        'axes.linewidth':   0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'xtick.direction':  'out',
        'ytick.direction':  'out',
    })

    groups   = sorted(np.unique(y_labels))          # alphabetical
    n_groups = len(groups)
    palette  = sns.color_palette('colorblind', n_colors=n_groups)

    # Log floor: half of the 1st-percentile positive value (zeros → defined baseline)
    pos_vals  = X_binned[X_binned > 0]
    log_floor = float(np.percentile(pos_vals, 1)) * 0.5 if len(pos_vals) else 1.0

    # ── Figure & GridSpec ────────────────────────────────────────────────────
    # Each group block = 2 stacked axes; blocks separated by generous hspace.
    # Within a block: linear panel is 1.8× the height of the log panel.
    fig = plt.figure(figsize=(14, 5.5 * n_groups), dpi=300)
    outer_gs = gridspec.GridSpec(
        n_groups, 1, figure=fig, hspace=0.55,
    )

    # Scientific notation formatter for linear y-axis
    def _sci_fmt():
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))   # always use ×10ⁿ notation
        return fmt

    # Log-axis formatter: show as 10ⁿ
    def _log_tick(x, _):
        if x <= 0:
            return ''
        exp = int(np.round(np.log10(x)))
        return f'$10^{{{exp}}}$'

    for i, group in enumerate(groups):
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer_gs[i],
            height_ratios=[1.8, 1.0], hspace=0.18,
        )
        ax_lin = fig.add_subplot(inner_gs[0])
        ax_log = fig.add_subplot(inner_gs[1])

        mask     = y_labels == group
        raw      = X_binned[mask]
        avg      = raw.mean(axis=0)
        std      = raw.std(axis=0)
        color    = palette[i]

        # ── TOP: linear ──────────────────────────────────────────────────────
        ax_lin.fill_between(
            mz,
            np.maximum(avg - std, 0), avg + std,
            color=color, alpha=0.15, linewidth=0, zorder=1,
        )
        ax_lin.plot(mz, avg, color='#222222', lw=0.9, zorder=2)

        # Y limits — set before drawing rug ticks
        y_top = float((avg + std).max()) * 1.12
        y_top = max(y_top, 1.0)
        ax_lin.set_ylim(0, y_top)
        ax_lin.set_xlim(mz.min(), mz.max())

        # Rug ticks at top of panel:  all features (thin) → high-conf (bold)
        rug_y0_thin  = y_top * 0.93
        rug_y0_bold  = y_top * 0.90
        for mz_val in feat_all:
            ax_lin.plot([mz_val, mz_val], [rug_y0_thin, y_top],
                        color=color, lw=0.5, alpha=0.45, solid_capstyle='butt', zorder=3)
        for mz_val in feat_high:
            ax_lin.plot([mz_val, mz_val], [rug_y0_bold, y_top],
                        color=color, lw=1.4, alpha=0.90, solid_capstyle='butt', zorder=4)

        ax_lin.set_ylabel('Intensity', fontsize=10, labelpad=4)
        ax_lin.set_title(group, fontsize=11, fontweight='bold', loc='left', pad=5)
        ax_lin.yaxis.set_major_formatter(_sci_fmt())
        ax_lin.yaxis.get_offset_text().set_fontsize(8)
        ax_lin.tick_params(axis='x', labelbottom=False, length=4)
        ax_lin.tick_params(axis='y', length=4)
        sns.despine(ax=ax_lin, trim=True)

        # ── BOTTOM: log₁₀ ───────────────────────────────────────────────────
        avg_clipped = np.where(avg > log_floor, avg, log_floor)
        lo_fill     = np.where(avg - std > log_floor, avg - std, log_floor)
        hi_fill     = np.where(avg + std > log_floor, avg + std, log_floor)

        ax_log.fill_between(mz, lo_fill, hi_fill,
                            color=color, alpha=0.12, linewidth=0, zorder=1)
        ax_log.plot(mz, avg_clipped, color='#222222', lw=0.9, zorder=2)
        ax_log.set_yscale('log')
        ax_log.set_xlim(mz.min(), mz.max())

        # Feature markers ON the log-scale trace
        # All features → '|' tick at their bin intensity
        for mz_val in feat_all:
            idx    = int(np.argmin(np.abs(mz - mz_val)))
            peak_y = max(float(avg_clipped[idx]), log_floor)
            ax_log.plot(mz[idx], peak_y, '|',
                        ms=7, mew=0.9, color=color, alpha=0.40, zorder=3)
        # High-confidence → filled circle with white edge
        for mz_val in feat_high:
            idx    = int(np.argmin(np.abs(mz - mz_val)))
            peak_y = max(float(avg_clipped[idx]), log_floor)
            ax_log.plot(mz[idx], peak_y, 'o',
                        ms=4.5, color=color, alpha=0.92, zorder=5,
                        markeredgecolor='white', markeredgewidth=0.6)

        ax_log.set_ylabel('Intensity (log₁₀)', fontsize=10, labelpad=4)
        ax_log.yaxis.set_major_formatter(ticker.FuncFormatter(_log_tick))
        ax_log.yaxis.set_minor_locator(ticker.NullLocator())
        ax_log.tick_params(axis='y', length=4, which='both')

        if i == n_groups - 1:
            ax_log.set_xlabel('m/z', fontsize=10, labelpad=6)
            ax_log.tick_params(axis='x', length=4)
        else:
            ax_log.tick_params(axis='x', labelbottom=False, length=4)

        sns.despine(ax=ax_log, trim=True)

    # ── Shared legend ────────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color='#222222', lw=0.9,
               label='Mean spectrum (raw counts)'),
        Line2D([0], [0], color='#444444', lw=0, marker='|', ms=9,
               markeredgewidth=1.0, alpha=0.45,
               label=f'Ensemble features (n ≥ {min_n_methods})'),
        Line2D([0], [0], color='#444444', lw=0, marker='o', ms=5,
               markeredgecolor='white', markeredgewidth=0.6,
               label=f'High-confidence features'),
        matplotlib.patches.Patch(
            facecolor='#aaaaaa', alpha=0.35, linewidth=0,
            label='±1 SD across samples'),
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower center', ncol=4,
        bbox_to_anchor=(0.5, -0.015),
        fontsize=9, frameon=True,
        framealpha=0.95, edgecolor='#cccccc',
    )

    # ── Figure title & caption ───────────────────────────────────────────────
    fig.suptitle(
        f'Group Spectra with Discriminating Features — {experiment_name}',
        fontsize=13, fontweight='bold', y=1.005,
    )
    fig.text(
        0.5, -0.035,
        'Bottom panel: log₁₀-scaled view of discriminating features to '
        'visualize markers with high dynamic range. '
        'Grey trace = mean raw binned counts; shaded band = ±1 SD.',
        ha='center', va='top', fontsize=8.5,
        style='italic', color='#555555',
    )

    pub_savefig(out_path)


# Make matplotlib.patches accessible for legend patch
import matplotlib.patches
