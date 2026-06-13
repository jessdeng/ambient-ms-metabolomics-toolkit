import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import seaborn as sns


# ---------------------------------------------------------------------------
# Provenance inspection helper
# ---------------------------------------------------------------------------

def _inspect_array(name, arr, context=""):
    """
    Print a diagnostic summary of an array immediately before it is plotted.
    """
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
    if len(unique_small) > 0:
        mode_candidate  = float(np.median(unique_small))
        n_at_mode       = int(np.sum(np.abs(flat - mode_candidate) < 1e-12))
        halfmin_warning = (n_at_mode / max(n_total, 1)) > 0.05
    else:
        halfmin_warning = False

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
        print(f"  half-min imputed? YES — many values near {float(np.median(unique_small)):.4g}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main spectrum plot — dual-panel: linear (global) + log10 (feature detail)
# ---------------------------------------------------------------------------

def plot_spectrum_with_features(
    X_binned, mz, y_labels, overlap_df, experiment_name, out_path,
    min_n_methods=1,
):
    """
    Dual-panel spectrum plot per condition group.

    TOP PANEL  (linear y-axis)
        Mean raw binned counts ± 1 SD across samples.
        A shaded 'zoom box' shows which m/z region is expanded below.
        Connection lines link the zoom box to the bottom panel.

    BOTTOM PANEL  (log₁₀ y-axis)
        Same raw data, same m/z region as the zoom box.
        Discriminating features (from overlap_df) are highlighted with
        coloured vertical lines and peak dots — now clearly visible
        above the noise floor regardless of absolute intensity.

    Parameters
    ----------
    X_binned     : ndarray (n_samples, n_features) — raw binned counts.
    mz           : ndarray (n_features,)            — bin centres.
    y_labels     : ndarray (n_samples,)             — group labels.
    overlap_df   : DataFrame with 'mz' and 'n_methods' columns.
    experiment_name : str
    out_path     : str — output PNG path.
    min_n_methods : int — minimum ensemble methods for high-confidence tier.
    """

    # ── Provenance inspection ────────────────────────────────────────────────
    _inspect_array("X_binned (as received)", X_binned,
                   context="raw_data — should NOT be log-transformed or scaled")

    # ── Feature streams ──────────────────────────────────────────────────────
    if 'n_methods' in overlap_df.columns:
        high_conf_df = overlap_df[overlap_df['n_methods'] >= min_n_methods]
    else:
        high_conf_df = overlap_df

    feat_mz_all  = overlap_df['mz'].values        # all ensemble candidates
    feat_mz_high = high_conf_df['mz'].values       # high-confidence subset

    # Verify mz alignment
    gap_all = np.array([np.abs(mz - m).min() for m in feat_mz_high])
    if gap_all.size > 0 and gap_all.max() > 0.001:
        print(f"[PROVENANCE] WARNING: max mz gap = {gap_all.max():.4f} Da — "
              f"feature mz may not match raw mz axis exactly")
    else:
        print(f"[PROVENANCE] mz alignment OK — max gap = "
              f"{gap_all.max() if gap_all.size else 0:.6f} Da")

    # ── Layout ───────────────────────────────────────────────────────────────
    groups   = sorted(np.unique(y_labels))
    n_groups = len(groups)
    palette  = sns.color_palette('colorblind', n_colors=n_groups)

    # 2 panels per group: linear (tall) + log (shorter)
    height_ratios = [2.0, 1.2] * n_groups
    fig = plt.figure(figsize=(16, 3.8 * n_groups))
    outer_gs = gridspec.GridSpec(
        n_groups * 2, 1,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.08,
    )

    # Collect (ax_lin, ax_log) pairs so we can add ConnectionPatch after
    # all axes are drawn (patch needs finalized axes positions).
    panel_axes = []

    # Pre-compute global log-floor: use 1% of the min positive value in the
    # whole matrix so zeros don't distort the log axis.
    pos_vals = X_binned[X_binned > 0]
    log_floor = float(np.percentile(pos_vals, 1)) * 0.5 if len(pos_vals) else 1.0

    # Compute zoom region from feature positions (pad 40 Da each side)
    if len(feat_mz_high) > 0:
        zoom_l = max(feat_mz_high.min() - 40, mz.min())
        zoom_r = min(feat_mz_high.max() + 40, mz.max())
    else:
        zoom_l, zoom_r = mz.min(), mz.max()

    for i, group in enumerate(groups):
        ax_lin = fig.add_subplot(outer_gs[i * 2])
        ax_log = fig.add_subplot(outer_gs[i * 2 + 1])
        panel_axes.append((ax_lin, ax_log, palette[i]))

        mask     = y_labels == group
        raw_data = X_binned[mask]
        avg      = raw_data.mean(axis=0)
        std      = raw_data.std(axis=0)

        # ── TOP PANEL: linear ────────────────────────────────────────────────
        ax_lin.fill_between(mz,
                            np.maximum(avg - std, 0),
                            avg + std,
                            color='lightgrey', alpha=0.35, label='±1 SD')
        ax_lin.plot(mz, avg,
                    color='#999999', lw=0.6,
                    label='Mean spectrum (raw binned counts)')

        # Light ticks for all ensemble features
        for mz_val in feat_mz_all:
            ax_lin.axvline(x=mz_val, color=palette[i], alpha=0.18, lw=0.5, zorder=2)
        # Bolder ticks for high-confidence features
        for mz_val in feat_mz_high:
            ax_lin.axvline(x=mz_val, color=palette[i], alpha=0.55, lw=0.9, zorder=3)

        # Y limits before drawing the zoom box
        y_top = float((avg + std).max()) * 1.08
        y_top = max(y_top, 1.0)
        ax_lin.set_ylim(0, y_top)
        ax_lin.set_xlim(mz.min(), mz.max())

        # Zoom box — shaded rectangle spanning the feature m/z range
        rect = FancyBboxPatch(
            (zoom_l, 0), zoom_r - zoom_l, y_top,
            boxstyle="square,pad=0",
            linewidth=1.4, edgecolor=palette[i],
            facecolor=palette[i], alpha=0.06, zorder=1,
        )
        ax_lin.add_patch(rect)
        ax_lin.annotate(
            'region expanded\nbelow ↓',
            xy=((zoom_l + zoom_r) / 2, y_top),
            xytext=(0, -4), textcoords='offset points',
            ha='center', va='top', fontsize=6.5, color=palette[i],
            fontweight='bold',
        )

        ax_lin.set_ylabel('Intensity\n(raw counts)', fontsize=8)
        ax_lin.set_title(f'{group}', fontsize=10, fontweight='bold', pad=3)
        ax_lin.tick_params(axis='x', labelbottom=False, length=0)
        ax_lin.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else
                                              f'{x/1e3:.0f}k' if x >= 1e3 else f'{x:.0f}')
        )
        # Small label clarifying data state
        ax_lin.text(0.01, 0.97,
                    'raw_data: binned counts  (not normalized / log / scaled)',
                    transform=ax_lin.transAxes, fontsize=6, va='top', color='#666666',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6))

        # ── BOTTOM PANEL: log₁₀, zoomed to feature region ───────────────────
        # Clip to log_floor so zeros map to a defined baseline
        avg_clipped = np.where(avg > log_floor, avg, log_floor)

        ax_log.fill_between(
            mz,
            np.where(np.maximum(avg - std, log_floor) > log_floor,
                     np.maximum(avg - std, log_floor), log_floor),
            np.where(avg + std > log_floor, avg + std, log_floor),
            color='lightgrey', alpha=0.30,
        )
        ax_log.plot(mz, avg_clipped, color='#777777', lw=0.7)
        ax_log.set_yscale('log')
        ax_log.set_xlim(zoom_l, zoom_r)

        # Feature markers on log panel — now clearly visible above noise
        for mz_val in feat_mz_all:
            ax_log.axvline(x=mz_val, color=palette[i], alpha=0.25, lw=0.6, zorder=2)
        for mz_val in feat_mz_high:
            ax_log.axvline(x=mz_val, color=palette[i], alpha=0.80, lw=1.1, zorder=3)
            nearest_idx = int(np.argmin(np.abs(mz - mz_val)))
            peak_y = max(float(avg_clipped[nearest_idx]), log_floor)
            ax_log.plot(mz[nearest_idx], peak_y,
                        'o', ms=4.5, color=palette[i], zorder=5, alpha=0.9)

        ax_log.set_ylabel('Log₁₀ Intensity\n(raw counts)', fontsize=8)
        ax_log.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f'{np.log10(x):.1f}' if x > 0 else '')
        )
        ax_log.set_xlabel('m/z' if i == n_groups - 1 else '', fontsize=9)
        if i < n_groups - 1:
            ax_log.tick_params(axis='x', labelbottom=False)

    # ── Connection patches: zoom-box corners → log-panel top corners ─────────
    # Must be added AFTER all axes are fully configured.
    # Use figure-level coordinates via ConnectionPatch with axesA / axesB.
    for (ax_lin, ax_log, color) in panel_axes:
        for x_data in [zoom_l, zoom_r]:
            con = ConnectionPatch(
                xyA=(x_data, 0),          coordsA='data',          axesA=ax_lin,
                xyB=(x_data, ax_log.get_ylim()[1] if ax_log.get_ylim()[1] > 0
                     else 1),              coordsB='data',          axesB=ax_log,
                color=color, lw=0.9, alpha=0.35, linestyle='--', zorder=0,
            )
            fig.add_artist(con)

    # ── Figure-level titles and caption ─────────────────────────────────────
    fig.suptitle(
        f'Group Spectra with Important Features — {experiment_name}\n'
        'Top: linear scale (global context, all counts)  │  '
        'Bottom: log₁₀ scale (feature detail, same m/z window)',
        fontsize=11, fontweight='bold', y=1.01,
    )
    fig.text(
        0.5, -0.005,
        'Bottom panel: Log-scaled detail of discriminating features to visualize '
        'markers with high dynamic range.\n'
        'Coloured vertical lines = ensemble features (processed-space selection); '
        'grey trace = raw binned ion counts.',
        ha='center', va='top', fontsize=8, style='italic', color='#444444',
        wrap=True,
    )

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {out_path}")
