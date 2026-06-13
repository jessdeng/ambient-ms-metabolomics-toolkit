import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import seaborn as sns

from src.shared.plot_style import apply_style, pub_savefig

apply_style()

# Okabe-Ito / Wong colorblind-safe palette
_CB_PALETTE = [
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#009E73',  # teal-green
    '#CC79A7',  # pink-purple
    '#E69F00',  # orange
    '#56B4E9',  # sky-blue
    '#F0E442',  # yellow
]

_GRAY_STEM   = '#BBBBBB'   # non-featured peak colour
_STEM_TOP    = 105         # y-axis ceiling (% RA) — 5 % headroom above 100


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _feature_tiers(overlap_df, min_n_methods):
    """Return (feat_all, feat_high) as numpy arrays of m/z values."""
    if 'n_methods' in overlap_df.columns:
        feat_df   = overlap_df[overlap_df['n_methods'] >= min_n_methods].copy()
        hi_thresh = max(int(np.percentile(feat_df['n_methods'], 75)),
                        min_n_methods + 1)
        feat_high = feat_df[feat_df['n_methods'] >= hi_thresh]['mz'].values
        feat_all  = feat_df['mz'].values
    else:
        feat_all  = overlap_df['mz'].values
        feat_high = feat_all
    return feat_all, feat_high


def _hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def _normalise_means(X_binned, y_labels, groups):
    """
    Compute per-group mean intensity vectors and normalise to 0–100 %
    relative abundance (100 = the highest mean peak across all groups).
    """
    avgs      = {g: X_binned[y_labels == g].mean(axis=0) for g in groups}
    global_max = max(v.max() for v in avgs.values())
    if global_max == 0:
        global_max = 1.0
    return {g: avgs[g] / global_max * 100.0 for g in groups}


def _stem_indices(mz, feat_all, feat_high_set):
    """
    Map feature m/z values onto their nearest bin indices, then split into:
      · feat_hi_idx   (list[int]) — high-confidence feature bins
      · feat_ens_idx  (list[int]) — ensemble-only bins (not high-conf)
      · nonfeat_mask  (bool ndarray) — True for all remaining (gray) bins

    Pre-computed once per figure; the same index sets are reused across
    all panels because every group is plotted on the same m/z grid.
    Duplicate hits (two feature m/z values landing on the same bin) are
    deduplicated via sets before converting to sorted lists.
    """
    hi_set  = set()
    ens_set = set()
    all_set = set()

    for mz_val in feat_all:
        idx = int(np.argmin(np.abs(mz - mz_val)))
        all_set.add(idx)
        if round(float(mz_val), 6) in feat_high_set:
            hi_set.add(idx)
        else:
            ens_set.add(idx)

    feat_hi_idx  = sorted(hi_set)
    feat_ens_idx = sorted(ens_set)

    nonfeat_mask = np.ones(len(mz), dtype=bool)
    for idx in all_set:
        nonfeat_mask[idx] = False

    return feat_hi_idx, feat_ens_idx, nonfeat_mask


def _plotly_stems(mz_vals, y_vals):
    """
    Build concatenated x / y arrays for Plotly line-mode stem traces.
    Each stem is encoded as three points: (x, 0) → (x, y) → gap (None).
    """
    xs, ys = [], []
    for x, y in zip(mz_vals, y_vals):
        xs += [float(x), float(x), None]
        ys += [0.0,       float(y), None]
    return xs, ys


def _select_labels(mz_vals, ra_vals, x_max=1200, min_gap_da=15, max_labels=15):
    """
    Choose a non-overlapping subset of high-confidence peak indices to label.

    Strategy
    --------
    1. Sort by relative abundance descending — tallest peaks get first pick.
    2. Greedily accept each candidate unless its m/z is within `min_gap_da`
       of an already-accepted label (prevents horizontal text collisions for
       90°-rotated annotations at ~6.5 pt).
    3. Hard cap at `max_labels` to keep panels readable.

    Parameters
    ----------
    mz_vals    : 1-D array of m/z values for the high-conf feature subset
    ra_vals    : 1-D array of relative abundance (%) for the same subset
    x_max      : right x-axis limit — features beyond this are skipped
    min_gap_da : minimum horizontal spacing between adjacent labels (Da)
    max_labels : absolute cap on labels per panel

    Returns
    -------
    Sorted list of integer indices into mz_vals / ra_vals.
    """
    if len(mz_vals) == 0:
        return []

    order    = np.argsort(ra_vals)[::-1]   # descending RA
    placed   = []                           # accepted label m/z values
    selected = []

    for idx in order:
        if len(selected) >= max_labels:
            break
        mz_v = float(mz_vals[idx])
        if mz_v > x_max:
            continue
        if any(abs(mz_v - p) < min_gap_da for p in placed):
            continue
        placed.append(mz_v)
        selected.append(int(idx))

    return sorted(selected)   # return in ascending m/z order for rendering


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_spectrum_with_features(
    X_binned, mz, y_labels, overlap_df, experiment_name, out_path,
    min_n_methods=1, interactive=False,
):
    """
    Centroid-style mass spectrum — 5-panel small-multiples facet grid.

    Parameters
    ----------
    X_binned        : (n_samples, n_bins) array of binned intensities
    mz              : (n_bins,) array of m/z bin centres
    y_labels        : (n_samples,) array of group labels
    overlap_df      : DataFrame with columns ['mz', 'n_methods']
    experiment_name : string for the figure title
    out_path        : output file path (.png for static, .html for interactive)
    min_n_methods   : minimum n_methods threshold for ensemble features
    interactive     : False → static Matplotlib PNG; True → Plotly HTML

    Design
    ------
    · Intensities shown as vertical stems (centroid MS convention), NOT
      continuous line plots — eliminates baseline noise carpeting.
    · Y-axis: linear relative abundance (0–100 %), normalised to the single
      highest mean peak across all groups.  Y-axis minimum locked to 0.
    · 5-panel small-multiples grid, shared X and Y axes — panels are
      directly comparable without any manual rescaling.
    · In each panel, the focal group's non-featured bins appear as thin
      silver-gray stems.  Ensemble and high-confidence features are drawn
      in the group's palette colour with a ● marker at the stem tip.
    · No ±1 SD clouds (SD fill does not translate to a centroid representation).
    """
    feat_all, feat_high = _feature_tiers(overlap_df, min_n_methods)
    groups              = sorted(np.unique(y_labels))
    avgs_ra             = _normalise_means(X_binned, y_labels, groups)

    # Pre-compute stem index sets once — identical for every panel
    feat_high_set                       = set(np.round(feat_high, 6))
    feat_hi_idx, feat_ens_idx, nf_mask  = _stem_indices(mz, feat_all,
                                                         feat_high_set)

    if interactive:
        out_html = str(out_path).rsplit('.', 1)[0] + '.html'
        _plot_plotly(mz, groups, avgs_ra,
                     feat_hi_idx, feat_ens_idx, nf_mask,
                     experiment_name, out_html)
    else:
        _plot_matplotlib(mz, groups, avgs_ra,
                         feat_hi_idx, feat_ens_idx, nf_mask,
                         experiment_name, out_path)


# ---------------------------------------------------------------------------
# Matplotlib backend  (static, publication PNG/PDF)
# ---------------------------------------------------------------------------

def _plot_matplotlib(mz, groups, avgs_ra,
                     feat_hi_idx, feat_ens_idx, nf_mask,
                     experiment_name, out_path,
                     x_max=1200, label_min_gap=15, label_max=15):
    """
    Small-multiples centroid MS figure — one panel per group.

    Stem tiers per panel
    --------------------
    Gray  (#BBBBBB, lw=0.5)   — non-featured peaks (background signal)
    Color (lw=0.85, α=0.62)   — ensemble-only features (colored stem, no label)
    Color (lw=1.4,  α=0.95)   — high-confidence features (bold stem +
                                 rotated m/z text label above peak apex)

    X-axis
    ------
    Spans mz.min() → x_max (default 1200) with major ticks every 200 Da.

    Label collision avoidance
    -------------------------
    _select_labels() sorts high-conf features by RA descending and greedily
    accepts labels only if they are ≥ label_min_gap Da from any already-placed
    label.  At most label_max labels appear per panel.
    """
    apply_style()

    n       = len(groups)
    palette = _CB_PALETTE[:n]

    fig, axes = plt.subplots(
        n, 1,
        figsize=(13, 2.8 * n),
        sharex=True,
        sharey=True,
        dpi=300,
    )
    if n == 1:
        axes = [axes]

    # numpy arrays for clean fancy-indexing
    hi_idx  = np.array(feat_hi_idx,  dtype=int) if feat_hi_idx  else np.array([], dtype=int)
    ens_idx = np.array(feat_ens_idx, dtype=int) if feat_ens_idx else np.array([], dtype=int)

    for ax, group, color in zip(axes, groups, palette):
        ra = avgs_ra[group]

        # ── Gray: all non-featured bins ───────────────────────────────────
        ax.vlines(
            mz[nf_mask], 0, ra[nf_mask],
            colors=_GRAY_STEM, linewidths=0.5, zorder=2,
        )

        # ── Ensemble-only features: colored stem (no label) ───────────────
        if len(ens_idx):
            ax.vlines(
                mz[ens_idx], 0, ra[ens_idx],
                colors=color, linewidths=0.85, alpha=0.62, zorder=3,
            )

        # ── High-confidence features: bold stem + m/z text label ──────────
        if len(hi_idx):
            mz_h = mz[hi_idx]
            ra_h = ra[hi_idx]

            ax.vlines(
                mz_h, 0, ra_h,
                colors=color, linewidths=1.4, alpha=0.95, zorder=5,
            )

            # Select non-overlapping label candidates (tallest peaks first)
            label_sel = _select_labels(
                mz_h, ra_h,
                x_max=x_max,
                min_gap_da=label_min_gap,
                max_labels=label_max,
            )
            for j in label_sel:
                ax.text(
                    mz_h[j],
                    ra_h[j] + 2.0,          # 2 RA-% above stem tip
                    f'{mz_h[j]:.1f}',
                    ha='center', va='bottom',
                    rotation=90,
                    fontsize=6.5,
                    color=color,
                    clip_on=False,           # allow overflow into top margin
                    zorder=7,
                )

        # ── Panel axes & group label ──────────────────────────────────────
        ax.set_xlim(mz.min(), x_max)
        ax.set_ylim(0, _STEM_TOP)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
        ax.tick_params(axis='both', length=3, which='major')
        ax.tick_params(axis='x',   length=2, which='minor')

        ax.text(
            0.012, 0.95, group,
            transform=ax.transAxes,
            fontsize=10, fontweight='bold', color=color,
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor=color, linewidth=0.8, alpha=0.85),
        )
        sns.despine(ax=ax, trim=True)

    axes[-1].set_xlabel('m/z', fontsize=10, labelpad=6)

    # Shared y-axis label via figure text
    fig.text(
        0.02, 0.5, 'Relative Abundance (%)',
        va='center', ha='center', rotation='vertical', fontsize=10,
    )

    # ── Figure-level legend ───────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color=_GRAY_STEM, lw=1.0,
               label='Non-featured peak (group mean)'),
        Line2D([0], [0], color='#555555', lw=0.9, alpha=0.65,
               label='Ensemble feature'),
        Line2D([0], [0], color='#333333', lw=1.4, alpha=0.95,
               label='High-confidence feature  (label = m/z, 1 d.p.)'),
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower center', ncol=3,
        bbox_to_anchor=(0.5, -0.025),
        fontsize=8.5, frameon=True, framealpha=0.95, edgecolor='#cccccc',
    )
    fig.suptitle(
        f'Group Spectra with Discriminating Features — {experiment_name}',
        fontsize=12, fontweight='bold', y=1.01,
    )

    fig.tight_layout(rect=[0.05, 0.03, 1, 1])
    pub_savefig(out_path)


# ---------------------------------------------------------------------------
# Plotly backend  (interactive, notebook-friendly)
# ---------------------------------------------------------------------------

def _plot_plotly(mz, groups, avgs_ra,
                 feat_hi_idx, feat_ens_idx, nf_mask,
                 experiment_name, out_path,
                 x_max=1200, label_min_gap=15, label_max=15):
    """
    Interactive small-multiples centroid MS figure (Plotly HTML).

    Each panel contains:
      · Gray stems for non-featured bins
      · Group-colored stems (medium) for ensemble-only features
      · Group-colored stems (bold) for high-confidence features,
        with m/z text labels (1 d.p.) above each labeled peak.
        Labels use the same _select_labels() greedy filter as the
        Matplotlib backend to prevent crowding.

    X-axis spans mz.min() → x_max (default 1200), ticks every 200 Da.
    Y-axis is linear 0–105 % relative abundance, shared across all panels.
    Hover on any stem tip shows exact m/z and RA %.
    """
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Plotly is not installed.  Run: pip install plotly")

    n       = len(groups)
    palette = _CB_PALETTE[:n]

    hi_idx  = np.array(feat_hi_idx,  dtype=int) if feat_hi_idx  else np.array([], dtype=int)
    ens_idx = np.array(feat_ens_idx, dtype=int) if feat_ens_idx else np.array([], dtype=int)

    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.04,
        subplot_titles=groups,
    )

    for i, (group, color) in enumerate(zip(groups, palette)):
        row    = i + 1
        ra     = avgs_ra[group]
        c_full = _hex_to_rgba(color, 1.00)
        c_ens  = _hex_to_rgba(color, 0.62)

        # ── Gray: non-featured stems ──────────────────────────────────────
        xs_g, ys_g = _plotly_stems(mz[nf_mask], ra[nf_mask])
        fig.add_trace(go.Scatter(
            x=xs_g, y=ys_g,
            mode='lines',
            line=dict(color=_GRAY_STEM, width=0.6),
            hoverinfo='skip',
            showlegend=(i == 0),
            legendgroup='nonfeat',
            name='Non-featured peak',
        ), row=row, col=1)

        # ── Ensemble-only: colored stems (invisible hover markers at tips) ─
        if len(ens_idx):
            mz_e, ra_e = mz[ens_idx], ra[ens_idx]
            xs_e, ys_e = _plotly_stems(mz_e, ra_e)
            fig.add_trace(go.Scatter(
                x=xs_e, y=ys_e,
                mode='lines',
                line=dict(color=c_ens, width=1.1),
                hoverinfo='skip',
                showlegend=(i == 0),
                legendgroup='ensemble',
                name='Ensemble feature',
            ), row=row, col=1)
            fig.add_trace(go.Scatter(
                x=mz_e.tolist(), y=ra_e.tolist(),
                mode='markers',
                marker=dict(size=6, color='rgba(0,0,0,0)'),
                showlegend=False,
                hovertemplate=(
                    f'<b>Ensemble — {group}</b><br>'
                    'm/z: %{x:.4f}<br>'
                    'RA: %{y:.1f} %<extra></extra>'
                ),
            ), row=row, col=1)

        # ── High-confidence: bold stems + m/z text labels ─────────────────
        if len(hi_idx):
            mz_h, ra_h = mz[hi_idx], ra[hi_idx]
            xs_h, ys_h = _plotly_stems(mz_h, ra_h)
            fig.add_trace(go.Scatter(
                x=xs_h, y=ys_h,
                mode='lines',
                line=dict(color=c_full, width=1.8),
                hoverinfo='skip',
                showlegend=(i == 0),
                legendgroup='highconf',
                name='High-confidence feature',
            ), row=row, col=1)
            # Invisible hover-only markers at stem tips (no visible symbol)
            fig.add_trace(go.Scatter(
                x=mz_h.tolist(), y=ra_h.tolist(),
                mode='markers',
                marker=dict(size=8, color='rgba(0,0,0,0)'),
                showlegend=False,
                hovertemplate=(
                    f'<b>High-confidence — {group}</b><br>'
                    'm/z: %{{x:.4f}}<br>'
                    'RA: %{{y:.1f}} %<extra></extra>'
                ),
            ), row=row, col=1)
            # m/z text labels — greedy collision-filtered subset
            label_sel = _select_labels(
                mz_h, ra_h,
                x_max=x_max,
                min_gap_da=label_min_gap,
                max_labels=label_max,
            )
            if label_sel:
                lbl_mz = mz_h[label_sel]
                lbl_ra = ra_h[label_sel]
                fig.add_trace(go.Scatter(
                    x=lbl_mz.tolist(),
                    y=(lbl_ra + 3.0).tolist(),
                    mode='text',
                    text=[f'{v:.1f}' for v in lbl_mz],
                    textposition='top center',
                    textfont=dict(size=8, color=c_full,
                                  family='Arial, sans-serif'),
                    showlegend=False,
                    hoverinfo='skip',
                ), row=row, col=1)

    # ── X-axis: mz.min() → x_max, ticks every 200 Da ────────────────────
    fig.update_xaxes(
        range=[float(mz.min()), x_max],
        dtick=200, tick0=0,
        showgrid=False, linecolor='#444', ticks='outside',
        tickfont=dict(size=9, family='Arial'),
    )
    fig.update_xaxes(
        title_text='m/z', title_font=dict(size=11, family='Arial'),
        row=n, col=1,
    )

    # ── Y-axis: linear 0–105 %, consistent ticks across all panels ────────
    fig.update_yaxes(
        range=[0, _STEM_TOP],
        tickvals=[0, 25, 50, 75, 100],
        showgrid=True, gridcolor='rgba(0,0,0,0.06)', gridwidth=0.5,
        linecolor='#444', ticks='outside',
        tickfont=dict(size=9, family='Arial'),
    )

    # Colour & left-align auto-generated subplot title annotations
    for j in range(n):
        fig.layout.annotations[j].update(
            font=dict(color=palette[j], size=11, family='Arial, sans-serif'),
            x=0.01, xanchor='left',
        )

    extra_anns = [
        dict(
            text='Relative Abundance (%)',
            xref='paper', yref='paper', x=-0.06, y=0.5,
            showarrow=False, textangle=-90,
            font=dict(size=11, family='Arial'),
        ),
        dict(
            text=(
                'Stem height = group mean  |  '
                'Gray = non-featured  |  '
                'Thin colored = ensemble feature  |  '
                'Bold colored + label = high-confidence (m/z, 1 d.p.)  |  '
                'Y-axis normalised to dataset maximum'
            ),
            xref='paper', yref='paper', x=0.5, y=-0.04,
            showarrow=False, align='center',
            font=dict(size=8.5, color='#666', family='Arial'),
        ),
    ]

    fig.update_layout(
        title=dict(
            text=(
                f'<b>Group Spectra with Discriminating Features</b><br>'
                f'<span style="font-size:11px;color:#555">{experiment_name}</span>'
            ),
            x=0.5, font=dict(size=14, family='Arial'),
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=280 * n,
        width=1150,
        margin=dict(t=90, b=70, l=95, r=40),
        hovermode='closest',
        legend=dict(
            title=dict(text='Feature tier', font=dict(size=10, family='Arial')),
            font=dict(size=10, family='Arial'),
            bordercolor='#cccccc', borderwidth=1,
            bgcolor='rgba(255,255,255,0.92)',
            x=1.01, y=1.0,
        ),
        annotations=list(fig.layout.annotations) + extra_anns,
    )

    fig.write_html(out_path, include_plotlyjs='cdn')
    print(f"  Saved interactive HTML -> {out_path}")
