"""
src/shared/plot_style.py
========================
Centralised publication-quality figure style for the ambient-MS pipeline.

Usage
-----
    from src.shared.plot_style import apply_style, pub_savefig

    apply_style()               # call once per script / at top of plotting fn
    ...build figure...
    pub_savefig(out_path)       # saves at 300 DPI, white bg, tight bbox, closes fig

Design targets
--------------
- 300 DPI  (Nature/Science single-column figures are 90 mm wide at 300 DPI)
- Arial / DejaVu Sans (sans-serif, clean)
- Font sizes: axes labels 10 pt, tick labels 9 pt, titles 11 pt
- No top / right spine (minimalist journal style)
- Outward-facing tick marks
- PDF with embedded fonts (pdf.fonttype = 42)
- colorblind-safe seaborn 'colorblind' palette (used by callers, not forced here)
"""

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# rcParams dict — all plots in the pipeline should conform to this spec
# ---------------------------------------------------------------------------

_FONT_FAMILY = ['Arial', 'DejaVu Sans', 'Helvetica Neue', 'Helvetica', 'sans-serif']

PUB_RCPARAMS = {
    # Font
    'font.family':          _FONT_FAMILY,
    'font.size':            10,
    'axes.labelsize':       10,
    'axes.titlesize':       11,
    'xtick.labelsize':      9,
    'ytick.labelsize':      9,
    'legend.fontsize':      9,
    'legend.title_fontsize': 9,
    # Axes / spines
    'axes.linewidth':       0.8,
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    # Ticks
    'xtick.major.width':    0.8,
    'ytick.major.width':    0.8,
    'xtick.minor.width':    0.5,
    'ytick.minor.width':    0.5,
    'xtick.major.size':     4.0,
    'ytick.major.size':     4.0,
    'xtick.minor.size':     2.0,
    'ytick.minor.size':     2.0,
    'xtick.direction':      'out',
    'ytick.direction':      'out',
    # Figure
    'figure.facecolor':     'white',
    'figure.dpi':           100,        # screen preview; savefig DPI is separate
    # Save defaults (used by pub_savefig)
    'savefig.dpi':          300,
    'savefig.bbox':         'tight',
    'savefig.facecolor':    'white',
    # Font embedding in PDF/PS output
    'pdf.fonttype':         42,
    'ps.fonttype':          42,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_style():
    """
    Apply publication-quality style globally.

    Call once at module level or at the start of each plotting function.
    Safe to call multiple times (idempotent).
    """
    sns.set_style('ticks')
    plt.rcParams.update(PUB_RCPARAMS)


def pub_savefig(path, dpi=300, close=True, **kwargs):
    """
    Save the current figure at publication quality and optionally close it.

    Drop-in replacement for::

        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()

    Parameters
    ----------
    path   : str  — output file path (.png, .pdf, .svg all work)
    dpi    : int  — default 300
    close  : bool — call plt.close() after saving (default True)
    **kwargs      — forwarded to plt.savefig()
    """
    kwargs.setdefault('dpi',          dpi)
    kwargs.setdefault('bbox_inches',  'tight')
    kwargs.setdefault('facecolor',    'white')
    plt.savefig(path, **kwargs)
    if close:
        plt.close()
    print(f"  Saved -> {path}")
