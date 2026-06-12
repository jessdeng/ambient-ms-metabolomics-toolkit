"""
r_comparable/pipeline.py — PLS-DA, VIP Scores, and Plots (R-compatible pipeline)
==================================================================================
Identical implementation to standard/pipeline.py. The R-comparable pipeline
uses different bin labels (MetaboAnalyst offset convention) but the same PLS-DA
and VIP mathematics.

Usage:
    python -m src.r_comparable.run_analysis
"""

# All functions are identical to standard/pipeline.py — re-export directly.
from standard.pipeline import (
    fit_plsda,
    compute_vip_1comp,
    plot_scores_3d,
    plot_vip,
)
