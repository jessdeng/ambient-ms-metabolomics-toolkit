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
# This includes the defensive helpers (adaptive_n_components,
# optimize_plsda_components) so both branches share the SAME dimensionality caps.
from standard.pipeline import (
    fit_plsda,
    compute_vip,
    compute_vip_1comp,
    compute_plsda_q2,
    compute_plsda_r2y,
    evaluate_plsda_q2,
    adaptive_n_components,
    optimize_plsda_components,
    plot_scores_3d,
    plot_vip,
)
