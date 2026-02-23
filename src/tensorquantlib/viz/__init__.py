"""Visualization utilities for TensorQuantLib.

Provides publication-quality plots for:
- pricing surfaces (2D heatmaps and 3D surfaces)
- Greeks surfaces
- TT rank profiles
- compression vs tolerance trade-off curves
- convergence diagnostics
"""

from .plots import (
    plot_compression_vs_tolerance,
    plot_convergence,
    plot_greeks_surface,
    plot_pricing_surface,
    plot_rank_profile,
    plot_tt_ranks,
)

__all__ = [
    "plot_compression_vs_tolerance",
    "plot_convergence",
    "plot_greeks_surface",
    "plot_pricing_surface",
    "plot_rank_profile",
    "plot_tt_ranks",
]
