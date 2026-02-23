"""Plotting functions for TensorQuantLib.

All functions return ``(fig, ax)`` tuples so callers can customise further.
Matplotlib is imported lazily — the rest of the library works without it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import matplotlib.axes
    import matplotlib.figure


def _import_mpl() -> tuple[Any, Any]:
    """Lazy-import matplotlib and return (plt, mpl) or raise ImportError."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        return plt, matplotlib
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting.  "
            "Install it with:  pip install 'tensorquantlib[dev]'"
        ) from exc


# ====================================================================== #
# Pricing Surface
# ====================================================================== #

def plot_pricing_surface(
    grid: np.ndarray,
    axis_values: Sequence[np.ndarray],
    dims: tuple[int, int] = (0, 1),
    fixed_indices: dict[int, int] | None = None,
    title: str = "Pricing Surface",
    xlabel: str | None = None,
    ylabel: str | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (8, 6),
    mode: str = "heatmap",
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot a 2D slice of a pricing grid as heatmap or 3D surface.

    Args:
        grid: N-dimensional pricing grid (NumPy array).
        axis_values: List of 1D arrays giving tick values along each axis.
        dims: Which two axes to plot. The remaining axes are sliced at
              ``fixed_indices`` (default: midpoints).
        fixed_indices: ``{axis: index}`` overrides for slicing.
        title: Plot title.
        xlabel: Label for x-axis (default: "Axis {dims[0]}").
        ylabel: Label for y-axis (default: "Axis {dims[1]}").
        cmap: Matplotlib colour-map name.
        figsize: Figure size in inches ``(width, height)``.
        mode: ``"heatmap"`` (default) or ``"surface"`` (3D).

    Returns:
        ``(fig, ax)`` tuple.
    """
    plt, _ = _import_mpl()
    fixed_indices = fixed_indices or {}

    # Build slicer for all dimensions
    slicer: list[Any] = []
    for i in range(grid.ndim):
        if i in dims:
            slicer.append(slice(None))
        else:
            idx = fixed_indices.get(i, grid.shape[i] // 2)
            slicer.append(idx)

    Z = grid[tuple(slicer)]
    # Ensure dims[0] is rows, dims[1] is cols — transpose if needed
    if dims[0] > dims[1]:
        Z = Z.T

    X = axis_values[dims[0]]
    Y = axis_values[dims[1]]

    xlabel = xlabel or f"Axis {dims[0]}"
    ylabel = ylabel or f"Axis {dims[1]}"

    if mode == "surface":
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        Xm, Ym = np.meshgrid(X, Y, indexing="ij")
        ax.plot_surface(Xm, Ym, Z, cmap=cmap, edgecolor="none", alpha=0.9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel("Price")
        ax.set_title(title)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(
            Z.T,
            origin="lower",
            aspect="auto",
            extent=[X[0], X[-1], Y[0], Y[-1]],
            cmap=cmap,
        )
        fig.colorbar(im, ax=ax, label="Price")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    return fig, ax


# ====================================================================== #
# Greeks Surface
# ====================================================================== #

def plot_greeks_surface(
    greek_grids: dict[str, np.ndarray],
    axis_values: Sequence[np.ndarray],
    dims: tuple[int, int] = (0, 1),
    fixed_indices: dict[int, int] | None = None,
    cmap: str = "RdBu_r",
    figsize: tuple[float, float] = (14, 4),
) -> tuple[matplotlib.figure.Figure, list[matplotlib.axes.Axes]]:
    """Plot multiple Greeks as side-by-side heatmaps.

    Args:
        greek_grids: ``{"Delta": array, "Gamma": array, ...}``.
        axis_values: Tick values per axis (same as ``plot_pricing_surface``).
        dims: Pair of axes to plot.
        fixed_indices: Override slice indices for remaining axes.
        cmap: Colour map.
        figsize: Figure size for the whole row.

    Returns:
        ``(fig, axes)`` tuple.
    """
    plt, _ = _import_mpl()
    fixed_indices = fixed_indices or {}
    n = len(greek_grids)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    X = axis_values[dims[0]]
    Y = axis_values[dims[1]]

    for ax, (name, grid) in zip(axes, greek_grids.items()):
        slicer: list[Any] = []
        for i in range(grid.ndim):
            if i in dims:
                slicer.append(slice(None))
            else:
                idx = fixed_indices.get(i, grid.shape[i] // 2)
                slicer.append(idx)
        Z = grid[tuple(slicer)]
        if dims[0] > dims[1]:
            Z = Z.T

        im = ax.imshow(
            Z.T,
            origin="lower",
            aspect="auto",
            extent=[X[0], X[-1], Y[0], Y[-1]],
            cmap=cmap,
        )
        fig.colorbar(im, ax=ax)
        ax.set_title(name)
        ax.set_xlabel(f"Axis {dims[0]}")
        ax.set_ylabel(f"Axis {dims[1]}")

    fig.tight_layout()
    return fig, list(axes)


# ====================================================================== #
# TT Rank Profile
# ====================================================================== #

def plot_tt_ranks(
    cores: list[np.ndarray],
    title: str = "TT-Rank Profile",
    figsize: tuple[float, float] = (6, 4),
    color: str = "#2563eb",
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Bar chart of TT-ranks across bonds.

    Args:
        cores: List of TT-cores.
        title: Plot title.
        figsize: Figure size.
        color: Bar colour.

    Returns:
        ``(fig, ax)`` tuple.
    """
    plt, _ = _import_mpl()
    from tensorquantlib.tt.ops import tt_ranks

    ranks = tt_ranks(cores)
    bonds = list(range(len(ranks)))

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(bonds, ranks, color=color, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Bond index")
    ax.set_ylabel("Rank")
    ax.set_title(title)
    ax.set_xticks(bonds)
    return fig, ax


def plot_rank_profile(
    rank_lists: dict[str, list[int]],
    title: str = "Rank Profiles",
    figsize: tuple[float, float] = (7, 4),
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Overlay multiple rank profiles for comparison.

    Args:
        rank_lists: ``{"eps=1e-4": [r0, r1, ...], ...}``.
        title: Plot title.
        figsize: Figure size.

    Returns:
        ``(fig, ax)`` tuple.
    """
    plt, _ = _import_mpl()

    fig, ax = plt.subplots(figsize=figsize)
    for label, ranks in rank_lists.items():
        ax.plot(range(len(ranks)), ranks, "o-", label=label, markersize=5)
    ax.set_xlabel("Bond index")
    ax.set_ylabel("Rank")
    ax.set_title(title)
    ax.legend()
    return fig, ax


# ====================================================================== #
# Compression vs Tolerance
# ====================================================================== #

def plot_compression_vs_tolerance(
    epsilons: Sequence[float],
    compression_ratios: Sequence[float],
    errors: Sequence[float] | None = None,
    title: str = "Compression vs Tolerance",
    figsize: tuple[float, float] = (7, 4),
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot compression ratio and (optionally) error vs SVD tolerance.

    Args:
        epsilons: SVD tolerance values.
        compression_ratios: Matching compression ratios.
        errors: Optional matching relative errors.
        title: Plot title.
        figsize: Figure size.

    Returns:
        ``(fig, ax)`` — second y-axis is ``ax.twinx()`` if errors given.
    """
    plt, _ = _import_mpl()

    fig, ax1 = plt.subplots(figsize=figsize)
    color1, color2 = "#2563eb", "#dc2626"

    ax1.semilogx(epsilons, compression_ratios, "o-", color=color1, label="Compression ratio")
    ax1.set_xlabel("SVD tolerance (ε)")
    ax1.set_ylabel("Compression ratio", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_title(title)

    ax_out = ax1
    if errors is not None:
        ax2 = ax1.twinx()
        ax2.loglog(epsilons, errors, "s--", color=color2, label="Relative error")
        ax2.set_ylabel("Relative error", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
        ax_out = ax1

    fig.tight_layout()
    return fig, ax_out


# ====================================================================== #
# Convergence
# ====================================================================== #

def plot_convergence(
    iterations: Sequence[int],
    values: Sequence[float],
    ylabel: str = "Error",
    title: str = "Convergence",
    log_y: bool = True,
    figsize: tuple[float, float] = (7, 4),
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Line plot of a convergence metric over iterations.

    Args:
        iterations: Iteration indices.
        values: Corresponding metric values.
        ylabel: Y-axis label.
        title: Plot title.
        log_y: Use log scale for y-axis.
        figsize: Figure size.

    Returns:
        ``(fig, ax)`` tuple.
    """
    plt, _ = _import_mpl()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(iterations, values, "o-", color="#2563eb", markersize=4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax
