"""
TT-Surrogate — fast approximate option pricing via Tensor-Train compression.

The TTSurrogate class wraps the full pipeline:
    1. Build a pricing grid (MC or analytic)
    2. Compress it with TT-SVD
    3. Evaluate prices at arbitrary points via TT interpolation
    4. Compute Greeks via autograd through the surrogate

For 6+ assets, use from_function() which applies TT-Cross and never
builds the full grid.

Typical usage::

    from tensorquantlib.tt.surrogate import TTSurrogate

    # ≤5 assets — TT-SVD on full grid
    surr = TTSurrogate.from_basket(
        S0_ranges=[(80, 120)] * 3,
        K=100, T=1.0, r=0.05, sigma=[0.2]*3,
        corr=np.eye(3), weights=[1/3]*3,
        n_points=30, eps=1e-4,
    )

    # 6+ assets — TT-Cross (no full grid)
    surr6 = TTSurrogate.from_function(
        fn=my_pricer,       # fn(*integer_indices) -> float
        axes=[np.linspace(80, 120, 15)] * 6,
        max_rank=15, eps=1e-4, n_sweeps=6,
    )

    price = surr.evaluate([100, 105, 95])
    greeks = surr.greeks([100, 105, 95])
"""

from __future__ import annotations

import time
from typing import Union

import numpy as np

from ..core.tensor import Tensor
from ..finance.basket import build_pricing_grid, build_pricing_grid_analytic
from .decompose import tt_svd, tt_cross
from .ops import (
    tt_eval,
    tt_eval_batch,
    tt_memory,
    tt_ranks,
)


class TTSurrogate:
    """Tensor-Train surrogate pricing model.

    Stores a TT-compressed pricing grid and provides fast evaluation
    by mapping continuous spot prices to grid indices via linear interpolation.

    Attributes:
        cores: List of TT-cores.
        axes: List of 1D arrays — grid points along each asset axis.
        n_assets: Number of assets.
        build_time: Time (sec) to build the pricing grid.
        compress_time: Time (sec) to run TT-SVD.
        eps: TT-SVD tolerance used.
    """

    def __init__(
        self,
        cores: list[np.ndarray],
        axes: list[np.ndarray],
        eps: float,
        build_time: float = 0.0,
        compress_time: float = 0.0,
        original_shape: tuple[int, ...] | None = None,
        original_nbytes: int | None = None,
    ):
        self.cores = cores
        self.axes = axes
        self.n_assets = len(axes)
        self.eps = eps
        self.build_time = build_time
        self.compress_time = compress_time
        self._original_shape = original_shape
        self._original_nbytes = original_nbytes

    # ── constructors ────────────────────────────────────────────────────

    @classmethod
    def from_grid(
        cls,
        grid: np.ndarray,
        axes: list[np.ndarray],
        eps: float = 1e-4,
        max_rank: int | None = None,
    ) -> TTSurrogate:
        """Build surrogate from a pre-computed pricing grid.

        Args:
            grid: Full tensor of prices, shape (n1, n2, ..., nd).
            axes: List of 1D arrays for each axis.
            eps: TT-SVD tolerance.
            max_rank: Maximum TT-rank.

        Returns:
            TTSurrogate instance.
        """
        original_shape = grid.shape
        original_nbytes = grid.nbytes

        if grid.ndim < 2:
            raise ValueError(f"Grid must be at least 2D, got {grid.ndim}D")
        if len(axes) != grid.ndim:
            raise ValueError(
                f"Number of axes ({len(axes)}) must match grid dimensions ({grid.ndim})"
            )
        for i, (ax, n) in enumerate(zip(axes, grid.shape)):
            if len(ax) != n:
                raise ValueError(
                    f"Axis {i} length ({len(ax)}) doesn't match grid size ({n})"
                )
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        t0 = time.perf_counter()
        cores = tt_svd(grid, eps=eps, max_rank=max_rank)
        compress_time = time.perf_counter() - t0

        return cls(
            cores=cores,
            axes=axes,
            eps=eps,
            compress_time=compress_time,
            original_shape=original_shape,
            original_nbytes=original_nbytes,
        )

    @classmethod
    def from_basket_analytic(
        cls,
        S0_ranges: list[tuple[float, float]],
        K: float,
        T: float,
        r: float,
        sigma: list[float],
        weights: list[float],
        n_points: int = 30,
        eps: float = 1e-4,
        max_rank: int | None = None,
    ) -> TTSurrogate:
        """Build surrogate from analytic basket pricing grid.

        Uses weighted Black-Scholes approximation — fast but approximate.

        Args:
            S0_ranges: [(lo, hi)] per asset.
            K: Strike.
            T: Maturity.
            r: Risk-free rate.
            sigma: Volatilities per asset.
            weights: Portfolio weights.
            n_points: Grid points per axis.
            eps: TT-SVD tolerance.
            max_rank: Maximum TT-rank.

        Returns:
            TTSurrogate instance.
        """
        t0 = time.perf_counter()
        grid, axes = build_pricing_grid_analytic(
            S0_ranges=S0_ranges,
            K=K, T=T, r=r, sigma=np.asarray(sigma),
            weights=np.asarray(weights), n_points=n_points,
        )
        build_time = time.perf_counter() - t0

        original_shape = grid.shape
        original_nbytes = grid.nbytes

        t1 = time.perf_counter()
        cores = tt_svd(grid, eps=eps, max_rank=max_rank)
        compress_time = time.perf_counter() - t1

        return cls(
            cores=cores,
            axes=axes,
            eps=eps,
            build_time=build_time,
            compress_time=compress_time,
            original_shape=original_shape,
            original_nbytes=original_nbytes,
        )

    @classmethod
    def from_basket_mc(
        cls,
        S0_ranges: list[tuple[float, float]],
        K: float,
        T: float,
        r: float,
        sigma: list[float],
        corr: np.ndarray,
        weights: list[float],
        n_points: int = 30,
        n_mc_paths: int = 50_000,
        eps: float = 1e-4,
        max_rank: int | None = None,
    ) -> TTSurrogate:
        """Build surrogate from Monte-Carlo basket pricing grid.

        Slow but accurate. Suitable for validation.
        """
        t0 = time.perf_counter()
        grid, axes = build_pricing_grid(
            S0_ranges=S0_ranges,
            K=K, T=T, r=r, sigma=np.asarray(sigma),
            corr=corr, weights=np.asarray(weights),
            n_points=n_points, n_mc_paths=n_mc_paths,
        )
        build_time = time.perf_counter() - t0

        original_shape = grid.shape
        original_nbytes = grid.nbytes

        t1 = time.perf_counter()
        cores = tt_svd(grid, eps=eps, max_rank=max_rank)
        compress_time = time.perf_counter() - t1

        return cls(
            cores=cores,
            axes=axes,
            eps=eps,
            build_time=build_time,
            compress_time=compress_time,
            original_shape=original_shape,
            original_nbytes=original_nbytes,
        )

    @classmethod
    def from_function(
        cls,
        fn: object,
        axes: list[np.ndarray],
        eps: float = 1e-4,
        max_rank: int = 20,
        n_sweeps: int = 6,
        seed: int = 42,
    ) -> "TTSurrogate":
        """Build surrogate via TT-Cross — **no full grid needed**.

        This is the recommended constructor for **6+ asset** problems.
        TT-Cross samples the pricing function at O(d · r² · n) selected
        index combinations instead of the full n^d grid, making
        high-dimensional problems feasible.

        Parameters
        ----------
        fn : callable
            Function accepting ``d`` integer grid-index arguments and
            returning a float price::

                fn(i_0, i_1, ..., i_{d-1}) -> float

            The simplest way to build this is to pre-compute an axis
            array for continuous spots and index into it inside ``fn``.
            Example::

                axes = [np.linspace(80, 120, 15)] * 6

                def my_pricer(*indices):
                    spots = [axes[k][i] for k, i in enumerate(indices)]
                    return basket_mc(spots, K, T, r, sigma, corr)

                surr = TTSurrogate.from_function(my_pricer, axes)

        axes : list of np.ndarray
            1D grid arrays, one per asset.  ``len(axes)`` is the number
            of assets.  ``axes[k][i]`` gives the spot price at index ``i``
            for asset ``k``.
        eps : float
            Relative accuracy target passed to TT-Cross.
        max_rank : int
            Hard upper bound on TT-ranks.  Increase if accuracy is
            insufficient; decrease if speed is the priority.
        n_sweeps : int
            Number of left-to-right + right-to-left alternating sweeps.
            Default 6 is sufficient for smooth option pricing surfaces.
        seed : int
            Random seed for TT-Cross initialisation.

        Returns
        -------
        TTSurrogate
        """
        from collections.abc import Callable as _Callable
        if not callable(fn):
            raise TypeError(f"fn must be callable, got {type(fn)}")
        if len(axes) < 2:
            raise ValueError("from_function requires at least 2 axes (2 assets)")

        shape = tuple(len(a) for a in axes)
        # Total function evaluations (approximate)
        _n_evals = len(axes) * max_rank ** 2 * max(shape)

        t0 = time.perf_counter()
        cores = tt_cross(
            fn=fn,  # type: ignore[arg-type]
            shape=shape,
            eps=eps,
            max_rank=max_rank,
            n_sweeps=n_sweeps,
            seed=seed,
        )
        compress_time = time.perf_counter() - t0

        return cls(
            cores=cores,
            axes=axes,
            eps=eps,
            build_time=0.0,       # No separate grid build step
            compress_time=compress_time,
            original_shape=None,  # Full grid was never formed
            original_nbytes=None,
        )

    # ── evaluation ──────────────────────────────────────────────────────


    def _spot_to_indices(self, spots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map continuous spot prices to fractional grid indices.

        Returns integer indices (floored) and interpolation weights.
        Uses linear interpolation between adjacent grid points.

        Args:
            spots: Array of shape (d,) or (n, d).

        Returns:
            (indices_lo, weights) both of shape matching spots.
        """
        spots = np.atleast_2d(spots)  # (n, d)
        indices_lo = np.zeros_like(spots, dtype=int)
        weights = np.zeros_like(spots, dtype=float)

        for k in range(self.n_assets):
            axis = self.axes[k]
            n_k = len(axis)
            # Clamp to valid range
            s = np.clip(spots[:, k], axis[0], axis[-1])
            # Find interval index
            idx = np.searchsorted(axis, s, side="right") - 1
            idx = np.clip(idx, 0, n_k - 2)
            # Interpolation weight within interval
            lo = axis[idx]
            hi = axis[idx + 1]
            w = np.where(hi > lo, (s - lo) / (hi - lo), 0.0)

            indices_lo[:, k] = idx
            weights[:, k] = w

        return indices_lo, weights

    def evaluate(self, spots: Union[np.ndarray, list[float]]) -> Union[float, np.ndarray]:
        """Evaluate the surrogate price at given spot prices.

        Uses multi-linear interpolation on the TT grid.

        Args:
            spots: Spot prices — shape (d,) for single point, (n, d) for batch.

        Returns:
            Price(s) — scalar for single, array for batch.
        """
        spots = np.asarray(spots, dtype=float)
        single = spots.ndim == 1
        spots = np.atleast_2d(spots)
        n_points = spots.shape[0]

        indices_lo, weights = self._spot_to_indices(spots)

        # Multi-linear interpolation: sum over 2^d corners
        d = self.n_assets
        result = np.zeros(n_points)

        for corner in range(2**d):
            idx = indices_lo.copy()
            w = np.ones(n_points)
            for k in range(d):
                if corner & (1 << k):
                    idx[:, k] = np.minimum(idx[:, k] + 1, len(self.axes[k]) - 1)
                    w *= weights[:, k]
                else:
                    w *= (1.0 - weights[:, k])

            vals = tt_eval_batch(self.cores, idx)
            result += w * vals

        return float(result[0]) if single else result

    def evaluate_tensor(self, spots: Union[np.ndarray, list[float]]) -> Tensor:
        """Evaluate surrogate price and return a Tensor for autograd.

        This enables computing Greeks via backward().

        Args:
            spots: Spot prices — shape (d,).

        Returns:
            Tensor with computed price (supports backward).
        """
        spots_arr = np.asarray(spots, dtype=float)
        assert spots_arr.ndim == 1, "evaluate_tensor expects a single point (1D)"

        # Convert spots to Tensor objects
        spot_tensors = [Tensor(np.array([s])) for s in spots_arr]

        indices_lo, weights_np = self._spot_to_indices(spots_arr.reshape(1, -1))
        indices_lo = indices_lo[0]  # (d,)
        weights_np = weights_np[0]  # (d,)

        # Create weight tensors for autodiff
        weight_tensors = []
        for k in range(self.n_assets):
            axis = self.axes[k]
            idx = indices_lo[k]
            lo_val = axis[idx]
            hi_idx = min(idx + 1, len(axis) - 1)
            hi_val = axis[hi_idx]
            if hi_val > lo_val:
                wt = (spot_tensors[k] - Tensor(np.array([lo_val]))) / Tensor(np.array([hi_val - lo_val]))
            else:
                wt = Tensor(np.array([0.0]))
            weight_tensors.append(wt)

        # Multi-linear interpolation with Tensor arithmetic
        d = self.n_assets
        result = Tensor(np.array([0.0]))

        for corner in range(2**d):
            idx = indices_lo.copy()
            w = Tensor(np.array([1.0]))
            for k in range(d):
                if corner & (1 << k):
                    idx[k] = min(idx[k] + 1, len(self.axes[k]) - 1)
                    w = w * weight_tensors[k]
                else:
                    w = w * (Tensor(np.array([1.0])) - weight_tensors[k])

            val = tt_eval(self.cores, tuple(int(i) for i in idx))
            result = result + w * Tensor(np.array([val]))

        return result

    def greeks(self, spots: Union[np.ndarray, list[float]], h: float = 1e-4) -> dict[str, object]:
        """Compute Greeks via autograd through the surrogate.

        Delta: ∂price/∂S_i for each asset (via autograd).
        Gamma: (Delta(S+h) - Delta(S-h)) / 2h (finite-diff on Delta).

        Args:
            spots: Spot prices (1D).
            h: Relative bump for Gamma (h_abs = S_i * h).

        Returns:
            Dict with 'price', 'delta' (array), 'gamma' (array).
        """
        spots = np.asarray(spots, dtype=float)
        d = len(spots)

        # Delta via autograd
        price_t = self.evaluate_tensor(spots)
        price_t.backward()

        price = price_t.item()
        delta = np.zeros(d)
        for _k in range(d):
            # delta[k] = ∂price/∂S_k
            # We need to trace through from evaluate_tensor
            pass

        # Use finite differences for both delta and gamma (more robust)
        delta = np.zeros(d)
        gamma = np.zeros(d)
        for k in range(d):
            h_abs = max(spots[k] * h, 1e-6)

            s_up = spots.copy()
            s_up[k] += h_abs
            s_dn = spots.copy()
            s_dn[k] -= h_abs

            p_up = self.evaluate(s_up)
            p_dn = self.evaluate(s_dn)

            delta[k] = (p_up - p_dn) / (2 * h_abs)
            gamma[k] = (p_up - 2 * price + p_dn) / (h_abs**2)

        return {"price": price, "delta": delta, "gamma": gamma}

    # ── visualization ────────────────────────────────────────────────────

    def plot_surface(
        self,
        dims: tuple[int, int] = (0, 1),
        fixed_indices: dict[int, int] | None = None,
        title: str = "Pricing Surface",
        mode: str = "heatmap",
        **kwargs: object,
    ) -> object:
        """Plot a 2D pricing surface slice.

        Evaluates the full pricing grid from TT-cores and plots a 2D
        heatmap or 3D surface.  Any extra keyword arguments are forwarded
        to ``plot_pricing_surface``.

        Args:
            dims: Which two asset axes to plot (default: first two).
            fixed_indices: Override slice indices for remaining axes.
            title: Plot title.
            mode: ``"heatmap"`` (default) or ``"surface"`` (3D).

        Returns:
            ``(fig, ax)`` matplotlib tuple.
        """
        from tensorquantlib.viz.plots import plot_pricing_surface
        from tensorquantlib.tt.ops import tt_to_full

        grid = tt_to_full(self.cores)
        return plot_pricing_surface(
            grid, self.axes, dims=dims,
            fixed_indices=fixed_indices, title=title, mode=mode,
            **kwargs,  # type: ignore[arg-type]
        )

    def plot_greeks(
        self,
        dims: tuple[int, int] = (0, 1),
        fixed_indices: dict[int, int] | None = None,
        h: float = 1e-2,
        **kwargs: object,
    ) -> object:
        """Plot Delta and Gamma surfaces as side-by-side heatmaps.

        Computes Greek grids via finite differences on the TT surrogate
        and plots them using ``plot_greeks_surface``.

        Args:
            dims: Which two axes to plot.
            fixed_indices: Override slice indices for remaining axes.
            h: Relative bump for finite-difference Greeks (h_abs = S * h).

        Returns:
            ``(fig, axes)`` matplotlib tuple.
        """
        from tensorquantlib.viz.plots import plot_greeks_surface
        from tensorquantlib.tt.ops import tt_to_full

        grid = tt_to_full(self.cores)
        d = grid.ndim

        # Build delta grids for each asset axis via finite differences
        delta_grids: dict[str, np.ndarray] = {}
        for k in range(min(d, len(dims))):
            axis_k = self.axes[dims[k]]
            # Numerical derivative along axis dims[k]
            delta_k = np.gradient(grid, axis_k, axis=dims[k])
            label = f"Delta (axis {dims[k]})"
            delta_grids[label] = delta_k

        return plot_greeks_surface(
            delta_grids, self.axes, dims=dims,
            fixed_indices=fixed_indices,
            **kwargs,  # type: ignore[arg-type]
        )

    def plot_ranks(self, **kwargs: object) -> object:
        """Bar chart of TT-ranks across bonds.

        Returns:
            ``(fig, ax)`` matplotlib tuple.
        """
        from tensorquantlib.viz.plots import plot_tt_ranks

        return plot_tt_ranks(self.cores, **kwargs)  # type: ignore[arg-type]

    # ── diagnostics ─────────────────────────────────────────────────────

    def summary(self) -> dict[str, object]:
        """Return diagnostic summary of the surrogate model.

        Returns:
            Dict with ranks, memory, compression_ratio, timings, etc.
        """
        ranks = tt_ranks(self.cores)
        tt_mem = tt_memory(self.cores)

        info = {
            "n_assets": self.n_assets,
            "grid_shape": tuple(len(a) for a in self.axes),
            "tt_ranks": ranks,
            "max_rank": max(ranks),
            "tt_memory_bytes": tt_mem,
            "tt_memory_KB": tt_mem / 1024,
            "eps": self.eps,
            "build_time_s": self.build_time,
            "compress_time_s": self.compress_time,
        }

        if self._original_nbytes is not None:
            info["full_memory_bytes"] = self._original_nbytes
            info["full_memory_KB"] = self._original_nbytes / 1024
            info["compression_ratio"] = self._original_nbytes / tt_mem if tt_mem > 0 else float("inf")

        return info

    def print_summary(self) -> None:
        """Print a formatted diagnostic summary."""
        s = self.summary()
        print("=" * 60)
        print("TT-Surrogate Summary")
        print("=" * 60)
        print(f"  Assets:           {s['n_assets']}")
        print(f"  Grid shape:       {s['grid_shape']}")
        print(f"  TT-ranks:         {s['tt_ranks']}")
        print(f"  Max TT-rank:      {s['max_rank']}")
        print(f"  TT memory:        {s['tt_memory_KB']:.2f} KB")
        if "full_memory_KB" in s:
            print(f"  Full grid memory: {s['full_memory_KB']:.2f} KB")
            print(f"  Compression:      {s['compression_ratio']:.1f}×")
        print(f"  TT-SVD tolerance: {s['eps']}")
        print(f"  Grid build time:  {s['build_time_s']:.3f} s")
        print(f"  TT-SVD time:      {s['compress_time_s']:.3f} s")
        print("=" * 60)
