"""
TT-accelerated pricing — build compressed surrogates for MC-based pricers.

Provides factory functions that build TT-surrogate models by evaluating
Monte-Carlo pricers on structured grids and compressing via TT-SVD.

Supported pricers:
    - Heston stochastic volatility (``heston_surrogate``)
    - American options via LSM (``american_surrogate``)
    - Exotic options — asian, barrier, lookback (``exotic_surrogate``)
    - Jump-diffusion — Merton analytic (``jump_diffusion_surrogate``)

Usage::

    from tensorquantlib.tt.pricing import heston_surrogate

    surr = heston_surrogate(
        S_range=(80, 120), K_range=(90, 110),
        T_range=(0.25, 2.0), n_points=15, eps=1e-4,
    )
    price = surr.evaluate([100, 105, 1.0])
"""

from __future__ import annotations

import itertools
import time

import numpy as np

from .decompose import tt_svd
from .surrogate import TTSurrogate


def _build_grid(
    axes: list[np.ndarray],
    pricer_fn,
) -> np.ndarray:
    """Evaluate *pricer_fn* on every point of the Cartesian product of *axes*.

    Args:
        axes: List of 1-D arrays — grid ticks per dimension.
        pricer_fn: Callable accepting ``*args`` where len(args) == len(axes).
                   Must return a scalar float.

    Returns:
        np.ndarray of shape ``(len(axes[0]), ..., len(axes[-1]))``.
    """
    shape = tuple(len(a) for a in axes)
    grid = np.empty(shape, dtype=np.float64)
    for idx in itertools.product(*(range(n) for n in shape)):
        args = tuple(axes[k][idx[k]] for k in range(len(axes)))
        grid[idx] = pricer_fn(*args)
    return grid


def _make_surrogate(
    axes: list[np.ndarray],
    pricer_fn,
    eps: float,
    max_rank: int | None,
) -> TTSurrogate:
    t0 = time.perf_counter()
    grid = _build_grid(axes, pricer_fn)
    build_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    cores = tt_svd(grid, eps=eps, max_rank=max_rank)
    compress_time = time.perf_counter() - t1

    return TTSurrogate(
        cores=cores,
        axes=axes,
        eps=eps,
        build_time=build_time,
        compress_time=compress_time,
        original_shape=grid.shape,
        original_nbytes=grid.nbytes,
    )


# ── Heston surrogate ───────────────────────────────────────────────────


def heston_surrogate(
    S_range: tuple[float, float] = (80, 120),
    K_range: tuple[float, float] = (90, 110),
    T_range: tuple[float, float] = (0.25, 2.0),
    r: float = 0.05,
    v0: float = 0.04,
    kappa: float = 2.0,
    theta: float = 0.04,
    xi: float = 0.3,
    rho: float = -0.7,
    n_points: int = 15,
    eps: float = 1e-4,
    max_rank: int | None = None,
    n_mc_paths: int = 50_000,
    option_type: str = "call",
) -> TTSurrogate:
    """Build a TT-surrogate for Heston MC prices over (S, K, T).

    Args:
        S_range: Spot price range.
        K_range: Strike range.
        T_range: Maturity range.
        r: Risk-free rate.
        v0, kappa, theta, xi, rho: Heston parameters.
        n_points: Grid points per axis.
        eps: TT-SVD tolerance.
        max_rank: Maximum TT-rank.
        n_mc_paths: Paths for Heston MC.
        option_type: "call" or "put".

    Returns:
        TTSurrogate with axes [S, K, T].
    """
    from ..finance.heston import HestonParams, heston_price_mc

    params = HestonParams(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)

    axes = [
        np.linspace(*S_range, n_points),
        np.linspace(*K_range, n_points),
        np.linspace(*T_range, n_points),
    ]

    def pricer(S, K, T):
        return heston_price_mc(S, K, T, r, params, n_paths=n_mc_paths, option_type=option_type)

    return _make_surrogate(axes, pricer, eps, max_rank)


# ── American option surrogate ───────────────────────────────────────────


def american_surrogate(
    S_range: tuple[float, float] = (80, 120),
    K_range: tuple[float, float] = (90, 110),
    T_range: tuple[float, float] = (0.25, 2.0),
    r: float = 0.05,
    sigma: float = 0.2,
    n_points: int = 15,
    eps: float = 1e-4,
    max_rank: int | None = None,
    n_paths: int = 50_000,
    n_steps: int = 100,
    option_type: str = "put",
) -> TTSurrogate:
    """Build a TT-surrogate for American option (LSM) prices over (S, K, T).

    Args:
        S_range: Spot price range.
        K_range: Strike range.
        T_range: Maturity range.
        r: Risk-free rate.
        sigma: Volatility.
        n_points: Grid points per axis.
        eps: TT-SVD tolerance.
        max_rank: Maximum TT-rank.
        n_paths: MC paths for LSM.
        n_steps: Time steps for LSM.
        option_type: "call" or "put".

    Returns:
        TTSurrogate with axes [S, K, T].
    """
    from ..finance.american import american_option_lsm

    axes = [
        np.linspace(*S_range, n_points),
        np.linspace(*K_range, n_points),
        np.linspace(*T_range, n_points),
    ]

    def pricer(S, K, T):
        return american_option_lsm(
            S, K, T, r, sigma, n_paths=n_paths, n_steps=n_steps, option_type=option_type
        )

    return _make_surrogate(axes, pricer, eps, max_rank)


# ── Exotic option surrogate ─────────────────────────────────────────────


def exotic_surrogate(
    exotic_type: str = "asian",
    S_range: tuple[float, float] = (80, 120),
    K_range: tuple[float, float] = (90, 110),
    T_range: tuple[float, float] = (0.25, 2.0),
    r: float = 0.05,
    sigma: float = 0.2,
    n_points: int = 15,
    eps: float = 1e-4,
    max_rank: int | None = None,
    n_paths: int = 50_000,
    **pricer_kwargs,
) -> TTSurrogate:
    """Build a TT-surrogate for exotic option MC prices over (S, K, T).

    Args:
        exotic_type: One of "asian", "barrier_up_out", "barrier_down_out",
                     "lookback_fixed", "lookback_floating".
        S_range: Spot price range.
        K_range: Strike range.
        T_range: Maturity range.
        r: Risk-free rate.
        sigma: Volatility.
        n_points: Grid points per axis.
        eps: TT-SVD tolerance.
        max_rank: Maximum TT-rank.
        n_paths: MC paths.
        **pricer_kwargs: Extra keyword arguments passed to the underlying pricer.

    Returns:
        TTSurrogate with axes [S, K, T].
    """
    from ..finance.exotics import (
        asian_price_mc,
        barrier_price_mc,
        lookback_price_mc,
    )

    axes = [
        np.linspace(*S_range, n_points),
        np.linspace(*K_range, n_points),
        np.linspace(*T_range, n_points),
    ]

    if exotic_type == "asian":

        def pricer(S, K, T):
            return asian_price_mc(S, K, T, r, sigma, n_paths=n_paths, **pricer_kwargs)
    elif exotic_type.startswith("barrier"):
        barrier = pricer_kwargs.pop("barrier", 130.0)
        # Convert "barrier_up_out" → "up-and-out"
        bt = exotic_type.replace("barrier_", "").replace("_", "-and-")

        def pricer(S, K, T):
            return barrier_price_mc(
                S,
                K,
                T,
                r,
                sigma,
                barrier=barrier,
                barrier_type=bt,
                n_paths=n_paths,
                **pricer_kwargs,
            )
    elif exotic_type.startswith("lookback"):
        strike_type = "fixed" if "fixed" in exotic_type else "floating"

        def pricer(S, K, T):
            result = lookback_price_mc(
                S, K, T, r, sigma, strike_type=strike_type, n_paths=n_paths, **pricer_kwargs
            )
            return result[0] if isinstance(result, tuple) else result
    else:
        raise ValueError(f"Unknown exotic_type: {exotic_type!r}")

    return _make_surrogate(axes, pricer, eps, max_rank)


# ── Jump-diffusion surrogate ────────────────────────────────────────────


def jump_diffusion_surrogate(
    S_range: tuple[float, float] = (80, 120),
    K_range: tuple[float, float] = (90, 110),
    T_range: tuple[float, float] = (0.25, 2.0),
    r: float = 0.05,
    sigma: float = 0.2,
    lam: float = 1.0,
    mu_j: float = -0.05,
    sigma_j: float = 0.1,
    n_points: int = 15,
    eps: float = 1e-4,
    max_rank: int | None = None,
    option_type: str = "call",
) -> TTSurrogate:
    """Build a TT-surrogate for Merton jump-diffusion prices over (S, K, T).

    Uses the analytic series expansion (fast) rather than MC.

    Args:
        S_range: Spot price range.
        K_range: Strike range.
        T_range: Maturity range.
        r: Risk-free rate.
        sigma: Diffusion volatility.
        lam: Jump intensity.
        mu_j: Mean jump size.
        sigma_j: Jump size volatility.
        n_points: Grid points per axis.
        eps: TT-SVD tolerance.
        max_rank: Maximum TT-rank.
        option_type: "call" or "put".

    Returns:
        TTSurrogate with axes [S, K, T].
    """
    from ..finance.jump_diffusion import merton_jump_price

    axes = [
        np.linspace(*S_range, n_points),
        np.linspace(*K_range, n_points),
        np.linspace(*T_range, n_points),
    ]

    def pricer(S, K, T):
        return merton_jump_price(S, K, T, r, sigma, lam, mu_j, sigma_j, option_type=option_type)

    return _make_surrogate(axes, pricer, eps, max_rank)
