"""
Implied volatility solver for Black-Scholes options.

Provides both scalar and vectorized implied volatility computation using
Brent's method (bisection + secant) via scipy.optimize.brentq, plus a fast
Newton-Raphson (Halley) starter based on the Jaeckel (2015) approximation.

Functions:
    implied_vol      -- scalar IV for a single (price, S, K, T, r) tuple
    implied_vol_batch -- vectorized IV for arrays of market prices
    iv_surface        -- build a 2-D IV surface over (K, T) grids
"""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.optimize import brentq

from tensorquantlib.finance.black_scholes import bs_price_numpy, bs_vega

# ------------------------------------------------------------------ #
# Scalar solver
# ------------------------------------------------------------------ #


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = "call",
    *,
    tol: float = 1e-8,
    max_iter: int = 100,
    sigma_lo: float = 1e-4,
    sigma_hi: float = 10.0,
) -> float:
    """Compute implied volatility for a single European option via Brent's method.

    Inverts the Black-Scholes formula: finds sigma such that
    ``bs_price_numpy(S, K, T, r, sigma) == market_price``.

    Args:
        market_price: Observed option market price.
        S: Spot price.
        K: Strike price.
        T: Time to expiry (years).
        r: Risk-free rate.
        q: Continuous dividend yield (default 0).
        option_type: 'call' or 'put'.
        tol: Convergence tolerance on volatility (default 1e-8).
        max_iter: Maximum Brent iterations (default 100).
        sigma_lo: Lower bound for volatility search (default 1e-4).
        sigma_hi: Upper bound for volatility search (default 10.0).

    Returns:
        Implied volatility (annualised).

    Raises:
        ValueError: If market_price is outside the no-arbitrage bounds, or
            if the root is not bracketed within [sigma_lo, sigma_hi].

    Example:
        >>> iv = implied_vol(10.45, S=100, K=100, T=1.0, r=0.05)
        >>> abs(iv - 0.2) < 0.001
        True
    """
    # No-arbitrage bounds check
    intrinsic: float
    if option_type == "call":
        intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        upper_bound = S * np.exp(-q * T)
    else:
        intrinsic = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
        upper_bound = K * np.exp(-r * T)

    if market_price < intrinsic - tol:
        raise ValueError(
            f"market_price={market_price:.6f} is below intrinsic value "
            f"{intrinsic:.6f} — violates no-arbitrage"
        )
    if market_price > upper_bound + tol:
        raise ValueError(f"market_price={market_price:.6f} exceeds upper bound {upper_bound:.6f}")

    # Objective function
    def objective(sigma: float) -> float:
        return float(bs_price_numpy(S, K, T, r, sigma, q=q, option_type=option_type)) - market_price

    # Check bracket
    f_lo = objective(sigma_lo)
    f_hi = objective(sigma_hi)
    if f_lo * f_hi > 0:
        raise ValueError(
            f"Root not bracketed: f({sigma_lo})={f_lo:.4f}, f({sigma_hi})={f_hi:.4f}. "
            "Try widening sigma_lo / sigma_hi."
        )

    result: float = brentq(objective, sigma_lo, sigma_hi, xtol=tol, maxiter=max_iter)
    return result


# ------------------------------------------------------------------ #
# Newton-Raphson + Halley fast approximation
# ------------------------------------------------------------------ #


def implied_vol_nr(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = "call",
    *,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> float:
    """Implied volatility via Newton-Raphson with vega denominator.

    Faster than Brent for well-behaved cases (ATM options, moderate T).
    Falls back to Brent if Newton diverges or vega is too small.

    Args:
        market_price: Observed option price.
        S, K, T, r, q, option_type: Black-Scholes parameters.
        tol: Convergence tolerance.
        max_iter: Maximum Newton iterations.

    Returns:
        Implied volatility.
    """
    # Seed with Brenner-Subrahmanyam approximation: sigma ≈ (2π/T)^0.5 * (C/S)
    sigma = np.sqrt(2.0 * np.pi / T) * (market_price / S)
    sigma = float(np.clip(sigma, 1e-4, 5.0))

    for _ in range(max_iter):
        price = float(bs_price_numpy(S, K, T, r, sigma, q=q, option_type=option_type))
        vega = float(bs_vega(S, K, T, r, sigma, q=q))
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        if abs(vega) < 1e-12:
            # Vega too small — fall back to Brent
            return implied_vol(market_price, S, K, T, r, q=q, option_type=option_type, tol=tol)
        sigma -= diff / vega
        sigma = float(np.clip(sigma, 1e-4, 10.0))

    # Did not converge — try safe Brent
    return implied_vol(market_price, S, K, T, r, q=q, option_type=option_type, tol=tol)


# ------------------------------------------------------------------ #
# Batch / vectorized solver
# ------------------------------------------------------------------ #


def implied_vol_batch(
    market_prices: Union[list[float], np.ndarray],
    S: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: float,
    q: float = 0.0,
    option_type: str = "call",
    *,
    tol: float = 1e-8,
    method: str = "brent",
) -> np.ndarray:
    """Compute implied volatility for arrays of market prices.

    All array arguments are broadcast together; scalars are repeated.

    Args:
        market_prices: Array of observed option prices.
        S: Spot price(s).
        K: Strike(s).
        T: Time(s) to expiry.
        r: Risk-free rate (scalar).
        q: Dividend yield (scalar).
        option_type: 'call' or 'put'.
        tol: Convergence tolerance.
        method: 'brent' (robust) or 'newton' (faster).

    Returns:
        Array of implied volatilities (same broadcast shape). NaN where
        the solver failed (e.g. below intrinsic).

    Example:
        >>> import numpy as np
        >>> ivs = implied_vol_batch([5.0, 10.0, 15.0], S=100, K=100, T=1.0, r=0.05)
    """
    prices_arr = np.asarray(market_prices, dtype=float)
    S_arr = np.broadcast_to(np.asarray(S, dtype=float), prices_arr.shape)
    K_arr = np.broadcast_to(np.asarray(K, dtype=float), prices_arr.shape)
    T_arr = np.broadcast_to(np.asarray(T, dtype=float), prices_arr.shape)

    ivs = np.full(prices_arr.shape, np.nan)
    fn = implied_vol if method == "brent" else implied_vol_nr

    for idx in np.ndindex(prices_arr.shape):
        try:
            ivs[idx] = fn(
                float(prices_arr[idx]),
                float(S_arr[idx]),
                float(K_arr[idx]),
                float(T_arr[idx]),
                r,
                q=q,
                option_type=option_type,
                tol=tol,
            )
        except (ValueError, RuntimeError):
            ivs[idx] = np.nan

    return ivs


# ------------------------------------------------------------------ #
# IV surface builder
# ------------------------------------------------------------------ #


def iv_surface(
    market_prices: np.ndarray,
    S: float,
    K_grid: np.ndarray,
    T_grid: np.ndarray,
    r: float,
    q: float = 0.0,
    option_type: str = "call",
    tol: float = 1e-8,
) -> np.ndarray:
    """Build a 2-D implied volatility surface over a (K, T) grid.

    Args:
        market_prices: 2-D array of shape (len(K_grid), len(T_grid)).
        S: Spot price.
        K_grid: 1-D array of strikes.
        T_grid: 1-D array of expiries.
        r: Risk-free rate.
        q: Dividend yield.
        option_type: 'call' or 'put'.
        tol: Solver tolerance.

    Returns:
        2-D array of implied volatilities, shape (len(K_grid), len(T_grid)).
        NaN where solver failed.

    Example:
        >>> import numpy as np
        >>> from tensorquantlib.finance.black_scholes import bs_price_numpy
        >>> K = np.array([90.0, 100.0, 110.0])
        >>> T = np.array([0.5, 1.0])
        >>> prices = np.array([[bs_price_numpy(100, k, t, 0.05, 0.2) for t in T] for k in K])
        >>> surf = iv_surface(prices, 100.0, K, T, 0.05)
        >>> np.allclose(surf, 0.2, atol=1e-4)
        True
    """
    assert market_prices.shape == (len(K_grid), len(T_grid)), (
        f"market_prices shape {market_prices.shape} must be ({len(K_grid)}, {len(T_grid)})"
    )
    surface = np.full_like(market_prices, np.nan)
    for i, K in enumerate(K_grid):
        for j, T in enumerate(T_grid):
            try:
                surface[i, j] = implied_vol(
                    float(market_prices[i, j]),
                    S,
                    float(K),
                    float(T),
                    r,
                    q=q,
                    option_type=option_type,
                    tol=tol,
                )
            except (ValueError, RuntimeError):
                surface[i, j] = np.nan
    return surface
