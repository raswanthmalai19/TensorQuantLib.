"""
American option pricing via the Longstaff-Schwartz (2001) Least-Squares Monte Carlo (LSM) algorithm.

American options grant the holder the right to exercise early. LSM approximates
the continuation value at each time step using polynomial regression on simulated
paths, then decides whether early exercise is optimal.

This module provides:
    american_option_lsm  -- Single-asset American option price
    american_option_grid -- Price a grid of (S, K) or (S, T) combinations
    american_greeks      -- Delta, Gamma, Theta via finite-difference bumps on LSM

References:
    Longstaff, F.A. & Schwartz, E.S. (2001). Valuing American Options by Simulation:
    A Simple Least-Squares Approach. RFS 14(1), 113-147.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np


# ------------------------------------------------------------------ #
# Core LSM pricer
# ------------------------------------------------------------------ #

def american_option_lsm(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "put",
    *,
    n_paths: int = 100_000,
    n_steps: int = 252,
    basis_degree: int = 3,
    seed: Optional[int] = None,
    return_stderr: bool = False,
) -> Union[float, tuple[float, float]]:
    """Price an American option via Longstaff-Schwartz LSM.

    Simulates GBM paths and uses polynomial regression to estimate the
    continuation value at each exercise date, then takes the maximum of
    early exercise and continuation.

    Args:
        S: Current spot price.
        K: Strike price.
        T: Time to expiry (years).
        r: Risk-free rate.
        sigma: Volatility (constant).
        q: Continuous dividend yield (default 0).
        option_type: 'put' or 'call'.
        n_paths: Number of Monte Carlo paths (default 100k).
        n_steps: Number of exercise dates (default 252 = daily).
        basis_degree: Degree of Laguerre polynomial basis (default 3).
        seed: Random seed for reproducibility.
        return_stderr: If True, returns (price, stderr).

    Returns:
        American option price, or (price, stderr) if return_stderr=True.

    Example:
        >>> price = american_option_lsm(100.0, 100.0, 1.0, 0.05, 0.2, seed=42)
        >>> 3.0 < price < 8.0
        True
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    discount = np.exp(-r * dt)

    # Simulate all paths using antithetic variates for variance reduction
    half_paths = n_paths // 2
    z = rng.standard_normal((n_steps, half_paths))
    z = np.concatenate([z, -z], axis=1)  # antithetic

    # Build path matrix (shape: n_steps+1, n_paths)
    log_S = np.log(S) + np.cumsum(
        (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z,
        axis=0,
    )
    S_paths = np.vstack([
        np.full(n_paths, S),
        np.exp(log_S),
    ])  # shape: (n_steps+1, n_paths)

    # Payoff at each step
    if option_type == "put":
        payoffs = np.maximum(K - S_paths, 0.0)
    elif option_type == "call":
        payoffs = np.maximum(S_paths - K, 0.0)
    else:
        raise ValueError(f"option_type must be 'put' or 'call', got {option_type!r}")

    # Cash flows: start with terminal payoff
    cash_flows = payoffs[-1].copy()  # shape: (n_paths,)

    # Backward induction
    for t in range(n_steps - 1, 0, -1):
        # Only consider paths where early exercise is worthwhile (ITM paths)
        immediate = payoffs[t]
        itm = immediate > 0.0
        n_itm = int(np.sum(itm))

        if n_itm < basis_degree + 1:
            # Not enough ITM paths to regress — just discount
            cash_flows = cash_flows * discount
            continue

        S_itm = S_paths[t, itm]
        cf_itm = cash_flows[itm] * discount

        # Laguerre polynomial basis: [1, L0(S), L1(S), ..., L_{d-1}(S)]
        X = _laguerre_basis(S_itm, basis_degree)

        # Regress continuation value
        coeffs, _, _, _ = np.linalg.lstsq(X, cf_itm, rcond=None)
        continuation = X @ coeffs

        # Optimal stopping: exercise where immediate > continuation
        exercise = immediate[itm] > continuation

        # Update cash flows
        cash_flows = cash_flows * discount
        cash_flows[itm] = np.where(exercise, immediate[itm], cash_flows[itm])

    # Discount back one more step
    cash_flows = cash_flows * discount

    price = float(np.mean(cash_flows))
    stderr = float(np.std(cash_flows) / np.sqrt(n_paths))

    if return_stderr:
        return price, stderr
    return price


def _laguerre_basis(x: np.ndarray, degree: int) -> np.ndarray:
    """Compute generalised Laguerre polynomial basis up to given degree.

    Uses the normalised form as suggested by Longstaff & Schwartz (2001).
    Shape of output: (len(x), degree + 1).
    """
    n = len(x)
    X = np.ones((n, degree + 1))
    if degree >= 1:
        X[:, 1] = 1.0 - x
    if degree >= 2:
        X[:, 2] = 1.0 - 2.0 * x + 0.5 * x ** 2
    if degree >= 3:
        X[:, 3] = 1.0 - 3.0 * x + 1.5 * x ** 2 - x ** 3 / 6.0
    if degree >= 4:
        X[:, 4] = 1.0 - 4.0 * x + 3.0 * x ** 2 - (2.0 / 3.0) * x ** 3 + x ** 4 / 24.0
    if degree > 4:
        # Fall back to Vandermonde (polynomial) basis for higher degrees
        for d in range(5, degree + 1):
            X[:, d] = x ** d
    return X


# ------------------------------------------------------------------ #
# Vectorized grid pricer
# ------------------------------------------------------------------ #

def american_option_grid(
    S_grid: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "put",
    *,
    n_paths: int = 50_000,
    n_steps: int = 100,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Price American options over a grid of spot prices.

    Uses the same path set (bumped spot) to reduce variance across prices.

    Args:
        S_grid: 1-D array of spot prices.
        K, T, r, sigma, q, option_type: Option parameters.
        n_paths, n_steps, seed: Simulation parameters.

    Returns:
        1-D array of prices, same length as S_grid.

    Example:
        >>> import numpy as np
        >>> S_grid = np.linspace(80, 120, 5)
        >>> prices = american_option_grid(S_grid, K=100, T=1.0, r=0.05, sigma=0.2, seed=0)
        >>> prices.shape[0] == 5
        True
    """
    prices = np.array([
        float(american_option_lsm(
            float(s), K, T, r, sigma, q=q, option_type=option_type,
            n_paths=n_paths, n_steps=n_steps, seed=seed,
        ))
        for s in S_grid
    ])
    return prices


# ------------------------------------------------------------------ #
# Greeks via finite differences
# ------------------------------------------------------------------ #

def american_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "put",
    *,
    n_paths: int = 50_000,
    n_steps: int = 100,
    seed: Optional[int] = None,
    dS: float = 0.5,
    dT: float = 1.0 / 365.0,
    dsigma: float = 0.01,
) -> dict[str, float]:
    """Compute American option Greeks via finite differences on LSM prices.

    Args:
        S, K, T, r, sigma, q, option_type: Option parameters.
        n_paths, n_steps, seed: Simulation settings.
        dS: Bump size for Delta/Gamma (default 0.5).
        dT: Bump size for Theta (default 1/365 year).
        dsigma: Bump size for Vega (default 0.01).

    Returns:
        Dictionary with keys: 'delta', 'gamma', 'theta', 'vega'.

    Example:
        >>> g = american_greeks(100.0, 100.0, 1.0, 0.05, 0.2, seed=0)
        >>> 0.0 < abs(g['delta']) < 1.0
        True
    """
    kw = dict(q=q, option_type=option_type, n_paths=n_paths, n_steps=n_steps, seed=seed)

    p0 = float(american_option_lsm(S, K, T, r, sigma, **kw))  # type: ignore[call-arg, arg-type]
    pu = float(american_option_lsm(S + dS, K, T, r, sigma, **kw))  # type: ignore[call-arg, arg-type]
    pd = float(american_option_lsm(S - dS, K, T, r, sigma, **kw))  # type: ignore[call-arg, arg-type]

    delta = (pu - pd) / (2.0 * dS)
    gamma = (pu - 2.0 * p0 + pd) / (dS ** 2)

    pt = float(american_option_lsm(S, K, T - dT, r, sigma, **kw))  # type: ignore[call-arg, arg-type]
    theta = (pt - p0) / dT

    pv = float(american_option_lsm(S, K, T, r, sigma + dsigma, **kw))  # type: ignore[call-arg, arg-type]
    vega = (pv - p0) / dsigma

    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}
