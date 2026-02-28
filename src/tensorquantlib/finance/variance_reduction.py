"""
Variance reduction techniques for Monte Carlo option pricing.

This module provides drop-in variance-reduced Monte Carlo pricers
that remain compatible with the rest of TensorQuantLib.

Techniques:
    - Antithetic variates         (halves variance for symmetric payoffs)
    - Control variates            (using geometric average as control for arithmetic Asian)
    - Quasi-Monte Carlo (Sobol)   (faster convergence with low-discrepancy sequences)
    - Importance sampling         (shift drift to reduce rare-event variance)
    - Stratified sampling         (Gaussian stratification via quantile transform)

Each function mirrors the signature of the standard Monte Carlo pricers
but adds a ``method`` or ``technique`` argument.

References:
    Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering. Springer.
    Joe, S. & Kuo, F.Y. (2008). Constructing Sobol sequences with better two-dimensional projections.
    SIAM J. Sci. Comput. 30(5), 2635-2654.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy.stats import norm, qmc


# ------------------------------------------------------------------ #
# Antithetic variates — European Black-Scholes
# ------------------------------------------------------------------ #

def bs_price_antithetic(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    *,
    n_paths: int = 100_000,
    seed: Optional[int] = None,
    return_stderr: bool = False,
) -> Union[float, tuple[float, float]]:
    """European option price via MC with antithetic variates.

    For a standard GBM payoff, using the paired (z, -z) construction reduces
    variance by roughly 50% compared to crude Monte Carlo.

    Args:
        S, K, T, r, sigma, q, option_type: Standard Black-Scholes parameters.
        n_paths: Total number of paths (must be even; half are antithetic).
        seed: Random seed.
        return_stderr: If True, return (price, stderr).

    Returns:
        Price, or (price, stderr).

    Example:
        >>> p = bs_price_antithetic(100, 100, 1.0, 0.05, 0.2, seed=42)
        >>> 9.0 < p < 12.0
        True
    """
    half = n_paths // 2
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(half)
    z_full = np.concatenate([z, -z])

    S_T = S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z_full)
    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    discount = np.exp(-r * T)
    price = discount * float(np.mean(payoffs))
    stderr = discount * float(np.std(payoffs) / np.sqrt(n_paths))

    return (price, stderr) if return_stderr else price


# ------------------------------------------------------------------ #
# Control variates — Asian option (geometric as control)
# ------------------------------------------------------------------ #

def asian_price_cv(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    *,
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: Optional[int] = None,
    return_stderr: bool = False,
) -> Union[float, tuple[float, float]]:
    """Asian arithmetic-average option with geometric control variate.

    Uses the closed-form geometric Asian price (Kemna & Vorst 1990) as a
    control variate, exploiting the high correlation between arithmetic
    and geometric averages to dramatically reduce estimator variance.

    Args:
        S, K, T, r, sigma, q, option_type: Option parameters.
        n_paths: MC paths.
        n_steps: Time steps for averaging.
        seed: Random seed.
        return_stderr: Return (price, stderr) tuple.

    Returns:
        Asian option price (control-variate-adjusted), or (price, stderr).

    Example:
        >>> p = asian_price_cv(100, 100, 1.0, 0.05, 0.2, seed=42)
        >>> 5.0 < p < 12.0
        True
    """
    from tensorquantlib.finance.exotics import asian_geometric_price

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_steps, n_paths))
    log_increments = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    S_paths = S * np.exp(np.vstack([np.zeros(n_paths), np.cumsum(log_increments, axis=0)]))
    obs = S_paths[1:]  # exclude S_0

    arith_avg = obs.mean(axis=0)
    geo_avg = np.exp(np.log(obs).mean(axis=0))

    if option_type == "call":
        arith_payoffs = np.maximum(arith_avg - K, 0.0)
        geo_payoffs = np.maximum(geo_avg - K, 0.0)
    else:
        arith_payoffs = np.maximum(K - arith_avg, 0.0)
        geo_payoffs = np.maximum(K - geo_avg, 0.0)

    discount = np.exp(-r * T)
    geo_price_analytic = asian_geometric_price(S, K, T, r, sigma, q, option_type)

    # Optimal control variate coefficient: cov(arith, geo) / var(geo)
    cov_matrix = np.cov(arith_payoffs, geo_payoffs)
    if cov_matrix[1, 1] > 1e-12:
        c_star = -cov_matrix[0, 1] / cov_matrix[1, 1]
    else:
        c_star = 0.0

    cv_payoffs = arith_payoffs + c_star * (geo_payoffs - float(geo_price_analytic / discount))
    price = discount * float(np.mean(cv_payoffs))
    stderr = discount * float(np.std(cv_payoffs) / np.sqrt(n_paths))

    return (price, stderr) if return_stderr else price


# ------------------------------------------------------------------ #
# Quasi-Monte Carlo — Sobol sequence
# ------------------------------------------------------------------ #

def bs_price_qmc(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    *,
    n_paths: int = 65_536,  # power of 2 recommended for Sobol
    scramble: bool = True,
    seed: Optional[int] = None,
    return_stderr: bool = False,
) -> Union[float, tuple[float, float]]:
    """European option price via Quasi-Monte Carlo (1-D Sobol sequence).

    Low-discrepancy sequences converge at O(1/N) vs O(1/sqrt(N)) for
    pseudo-random MC, giving ~10-100x better accuracy for the same N.

    Requires scipy >= 1.7 for qmc.Sobol.

    Args:
        S, K, T, r, sigma, q, option_type: Black-Scholes parameters.
        n_paths: Number of QMC samples (ideally a power of 2).
        scramble: Use Owen scrambling (recommended, better uniformity).
        seed: Scrambling seed.
        return_stderr: Return (price, stderr).

    Returns:
        Price, or (price, stderr).

    Example:
        >>> p = bs_price_qmc(100, 100, 1.0, 0.05, 0.2, seed=42)
        >>> 9.0 < p < 12.0
        True
    """
    sampler = qmc.Sobol(d=1, scramble=scramble, seed=seed)
    u = sampler.random(n_paths).flatten()
    z = norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))

    S_T = S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    discount = np.exp(-r * T)
    price = discount * float(np.mean(payoffs))
    stderr = discount * float(np.std(payoffs) / np.sqrt(n_paths))

    return (price, stderr) if return_stderr else price


# ------------------------------------------------------------------ #
# Importance sampling — OTM options
# ------------------------------------------------------------------ #

def bs_price_importance(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    *,
    n_paths: int = 100_000,
    seed: Optional[int] = None,
    return_stderr: bool = False,
) -> Union[float, tuple[float, float]]:
    """European option price via importance sampling.

    Shifts the sampling distribution to centre around the region where
    the option pays off. Particularly effective for deep OTM options
    where crude MC needs huge N.

    The optimal drift shift is: mu* = log(K/S) / (sigma * sqrt(T)) - sigma * sqrt(T) / 2

    Args:
        S, K, T, r, sigma, q, option_type: Standard parameters.
        n_paths: Number of paths.
        seed: Random seed.
        return_stderr: Return (price, stderr).

    Returns:
        Price, or (price, stderr).

    Example:
        >>> p = bs_price_importance(100, 100, 1.0, 0.05, 0.2, seed=0)
        >>> 9.0 < p < 12.0
        True
    """
    rng = np.random.default_rng(seed)

    sqrt_T = np.sqrt(T)
    drift = (r - q - 0.5 * sigma ** 2) * T

    # Optimal IS shift: move the Brownian motion so that S_T = K on average.
    # Under the shifted measure Q̃, we draw  W̃ ~ N(mu_star, 1)  ↔  draw z~N(0,1),
    # set W̃ = z + mu_star.  Then S_T = S * exp(drift + sigma*sqrt_T*(z+mu_star)).
    # Radon-Nikodym derivative dP/dQ̃ = exp(-mu_star*z - 0.5*mu_star^2).
    if option_type == "call":
        mu_star = (np.log(K / S) - drift) / (sigma * sqrt_T)
    else:
        mu_star = -(np.log(K / S) - drift) / (sigma * sqrt_T)

    z = rng.standard_normal(n_paths)           # z ~ N(0,1) under Q̃
    W_tilde = z + mu_star                       # shifted Brownian increment
    S_T = S * np.exp(drift + sigma * sqrt_T * W_tilde)

    # Likelihood ratio  dP/dQ̃  (original measure / importance measure)
    lr = np.exp(-mu_star * z - 0.5 * mu_star ** 2)

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    weighted = payoffs * lr
    discount = np.exp(-r * T)
    price = discount * float(np.mean(weighted))
    stderr = discount * float(np.std(weighted) / np.sqrt(n_paths))

    return (price, stderr) if return_stderr else price


# ------------------------------------------------------------------ #
# Stratified sampling
# ------------------------------------------------------------------ #

def bs_price_stratified(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    *,
    n_paths: int = 100_000,
    n_strata: int = 100,
    seed: Optional[int] = None,
    return_stderr: bool = False,
) -> Union[float, tuple[float, float]]:
    """European option price via stratified sampling over the unit interval.

    Divides [0,1] into n_strata equal strata and draws one uniform sample
    from each, then maps via the quantile function. This reduces variance
    by eliminating within-stratum variability.

    Args:
        S, K, T, r, sigma, q, option_type: Parameters.
        n_paths: Total paths (rounded to nearest multiple of n_strata).
        n_strata: Number of strata.
        seed: Random seed.
        return_stderr: Return (price, stderr).

    Returns:
        Price, or (price, stderr).

    Example:
        >>> p = bs_price_stratified(100, 100, 1.0, 0.05, 0.2, seed=0)
        >>> 9.0 < p < 12.0
        True
    """
    rng = np.random.default_rng(seed)
    paths_per_stratum = max(1, n_paths // n_strata)
    total = paths_per_stratum * n_strata

    # Stratified uniform samples
    strata_bounds = np.linspace(0.0, 1.0, n_strata + 1)
    u = np.concatenate([
        rng.uniform(lo, hi, paths_per_stratum)
        for lo, hi in zip(strata_bounds[:-1], strata_bounds[1:])
    ])
    z = norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))

    S_T = S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    discount = np.exp(-r * T)
    price = discount * float(np.mean(payoffs))
    stderr = discount * float(np.std(payoffs) / np.sqrt(total))

    return (price, stderr) if return_stderr else price


# ------------------------------------------------------------------ #
# Variance reduction comparison utility
# ------------------------------------------------------------------ #

def compare_variance_reduction(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    n_paths: int = 50_000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compare all variance reduction methods for a European option.

    Returns a dict mapping method name -> {'price': ..., 'stderr': ..., 'vr_ratio': ...}
    where vr_ratio = stderr_crude_mc / stderr_method (higher is better).

    Args:
        S, K, T, r, sigma, q, option_type: Black-Scholes parameters.
        n_paths: Paths used for each method.
        seed: Random seed.

    Returns:
        Dictionary of results per method.

    Example:
        >>> results = compare_variance_reduction(100, 100, 1.0, 0.05, 0.2, seed=0)
        >>> 'crude_mc' in results and 'antithetic' in results
        True
    """
    from tensorquantlib.finance.basket import simulate_basket  # for crude MC

    # Crude MC reference
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    S_T = S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
    if option_type == "call":
        pf = np.maximum(S_T - K, 0.0)
    else:
        pf = np.maximum(K - S_T, 0.0)
    d = np.exp(-r * T)
    crude_price = d * float(np.mean(pf))
    crude_stderr = d * float(np.std(pf) / np.sqrt(n_paths))

    methods: dict[str, dict[str, float]] = {}
    methods["crude_mc"] = {"price": crude_price, "stderr": crude_stderr, "vr_ratio": 1.0}

    for name, fn in [
        ("antithetic", bs_price_antithetic),
        ("qmc_sobol", bs_price_qmc),
        ("importance_sampling", bs_price_importance),
        ("stratified", bs_price_stratified),
    ]:
        try:
            price, stderr = fn(  # type: ignore[call-arg]
                S, K, T, r, sigma, q=q, option_type=option_type,
                n_paths=n_paths, seed=seed, return_stderr=True,
            )
            vr_ratio = crude_stderr / stderr if stderr > 1e-12 else float("inf")
            methods[name] = {"price": float(price), "stderr": float(stderr), "vr_ratio": float(vr_ratio)}
        except Exception as e:
            methods[name] = {"price": float("nan"), "stderr": float("nan"), "vr_ratio": float("nan"), "error": str(e)}  # type: ignore[assignment]

    return methods
