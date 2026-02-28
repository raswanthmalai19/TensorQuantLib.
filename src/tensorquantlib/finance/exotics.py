"""
Exotic options: Asian, digital (binary), and barrier options.

Provides both analytic closed-forms (where available) and Monte Carlo pricing
for the following option types built on log-normal GBM dynamics:

    Asian (arithmetic average):
        asian_price_mc       -- Monte Carlo arithmetic Asian call/put
        asian_geometric_price -- Analytic geometric-average Asian (closed-form)

    Digital (binary):
        digital_price        -- Analytic cash-or-nothing and asset-or-nothing
        digital_price_mc     -- Monte Carlo digital price (validation)

    Barrier options (single barrier, European):
        barrier_price        -- Analytic single-barrier option (Reiner-Rubinstein)
        barrier_price_mc     -- Monte Carlo barrier option price

All analytic formulas assume GBM with constant parameters.

References:
    Kemna & Vorst (1990). A Pricing Method for Options Based on Average Asset Values.
    Journal of Banking and Finance, 14(1), 113-129.

    Rubinstein, M. & Reiner, E. (1991). Breaking Down the Barriers. Risk 4(8), 28-35.

    Reiner, E. (1992). Quanto Mechanics. Risk 5(3), 59-63.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy.stats import norm


# ------------------------------------------------------------------ #
# Helper
# ------------------------------------------------------------------ #

def _gbm_paths(
    S: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    n_paths: int,
    n_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate GBM paths. Returns shape (n_steps+1, n_paths)."""
    dt = T / n_steps
    z = rng.standard_normal((n_steps, n_paths))
    log_increments = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    log_S = np.vstack([
        np.full(n_paths, np.log(S)),
        np.log(S) + np.cumsum(log_increments, axis=0),
    ])
    return np.exp(log_S)


# ================================================================== #
# ASIAN OPTIONS
# ================================================================== #

def asian_price_mc(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    average_type: str = "arithmetic",
    *,
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: Optional[int] = None,
    return_stderr: bool = False,
) -> Union[float, tuple[float, float]]:
    """Price an Asian average-rate option by Monte Carlo.

    The payoff is based on the average of the asset price over [0, T]:
        Call: max(avg(S) - K, 0) * exp(-rT)
        Put:  max(K - avg(S), 0) * exp(-rT)

    Args:
        S: Spot price.
        K: Strike.
        T: Time to expiry (years).
        r: Risk-free rate.
        sigma: Volatility.
        q: Dividend yield.
        option_type: 'call' or 'put'.
        average_type: 'arithmetic' or 'geometric'.
        n_paths: Monte Carlo paths.
        n_steps: Averaging time steps.
        seed: Random seed.
        return_stderr: If True, return (price, stderr).

    Returns:
        Asian option price, or (price, stderr).

    Example:
        >>> price = asian_price_mc(100, 100, 1.0, 0.05, 0.2, seed=0)
        >>> 5.0 < price < 12.0
        True
    """
    rng = np.random.default_rng(seed)
    paths = _gbm_paths(S, T, r, sigma, q, n_paths, n_steps, rng)

    # Exclude time-0 from average (average over [dt, T])
    obs = paths[1:]  # shape: (n_steps, n_paths)

    if average_type == "arithmetic":
        avg = obs.mean(axis=0)
    elif average_type == "geometric":
        avg = np.exp(np.log(obs).mean(axis=0))
    else:
        raise ValueError(f"average_type must be 'arithmetic' or 'geometric', got {average_type!r}")

    if option_type == "call":
        payoffs = np.maximum(avg - K, 0.0)
    else:
        payoffs = np.maximum(K - avg, 0.0)

    discount = np.exp(-r * T)
    price = discount * float(np.mean(payoffs))
    stderr = discount * float(np.std(payoffs) / np.sqrt(n_paths))

    return (price, stderr) if return_stderr else price


def asian_geometric_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
) -> float:
    """Closed-form price for continuous geometric-average Asian option (Kemna & Vorst 1990).

    Applicable to continuous monitoring (limit of n_steps → ∞).

    Args:
        S, K, T, r, sigma, q: Standard Black-Scholes inputs.
        option_type: 'call' or 'put'.

    Returns:
        Geometric Asian option price.

    Example:
        >>> p = asian_geometric_price(100, 100, 1.0, 0.05, 0.2)
        >>> 5.0 < p < 12.0
        True
    """
    # Adjusted parameters for geometric average
    sigma_geo = sigma / np.sqrt(3.0)
    b = 0.5 * (r - q - sigma ** 2 / 6.0)

    d1 = (np.log(S / K) + (b + 0.5 * sigma_geo ** 2) * T) / (sigma_geo * np.sqrt(T))
    d2 = d1 - sigma_geo * np.sqrt(T)

    if option_type == "call":
        price = S * np.exp((b - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp((b - r) * T) * norm.cdf(-d1)

    return float(price)


# ================================================================== #
# DIGITAL (BINARY) OPTIONS
# ================================================================== #

def digital_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    payoff_type: str = "cash",
    payoff_amount: float = 1.0,
) -> float:
    """Analytic Black-Scholes price for a digital (binary) option.

    Args:
        S: Spot price.
        K: Strike.
        T: Time to expiry (years).
        r: Risk-free rate.
        sigma: Volatility.
        q: Dividend yield.
        option_type: 'call' (pays if S_T > K) or 'put' (pays if S_T < K).
        payoff_type: 'cash' (fixed cash) or 'asset' (deliver asset if triggered).
        payoff_amount: Size of the cash payment if payoff_type='cash' (default 1.0).

    Returns:
        Digital option price.

    Example:
        >>> p = digital_price(100, 100, 1.0, 0.05, 0.2, payoff_type='cash')
        >>> 0.0 < p < 1.0
        True
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if payoff_type == "cash":
        # Cash-or-nothing: pays payoff_amount if in the money at expiry
        if option_type == "call":
            price = payoff_amount * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = payoff_amount * np.exp(-r * T) * norm.cdf(-d2)
    elif payoff_type == "asset":
        # Asset-or-nothing: delivers the asset if in the money
        if option_type == "call":
            price = S * np.exp(-q * T) * norm.cdf(d1)
        else:
            price = S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError(f"payoff_type must be 'cash' or 'asset', got {payoff_type!r}")

    return float(price)


def digital_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    payoff_type: str = "cash",
    payoff_amount: float = 1.0,
) -> dict[str, float]:
    """Analytic Black-Scholes Greeks for digital options.

    Returns delta, gamma, vega, theta, rho.

    Example:
        >>> g = digital_greeks(100, 100, 1.0, 0.05, 0.2)
        >>> 'delta' in g and 'gamma' in g
        True
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    phi_d2 = float(norm.pdf(d2))
    phi_d1 = float(norm.pdf(d1))

    if payoff_type == "cash":
        sign = 1.0 if option_type == "call" else -1.0
        factor = payoff_amount * np.exp(-r * T)
        delta = sign * factor * phi_d2 / (S * sigma * np.sqrt(T))
        gamma = -sign * factor * phi_d2 * d1 / (S ** 2 * sigma ** 2 * T)
        vega = -sign * factor * phi_d2 * d1 / sigma
        theta = (sign * factor * r * float(norm.cdf(sign * d2)) +
                 sign * factor * phi_d2 * (d1 / (2 * T) - r / (sigma * np.sqrt(T))))
        rho = -T * float(digital_price(S, K, T, r, sigma, q, option_type, payoff_type, payoff_amount))
    else:
        # Asset-or-nothing greeks
        sign = 1.0 if option_type == "call" else -1.0
        factor = S * np.exp(-q * T)
        ncdf_d1 = float(norm.cdf(sign * d1))
        delta = np.exp(-q * T) * (ncdf_d1 + sign * phi_d1 / (sigma * np.sqrt(T)))
        gamma = sign * np.exp(-q * T) * phi_d1 * (1.0 / (S * sigma * np.sqrt(T))) * (1.0 - d2 / (sigma * np.sqrt(T)))
        vega = sign * factor * phi_d1 * (np.sqrt(T) - d1 / sigma)
        theta = float("nan")  # complex expression omitted here; use FD
        rho = float("nan")

    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega), "theta": float(theta), "rho": float(rho)}


def digital_price_mc(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    payoff_type: str = "cash",
    payoff_amount: float = 1.0,
    *,
    n_paths: int = 100_000,
    seed: Optional[int] = None,
    return_stderr: bool = False,
) -> Union[float, tuple[float, float]]:
    """Monte Carlo price for a digital option (validation).

    Example:
        >>> p = digital_price_mc(100, 100, 1.0, 0.05, 0.2, seed=0)
        >>> 0.0 < p < 1.0
        True
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    S_T = S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)

    if option_type == "call":
        triggered = S_T > K
    else:
        triggered = S_T < K

    if payoff_type == "cash":
        payoffs = np.where(triggered, payoff_amount, 0.0)
    else:
        payoffs = np.where(triggered, S_T, 0.0)

    discount = np.exp(-r * T)
    price = discount * float(np.mean(payoffs))
    stderr = discount * float(np.std(payoffs) / np.sqrt(n_paths))

    return (price, stderr) if return_stderr else price


# ================================================================== #
# BARRIER OPTIONS
# ================================================================== #

def _bs_call(S: float, K: float, T: float, r: float, b: float, sigma: float) -> float:
    """Generalised Black-Scholes call price with cost-of-carry b = r - q."""
    if T <= 0 or K <= 0 or S <= 0:
        return max(S - K, 0.0)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (b + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return float(S * np.exp((b - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def _bs_put(S: float, K: float, T: float, r: float, b: float, sigma: float) -> float:
    """Generalised Black-Scholes put price with cost-of-carry b = r - q."""
    if T <= 0 or K <= 0 or S <= 0:
        return max(K - S, 0.0)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (b + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return float(-S * np.exp((b - r) * T) * norm.cdf(-d1) + K * np.exp(-r * T) * norm.cdf(-d2))


def barrier_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    barrier: float,
    barrier_type: str,
    q: float = 0.0,
    option_type: str = "call",
    rebate: float = 0.0,
) -> float:
    """Analytic price for European single-barrier options (Rubinstein-Reiner 1991).

    Supports all 8 standard barrier option types:

        'down-and-in'  call/put  -- activated if S crosses H from above
        'down-and-out' call/put  -- extinguished if S crosses H from above
        'up-and-in'    call/put  -- activated if S crosses H from below
        'up-and-out'   call/put  -- extinguished if S crosses H from below

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry.
        r: Risk-free rate.
        sigma: Volatility.
        barrier: Barrier level H.
        barrier_type: One of {'down-and-in', 'down-and-out', 'up-and-in', 'up-and-out'}.
        q: Dividend yield.
        option_type: 'call' or 'put'.
        rebate: Cash rebate paid if barrier is not hit (for out options).

    Returns:
        Barrier option price.

    Raises:
        ValueError: If barrier_type or option_type are invalid.

    Example:
        >>> p = barrier_price(100, 100, 1.0, 0.05, 0.2, barrier=90, barrier_type='down-and-out')
        >>> p > 0
        True
    """
    valid_barrier_types = {"down-and-in", "down-and-out", "up-and-in", "up-and-out"}
    if barrier_type not in valid_barrier_types:
        raise ValueError(f"barrier_type must be one of {valid_barrier_types}, got {barrier_type!r}")
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    H = float(barrier)
    b = r - q  # cost-of-carry
    sqrt_T = np.sqrt(T)
    v = sigma
    v2 = v * v

    # Vanilla prices
    c = _bs_call(S, K, T, r, b, v)
    p = _bs_put(S, K, T, r, b, v)

    # Haug/Rubinstein-Reiner notation (following FinancePy fx_barrier_option.py exactly)
    # ll = (b + sigma^2/2) / sigma^2
    # y  = log(H^2/(S*K))/(sigma*sqrt(T)) + ll*sigma*sqrt(T)
    # x1 = log(S/H)/(sigma*sqrt(T)) + ll*sigma*sqrt(T)
    # y1 = log(H/S)/(sigma*sqrt(T)) + ll*sigma*sqrt(T)
    sigma_rt = v * sqrt_T
    ll = (b + 0.5 * v2) / v2
    y  = np.log(H * H / (S * K)) / sigma_rt + ll * sigma_rt
    x1 = np.log(S / H)           / sigma_rt + ll * sigma_rt
    y1 = np.log(H / S)           / sigma_rt + ll * sigma_rt

    dq = np.exp((b - r) * T)   # S discount factor (= exp(-q*T))
    df = np.exp(-r * T)         # K discount factor
    h_over_s = H / S
    pow_ll   = h_over_s ** (2.0 * ll)
    pow_ll2  = h_over_s ** (2.0 * ll - 2.0)

    N = norm.cdf

    def c_di() -> float:
        """Down-and-in call, H <= K."""
        return (
            S * dq * pow_ll * N(y)
            - K * df * pow_ll2 * N(y - sigma_rt)
        )

    def c_di_H_gt_K() -> float:
        """Down-and-in call, H > K."""
        return (
            S * dq * N(x1) - K * df * N(x1 - sigma_rt)
            - S * dq * pow_ll * (N(-y) - N(-y1))
            + K * df * pow_ll2 * (N(-y + sigma_rt) - N(-y1 + sigma_rt))
        )

    def c_uo() -> float:
        """Up-and-out call, H > K (the only meaningful case for up-out call)."""
        return (
            S * dq * N(x1) - K * df * N(x1 - sigma_rt)
            - S * dq * pow_ll * N(y1)
            + K * df * pow_ll2 * N(y1 - sigma_rt)
        )

    def c_ui() -> float:
        """Up-and-in call, H >= K."""
        return (
            S * dq * N(x1) - K * df * N(x1 - sigma_rt)
            - S * dq * pow_ll * (N(-y) - N(-y1))
            + K * df * pow_ll2 * (N(-y + sigma_rt) - N(-y1 + sigma_rt))
        )

    def p_ui() -> float:
        """Up-and-in put, H >= K."""
        return (
            -S * dq * pow_ll * N(-y)
            + K * df * pow_ll2 * N(-y + sigma_rt)
        )

    def p_ui_H_lt_K() -> float:
        """Up-and-in put, H < K."""
        return (
            -S * dq * N(-x1) + K * df * N(-x1 + sigma_rt)
            + S * dq * pow_ll * (N(y) - N(y1))
            - K * df * pow_ll2 * (N(y - sigma_rt) - N(y1 - sigma_rt))
        )

    def p_di() -> float:
        """Down-and-in put, H < K."""
        return (
            -S * dq * N(-x1) + K * df * N(-x1 + sigma_rt)
            + S * dq * pow_ll * (N(y) - N(y1))
            - K * df * pow_ll2 * (N(y - sigma_rt) - N(y1 - sigma_rt))
        )

    # Main dispatch
    if option_type == "call":
        if barrier_type == "down-and-in":
            price = c_di() if H <= K else c_di_H_gt_K()
        elif barrier_type == "down-and-out":
            price = c - (c_di() if H <= K else c_di_H_gt_K())
        elif barrier_type == "up-and-in":
            if H >= K:
                price = c_ui()
            else:
                price = c  # barrier is below strike: always knocked in already
        else:  # up-and-out
            if H >= K:
                price = c - c_ui()
            else:
                price = 0.0  # barrier <= strike: call knocked out before it can pay
    else:  # put
        if barrier_type == "up-and-in":
            price = p_ui() if H >= K else p_ui_H_lt_K()
        elif barrier_type == "up-and-out":
            price = p - (p_ui() if H >= K else p_ui_H_lt_K())
        elif barrier_type == "down-and-in":
            if H < K:
                price = p_di()
            else:  # H >= K: barrier is at or above strike, always knocked in
                price = p
        else:  # down-and-out
            if H < K:
                price = p - p_di()
            else:
                price = 0.0

    return float(max(price, 0.0))


def _d1(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    return _d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


def _indicator_hit(S: float, H: float, barrier_type: str, T: float, r: float, sigma: float, q: float) -> float:
    """Probability of hitting the barrier (approximation via reflection principle)."""
    mu = (r - q - 0.5 * sigma ** 2) / (sigma ** 2)
    x = np.log(H / S) / (sigma * np.sqrt(T))
    if "down" in barrier_type:
        return float(norm.cdf(-x + mu * sigma * np.sqrt(T)) + (H / S) ** (2 * mu) * norm.cdf(-x - mu * sigma * np.sqrt(T)))
    else:
        return float(norm.cdf(x - mu * sigma * np.sqrt(T)) + (H / S) ** (2 * mu) * norm.cdf(x + mu * sigma * np.sqrt(T)))


def barrier_price_mc(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    barrier: float,
    barrier_type: str,
    q: float = 0.0,
    option_type: str = "call",
    rebate: float = 0.0,
    *,
    n_paths: int = 200_000,
    n_steps: int = 252,
    seed: Optional[int] = None,
    return_stderr: bool = False,
) -> Union[float, tuple[float, float]]:
    """Monte Carlo price for a European single-barrier option.

    Args:
        S, K, T, r, sigma, q: Standard parameters.
        barrier: Barrier level.
        barrier_type: 'down-and-in', 'down-and-out', 'up-and-in', 'up-and-out'.
        option_type: 'call' or 'put'.
        rebate: Rebate paid when knocked out/never knocked in.
        n_paths, n_steps, seed: Simulation parameters.
        return_stderr: Return (price, stderr) if True.

    Returns:
        Price, or (price, stderr).

    Example:
        >>> p = barrier_price_mc(100, 100, 1.0, 0.05, 0.2, barrier=90, barrier_type='down-and-out', seed=0)
        >>> p > 0
        True
    """
    rng = np.random.default_rng(seed)
    paths = _gbm_paths(S, T, r, sigma, q, n_paths, n_steps, rng)
    H = barrier

    # Track barrier crossing
    if "down" in barrier_type:
        crossed = np.any(paths <= H, axis=0)  # shape: (n_paths,)
    else:
        crossed = np.any(paths >= H, axis=0)

    S_T = paths[-1]
    if option_type == "call":
        payoff_vanilla = np.maximum(S_T - K, 0.0)
    else:
        payoff_vanilla = np.maximum(K - S_T, 0.0)

    if "out" in barrier_type:
        # Knocked out when barrier is crossed
        payoffs = np.where(crossed, rebate, payoff_vanilla)
    else:
        # Knocked in: only alive when barrier was crossed
        payoffs = np.where(crossed, payoff_vanilla, rebate)

    discount = np.exp(-r * T)
    price = discount * float(np.mean(payoffs))
    stderr = discount * float(np.std(payoffs) / np.sqrt(n_paths))

    return (price, stderr) if return_stderr else price


# ---------------------------------------------------------------------------
# Lookback options
# ---------------------------------------------------------------------------

def lookback_fixed_analytic(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
) -> float:
    """Analytic fixed-strike lookback option price (Goldman-Sosin-Gatto 1979).

    For a fixed-strike lookback call, the payoff is max(S_max - K, 0).
    For a fixed-strike lookback put, the payoff is max(K - S_min, 0).

    Parameters
    ----------
    S, K, T, r, sigma, q, option_type : standard option parameters.

    Returns
    -------
    float
        Option price.
    """
    from scipy.stats import norm

    b = r - q  # cost of carry
    s2 = sigma ** 2

    if option_type == "call":
        d1 = (np.log(S / K) + (b + 0.5 * s2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        price = (
            S * np.exp((b - r) * T) * norm.cdf(d1)
            - K * np.exp(-r * T) * norm.cdf(d2)
            + S * np.exp(-r * T) * (s2 / (2.0 * b)) * (
                -(S / K) ** (-2.0 * b / s2) * norm.cdf(d1 - 2.0 * b * np.sqrt(T) / sigma)
                + np.exp(b * T) * norm.cdf(d1)
            )
        )
    else:
        d1 = (np.log(S / K) + (b + 0.5 * s2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        price = (
            K * np.exp(-r * T) * norm.cdf(-d2)
            - S * np.exp((b - r) * T) * norm.cdf(-d1)
            + S * np.exp(-r * T) * (s2 / (2.0 * b)) * (
                (S / K) ** (-2.0 * b / s2) * norm.cdf(-d1 + 2.0 * b * np.sqrt(T) / sigma)
                - np.exp(b * T) * norm.cdf(-d1)
            )
        )

    return float(price)


def lookback_floating_analytic(
    S: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
) -> float:
    """Analytic floating-strike lookback option price.

    At inception S_min = S_max = S.

    Returns
    -------
    float
        Option price.
    """
    from scipy.stats import norm

    b = r - q
    s2 = sigma ** 2
    sqT = sigma * np.sqrt(T)
    a1 = (b + 0.5 * s2) * T / sqT
    a2 = a1 - sqT

    if option_type == "call":
        if abs(b) < 1e-12:
            price = S * sqT * (2.0 * norm.pdf(a1) + a1 * (2.0 * norm.cdf(a1) - 1.0))
        else:
            price = (
                S * np.exp((b - r) * T) * norm.cdf(a1)
                - S * np.exp(-r * T) * norm.cdf(a2)
                + S * np.exp(-r * T) * (s2 / (2.0 * b)) * (
                    np.exp(b * T) * norm.cdf(a1) - norm.cdf(a2)
                )
            )
    else:
        if abs(b) < 1e-12:
            price = S * sqT * (2.0 * norm.pdf(a1) - a1 * (2.0 * norm.cdf(a1) - 1.0))
        else:
            price = (
                -S * np.exp((b - r) * T) * norm.cdf(-a1)
                + S * np.exp(-r * T) * norm.cdf(-a2)
                + S * np.exp(-r * T) * (s2 / (2.0 * b)) * (
                    -np.exp(b * T) * norm.cdf(-a1) + norm.cdf(-a2)
                )
            )

    return float(max(price, 0.0))


def lookback_price_mc(
    S: float,
    K: float | None,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    strike_type: str = "fixed",
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int | None = None,
) -> tuple[float, float]:
    """Monte Carlo lookback option price.

    Returns
    -------
    tuple
        (price, standard_error)
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma ** 2) * dt
    vol = sigma * np.sqrt(dt)

    Z = rng.standard_normal((n_paths, n_steps))
    log_S = np.log(S) + np.cumsum(drift + vol * Z, axis=1)
    paths = np.exp(log_S)
    paths = np.concatenate([np.full((n_paths, 1), S), paths], axis=1)

    S_T = paths[:, -1]
    S_max = np.max(paths, axis=1)
    S_min = np.min(paths, axis=1)

    if strike_type == "fixed":
        if K is None:
            raise ValueError("K required for fixed-strike lookback")
        if option_type == "call":
            payoffs = np.maximum(S_max - K, 0.0)
        else:
            payoffs = np.maximum(K - S_min, 0.0)
    else:
        if option_type == "call":
            payoffs = S_T - S_min
        else:
            payoffs = S_max - S_T

    discount = np.exp(-r * T)
    price = discount * float(np.mean(payoffs))
    stderr = discount * float(np.std(payoffs) / np.sqrt(n_paths))
    return price, stderr


# ---------------------------------------------------------------------------
# Cliquet (ratchet) options
# ---------------------------------------------------------------------------

def cliquet_price_mc(
    S: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    n_periods: int = 4,
    cap: float | None = None,
    floor: float | None = None,
    global_cap: float | None = None,
    global_floor: float | None = None,
    n_paths: int = 100_000,
    n_steps_per_period: int = 63,
    seed: int | None = None,
) -> tuple[float, float]:
    """Monte Carlo cliquet (ratchet) option pricer.

    Returns
    -------
    tuple
        (price, standard_error)
    """
    rng = np.random.default_rng(seed)
    dt = T / (n_periods * n_steps_per_period)
    drift = (r - q - 0.5 * sigma ** 2) * dt
    vol = sigma * np.sqrt(dt)

    total_returns = np.zeros(n_paths)
    S_start = np.full(n_paths, S)

    for _ in range(n_periods):
        log_S = np.log(S_start)
        for _step in range(n_steps_per_period):
            Z = rng.standard_normal(n_paths)
            log_S = log_S + drift + vol * Z
        S_end = np.exp(log_S)
        period_return = (S_end - S_start) / S_start

        if cap is not None:
            period_return = np.minimum(period_return, cap)
        if floor is not None:
            period_return = np.maximum(period_return, floor)

        total_returns += period_return
        S_start = S_end

    if global_cap is not None:
        total_returns = np.minimum(total_returns, global_cap)
    if global_floor is not None:
        total_returns = np.maximum(total_returns, global_floor)

    payoffs = S * np.maximum(total_returns, 0.0)

    discount = np.exp(-r * T)
    price = discount * float(np.mean(payoffs))
    stderr = discount * float(np.std(payoffs) / np.sqrt(n_paths))
    return price, stderr


# ---------------------------------------------------------------------------
# Rainbow options (best-of / worst-of)
# ---------------------------------------------------------------------------

def rainbow_price_mc(
    spots: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigmas: np.ndarray,
    corr: np.ndarray,
    q: np.ndarray | None = None,
    option_type: str = "call",
    rainbow_type: str = "best-of",
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int | None = None,
) -> tuple[float, float]:
    """Monte Carlo rainbow option pricer (best-of / worst-of).

    Returns
    -------
    tuple
        (price, standard_error)
    """
    spots = np.asarray(spots, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    corr = np.asarray(corr, dtype=float)
    n_assets = len(spots)

    q_arr = np.zeros(n_assets) if q is None else np.asarray(q, dtype=float)

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    L = np.linalg.cholesky(corr)

    log_S = np.tile(np.log(spots), (n_paths, 1))
    for _step in range(n_steps):
        Z = rng.standard_normal((n_paths, n_assets))
        Z_corr = Z @ L.T
        drift = (r - q_arr - 0.5 * sigmas ** 2) * dt
        vol = sigmas * np.sqrt(dt) * Z_corr
        log_S += drift + vol

    S_T = np.exp(log_S)

    if rainbow_type == "best-of":
        selected = np.max(S_T, axis=1)
    else:
        selected = np.min(S_T, axis=1)

    if option_type == "call":
        payoffs = np.maximum(selected - K, 0.0)
    else:
        payoffs = np.maximum(K - selected, 0.0)

    discount = np.exp(-r * T)
    price = discount * float(np.mean(payoffs))
    stderr = discount * float(np.std(payoffs) / np.sqrt(n_paths))
    return price, stderr
