"""Interest rate derivatives: swaptions, caps, and floors.

Pricing under the Black (1976) model for European swaptions and caplets.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Cap / Floor pricing (Black 76)
# ---------------------------------------------------------------------------

def black76_caplet(forward: float, strike: float, T: float, sigma: float,
                   df: float, notional: float = 1.0,
                   tau: float = 0.25) -> float:
    """Price a single caplet using the Black (1976) model.

    Parameters
    ----------
    forward : float
        Forward rate for the period.
    strike : float
        Cap strike rate.
    T : float
        Time to caplet expiry (option maturity).
    sigma : float
        Black implied volatility for the forward rate.
    df : float
        Discount factor to payment date.
    notional : float
        Notional amount.
    tau : float
        Day count fraction for the period.

    Returns
    -------
    float
        Caplet price.
    """
    if T <= 0:
        return max(forward - strike, 0.0) * tau * df * notional
    d1 = (np.log(forward / strike) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(df * tau * notional * (forward * norm.cdf(d1) - strike * norm.cdf(d2)))


def black76_floorlet(forward: float, strike: float, T: float, sigma: float,
                     df: float, notional: float = 1.0,
                     tau: float = 0.25) -> float:
    """Price a single floorlet using Black (1976).

    Parameters
    ----------
    forward, strike, T, sigma, df, notional, tau
        Same as :func:`black76_caplet`.

    Returns
    -------
    float
        Floorlet price.
    """
    if T <= 0:
        return max(strike - forward, 0.0) * tau * df * notional
    d1 = (np.log(forward / strike) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(df * tau * notional * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1)))


def cap_price(forwards: np.ndarray, strike: float, expiries: np.ndarray,
              sigma: float | np.ndarray, dfs: np.ndarray,
              notional: float = 1.0, tau: float = 0.25) -> float:
    """Price an interest rate cap (sum of caplets).

    Parameters
    ----------
    forwards : array, shape (n,)
        Forward rates for each period.
    strike : float
        Cap strike.
    expiries : array, shape (n,)
        Expiry time for each caplet.
    sigma : float or array
        Black vol (flat or per-caplet).
    dfs : array, shape (n,)
        Discount factors to each payment date.
    notional : float
        Notional.
    tau : float
        Day count fraction.

    Returns
    -------
    float
        Total cap price.
    """
    sigmas = np.broadcast_to(np.asarray(sigma, dtype=float), forwards.shape)
    total = 0.0
    for i in range(len(forwards)):
        total += black76_caplet(forwards[i], strike, expiries[i],
                                sigmas[i], dfs[i], notional, tau)
    return total


def floor_price(forwards: np.ndarray, strike: float, expiries: np.ndarray,
                sigma: float | np.ndarray, dfs: np.ndarray,
                notional: float = 1.0, tau: float = 0.25) -> float:
    """Price an interest rate floor (sum of floorlets).

    Parameters same as :func:`cap_price`.
    """
    sigmas = np.broadcast_to(np.asarray(sigma, dtype=float), forwards.shape)
    total = 0.0
    for i in range(len(forwards)):
        total += black76_floorlet(forwards[i], strike, expiries[i],
                                  sigmas[i], dfs[i], notional, tau)
    return total


# ---------------------------------------------------------------------------
# Swaption pricing (Black 76)
# ---------------------------------------------------------------------------

def swap_rate(dfs: np.ndarray, tau: float = 0.5) -> float:
    """Par swap rate from discount factors.

    S = (df_0 - df_n) / (tau * sum(df_i))

    Parameters
    ----------
    dfs : array, shape (n+1,)
        Discount factors at each payment date, including the start (dfs[0]).
    tau : float
        Day count fraction for each period.

    Returns
    -------
    float
        Par swap rate.
    """
    annuity = tau * np.sum(dfs[1:])
    if annuity < 1e-15:
        return 0.0
    return float((dfs[0] - dfs[-1]) / annuity)


def swaption_price(swap_r: float, strike: float, T_option: float,
                   sigma: float, annuity: float,
                   notional: float = 1.0,
                   payer: bool = True) -> float:
    """Price a European swaption using Black (1976).

    Parameters
    ----------
    swap_r : float
        Forward swap rate.
    strike : float
        Swaption strike.
    T_option : float
        Time to swaption expiry.
    sigma : float
        Black implied vol for the swap rate.
    annuity : float
        PV of the swap's fixed-leg annuity factor.
    notional : float
        Notional.
    payer : bool
        If True, price a payer swaption (right to pay fixed).
        If False, price a receiver swaption.

    Returns
    -------
    float
        Swaption price.
    """
    if T_option <= 0:
        if payer:
            return max(swap_r - strike, 0.0) * annuity * notional
        return max(strike - swap_r, 0.0) * annuity * notional

    d1 = (np.log(swap_r / strike) + 0.5 * sigma**2 * T_option) / (sigma * np.sqrt(T_option))
    d2 = d1 - sigma * np.sqrt(T_option)

    if payer:
        price = annuity * notional * (swap_r * norm.cdf(d1) - strike * norm.cdf(d2))
    else:
        price = annuity * notional * (strike * norm.cdf(-d2) - swap_r * norm.cdf(-d1))

    return float(price)


def swaption_parity(payer: float, receiver: float, swap_r: float,
                    strike: float, annuity: float,
                    notional: float = 1.0) -> float:
    """Check put-call parity for swaptions.

    Payer - Receiver = (S - K) * A * N

    Returns the parity residual (should be ~0).
    """
    expected = (swap_r - strike) * annuity * notional
    return float(payer - receiver - expected)
