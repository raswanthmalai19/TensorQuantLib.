"""FX options: Garman-Kohlhagen and Quanto models.

Implements FX-specific option pricing:
- Garman-Kohlhagen (1983): Black-Scholes adapted for FX with foreign rate
- Quanto options: options on foreign assets settled in domestic currency
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Garman-Kohlhagen model
# ---------------------------------------------------------------------------

def garman_kohlhagen(
    S: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """Garman-Kohlhagen FX option pricing model.

    This is the Black-Scholes formula adapted for FX, where the foreign
    risk-free rate acts as a continuous dividend yield.

    Parameters
    ----------
    S : float
        Spot FX rate (domestic per foreign).
    K : float
        Strike FX rate.
    T : float
        Time to expiry.
    r_d : float
        Domestic risk-free rate.
    r_f : float
        Foreign risk-free rate.
    sigma : float
        FX volatility.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    float
        Option price in domestic currency.
    """
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * np.exp(-r_f * T) * norm.cdf(d1) - K * np.exp(-r_d * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r_d * T) * norm.cdf(-d2) - S * np.exp(-r_f * T) * norm.cdf(-d1)

    return float(price)


def gk_greeks(
    S: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    sigma: float,
    option_type: str = "call",
) -> dict[str, float]:
    """Compute Garman-Kohlhagen Greeks.

    Returns
    -------
    dict
        {'delta': ..., 'gamma': ..., 'vega': ..., 'theta': ..., 'rho_d': ..., 'rho_f': ...}
    """
    sqT = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / sqT
    d2 = d1 - sqT

    exp_rf = np.exp(-r_f * T)
    exp_rd = np.exp(-r_d * T)

    # Gamma is the same for calls and puts
    gamma = exp_rf * norm.pdf(d1) / (S * sqT)

    # Vega is the same for calls and puts
    vega = S * exp_rf * norm.pdf(d1) * np.sqrt(T)

    if option_type == "call":
        delta = exp_rf * norm.cdf(d1)
        theta = (
            -S * exp_rf * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
            + r_f * S * exp_rf * norm.cdf(d1)
            - r_d * K * exp_rd * norm.cdf(d2)
        )
        rho_d = K * T * exp_rd * norm.cdf(d2)
        rho_f = -S * T * exp_rf * norm.cdf(d1)
    else:
        delta = exp_rf * (norm.cdf(d1) - 1.0)
        theta = (
            -S * exp_rf * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
            - r_f * S * exp_rf * norm.cdf(-d1)
            + r_d * K * exp_rd * norm.cdf(-d2)
        )
        rho_d = -K * T * exp_rd * norm.cdf(-d2)
        rho_f = S * T * exp_rf * norm.cdf(-d1)

    return {
        'delta': float(delta),
        'gamma': float(gamma),
        'vega': float(vega),
        'theta': float(theta),
        'rho_d': float(rho_d),
        'rho_f': float(rho_f),
    }


def fx_forward(
    S: float,
    r_d: float,
    r_f: float,
    T: float,
) -> float:
    """FX forward rate via covered interest rate parity.

    F = S * exp((r_d - r_f) * T)

    Returns
    -------
    float
        Forward FX rate.
    """
    return float(S * np.exp((r_d - r_f) * T))


# ---------------------------------------------------------------------------
# Quanto option
# ---------------------------------------------------------------------------

def quanto_option(
    S: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    sigma_s: float,
    sigma_fx: float,
    rho: float,
    fx_rate: float,
    option_type: str = "call",
) -> float:
    """Quanto option pricing (option on foreign asset, settled in domestic currency).

    The quanto adjustment modifies the drift of the foreign asset due to
    the correlation between the asset and the FX rate.

    Parameters
    ----------
    S : float
        Spot price of the foreign asset (in foreign currency).
    K : float
        Strike price (in foreign currency).
    T : float
        Time to expiry.
    r_d : float
        Domestic risk-free rate.
    r_f : float
        Foreign risk-free rate.
    sigma_s : float
        Volatility of the foreign asset.
    sigma_fx : float
        Volatility of the FX rate.
    rho : float
        Correlation between the asset and FX rate.
    fx_rate : float
        Fixed FX rate at which payoff is converted.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    float
        Quanto option price in domestic currency.
    """
    # Quanto-adjusted drift rate
    r_q = r_d - rho * sigma_s * sigma_fx

    d1 = (np.log(S / K) + (r_q + 0.5 * sigma_s ** 2) * T) / (sigma_s * np.sqrt(T))
    d2 = d1 - sigma_s * np.sqrt(T)

    if option_type == "call":
        price = fx_rate * np.exp(-r_d * T) * (
            S * np.exp(r_q * T) * norm.cdf(d1) - K * norm.cdf(d2)
        )
    else:
        price = fx_rate * np.exp(-r_d * T) * (
            K * norm.cdf(-d2) - S * np.exp(r_q * T) * norm.cdf(-d1)
        )

    return float(price)
