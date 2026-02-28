"""Credit risk models: Merton structural model and reduced-form CDS pricing.

Merton (1974) structural model treats equity as a call option on firm assets.
Reduced-form models price CDS using hazard rates and survival probabilities.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Merton structural model
# ---------------------------------------------------------------------------

def merton_default_prob(V: float, D: float, T: float, r: float,
                        sigma_V: float) -> float:
    """Probability of default under Merton (1974) structural model.

    Default occurs when firm value V_T < D at maturity.

    Parameters
    ----------
    V : float
        Current firm asset value.
    D : float
        Face value of debt (default barrier).
    T : float
        Time to maturity in years.
    r : float
        Risk-free rate.
    sigma_V : float
        Volatility of firm assets.

    Returns
    -------
    float
        Risk-neutral probability of default.
    """
    d2 = (np.log(V / D) + (r - 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    return float(norm.cdf(-d2))


def merton_credit_spread(V: float, D: float, T: float, r: float,
                         sigma_V: float) -> float:
    """Credit spread implied by the Merton model.

    spread = -(1/T) * ln(B / (D * exp(-r*T)))
    where B is the risky bond price = D*exp(-r*T) - Put(V, D, T).

    Parameters
    ----------
    V, D, T, r, sigma_V
        Same as :func:`merton_default_prob`.

    Returns
    -------
    float
        Annual credit spread (continuously compounded).
    """
    d1 = (np.log(V / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    d2 = d1 - sigma_V * np.sqrt(T)
    # Risky bond = D*exp(-rT) - put on V with strike D
    put = D * np.exp(-r * T) * norm.cdf(-d2) - V * norm.cdf(-d1)
    risky_bond = D * np.exp(-r * T) - put
    risk_free_bond = D * np.exp(-r * T)
    # spread = yield_risky - r
    yield_risky = -np.log(risky_bond / D) / T
    return float(yield_risky - r)


# ---------------------------------------------------------------------------
# Survival / hazard rate utilities
# ---------------------------------------------------------------------------

def survival_probability(hazard_rate: float, T: float) -> float:
    """Survival probability under constant hazard rate.

    Q(T) = exp(-lambda * T)
    """
    return float(np.exp(-hazard_rate * T))


def hazard_rate_from_spread(spread: float, recovery: float = 0.4) -> float:
    """Implied constant hazard rate from CDS spread.

    lambda ≈ spread / (1 - R)

    Parameters
    ----------
    spread : float
        CDS spread (annualised, e.g. 0.01 = 100bp).
    recovery : float
        Recovery rate (default 0.4 = 40%).

    Returns
    -------
    float
        Implied hazard rate.
    """
    return spread / (1.0 - recovery)


# ---------------------------------------------------------------------------
# CDS pricing
# ---------------------------------------------------------------------------

def cds_spread(hazard_rate: float, T: float, recovery: float = 0.4,
               r: float = 0.05, n_premium_dates: int = 4) -> float:
    """Par CDS spread for constant hazard rate.

    Premium leg = spread * sum( DF_i * Q_i * delta_i )
    Protection leg = (1-R) * sum( DF_i * (Q_{i-1} - Q_i) )

    We solve for spread = protection_leg / risky_annuity.

    Parameters
    ----------
    hazard_rate : float
        Constant hazard rate (annualised).
    T : float
        CDS maturity in years.
    recovery : float
        Recovery rate.
    r : float
        Risk-free rate.
    n_premium_dates : int
        Number of premium payment dates per year.

    Returns
    -------
    float
        Par CDS spread (annualised).
    """
    n_periods = int(T * n_premium_dates)
    dt = T / n_periods if n_periods > 0 else T

    risky_annuity = 0.0
    protection_leg = 0.0
    for i in range(1, n_periods + 1):
        t_i = i * dt
        df = np.exp(-r * t_i)
        q_i = np.exp(-hazard_rate * t_i)
        q_prev = np.exp(-hazard_rate * (t_i - dt))
        risky_annuity += df * q_i * dt
        protection_leg += df * (q_prev - q_i)

    protection_leg *= (1.0 - recovery)

    if risky_annuity < 1e-15:
        return 0.0
    return float(protection_leg / risky_annuity)


def cds_price(hazard_rate: float, T: float, recovery: float = 0.4,
              r: float = 0.05, spread: float = 0.01,
              notional: float = 1e6, n_premium_dates: int = 4) -> float:
    """Mark-to-market value of a CDS position (protection buyer).

    MTM = protection_leg - spread * risky_annuity

    Parameters
    ----------
    hazard_rate : float
        Current constant hazard rate.
    T : float
        Remaining maturity.
    recovery : float
        Recovery rate.
    r : float
        Risk-free rate.
    spread : float
        Contracted CDS spread (annualised).
    notional : float
        Notional amount.
    n_premium_dates : int
        Premium payment frequency per year.

    Returns
    -------
    float
        MTM value of the CDS position for the protection buyer.
    """
    n_periods = int(T * n_premium_dates)
    dt = T / n_periods if n_periods > 0 else T

    risky_annuity = 0.0
    protection_leg = 0.0
    for i in range(1, n_periods + 1):
        t_i = i * dt
        df = np.exp(-r * t_i)
        q_i = np.exp(-hazard_rate * t_i)
        q_prev = np.exp(-hazard_rate * (t_i - dt))
        risky_annuity += df * q_i * dt
        protection_leg += df * (q_prev - q_i)

    protection_leg *= (1.0 - recovery)
    mtm = (protection_leg - spread * risky_annuity) * notional
    return float(mtm)
