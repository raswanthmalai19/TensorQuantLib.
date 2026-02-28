"""
Risk metrics for option portfolios and underlying assets.

Provides:
    PortfolioRisk    -- Class to compute VaR, CVaR, volatility, Sharpe, max drawdown
    var_historical   -- Historical simulation VaR
    var_parametric   -- Parametric (normal) VaR
    var_mc           -- Monte Carlo VaR via GBM
    cvar             -- Conditional VaR (Expected Shortfall)
    scenario_analysis -- P&L under a set of stress scenarios
    greeks_portfolio  -- Aggregate Greeks across a book of options

Definitions:
    VaR(alpha):   loss L such that P(L > VaR) = 1 - alpha  (e.g. 95% VaR)
    CVaR(alpha):  E[L | L > VaR(alpha)]  (also called Expected Shortfall / ES)

All loss figures are expressed as a fraction of current portfolio value
unless stated otherwise.

References:
    McNeil, A.J., Frey, R. & Embrechts, P. (2005). Quantitative Risk Management.
    Princeton University Press.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from scipy.stats import norm


# ------------------------------------------------------------------ #
# Scalar VaR / CVaR functions
# ------------------------------------------------------------------ #

def var_parametric(
    mu: float,
    sigma: float,
    alpha: float = 0.95,
    horizon: float = 1.0 / 252.0,
) -> float:
    """Parametric (Gaussian) VaR at confidence level alpha.

    Assumes P&L is normally distributed: P&L ~ N(mu*h, sigma^2*h).

    Args:
        mu: Daily drift (annualised, then scaled by horizon).
        sigma: Annualised volatility of returns.
        alpha: Confidence level (default 0.95).
        horizon: Time horizon in years (default 1/252 = 1 trading day).

    Returns:
        VaR as a positive number (= loss at alpha-quantile).

    Example:
        >>> v = var_parametric(0.0, 0.20, alpha=0.95)
        >>> v > 0
        True
    """
    z = norm.ppf(1.0 - alpha)
    return float(-(mu * horizon + sigma * np.sqrt(horizon) * z))


def var_historical(
    returns: np.ndarray,
    alpha: float = 0.95,
) -> float:
    """Historical simulation VaR.

    Args:
        returns: 1-D array of observed log-returns (or P&L changes).
        alpha: Confidence level (default 0.95).

    Returns:
        VaR as a positive number (lower tail quantile of losses).

    Example:
        >>> import numpy as np
        >>> r = np.random.default_rng(0).normal(0, 0.01, 1000)
        >>> v = var_historical(r, alpha=0.95)
        >>> v > 0
        True
    """
    losses = -np.sort(returns)  # sort ascending losses
    idx = int(np.floor((1.0 - alpha) * len(losses)))
    return float(losses[idx])


def cvar(
    returns: np.ndarray,
    alpha: float = 0.95,
) -> float:
    """Conditional VaR (Expected Shortfall) at confidence level alpha.

    CVaR = E[loss | loss > VaR(alpha)] — the average loss in the worst (1-alpha) tail.

    Args:
        returns: 1-D array of log-returns or P&L.
        alpha: Confidence level (default 0.95).

    Returns:
        CVaR as a positive number.

    Example:
        >>> import numpy as np
        >>> r = np.random.default_rng(0).normal(0, 0.01, 1000)
        >>> es = cvar(r, alpha=0.95)
        >>> es >= var_historical(r, alpha=0.95)
        True
    """
    v = var_historical(returns, alpha)
    tail_losses = -returns[-returns >= v]  # losses exceeding VaR
    if len(tail_losses) == 0:
        return v
    return float(np.mean(tail_losses))


def var_mc(
    S: float,
    sigma: float,
    horizon: float = 1.0 / 252.0,
    r: float = 0.0,
    q: float = 0.0,
    alpha: float = 0.95,
    n_paths: int = 100_000,
    seed: Optional[int] = None,
) -> tuple[float, float]:
    """Monte Carlo VaR and CVaR for a single equity position.

    Simulates GBM returns over the given horizon and computes VaR and CVaR
    from the simulated distribution.

    Args:
        S: Current spot price.
        sigma: Annualised volatility.
        horizon: Time horizon (years).
        r: Risk-free rate (annualised).
        q: Dividend yield.
        alpha: Confidence level.
        n_paths: Number of simulation paths.
        seed: Random seed.

    Returns:
        (VaR, CVaR) as positive numbers representing fractional loss of S.

    Example:
        >>> v, es = var_mc(100.0, 0.20, alpha=0.95, seed=42)
        >>> v > 0 and es >= v
        True
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    S_T = S * np.exp((r - q - 0.5 * sigma ** 2) * horizon + sigma * np.sqrt(horizon) * z)
    returns = (S_T - S) / S  # fractional P&L
    v = var_historical(returns, alpha)
    es = cvar(returns, alpha)
    return v, es


# ------------------------------------------------------------------ #
# Scenario analysis
# ------------------------------------------------------------------ #

def scenario_analysis(
    S: float,
    position_value_fn: object,  # callable: (S_new: float) -> float
    scenarios: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Compute P&L under a set of spot-price stress scenarios.

    Args:
        S: Current spot price.
        position_value_fn: Callable mapping spot price -> position value.
        scenarios: Dict mapping scenario name -> stressed spot price.
            E.g. {'crash -20%': 80.0, 'rally +20%': 120.0}

    Returns:
        Dict mapping scenario name -> {'S_stressed', 'value', 'pnl', 'pnl_pct'}.

    Example:
        >>> def v(s): return max(s - 100, 0)  # long call
        >>> results = scenario_analysis(100.0, v, {'down': 90.0, 'up': 110.0})
        >>> 'down' in results and 'up' in results
        True
    """
    v0 = float(position_value_fn(S))  # type: ignore[operator]
    output: dict[str, dict[str, float]] = {}
    for name, S_new in scenarios.items():
        v_new = float(position_value_fn(S_new))  # type: ignore[operator]
        pnl = v_new - v0
        pnl_pct = pnl / abs(v0) * 100.0 if abs(v0) > 1e-12 else float("nan")
        output[name] = {
            "S_stressed": S_new,
            "value": v_new,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        }
    return output


# ------------------------------------------------------------------ #
# Aggregate portfolio Greeks
# ------------------------------------------------------------------ #

@dataclass
class OptionPosition:
    """A single option position in a portfolio.

    Attributes:
        option_type: 'call' or 'put'.
        K: Strike.
        T: Time to expiry.
        sigma: Implied volatility.
        quantity: Number of contracts (negative = short).
        multiplier: Contract multiplier (e.g. 100 for equity options).
    """
    option_type: str
    K: float
    T: float
    sigma: float
    quantity: float = 1.0
    multiplier: float = 1.0


def greeks_portfolio(
    positions: list[OptionPosition],
    S: float,
    r: float,
    q: float = 0.0,
) -> dict[str, float]:
    """Aggregate Black-Scholes Greeks for a portfolio of options on the same underlying.

    Args:
        positions: List of OptionPosition objects.
        S: Current spot price.
        r: Risk-free rate.
        q: Dividend yield.

    Returns:
        Dict with aggregated 'delta', 'gamma', 'vega', 'theta', 'rho', and 'value'.

    Example:
        >>> pos = [OptionPosition('call', K=100, T=1.0, sigma=0.2, quantity=1)]
        >>> g = greeks_portfolio(pos, S=100, r=0.05)
        >>> 0.0 < g['delta'] < 1.0
        True
    """
    from tensorquantlib.finance.black_scholes import (
        bs_price_numpy, bs_delta, bs_gamma, bs_vega, bs_theta, bs_rho,
    )

    total: dict[str, float] = {"delta": 0.0, "gamma": 0.0, "vega": 0.0,
                                "theta": 0.0, "rho": 0.0, "value": 0.0}

    for pos in positions:
        w = pos.quantity * pos.multiplier
        total["value"] += w * float(bs_price_numpy(S, pos.K, pos.T, r, pos.sigma, q=q, option_type=pos.option_type))
        total["delta"] += w * float(bs_delta(S, pos.K, pos.T, r, pos.sigma, q=q, option_type=pos.option_type))
        total["gamma"] += w * float(bs_gamma(S, pos.K, pos.T, r, pos.sigma, q=q))
        total["vega"] += w * float(bs_vega(S, pos.K, pos.T, r, pos.sigma, q=q))
        total["theta"] += w * float(bs_theta(S, pos.K, pos.T, r, pos.sigma, q=q, option_type=pos.option_type))
        total["rho"] += w * float(bs_rho(S, pos.K, pos.T, r, pos.sigma, q=q, option_type=pos.option_type))

    return total


# ------------------------------------------------------------------ #
# Portfolio risk container
# ------------------------------------------------------------------ #

@dataclass
class PortfolioRisk:
    """Compute risk metrics for a time series of portfolio returns.

    Usage::

        import numpy as np
        risk = PortfolioRisk(returns=np.random.normal(0.0002, 0.01, 252))
        risk.summary()

    Attributes:
        returns: 1-D array of daily log-returns or P&L fractions.
        alpha: Confidence level for VaR/CVaR (default 0.95).
        risk_free_daily: Daily risk-free rate for Sharpe ratio.
    """

    returns: np.ndarray
    alpha: float = 0.95
    risk_free_daily: float = 0.0

    def var(self) -> float:
        """Historical VaR at self.alpha."""
        return var_historical(self.returns, self.alpha)

    def cvar(self) -> float:
        """Historical CVaR (Expected Shortfall) at self.alpha."""
        return cvar(self.returns, self.alpha)

    def volatility(self, annualise: bool = True) -> float:
        """Return volatility (std dev). Annualised by sqrt(252) if annualise=True."""
        vol = float(np.std(self.returns))
        return vol * np.sqrt(252.0) if annualise else vol

    def sharpe(self, annualise: bool = True) -> float:
        """Sharpe ratio: (mean - rf) / std."""
        excess = self.returns - self.risk_free_daily
        s = float(np.mean(excess)) / float(np.std(self.returns)) if np.std(self.returns) > 0 else 0.0
        return s * np.sqrt(252.0) if annualise else s

    def max_drawdown(self) -> float:
        """Maximum drawdown from cumulative return series."""
        cum = np.cumprod(1.0 + self.returns)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        return float(np.min(dd))

    def calmar(self) -> float:
        """Calmar ratio: annualised return / abs(max drawdown)."""
        ann_return = float(np.mean(self.returns)) * 252.0
        mdd = abs(self.max_drawdown())
        return ann_return / mdd if mdd > 1e-12 else float("inf")

    def summary(self) -> dict[str, float]:
        """Return all risk metrics as a dictionary.

        Example:
            >>> import numpy as np
            >>> pr = PortfolioRisk(np.random.default_rng(0).normal(0.001, 0.015, 500))
            >>> s = pr.summary()
            >>> 'var_95' in s and 'sharpe' in s
            True
        """
        return {
            f"var_{int(self.alpha*100)}": self.var(),
            f"cvar_{int(self.alpha*100)}": self.cvar(),
            "volatility_ann": self.volatility(),
            "sharpe": self.sharpe(),
            "max_drawdown": self.max_drawdown(),
            "calmar": self.calmar(),
            "mean_daily_return": float(np.mean(self.returns)),
            "n_days": len(self.returns),
        }
