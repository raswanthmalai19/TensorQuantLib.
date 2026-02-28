"""Performance metrics for backtesting."""
from __future__ import annotations

import numpy as np


def sharpe_ratio(returns: np.ndarray, rf: float = 0.0,
                 periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio.

    Parameters
    ----------
    returns : array-like
        Period returns (e.g. daily).
    rf : float
        Risk-free rate per period.
    periods_per_year : int
        Annualisation factor (252 for daily).

    Returns
    -------
    float
        Annualised Sharpe ratio.
    """
    r = np.asarray(returns, dtype=float)
    excess = r - rf
    std = np.std(excess, ddof=1)
    if std < 1e-15:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    """Maximum drawdown from peak.

    Parameters
    ----------
    equity : array-like
        Equity curve (absolute values).

    Returns
    -------
    float
        Maximum drawdown as a positive fraction (e.g. 0.15 = 15%).
    """
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak > 0, peak, 1.0)
    return float(np.max(dd))


def sortino_ratio(returns: np.ndarray, rf: float = 0.0,
                  periods_per_year: int = 252) -> float:
    """Annualised Sortino ratio (downside deviation only).

    Parameters
    ----------
    returns : array-like
        Period returns.
    rf : float
        Risk-free rate per period.
    periods_per_year : int
        Annualisation factor.

    Returns
    -------
    float
        Annualised Sortino ratio.
    """
    r = np.asarray(returns, dtype=float)
    excess = r - rf
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf") if np.mean(excess) > 0 else 0.0
    down_std = np.sqrt(np.mean(downside**2))
    if down_std < 1e-15:
        return 0.0
    return float(np.mean(excess) / down_std * np.sqrt(periods_per_year))


def win_rate(trades: np.ndarray) -> float:
    """Fraction of profitable trades.

    Parameters
    ----------
    trades : array-like
        P&L per trade (positive = profit).

    Returns
    -------
    float
        Win rate in [0, 1].
    """
    t = np.asarray(trades, dtype=float)
    if len(t) == 0:
        return 0.0
    return float(np.sum(t > 0) / len(t))


def profit_factor(trades: np.ndarray) -> float:
    """Profit factor = gross profits / gross losses.

    Parameters
    ----------
    trades : array-like
        P&L per trade.

    Returns
    -------
    float
        Profit factor. Returns inf if no losses.
    """
    t = np.asarray(trades, dtype=float)
    gross_profit = float(np.sum(t[t > 0]))
    gross_loss = float(np.abs(np.sum(t[t < 0])))
    if gross_loss < 1e-15:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss
