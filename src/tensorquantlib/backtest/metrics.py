"""Performance metrics for backtesting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


def sharpe_ratio(returns: np.ndarray, rf: float = 0.0, periods_per_year: int = 252) -> float:
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


def sortino_ratio(returns: np.ndarray, rf: float = 0.0, periods_per_year: int = 252) -> float:
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


# ------------------------------------------------------------------ #
# Additional metrics
# ------------------------------------------------------------------ #


def annualized_return(equity: np.ndarray, periods_per_year: int = 252) -> float:
    """Compound annualized growth rate (CAGR) from an equity curve.

    Parameters
    ----------
    equity : array-like
        Equity value at each step (must start > 0).
    periods_per_year : int
        Annualisation factor (252 for daily).

    Returns
    -------
    float
        Annualized return, e.g. 0.12 = 12 %.
    """
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 2 or eq[0] <= 0:
        return 0.0
    total_periods = len(eq) - 1
    years = total_periods / periods_per_year
    if years <= 0:
        return 0.0
    return float((eq[-1] / eq[0]) ** (1.0 / years) - 1.0)


def calmar_ratio(equity: np.ndarray, periods_per_year: int = 252) -> float:
    """Calmar ratio = annualized return / maximum drawdown.

    Parameters
    ----------
    equity : array-like
        Equity curve.
    periods_per_year : int
        Annualisation factor.

    Returns
    -------
    float
        Calmar ratio.  Returns ``inf`` if max drawdown is zero.
    """
    ann_ret = annualized_return(equity, periods_per_year)
    mdd = max_drawdown(equity)
    if mdd < 1e-15:
        return float("inf") if ann_ret > 0 else 0.0
    return float(ann_ret / mdd)


def information_ratio(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """Annualised information ratio (IR).

    IR = E[active_return] / std(active_return) * sqrt(periods_per_year)

    Parameters
    ----------
    strategy_returns : array-like
        Period returns of the strategy.
    benchmark_returns : array-like
        Period returns of the benchmark.
    periods_per_year : int
        Annualisation factor.

    Returns
    -------
    float
        Annualised information ratio.
    """
    active = np.asarray(strategy_returns, dtype=float) - np.asarray(benchmark_returns, dtype=float)
    std = np.std(active, ddof=1)
    if std < 1e-15:
        return 0.0
    return float(np.mean(active) / std * np.sqrt(periods_per_year))


def turnover(trades: list, initial_capital: float = 1_000_000.0) -> float:
    """Annualized one-way turnover as a fraction of initial capital.

    turnover = total_notional_traded / initial_capital

    Parameters
    ----------
    trades : list of Trade
        Executed trades (each must have ``.quantity`` and ``.price``).
    initial_capital : float
        Starting capital used as the denominator.

    Returns
    -------
    float
        Turnover (e.g. 2.5 = 250 % of capital traded).
    """
    if not trades:
        return 0.0
    total_notional = sum(abs(t.quantity) * t.price for t in trades)
    return float(total_notional / initial_capital)


def hedge_pnl_attribution(
    equity_curve: np.ndarray,
    deltas: list,
    gammas: list,
    prices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Decompose per-step portfolio P&L into Delta, Gamma, and residual.

    For a continuously delta-hedged portfolio, the per-step P&L can be
    approximated as:

    .. math::

        \\text{Total P\\&L} \\approx
        \\underbrace{\\Delta \\cdot \\Delta S}_{\\text{delta P\\&L}}
        + \\underbrace{\\tfrac{1}{2}\\Gamma (\\Delta S)^2}_{\\text{gamma P\\&L}}
        + \\text{residual}

    The residual captures higher-order terms (Theta decay, vanna, etc.) and
    hedging error from discrete rebalancing.

    Parameters
    ----------
    equity_curve : array-like
        Portfolio equity at each step (length N).
    deltas : list of float
        Per-step delta of the overall position (length N).
    gammas : list of float
        Per-step gamma (length N).
    prices : array-like
        Underlying asset price at each step (length N).

    Returns
    -------
    dict
        Keys ``'delta_pnl'``, ``'gamma_pnl'``, ``'residual_pnl'``.
        Each is an array of length N-1.
    """
    equity = np.asarray(equity_curve, dtype=float)
    prices_arr = np.asarray(prices, dtype=float)
    delta_arr = np.asarray(deltas, dtype=float)
    gamma_arr = np.asarray(gammas, dtype=float)

    n = len(equity) - 1
    total_pnl = np.diff(equity)
    dS = np.diff(prices_arr[: n + 1])

    delta_pnl = delta_arr[:n] * dS
    gamma_pnl = 0.5 * gamma_arr[:n] * dS**2
    residual_pnl = total_pnl - delta_pnl - gamma_pnl

    return {
        "delta_pnl": delta_pnl,
        "gamma_pnl": gamma_pnl,
        "residual_pnl": residual_pnl,
    }


def hedge_efficiency(
    hedged_equity: np.ndarray,
    unhedged_equity: np.ndarray,
) -> float:
    """Variance reduction achieved by a hedge.

    .. math::

        \\text{efficiency} = 1 - \\frac{\\operatorname{Var}(\\text{hedged P\\&L})}{
        \\operatorname{Var}(\\text{unhedged P\\&L})}

    Returns 1.0 for a perfect hedge, 0.0 if the hedge adds no value,
    and a negative number if the hedge increases variance.

    Parameters
    ----------
    hedged_equity : array-like
        Equity curve of the hedged portfolio.
    unhedged_equity : array-like
        Equity curve of the same portfolio *without* hedging.

    Returns
    -------
    float
        Hedge efficiency in (-inf, 1].
    """
    hedged_ret = np.diff(np.asarray(hedged_equity, dtype=float))
    unhedged_ret = np.diff(np.asarray(unhedged_equity, dtype=float))
    var_unhedged = float(np.var(unhedged_ret, ddof=1))
    if var_unhedged < 1e-15:
        return 0.0
    var_hedged = float(np.var(hedged_ret, ddof=1))
    return float(1.0 - var_hedged / var_unhedged)
