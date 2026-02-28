"""Backtesting engine — runs a strategy on a price series."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tensorquantlib.backtest.strategy import Strategy, Trade


@dataclass
class BacktestResult:
    """Container for backtest output."""

    equity_curve: np.ndarray
    """Equity value at each step."""

    trades: list[Trade]
    """List of all executed trades."""

    returns: np.ndarray
    """Per-step returns of the equity curve."""

    final_equity: float
    """Final portfolio value."""


class BacktestEngine:
    """Run a :class:`Strategy` over a price series.

    Parameters
    ----------
    strategy : Strategy
        Strategy instance to run.
    prices : array-like
        1-D array of asset prices (one per time step).
    initial_capital : float
        Starting cash.
    """

    def __init__(self, strategy: Strategy, prices, initial_capital: float = 1e6):
        self.strategy = strategy
        self.prices = np.asarray(prices, dtype=float)
        self.initial_capital = initial_capital

    def run(self) -> BacktestResult:
        """Execute the backtest.

        Returns
        -------
        BacktestResult
        """
        strat = self.strategy
        prices = self.prices
        n = len(prices)

        strat.cash = self.initial_capital
        strat.position = 0.0
        strat.trades = []

        equity = np.zeros(n)

        for i in range(n):
            desired = strat.on_data(i, prices[i])
            delta_pos = desired - strat.position
            if abs(delta_pos) > 1e-12:
                trade = Trade(step=i, quantity=delta_pos, price=prices[i])
                strat.cash -= delta_pos * prices[i]
                strat.position = desired
                strat.trades.append(trade)
                strat.on_fill(trade)

            equity[i] = strat.cash + strat.position * prices[i]

        returns = np.diff(equity) / np.where(np.abs(equity[:-1]) > 1e-12,
                                              equity[:-1], 1.0)
        return BacktestResult(
            equity_curve=equity,
            trades=strat.trades,
            returns=returns,
            final_equity=float(equity[-1]),
        )
