"""Abstract base strategy and concrete strategy implementations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Trade:
    """Record of a single trade."""
    step: int
    quantity: float  # positive = buy, negative = sell
    price: float
    label: str = ""


class Strategy(ABC):
    """Base class for backtesting strategies.

    Sub-classes must implement :meth:`on_data`.
    """

    def __init__(self):
        self.position: float = 0.0
        self.trades: list[Trade] = []
        self.cash: float = 0.0

    @abstractmethod
    def on_data(self, step: int, price: float, **kwargs) -> float:
        """Called each time step with current price.

        Parameters
        ----------
        step : int
            Current time step index.
        price : float
            Current asset price.

        Returns
        -------
        float
            Desired position size (signed). The engine will trade the
            difference between current and desired position.
        """

    def on_fill(self, trade: Trade) -> None:
        """Called after a trade is executed. Override for bookkeeping."""


class DeltaHedgeStrategy(Strategy):
    """Delta-hedge a short option position using Black-Scholes delta.

    At each step compute BS delta and rebalance the hedge portfolio.

    Parameters
    ----------
    K : float
        Option strike.
    T_total : float
        Total time to expiry at inception (years).
    r : float
        Risk-free rate.
    sigma : float
        Implied volatility.
    option_type : str
        ``'call'`` or ``'put'``.
    n_steps : int
        Total number of rebalancing steps.
    """

    def __init__(self, K: float, T_total: float, r: float, sigma: float,
                 option_type: str = "call", n_steps: int = 252):
        super().__init__()
        self.K = K
        self.T_total = T_total
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.n_steps = n_steps

    def _bs_delta(self, S: float, T_remain: float) -> float:
        from scipy.stats import norm
        if T_remain <= 0:
            if self.option_type == "call":
                return 1.0 if S > self.K else 0.0
            else:
                return -1.0 if S < self.K else 0.0
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma**2) * T_remain) / \
             (self.sigma * np.sqrt(T_remain))
        if self.option_type == "call":
            return float(norm.cdf(d1))
        return float(norm.cdf(d1) - 1.0)

    def on_data(self, step: int, price: float, **kwargs) -> float:
        T_remain = self.T_total * (1 - step / self.n_steps)
        T_remain = max(T_remain, 0.0)
        return self._bs_delta(price, T_remain)


class StraddleStrategy(Strategy):
    """Buy a straddle at regular intervals.

    This is a simple strategy that opens a new straddle position every
    ``interval`` steps by buying one unit of the asset and recording
    the entry price.

    Parameters
    ----------
    interval : int
        Number of steps between new straddle entries.
    """

    def __init__(self, interval: int = 21):
        super().__init__()
        self.interval = interval
        self._entries: list[float] = []

    def on_data(self, step: int, price: float, **kwargs) -> float:
        if step % self.interval == 0:
            self._entries.append(price)
            return self.position + 1.0  # add one unit
        return self.position
