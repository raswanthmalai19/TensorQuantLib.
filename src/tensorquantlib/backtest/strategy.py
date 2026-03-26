"""Abstract base strategy and concrete strategy implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Trade:
    """Record of a single trade."""

    step: int
    quantity: float  # positive = buy, negative = sell
    price: float
    label: str = ""
    slippage: float = 0.0  # slippage cost paid on this trade
    commission: float = 0.0  # commission paid on this trade

    @property
    def notional(self) -> float:
        """Gross notional value of the trade."""
        return abs(self.quantity) * self.price

    @property
    def total_cost(self) -> float:
        """Total execution cost (slippage + commission)."""
        return self.slippage + self.commission


class Strategy(ABC):
    """Base class for backtesting strategies.

    Sub-classes must implement :meth:`on_data`.
    """

    def __init__(self):
        self.position: float = 0.0
        self.trades: list[Trade] = []
        self.cash: float = 0.0
        self._greeks_history: dict[str, list] = {}

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
        # Default no-op; strategies override when they need fill bookkeeping.
        return None


class DeltaHedgeStrategy(Strategy):
    """Delta-hedge a short option position using Black-Scholes delta.

    At each step compute BS delta and rebalance the hedge portfolio.
    Also tracks per-step Delta and Gamma in ``_greeks_history`` for
    P&L attribution via :func:`~tensorquantlib.backtest.metrics.hedge_pnl_attribution`.

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

    def __init__(
        self,
        K: float,
        T_total: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        n_steps: int = 252,
    ):
        super().__init__()
        self.K = K
        self.T_total = T_total
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.n_steps = n_steps
        self._greeks_history: dict[str, list] = {"delta": [], "gamma": [], "T_remain": []}

    def _bs_delta(self, S: float, T_remain: float) -> float:
        from scipy.stats import norm

        if T_remain <= 0:
            if self.option_type == "call":
                return 1.0 if S > self.K else 0.0
            else:
                return -1.0 if S < self.K else 0.0
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma**2) * T_remain) / (
            self.sigma * np.sqrt(T_remain)
        )
        if self.option_type == "call":
            return float(norm.cdf(d1))
        return float(norm.cdf(d1) - 1.0)

    def _bs_gamma(self, S: float, T_remain: float) -> float:
        from scipy.stats import norm

        if T_remain <= 0:
            return 0.0
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma**2) * T_remain) / (
            self.sigma * np.sqrt(T_remain)
        )
        return float(norm.pdf(d1) / (S * self.sigma * np.sqrt(T_remain)))

    def on_data(self, step: int, price: float, **kwargs) -> float:
        T_remain = max(self.T_total * (1 - step / self.n_steps), 0.0)
        delta = self._bs_delta(price, T_remain)
        gamma = self._bs_gamma(price, T_remain)
        self._greeks_history["delta"].append(delta)
        self._greeks_history["gamma"].append(gamma)
        self._greeks_history["T_remain"].append(T_remain)
        return delta


class GammaScalpingStrategy(Strategy):
    """Gamma scalping: delta-hedge a long straddle and harvest realized volatility.

    The strategy holds a synthetic long straddle (long gamma) and
    continuously delta-hedges to remain directionally neutral.  It
    profits when *realized* volatility exceeds the *implied* volatility
    embedded in the initial option price.

    Per-step P&L attribution (stored in ``_greeks_history``):

    .. code-block:: text

        Daily P&L ≈  ½Γ(ΔS)²          (gamma P&L — profits from large moves)
                   + Θ · Δt             (theta P&L — loses time value daily)
                   + residual           (higher-order / hedging error)

    When realized_vol > implied_vol the gamma P&L dominates and the
    strategy is profitable over the full holding period.

    Parameters
    ----------
    K : float
        Straddle strike (ideally near-ATM).
    T_total : float
        Time to expiry at inception (years).
    r : float
        Risk-free rate.
    sigma_implied : float
        Implied volatility at inception.
    n_steps : int
        Total rebalancing steps.
    """

    def __init__(
        self, K: float, T_total: float, r: float, sigma_implied: float, n_steps: int = 252
    ):
        super().__init__()
        self.K = K
        self.T_total = T_total
        self.r = r
        self.sigma = sigma_implied
        self.n_steps = n_steps
        self._greeks_history: dict[str, list] = {
            "delta": [],
            "gamma": [],
            "theta": [],
            "theoretical_gamma_pnl": [],
            "theoretical_theta_pnl": [],
        }
        self._prev_price: float | None = None

    def _straddle_delta(self, S: float, T_remain: float) -> float:
        """Straddle delta = N(d1) − N(−d1) = 2N(d1) − 1."""
        from scipy.stats import norm

        if T_remain <= 0:
            return 0.0
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma**2) * T_remain) / (
            self.sigma * np.sqrt(T_remain)
        )
        return float(2.0 * norm.cdf(d1) - 1.0)

    def _straddle_gamma(self, S: float, T_remain: float) -> float:
        """Straddle Gamma = 2 × call Gamma (call and put share the same Gamma)."""
        from scipy.stats import norm

        if T_remain <= 0:
            return 0.0
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma**2) * T_remain) / (
            self.sigma * np.sqrt(T_remain)
        )
        return float(2.0 * norm.pdf(d1) / (S * self.sigma * np.sqrt(T_remain)))

    def _straddle_theta(self, S: float, T_remain: float) -> float:
        """Straddle Theta per year (negative = time decay)."""
        from scipy.stats import norm

        if T_remain <= 0:
            return 0.0
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma**2) * T_remain) / (
            self.sigma * np.sqrt(T_remain)
        )
        d2 = d1 - self.sigma * np.sqrt(T_remain)
        theta_call = -S * norm.pdf(d1) * self.sigma / (
            2.0 * np.sqrt(T_remain)
        ) - self.r * self.K * np.exp(-self.r * T_remain) * norm.cdf(d2)
        theta_put = -S * norm.pdf(d1) * self.sigma / (
            2.0 * np.sqrt(T_remain)
        ) + self.r * self.K * np.exp(-self.r * T_remain) * norm.cdf(-d2)
        return float(theta_call + theta_put)

    def on_data(self, step: int, price: float, **kwargs) -> float:
        T_remain = max(self.T_total * (1 - step / self.n_steps), 0.0)
        dt = self.T_total / self.n_steps
        delta = self._straddle_delta(price, T_remain)
        gamma = self._straddle_gamma(price, T_remain)
        theta = self._straddle_theta(price, T_remain)

        if self._prev_price is not None:
            dS = price - self._prev_price
            theoretical_gamma_pnl = 0.5 * gamma * dS**2
            theoretical_theta_pnl = theta * dt  # negative (time decay)
        else:
            theoretical_gamma_pnl = 0.0
            theoretical_theta_pnl = 0.0

        self._prev_price = price
        self._greeks_history["delta"].append(delta)
        self._greeks_history["gamma"].append(gamma)
        self._greeks_history["theta"].append(theta)
        self._greeks_history["theoretical_gamma_pnl"].append(theoretical_gamma_pnl)
        self._greeks_history["theoretical_theta_pnl"].append(theoretical_theta_pnl)
        return delta


class DeltaGammaHedgeStrategy(Strategy):
    """Delta-Gamma neutral hedge using two options and the underlying.

    Holds a primary long option (strike ``K1``) and hedges **Gamma**
    using a second option (strike ``K2``).  Any residual Delta is
    neutralised using the underlying asset.

    At each rebalancing step the hedge ratios are:

    .. math::

        Q_{\\text{hedge}} = \\frac{\\Gamma_1(S,T)}{\\Gamma_2(S,T)}
        \\qquad \\text{(makes net Gamma = 0)}

        N_{\\text{stock}} = -(\\Delta_1 - Q_{\\text{hedge}} \\cdot \\Delta_2)
        \\qquad \\text{(makes net Delta = 0)}

    The ``_greeks_history`` dict records ``net_delta``, ``net_gamma``,
    ``hedge_ratio``, and ``stock_position`` at every step.

    Parameters
    ----------
    K1 : float
        Strike of the *primary* (long) option.
    K2 : float
        Strike of the *hedge* option (must differ from K1 for the
        gamma hedge to be non-trivial).
    T_total : float
        Time to expiry at inception (years).
    r : float
        Risk-free rate.
    sigma : float
        Implied volatility (flat smile assumed).
    n_steps : int
        Total rebalancing steps.
    option_type : str
        ``'call'`` or ``'put'`` for both legs.
    """

    def __init__(
        self,
        K1: float,
        K2: float,
        T_total: float,
        r: float,
        sigma: float,
        n_steps: int = 252,
        option_type: str = "call",
    ):
        super().__init__()
        self.K1 = K1
        self.K2 = K2
        self.T_total = T_total
        self.r = r
        self.sigma = sigma
        self.n_steps = n_steps
        self.option_type = option_type
        self._greeks_history: dict[str, list] = {
            "net_delta": [],
            "net_gamma": [],
            "hedge_ratio": [],
            "stock_position": [],
        }

    def _delta(self, S: float, K: float, T_remain: float) -> float:
        from scipy.stats import norm

        if T_remain <= 0:
            return 0.0
        d1 = (np.log(S / K) + (self.r + 0.5 * self.sigma**2) * T_remain) / (
            self.sigma * np.sqrt(T_remain)
        )
        if self.option_type == "call":
            return float(norm.cdf(d1))
        return float(norm.cdf(d1) - 1.0)

    def _gamma(self, S: float, K: float, T_remain: float) -> float:
        from scipy.stats import norm

        if T_remain <= 0:
            return 0.0
        d1 = (np.log(S / K) + (self.r + 0.5 * self.sigma**2) * T_remain) / (
            self.sigma * np.sqrt(T_remain)
        )
        return float(norm.pdf(d1) / (S * self.sigma * np.sqrt(T_remain)))

    def on_data(self, step: int, price: float, **kwargs) -> float:
        T_remain = max(self.T_total * (1 - step / self.n_steps), 0.0)
        delta1 = self._delta(price, self.K1, T_remain)
        gamma1 = self._gamma(price, self.K1, T_remain)
        delta2 = self._delta(price, self.K2, T_remain)
        gamma2 = self._gamma(price, self.K2, T_remain)

        # Hedge ratio neutralises gamma
        Q_hedge = gamma1 / gamma2 if abs(gamma2) > 1e-12 else 0.0
        # Net delta after gamma hedge; we hold -net_delta in underlying
        net_delta = delta1 - Q_hedge * delta2
        stock_position = -net_delta

        self._greeks_history["net_delta"].append(net_delta + stock_position)  # ≈ 0
        self._greeks_history["net_gamma"].append(gamma1 - Q_hedge * gamma2)  # ≈ 0
        self._greeks_history["hedge_ratio"].append(Q_hedge)
        self._greeks_history["stock_position"].append(stock_position)
        return stock_position


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
