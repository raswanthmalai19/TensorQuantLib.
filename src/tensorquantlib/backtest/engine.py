"""Backtesting engine with realistic execution cost models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tensorquantlib.backtest.strategy import Strategy, Trade

# ------------------------------------------------------------------ #
# Execution cost models
# ------------------------------------------------------------------ #


@dataclass
class SlippageModel:
    """Market-impact and bid-ask spread model.

    Total slippage = half-spread cost + square-root market-impact cost.

    Parameters
    ----------
    fixed_spread : float
        One-way half-spread as a fraction of price.
        E.g. ``0.0005`` = 5 bps one-way (10 bps round-trip).
    market_impact : float
        Square-root market-impact coefficient.
        Cost per unit = ``price × market_impact × sqrt(|qty| / adv)``.
        Set to 0 to disable market impact.
    adv : float
        Average daily volume (units). Used for impact scaling only.
    """

    fixed_spread: float = 0.0
    market_impact: float = 0.0
    adv: float = 1_000_000.0

    def cost(self, price: float, quantity: float) -> float:
        """Compute slippage cost for one trade (always non-negative)."""
        spread_cost = abs(quantity) * price * self.fixed_spread
        if self.market_impact > 0.0 and self.adv > 0.0:
            impact_cost = (
                abs(quantity) * price * self.market_impact * np.sqrt(abs(quantity) / self.adv)
            )
        else:
            impact_cost = 0.0
        return float(spread_cost + impact_cost)


@dataclass
class CommissionModel:
    """Commission / brokerage fee model.

    Total commission = max(per_trade + per_unit×|qty| + percentage×notional,
    minimum).

    Parameters
    ----------
    per_trade : float
        Fixed fee per order (e.g. ``1.0`` = $1 per trade).
    per_unit : float
        Fee per unit traded (e.g. ``0.005`` = half a cent per share).
    percentage : float
        Fraction of notional (e.g. ``0.001`` = 10 bps of notional).
    minimum : float
        Minimum commission per trade.
    """

    per_trade: float = 0.0
    per_unit: float = 0.0
    percentage: float = 0.0
    minimum: float = 0.0

    def cost(self, price: float, quantity: float) -> float:
        """Compute commission for one trade (always non-negative)."""
        notional = abs(quantity) * price
        c = self.per_trade + abs(quantity) * self.per_unit + notional * self.percentage
        return float(max(c, self.minimum))


# Pre-built convenience presets
#: Zero-cost model (default when no model is supplied).
ZERO_COST = CommissionModel()
#: Interactive Brokers-style equity commission ($0.005/share, $1 min).
EQUITY_COMM = CommissionModel(per_unit=0.005, minimum=1.0)
#: Typical institutional FX desk (0.2 bps of notional).
FX_COMM = CommissionModel(percentage=0.00002)
#: Liquid-equity half-spread slippage (5 bps one-way).
EQUITY_SLIP = SlippageModel(fixed_spread=0.0005)
#: Illiquid name: 20 bps spread + square-root market impact.
ILLIQUID_SLIP = SlippageModel(fixed_spread=0.002, market_impact=0.1, adv=50_000)


# ------------------------------------------------------------------ #
# Backtest result
# ------------------------------------------------------------------ #


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

    total_commission: float = 0.0
    """Total commissions paid during the backtest."""

    total_slippage: float = 0.0
    """Total slippage cost paid during the backtest."""

    total_turnover: float = 0.0
    """Total gross notional traded (sum of |qty| * price)."""

    n_trades: int = 0
    """Number of trades executed."""

    greeks_history: dict = field(default_factory=dict)
    """Per-step Greeks recorded by the strategy (if any)."""


# ------------------------------------------------------------------ #
# Engine
# ------------------------------------------------------------------ #


class BacktestEngine:
    """Run a :class:`Strategy` over a price series with realistic execution.

    Parameters
    ----------
    strategy : Strategy
        Strategy instance to run.
    prices : array-like
        1-D array of asset prices (one per time step).
    initial_capital : float
        Starting cash.
    slippage : SlippageModel, optional
        Slippage model.  Defaults to ``SlippageModel()`` (zero slippage).
        Use :data:`EQUITY_SLIP` for a realistic liquid-equity preset.
    commission : CommissionModel, optional
        Commission model.  Defaults to ``CommissionModel()`` (zero fees).
        Use :data:`EQUITY_COMM` for an Interactive Brokers-style preset.

    Examples
    --------
    Zero-cost (default)::

        engine = BacktestEngine(strategy, prices)

    With realistic costs::

        from tensorquantlib.backtest.engine import EQUITY_SLIP, EQUITY_COMM
        engine = BacktestEngine(
            strategy, prices,
            slippage=EQUITY_SLIP,
            commission=EQUITY_COMM,
        )
    """

    def __init__(
        self,
        strategy: Strategy,
        prices,
        initial_capital: float = 1_000_000.0,
        slippage: SlippageModel | None = None,
        commission: CommissionModel | None = None,
    ):
        self.strategy = strategy
        self.prices = np.asarray(prices, dtype=float)
        self.initial_capital = initial_capital
        self.slippage = slippage if slippage is not None else SlippageModel()
        self.commission = commission if commission is not None else CommissionModel()

    def run(self) -> BacktestResult:
        """Execute the backtest and return a :class:`BacktestResult`."""
        strat = self.strategy
        prices = self.prices
        n = len(prices)

        strat.cash = self.initial_capital
        strat.position = 0.0
        strat.trades = []

        equity = np.zeros(n)
        total_commission = 0.0
        total_slippage = 0.0
        total_turnover = 0.0

        for i in range(n):
            desired = strat.on_data(i, prices[i])
            delta_pos = desired - strat.position

            if abs(delta_pos) > 1e-12:
                slip = self.slippage.cost(prices[i], delta_pos)
                comm = self.commission.cost(prices[i], delta_pos)
                notional = abs(delta_pos) * prices[i]

                total_slippage += slip
                total_commission += comm
                total_turnover += notional

                # Cash: pay for shares + execution costs
                strat.cash -= delta_pos * prices[i] + slip + comm
                strat.position = desired

                trade = Trade(
                    step=i,
                    quantity=delta_pos,
                    price=prices[i],
                    slippage=slip,
                    commission=comm,
                )
                strat.trades.append(trade)
                strat.on_fill(trade)

            equity[i] = strat.cash + strat.position * prices[i]

        returns = np.diff(equity) / np.where(np.abs(equity[:-1]) > 1e-12, equity[:-1], 1.0)
        return BacktestResult(
            equity_curve=equity,
            trades=strat.trades,
            returns=returns,
            final_equity=float(equity[-1]),
            total_commission=total_commission,
            total_slippage=total_slippage,
            total_turnover=total_turnover,
            n_trades=len(strat.trades),
            greeks_history=dict(getattr(strat, "_greeks_history", {})),
        )
