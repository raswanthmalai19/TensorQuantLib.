"""Backtesting framework: strategy simulation and P&L tracking."""

from tensorquantlib.backtest.engine import (
    EQUITY_COMM,
    EQUITY_SLIP,
    FX_COMM,
    ILLIQUID_SLIP,
    ZERO_COST,
    BacktestEngine,
    BacktestResult,
    CommissionModel,
    SlippageModel,
)
from tensorquantlib.backtest.metrics import (
    annualized_return,
    calmar_ratio,
    hedge_efficiency,
    hedge_pnl_attribution,
    information_ratio,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    turnover,
    win_rate,
)
from tensorquantlib.backtest.strategy import (
    DeltaGammaHedgeStrategy,
    DeltaHedgeStrategy,
    GammaScalpingStrategy,
    StraddleStrategy,
    Strategy,
    Trade,
)

__all__ = [
    # metrics
    "annualized_return",
    "calmar_ratio",
    "hedge_efficiency",
    "hedge_pnl_attribution",
    "information_ratio",
    "max_drawdown",
    "profit_factor",
    "sharpe_ratio",
    "sortino_ratio",
    "turnover",
    "win_rate",
    # engine
    "BacktestEngine",
    "BacktestResult",
    "CommissionModel",
    "EQUITY_COMM",
    "EQUITY_SLIP",
    "FX_COMM",
    "ILLIQUID_SLIP",
    "SlippageModel",
    "ZERO_COST",
    # strategy
    "DeltaGammaHedgeStrategy",
    "DeltaHedgeStrategy",
    "GammaScalpingStrategy",
    "StraddleStrategy",
    "Strategy",
    "Trade",
]
