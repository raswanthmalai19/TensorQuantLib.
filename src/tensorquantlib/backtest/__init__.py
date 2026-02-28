"""Backtesting framework: strategy simulation and P&L tracking."""
from tensorquantlib.backtest.metrics import (
    sharpe_ratio,
    max_drawdown,
    sortino_ratio,
    win_rate,
    profit_factor,
)
from tensorquantlib.backtest.engine import BacktestEngine, BacktestResult
from tensorquantlib.backtest.strategy import (
    Strategy,
    DeltaHedgeStrategy,
    StraddleStrategy,
)

__all__ = [
    "sharpe_ratio",
    "max_drawdown",
    "sortino_ratio",
    "win_rate",
    "profit_factor",
    "BacktestEngine",
    "BacktestResult",
    "Strategy",
    "DeltaHedgeStrategy",
    "StraddleStrategy",
]
