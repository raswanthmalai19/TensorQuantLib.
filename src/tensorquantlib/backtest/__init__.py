"""Backtesting framework: strategy simulation and P&L tracking."""
from tensorquantlib.backtest.metrics import (
    sharpe_ratio,
    max_drawdown,
    sortino_ratio,
    win_rate,
    profit_factor,
    annualized_return,
    calmar_ratio,
    information_ratio,
    turnover,
    hedge_pnl_attribution,
    hedge_efficiency,
)
from tensorquantlib.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    SlippageModel,
    CommissionModel,
    ZERO_COST,
    EQUITY_COMM,
    FX_COMM,
    EQUITY_SLIP,
    ILLIQUID_SLIP,
)
from tensorquantlib.backtest.strategy import (
    Strategy,
    Trade,
    DeltaHedgeStrategy,
    GammaScalpingStrategy,
    DeltaGammaHedgeStrategy,
    StraddleStrategy,
)

__all__ = [
    # metrics
    "sharpe_ratio", "max_drawdown", "sortino_ratio", "win_rate", "profit_factor",
    "annualized_return", "calmar_ratio", "information_ratio", "turnover",
    "hedge_pnl_attribution", "hedge_efficiency",
    # engine
    "BacktestEngine", "BacktestResult",
    "SlippageModel", "CommissionModel",
    "ZERO_COST", "EQUITY_COMM", "FX_COMM", "EQUITY_SLIP", "ILLIQUID_SLIP",
    # strategy
    "Strategy", "Trade",
    "DeltaHedgeStrategy", "GammaScalpingStrategy",
    "DeltaGammaHedgeStrategy", "StraddleStrategy",
]
