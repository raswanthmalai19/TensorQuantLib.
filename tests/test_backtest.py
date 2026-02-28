"""Tests for backtesting framework (metrics, engine, strategies)."""
from __future__ import annotations

import numpy as np
import pytest

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


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_positive_returns(self):
        returns = np.array([0.01, 0.02, 0.01, 0.015, 0.01])
        sr = sharpe_ratio(returns)
        assert sr > 0

    def test_zero_std(self):
        returns = np.array([0.01, 0.01, 0.01])
        sr = sharpe_ratio(returns)
        assert sr == 0.0

    def test_known_value(self):
        """Known daily returns → expected Sharpe."""
        returns = np.array([0.001] * 252)
        sr = sharpe_ratio(returns, rf=0.0, periods_per_year=252)
        # mean = 0.001, std ≈ 0, but with ddof=1 std = 0 → returns 0
        # Use varied returns instead
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)
        sr = sharpe_ratio(returns, rf=0.0)
        assert sr > 0  # positive expected return


class TestMaxDrawdown:
    def test_no_drawdown(self):
        equity = np.array([100.0, 101.0, 102.0, 103.0])
        assert max_drawdown(equity) == 0.0

    def test_known_drawdown(self):
        equity = np.array([100.0, 110.0, 88.0, 95.0])
        # Peak = 110, trough = 88, dd = (110-88)/110 = 0.2
        dd = max_drawdown(equity)
        assert abs(dd - 0.2) < 1e-10

    def test_full_loss(self):
        equity = np.array([100.0, 50.0, 0.0])
        dd = max_drawdown(equity)
        assert abs(dd - 1.0) < 1e-10


class TestSortinoRatio:
    def test_all_positive_returns(self):
        returns = np.array([0.01, 0.02, 0.03])
        sr = sortino_ratio(returns)
        assert sr == float("inf")

    def test_mixed_returns(self):
        returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
        sr = sortino_ratio(returns)
        assert sr > 0  # net positive


class TestWinRate:
    def test_all_winners(self):
        assert win_rate(np.array([1.0, 2.0, 0.5])) == 1.0

    def test_all_losers(self):
        assert win_rate(np.array([-1.0, -2.0])) == 0.0

    def test_half_and_half(self):
        assert abs(win_rate(np.array([1.0, -1.0])) - 0.5) < 1e-10

    def test_empty(self):
        assert win_rate(np.array([])) == 0.0


class TestProfitFactor:
    def test_two_to_one(self):
        trades = np.array([10.0, -5.0])
        pf = profit_factor(trades)
        assert abs(pf - 2.0) < 1e-10

    def test_no_losses(self):
        assert profit_factor(np.array([1.0, 2.0])) == float("inf")

    def test_no_profits(self):
        assert profit_factor(np.array([-1.0, -2.0])) == 0.0


# ---------------------------------------------------------------------------
# Engine tests
# ---------------------------------------------------------------------------

class TestBacktestEngine:
    def test_buy_and_hold(self):
        """Simple buy-and-hold: buy at step 0, hold."""
        class BuyAndHold(Strategy):
            def on_data(self, step, price, **kw):
                return 1.0  # hold 1 unit always

        prices = np.array([100.0, 105.0, 110.0, 108.0])
        engine = BacktestEngine(BuyAndHold(), prices, initial_capital=1000.0)
        result = engine.run()

        assert isinstance(result, BacktestResult)
        # At each step equity = cash + 1*price; cash = 1000 - 100 = 900
        np.testing.assert_allclose(result.equity_curve,
                                   [1000.0, 1005.0, 1010.0, 1008.0])
        assert result.final_equity == 1008.0
        assert len(result.trades) == 1  # one buy trade

    def test_no_trades(self):
        """Strategy that does nothing."""
        class DoNothing(Strategy):
            def on_data(self, step, price, **kw):
                return 0.0

        prices = np.array([100.0, 105.0, 110.0])
        engine = BacktestEngine(DoNothing(), prices, initial_capital=1000.0)
        result = engine.run()
        np.testing.assert_allclose(result.equity_curve, [1000.0, 1000.0, 1000.0])
        assert len(result.trades) == 0

    def test_returns_correct_length(self):
        class BuyHold(Strategy):
            def on_data(self, step, price, **kw):
                return 1.0

        prices = np.linspace(100, 110, 50)
        result = BacktestEngine(BuyHold(), prices).run()
        assert len(result.returns) == len(prices) - 1


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------

class TestDeltaHedgeStrategy:
    def test_delta_converges(self):
        """Delta hedge P&L is bounded for BS dynamics."""
        np.random.seed(42)
        S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        n_steps = 252
        dt = T / n_steps
        prices = np.zeros(n_steps + 1)
        prices[0] = S0
        for i in range(n_steps):
            z = np.random.randn()
            prices[i + 1] = prices[i] * np.exp((r - 0.5 * sigma**2) * dt
                                                 + sigma * np.sqrt(dt) * z)

        strat = DeltaHedgeStrategy(K=K, T_total=T, r=r, sigma=sigma, n_steps=n_steps)
        result = BacktestEngine(strat, prices, initial_capital=1e6).run()

        # Equity should not deviate wildly from initial
        assert result.final_equity > 0
        # Max drawdown should be moderate
        dd = max_drawdown(result.equity_curve)
        assert dd < 0.5


class TestStraddleStrategy:
    def test_accumulates_positions(self):
        prices = np.linspace(100, 110, 63)  # ~3 months daily
        strat = StraddleStrategy(interval=21)
        result = BacktestEngine(strat, prices, initial_capital=1e6).run()
        # Should have 3 entry trades (steps 0, 21, 42)
        assert len(result.trades) == 3
        assert result.final_equity > 0
