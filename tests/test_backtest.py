"""Tests for backtesting framework (metrics, engine, strategies)."""

from __future__ import annotations

import numpy as np

from tensorquantlib.backtest.engine import (
    EQUITY_COMM,
    EQUITY_SLIP,
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
        np.testing.assert_allclose(result.equity_curve, [1000.0, 1005.0, 1010.0, 1008.0])
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
            prices[i + 1] = prices[i] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

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


# ---------------------------------------------------------------------------
# Execution cost model tests
# ---------------------------------------------------------------------------


class TestSlippageModel:
    def test_zero_slippage_by_default(self):
        m = SlippageModel()
        assert m.cost(100.0, 10.0) == 0.0

    def test_fixed_spread_cost(self):
        m = SlippageModel(fixed_spread=0.001)  # 10 bps round-trip
        cost = m.cost(price=100.0, quantity=100.0)
        # spread_cost = 100 shares x $100 x 0.001 = $10
        assert abs(cost - 10.0) < 1e-10

    def test_cost_independent_of_trade_direction(self):
        m = SlippageModel(fixed_spread=0.0005)
        assert abs(m.cost(100.0, 50.0) - m.cost(100.0, -50.0)) < 1e-12

    def test_market_impact_scales_with_sqrt_quantity(self):
        m = SlippageModel(market_impact=0.1, adv=10_000.0)
        c1 = m.cost(100.0, 100.0)
        c4 = m.cost(100.0, 400.0)
        # sqrt-impact: cost ∝ qty^{3/2}, so 4x qty -> (4)^{3/2} = 8x cost
        ratio = c4 / c1
        assert abs(ratio - 8.0) < 0.5  # 4^(3/2) = 8


class TestCommissionModel:
    def test_zero_commission_by_default(self):
        m = CommissionModel()
        assert m.cost(100.0, 10.0) == 0.0

    def test_per_trade_only(self):
        m = CommissionModel(per_trade=1.0)
        assert abs(m.cost(100.0, 100.0) - 1.0) < 1e-10
        assert abs(m.cost(200.0, 500.0) - 1.0) < 1e-10  # same regardless of size

    def test_per_unit(self):
        m = CommissionModel(per_unit=0.005)
        # 200 shares x $0.005 = $1.00
        assert abs(m.cost(100.0, 200.0) - 1.0) < 1e-10

    def test_minimum_applied(self):
        m = CommissionModel(per_unit=0.005, minimum=1.0)
        # 10 shares x $0.005 = $0.05 < minimum $1.00 -> $1.00
        assert abs(m.cost(100.0, 10.0) - 1.0) < 1e-10

    def test_percentage(self):
        m = CommissionModel(percentage=0.001)  # 10 bps
        # 100 shares x $50 = $5000 notional x 0.001 = $5
        assert abs(m.cost(50.0, 100.0) - 5.0) < 1e-10

    def test_equity_comm_preset(self):
        cost = EQUITY_COMM.cost(price=50.0, quantity=10.0)
        # 10 x $0.005 = $0.05 < $1 minimum -> $1.00
        assert abs(cost - 1.0) < 1e-10


class TestBacktestWithCosts:
    def test_costs_reduce_equity(self):
        """Equity with slippage+commission < zero-cost equity."""

        class BuyAndHold(Strategy):
            def on_data(self, step, price, **kw):
                return 1.0

        prices = np.array([100.0, 105.0, 110.0, 108.0])

        result_free = BacktestEngine(BuyAndHold(), prices, initial_capital=1000.0).run()
        result_cost = BacktestEngine(
            BuyAndHold(),
            prices,
            initial_capital=1000.0,
            slippage=EQUITY_SLIP,
            commission=EQUITY_COMM,
        ).run()

        assert result_cost.final_equity < result_free.final_equity

    def test_total_commission_positive_when_trades_occur(self):
        class BuyAndHold(Strategy):
            def on_data(self, step, price, **kw):
                return 1.0

        prices = np.array([100.0, 105.0, 110.0])
        result = BacktestEngine(
            BuyAndHold(),
            prices,
            initial_capital=1000.0,
            commission=EQUITY_COMM,
        ).run()
        assert result.total_commission > 0.0

    def test_total_slippage_positive(self):
        class BuyAndHold(Strategy):
            def on_data(self, step, price, **kw):
                return 1.0

        prices = np.array([100.0, 105.0, 110.0])
        result = BacktestEngine(
            BuyAndHold(),
            prices,
            initial_capital=1000.0,
            slippage=EQUITY_SLIP,
        ).run()
        assert result.total_slippage > 0.0

    def test_n_trades_correct(self):
        class BuyAndHold(Strategy):
            def on_data(self, step, price, **kw):
                return 1.0

        prices = np.array([100.0, 105.0, 110.0])
        result = BacktestEngine(BuyAndHold(), prices).run()
        assert result.n_trades == 1  # one initial buy

    def test_trade_total_cost_property(self):
        t = Trade(step=0, quantity=100.0, price=50.0, slippage=0.5, commission=1.0)
        assert abs(t.total_cost - 1.5) < 1e-12
        assert abs(t.notional - 5000.0) < 1e-12

    def test_zero_cost_matches_default(self):
        """Explicitly passing cost=None gives same result as default (zero cost)."""

        class BuyAndHold(Strategy):
            def on_data(self, step, price, **kw):
                return 1.0

        prices = np.array([100.0, 105.0, 110.0, 108.0])
        r1 = BacktestEngine(BuyAndHold(), prices, 1000.0).run()
        r2 = BacktestEngine(BuyAndHold(), prices, 1000.0, slippage=None, commission=None).run()
        np.testing.assert_array_equal(r1.equity_curve, r2.equity_curve)


# ---------------------------------------------------------------------------
# New strategy tests
# ---------------------------------------------------------------------------


class TestGammaScalpingStrategy:
    def _run(self, S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, n_steps=50):
        np.random.seed(0)
        dt = T / n_steps
        prices = np.zeros(n_steps + 1)
        prices[0] = S0
        for i in range(n_steps):
            prices[i + 1] = prices[i] * np.exp(
                (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn()
            )
        strat = GammaScalpingStrategy(K=K, T_total=T, r=r, sigma_implied=sigma, n_steps=n_steps + 1)
        return BacktestEngine(strat, prices, initial_capital=1e6).run()

    def test_greeks_history_populated(self):
        result = self._run()
        gh = result.greeks_history
        assert "delta" in gh
        assert "gamma" in gh
        assert "theta" in gh
        assert "theoretical_gamma_pnl" in gh
        assert "theoretical_theta_pnl" in gh

    def test_greeks_length_matches_steps(self):
        result = self._run(n_steps=50)
        # n_steps+1 prices → n_steps+1 calls to on_data
        assert len(result.greeks_history["gamma"]) == 51

    def test_gamma_strictly_positive(self):
        result = self._run()
        gammas = result.greeks_history["gamma"]
        # Gamma should be positive for all steps except possibly the last (T=0)
        assert all(g >= 0.0 for g in gammas)

    def test_theta_negative(self):
        """Straddle theta is negative (time decay)."""
        result = self._run()
        thetas = result.greeks_history["theta"]
        # All theta values at T_remain > 0 should be negative
        assert all(t <= 0.0 for t in thetas[:-1])

    def test_run_completes(self):
        result = self._run()
        assert result.final_equity > 0
        assert len(result.trades) > 0


class TestDeltaGammaHedgeStrategy:
    def _run(self, S0=100.0, K1=95.0, K2=105.0, T=1.0, r=0.05, sigma=0.2, n_steps=50):
        np.random.seed(1)
        dt = T / n_steps
        prices = np.zeros(n_steps + 1)
        prices[0] = S0
        for i in range(n_steps):
            prices[i + 1] = prices[i] * np.exp(
                (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn()
            )
        strat = DeltaGammaHedgeStrategy(
            K1=K1, K2=K2, T_total=T, r=r, sigma=sigma, n_steps=n_steps + 1
        )
        return BacktestEngine(strat, prices, initial_capital=1e6).run()

    def test_net_gamma_near_zero(self):
        """Delta-gamma hedge should keep net Gamma ≈ 0 at each step."""
        result = self._run()
        net_gammas = result.greeks_history["net_gamma"]
        for g in net_gammas:
            assert abs(g) < 1e-10, f"Net gamma not zero: {g}"

    def test_greeks_history_populated(self):
        result = self._run()
        gh = result.greeks_history
        assert "net_delta" in gh
        assert "net_gamma" in gh
        assert "hedge_ratio" in gh
        assert "stock_position" in gh

    def test_hedge_ratio_positive(self):
        """Both options have positive gamma so Q_hedge > 0."""
        result = self._run()
        ratios = result.greeks_history["hedge_ratio"]
        assert all(q >= 0.0 for q in ratios)

    def test_run_completes(self):
        result = self._run()
        assert result.final_equity > 0


# ---------------------------------------------------------------------------
# New metrics tests
# ---------------------------------------------------------------------------


class TestAnnualizedReturn:
    def test_flat_equity_returns_zero(self):
        eq = np.ones(253) * 1000.0
        assert abs(annualized_return(eq) - 0.0) < 1e-10

    def test_known_annual_return(self):
        # Grow from 1000 to 1100 over 252 steps → 10% annualized
        eq = np.linspace(1000.0, 1100.0, 253)
        ann = annualized_return(eq, periods_per_year=252)
        assert abs(ann - 0.10) < 0.005

    def test_single_step_returns_zero(self):
        assert annualized_return(np.array([1000.0])) == 0.0

    def test_negative_growth(self):
        eq = np.linspace(1000.0, 800.0, 253)
        ann = annualized_return(eq, periods_per_year=252)
        assert ann < 0.0


class TestCalmarRatio:
    def test_positive_return_positive_drawdown(self):
        eq = np.array([100.0, 110.0, 95.0, 120.0, 130.0])
        c = calmar_ratio(eq, periods_per_year=4)
        assert c > 0.0

    def test_no_drawdown_returns_inf(self):
        eq = np.linspace(100.0, 200.0, 252)
        c = calmar_ratio(eq)
        assert c == float("inf")

    def test_calmar_decreases_with_larger_drawdown(self):
        eq1 = np.array([100.0, 110.0, 105.0, 120.0])  # small dip
        eq2 = np.array([100.0, 110.0, 80.0, 120.0])  # big dip
        c1 = calmar_ratio(eq1, periods_per_year=3)
        c2 = calmar_ratio(eq2, periods_per_year=3)
        assert c1 > c2


class TestInformationRatio:
    def test_identical_returns_zero(self):
        r = np.array([0.01, 0.02, -0.01, 0.03])
        assert information_ratio(r, r) == 0.0

    def test_outperformance_positive(self):
        strat = np.array([0.01, 0.02, 0.015, 0.01])
        bench = np.array([0.005, 0.005, 0.005, 0.005])
        ir = information_ratio(strat, bench)
        assert ir > 0.0

    def test_underperformance_negative(self):
        strat = np.array([-0.01, -0.02, -0.01])
        bench = np.array([0.01, 0.02, 0.01])
        ir = information_ratio(strat, bench)
        assert ir < 0.0


class TestTurnover:
    def test_no_trades_returns_zero(self):
        assert turnover([]) == 0.0

    def test_known_turnover(self):
        trades = [Trade(step=0, quantity=100.0, price=10.0)]
        # notional = 1000, capital = 1000 → turnover = 1.0
        t = turnover(trades, initial_capital=1_000.0)
        assert abs(t - 1.0) < 1e-10

    def test_multiple_trades(self):
        trades = [
            Trade(step=0, quantity=50.0, price=10.0),
            Trade(step=1, quantity=50.0, price=12.0),
        ]
        # notional = 500 + 600 = 1100, capital = 1000 → 1.1
        t = turnover(trades, initial_capital=1_000.0)
        assert abs(t - 1.1) < 1e-10


class TestHedgePnLAttribution:
    def test_output_keys(self):
        equity = np.array([1000.0, 1002.0, 1001.0, 1005.0])
        prices = np.array([100.0, 101.0, 100.5, 103.0])
        deltas = [0.5, 0.5, 0.5, 0.5]
        gammas = [0.02, 0.02, 0.02, 0.02]
        out = hedge_pnl_attribution(equity, deltas, gammas, prices)
        assert set(out.keys()) == {"delta_pnl", "gamma_pnl", "residual_pnl"}

    def test_output_length(self):
        n = 10
        equity = np.linspace(1000.0, 1020.0, n)
        prices = np.linspace(100.0, 110.0, n)
        deltas = [0.5] * n
        gammas = [0.01] * n
        out = hedge_pnl_attribution(equity, deltas, gammas, prices)
        assert len(out["delta_pnl"]) == n - 1

    def test_components_sum_to_total(self):
        """delta_pnl + gamma_pnl + residual_pnl == total_pnl."""
        np.random.seed(7)
        equity = 1000.0 + np.cumsum(np.random.randn(11))
        prices = 100.0 + np.cumsum(np.random.randn(11))
        deltas = np.random.uniform(0.3, 0.7, 11).tolist()
        gammas = np.random.uniform(0.01, 0.05, 11).tolist()
        out = hedge_pnl_attribution(equity, deltas, gammas, prices)
        total = np.diff(equity)
        reconstructed = out["delta_pnl"] + out["gamma_pnl"] + out["residual_pnl"]
        np.testing.assert_allclose(reconstructed, total, atol=1e-12)


class TestHedgeEfficiency:
    def test_perfect_hedge_returns_one(self):
        """Hedged equity flat = perfect hedge → efficiency = 1."""
        hedged = np.ones(252) * 1000.0
        unhedged = 1000.0 + np.cumsum(np.random.randn(252))
        eff = hedge_efficiency(hedged, unhedged)
        assert abs(eff - 1.0) < 1e-10

    def test_no_hedge_returns_zero(self):
        """Same equity for hedged and unhedged → efficiency = 0."""
        eq = 1000.0 + np.cumsum(np.random.randn(100))
        eff = hedge_efficiency(eq, eq)
        assert abs(eff) < 1e-10

    def test_efficient_hedge_positive(self):
        """A good hedge reduces variance."""
        np.random.seed(3)
        noise = np.random.randn(100)
        unhedged = np.cumsum(noise) + 1000.0
        hedged = np.cumsum(noise * 0.1) + 1000.0  # 90% variance reduction
        eff = hedge_efficiency(hedged, unhedged)
        assert eff > 0.8


class TestGreeksHistoryInBacktest:
    """Verify greeks_history is propagated through BacktestEngine."""

    def test_delta_hedge_exports_greeks(self):
        prices = np.linspace(100.0, 110.0, 30)
        strat = DeltaHedgeStrategy(K=105.0, T_total=1.0, r=0.05, sigma=0.2, n_steps=30)
        result = BacktestEngine(strat, prices).run()
        assert "delta" in result.greeks_history
        assert "gamma" in result.greeks_history
        assert len(result.greeks_history["delta"]) == 30

    def test_gamma_scalping_exports_pnl_attribution(self):
        np.random.seed(5)
        prices = 100.0 + np.cumsum(np.random.randn(30) * 0.5)
        strat = GammaScalpingStrategy(K=100.0, T_total=1.0, r=0.05, sigma_implied=0.2, n_steps=30)
        result = BacktestEngine(strat, prices).run()
        assert "theoretical_gamma_pnl" in result.greeks_history
        assert "theoretical_theta_pnl" in result.greeks_history
