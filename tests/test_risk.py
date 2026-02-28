"""Tests for risk metrics module."""
from __future__ import annotations

import numpy as np
import pytest

from tensorquantlib.finance.risk import (
    var_parametric,
    var_historical,
    cvar,
    var_mc,
    scenario_analysis,
    greeks_portfolio,
    OptionPosition,
    PortfolioRisk,
)


class TestVaRParametric:
    def test_positive(self):
        v = var_parametric(0.0, 0.20, alpha=0.95)
        assert v > 0

    def test_higher_sigma_higher_var(self):
        v1 = var_parametric(0.0, 0.20, alpha=0.95)
        v2 = var_parametric(0.0, 0.40, alpha=0.95)
        assert v2 > v1

    def test_higher_alpha_higher_var(self):
        v95 = var_parametric(0.0, 0.20, alpha=0.95)
        v99 = var_parametric(0.0, 0.20, alpha=0.99)
        assert v99 > v95

    def test_known_value(self):
        """1-day 95% VaR for sigma=20% annual ~ sigma/sqrt(252) * 1.645."""
        v = var_parametric(0.0, 0.20, alpha=0.95, horizon=1.0 / 252.0)
        expected = 0.20 / np.sqrt(252.0) * 1.6449
        assert abs(v - expected) < 0.001


class TestVaRHistorical:
    def test_positive(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0, 0.01, 1000)
        assert var_historical(r, alpha=0.95) > 0

    def test_higher_alpha_higher_var(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0, 0.01, 10000)
        v95 = var_historical(r, alpha=0.95)
        v99 = var_historical(r, alpha=0.99)
        assert v99 >= v95


class TestCVaR:
    def test_cvar_ge_var(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0, 0.01, 10000)
        v = var_historical(r, alpha=0.95)
        es = cvar(r, alpha=0.95)
        assert es >= v - 1e-10

    def test_positive(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0, 0.01, 5000)
        assert cvar(r, alpha=0.95) > 0


class TestVaRMC:
    def test_positive(self):
        v, es = var_mc(100.0, 0.20, alpha=0.95, seed=42)
        assert v > 0 and es >= v

    def test_longer_horizon_higher_var(self):
        v1, _ = var_mc(100.0, 0.20, horizon=1.0 / 252, alpha=0.95, seed=42)
        v2, _ = var_mc(100.0, 0.20, horizon=10.0 / 252, alpha=0.95, seed=42)
        assert v2 > v1


class TestScenarioAnalysis:
    def test_basic_call(self):
        # Long call: value at S=100 is max(100-100,0)=0; at S=120 it's 20; at S=80 it's 0
        # Use a slightly ITM payoff to get meaningful P&L in both directions
        def value(s: float) -> float:
            return max(s - 90.0, 0.0)  # strike 90, so both S=80 and S=120 have positive value

        results = scenario_analysis(100.0, value, {"crash": 80.0, "rally": 120.0})
        assert "crash" in results and "rally" in results
        # At S=100: value = 10; at crash S=80: value = 0 → pnl = -10
        assert results["crash"]["pnl"] < 0   # loses value in crash
        assert results["rally"]["pnl"] > 0   # gains value in rally


class TestGreeksPortfolio:
    def test_single_call(self):
        pos = [OptionPosition("call", K=100, T=1.0, sigma=0.20)]
        g = greeks_portfolio(pos, S=100, r=0.05)
        assert 0.0 < g["delta"] < 1.0
        assert g["gamma"] > 0
        assert g["value"] > 0

    def test_straddle_zero_delta(self):
        """Long call + short call = zero position; delta should be ~0."""
        pos = [
            OptionPosition("call", K=100, T=1.0, sigma=0.20, quantity=1),
            OptionPosition("call", K=100, T=1.0, sigma=0.20, quantity=-1),
        ]
        g = greeks_portfolio(pos, S=100, r=0.05)
        assert abs(g["delta"]) < 1e-10  # exact cancellation

    def test_short_call_negative_delta(self):
        pos = [OptionPosition("call", K=100, T=1.0, sigma=0.20, quantity=-1)]
        g = greeks_portfolio(pos, S=100, r=0.05)
        assert g["delta"] < 0


class TestPortfolioRisk:
    @pytest.fixture
    def sample_returns(self) -> np.ndarray:
        return np.random.default_rng(42).normal(0.0005, 0.015, 252)

    def test_summary_keys(self, sample_returns):
        pr = PortfolioRisk(sample_returns)
        s = pr.summary()
        assert "var_95" in s
        assert "cvar_95" in s
        assert "sharpe" in s
        assert "max_drawdown" in s

    def test_max_drawdown_negative(self, sample_returns):
        pr = PortfolioRisk(sample_returns)
        assert pr.max_drawdown() <= 0

    def test_volatility_positive(self, sample_returns):
        pr = PortfolioRisk(sample_returns)
        assert pr.volatility() > 0

    def test_cvar_ge_var(self, sample_returns):
        pr = PortfolioRisk(sample_returns)
        assert pr.cvar() >= pr.var() - 1e-10
