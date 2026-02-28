"""Tests for Heston stochastic volatility model."""
from __future__ import annotations

import numpy as np
import pytest

from tensorquantlib.finance.heston import (
    HestonParams,
    heston_price,
    heston_price_mc,
    heston_greeks,
)


@pytest.fixture
def default_params() -> HestonParams:
    return HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)


class TestHestonParams:
    def test_feller_satisfied(self):
        p = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=0.0, v0=0.04)
        # 2*2*0.04 = 0.16 > 0.09 = 0.3^2
        assert p.feller_satisfied()

    def test_feller_violated(self):
        p = HestonParams(kappa=0.1, theta=0.01, xi=1.0, rho=0.0, v0=0.01)
        # 2*0.1*0.01 = 0.002 < 1.0 = 1.0^2
        assert not p.feller_satisfied()

    def test_roundtrip_array(self):
        p = HestonParams(kappa=1.5, theta=0.05, xi=0.4, rho=-0.5, v0=0.06)
        p2 = HestonParams.from_array(p.to_array())
        assert abs(p.kappa - p2.kappa) < 1e-10


class TestHestonPrice:
    def test_call_positive(self, default_params):
        price = heston_price(100, 100, 1.0, 0.05, default_params)
        assert price > 0

    def test_put_positive(self, default_params):
        price = heston_price(100, 100, 1.0, 0.05, default_params, option_type="put")
        assert price > 0

    def test_put_call_parity(self, default_params):
        """C - P = S*exp(-qT) - K*exp(-rT)"""
        S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0
        call = heston_price(S, K, T, r, default_params, q=q, option_type="call")
        put  = heston_price(S, K, T, r, default_params, q=q, option_type="put")
        parity = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert abs((call - put) - parity) < 0.5  # loose tolerance due to quad

    def test_mc_vs_analytic_atm(self, default_params):
        """MC and analytic price should agree within reasonable tolerance for ATM."""
        price_analytic = heston_price(100, 100, 1.0, 0.05, default_params)
        price_mc, stderr = heston_price_mc(100, 100, 1.0, 0.05, default_params,
                                           n_paths=300_000, n_steps=100, seed=42,
                                           return_stderr=True)
        # Allow up to 10*stderr + 1.5 to account for Euler discretisation bias
        assert abs(price_analytic - price_mc) < 10 * float(stderr) + 1.5

    def test_itm_call_bounded(self, default_params):
        """Deep ITM call should approach S - K*exp(-rT)."""
        S, K, T, r = 200, 100, 1.0, 0.05
        price = heston_price(S, K, T, r, default_params)
        intrinsic = S - K * np.exp(-r * T)
        assert price >= intrinsic - 0.5  # within numeric tolerance


class TestHestonGreeks:
    def test_delta_call_in_01(self, default_params):
        greeks = heston_greeks(100, 100, 1.0, 0.05, default_params)
        assert 0.0 < greeks["delta"] < 1.0

    def test_gamma_positive(self, default_params):
        greeks = heston_greeks(100, 100, 1.0, 0.05, default_params)
        assert greeks["gamma"] > 0.0

    def test_vega_positive_for_call(self, default_params):
        greeks = heston_greeks(100, 100, 1.0, 0.05, default_params)
        assert greeks["vega"] > 0.0

    def test_mc_vs_mc_price(self, default_params):
        """Heston MC price consistency check (same params, different seed = close)."""
        p1 = heston_price_mc(100, 100, 1.0, 0.05, default_params, n_paths=50_000, seed=1)
        p2 = heston_price_mc(100, 100, 1.0, 0.05, default_params, n_paths=50_000, seed=2)
        assert abs(float(p1) - float(p2)) < 2.0  # within 2 dollars
