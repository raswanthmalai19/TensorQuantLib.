"""Tests for Black-Scholes pricing and Greeks — analytic vs autograd."""

import numpy as np
import pytest
from tensorquantlib.core.tensor import Tensor
from tensorquantlib.finance.black_scholes import (
    bs_price_numpy, bs_price_tensor,
    bs_delta, bs_gamma, bs_vega, bs_theta, bs_rho,
)
from tensorquantlib.finance.greeks import compute_greeks


# Standard test parameters
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2


class TestBSPriceNumpy:
    def test_call_price_positive(self):
        price = bs_price_numpy(S, K, T, r, sigma)
        assert price > 0

    def test_put_price_positive(self):
        price = bs_price_numpy(S, K, T, r, sigma, option_type="put")
        assert price > 0

    def test_put_call_parity(self):
        """C - P = S*exp(-qT) - K*exp(-rT)."""
        C = bs_price_numpy(S, K, T, r, sigma)
        P = bs_price_numpy(S, K, T, r, sigma, option_type="put")
        parity = S - K * np.exp(-r * T)  # q=0
        np.testing.assert_almost_equal(C - P, parity, decimal=10)

    def test_deep_itm_call(self):
        """Deep ITM call ≈ S - K*exp(-rT)."""
        price = bs_price_numpy(200.0, 100.0, 1.0, r, sigma)
        intrinsic = 200.0 - 100.0 * np.exp(-r * 1.0)
        assert price >= intrinsic - 0.01

    def test_deep_otm_call_near_zero(self):
        """Deep OTM call is near zero."""
        price = bs_price_numpy(50.0, 200.0, 0.1, r, sigma)
        assert price < 0.01

    def test_vectorized(self):
        """BS price works on arrays."""
        S_arr = np.array([90.0, 100.0, 110.0])
        prices = bs_price_numpy(S_arr, K, T, r, sigma)
        assert prices.shape == (3,)
        assert all(p > 0 for p in prices)


class TestBSPriceTensor:
    def test_matches_numpy(self):
        """Tensor-based BS must match analytic NumPy version."""
        price_np = bs_price_numpy(S, K, T, r, sigma)
        price_t = bs_price_tensor(S, K, T, r, sigma)
        np.testing.assert_almost_equal(float(price_t.data), price_np, decimal=10)

    def test_put_matches_numpy(self):
        price_np = bs_price_numpy(S, K, T, r, sigma, option_type="put")
        price_t = bs_price_tensor(S, K, T, r, sigma, option_type="put")
        np.testing.assert_almost_equal(float(price_t.data), price_np, decimal=10)

    def test_with_dividend(self):
        q = 0.02
        price_np = bs_price_numpy(S, K, T, r, sigma, q=q)
        price_t = bs_price_tensor(S, K, T, r, sigma, q=q)
        np.testing.assert_almost_equal(float(price_t.data), price_np, decimal=10)


class TestAutoGradDelta:
    def test_delta_call(self):
        """Autograd Delta matches analytic Delta within 1e-4."""
        greeks = compute_greeks(bs_price_tensor, S, K, T, r, sigma)
        analytic = bs_delta(S, K, T, r, sigma)
        np.testing.assert_almost_equal(greeks["delta"], analytic, decimal=4)

    def test_delta_put(self):
        greeks = compute_greeks(bs_price_tensor, S, K, T, r, sigma, option_type="put")
        analytic = bs_delta(S, K, T, r, sigma, option_type="put")
        np.testing.assert_almost_equal(greeks["delta"], analytic, decimal=4)

    def test_delta_range(self):
        """Call delta should be in [0, 1]."""
        greeks = compute_greeks(bs_price_tensor, S, K, T, r, sigma)
        assert 0 <= greeks["delta"] <= 1


class TestAutoGradVega:
    def test_vega_call(self):
        """Autograd Vega matches analytic Vega within 1e-3."""
        greeks = compute_greeks(bs_price_tensor, S, K, T, r, sigma)
        analytic = bs_vega(S, K, T, r, sigma)
        np.testing.assert_almost_equal(greeks["vega"], analytic, decimal=3)

    def test_vega_positive(self):
        """Vega should be positive (higher vol → higher price)."""
        greeks = compute_greeks(bs_price_tensor, S, K, T, r, sigma)
        assert greeks["vega"] > 0


class TestFiniteDiffGamma:
    def test_gamma_call(self):
        """Finite-diff Gamma matches analytic Gamma within 1e-2."""
        greeks = compute_greeks(bs_price_tensor, S, K, T, r, sigma)
        analytic = bs_gamma(S, K, T, r, sigma)
        np.testing.assert_almost_equal(greeks["gamma"], analytic, decimal=2)

    def test_gamma_positive(self):
        """Gamma should be positive for both calls and puts."""
        greeks = compute_greeks(bs_price_tensor, S, K, T, r, sigma)
        assert greeks["gamma"] > 0


class TestAnalyticGreeks:
    """Sanity checks on the analytic Greek formulas."""

    def test_delta_call_atm(self):
        """ATM call delta ≈ 0.5 (slightly above due to drift)."""
        d = bs_delta(S, K, T, r, sigma)
        assert 0.4 < d < 0.7

    def test_gamma_atm_positive(self):
        g = bs_gamma(S, K, T, r, sigma)
        assert g > 0

    def test_vega_atm_positive(self):
        v = bs_vega(S, K, T, r, sigma)
        assert v > 0

    def test_theta_call_negative(self):
        """Theta for a call is typically negative (time decay)."""
        th = bs_theta(S, K, T, r, sigma)
        assert th < 0

    def test_rho_call_positive(self):
        """Higher rates → higher call price."""
        rh = bs_rho(S, K, T, r, sigma)
        assert rh > 0
