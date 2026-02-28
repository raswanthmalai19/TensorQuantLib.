"""Tests for FX options (Garman-Kohlhagen and Quanto)."""
from __future__ import annotations

import numpy as np
import pytest

from tensorquantlib.finance.fx import (
    garman_kohlhagen,
    gk_greeks,
    fx_forward,
    quanto_option,
)
from tensorquantlib.finance.black_scholes import bs_price_numpy


class TestGarmanKohlhagen:
    """Tests for Garman-Kohlhagen FX option model."""

    def test_reduces_to_bs_with_zero_rf(self):
        """With r_f=0, GK should equal standard BS."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        gk_call = garman_kohlhagen(S, K, T, r_d=r, r_f=0.0, sigma=sigma)
        bs_call = float(bs_price_numpy(S, K, T, r, sigma))
        assert abs(gk_call - bs_call) < 1e-10

    def test_put_call_parity(self):
        """GK put-call parity: C - P = S*exp(-r_f*T) - K*exp(-r_d*T)."""
        S, K, T, r_d, r_f, sigma = 1.25, 1.30, 0.5, 0.03, 0.01, 0.1
        call = garman_kohlhagen(S, K, T, r_d, r_f, sigma, 'call')
        put = garman_kohlhagen(S, K, T, r_d, r_f, sigma, 'put')
        parity = S * np.exp(-r_f * T) - K * np.exp(-r_d * T)
        assert abs((call - put) - parity) < 1e-10

    def test_positive_prices(self):
        call = garman_kohlhagen(1.25, 1.30, 0.5, 0.03, 0.01, 0.1, 'call')
        put = garman_kohlhagen(1.25, 1.30, 0.5, 0.03, 0.01, 0.1, 'put')
        assert call > 0
        assert put > 0

    def test_deep_itm_call(self):
        """Deep ITM call: price ≈ S*exp(-r_f*T) - K*exp(-r_d*T)."""
        S, K, T, r_d, r_f, sigma = 150.0, 100.0, 1.0, 0.05, 0.02, 0.2
        call = garman_kohlhagen(S, K, T, r_d, r_f, sigma, 'call')
        intrinsic = S * np.exp(-r_f * T) - K * np.exp(-r_d * T)
        assert call >= intrinsic - 0.01

    def test_higher_vol_higher_price(self):
        c1 = garman_kohlhagen(1.25, 1.25, 1.0, 0.03, 0.01, 0.1)
        c2 = garman_kohlhagen(1.25, 1.25, 1.0, 0.03, 0.01, 0.2)
        assert c2 > c1


class TestGKGreeks:
    """Tests for Garman-Kohlhagen Greeks."""

    def test_call_delta_in_range(self):
        g = gk_greeks(1.25, 1.25, 1.0, 0.03, 0.01, 0.1, 'call')
        assert 0 < g['delta'] < 1

    def test_put_delta_negative(self):
        g = gk_greeks(1.25, 1.25, 1.0, 0.03, 0.01, 0.1, 'put')
        assert -1 < g['delta'] < 0

    def test_gamma_positive(self):
        g = gk_greeks(1.25, 1.25, 1.0, 0.03, 0.01, 0.1, 'call')
        assert g['gamma'] > 0

    def test_vega_positive(self):
        g = gk_greeks(1.25, 1.25, 1.0, 0.03, 0.01, 0.1, 'call')
        assert g['vega'] > 0

    def test_delta_fd_check(self):
        """Finite-difference delta check."""
        eps = 0.001
        S, K, T, r_d, r_f, sigma = 1.25, 1.25, 1.0, 0.03, 0.01, 0.1
        g = gk_greeks(S, K, T, r_d, r_f, sigma, 'call')
        p_up = garman_kohlhagen(S + eps, K, T, r_d, r_f, sigma, 'call')
        p_down = garman_kohlhagen(S - eps, K, T, r_d, r_f, sigma, 'call')
        fd_delta = (p_up - p_down) / (2 * eps)
        assert abs(g['delta'] - fd_delta) < 0.001


class TestFXForward:
    """Tests for FX forward rate."""

    def test_forward_rate(self):
        F = fx_forward(S=1.25, r_d=0.03, r_f=0.01, T=1.0)
        expected = 1.25 * np.exp((0.03 - 0.01) * 1.0)
        assert abs(F - expected) < 1e-10

    def test_equal_rates(self):
        """When r_d = r_f, forward = spot."""
        F = fx_forward(S=1.25, r_d=0.03, r_f=0.03, T=1.0)
        assert abs(F - 1.25) < 1e-10


class TestQuanto:
    """Tests for Quanto options."""

    def test_reduces_to_bs_when_no_fx_vol(self):
        """Quanto with sigma_fx=0 and rho=0 should be fx_rate * BS(r_d)."""
        S, K, T, r_d, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        quanto = quanto_option(S, K, T, r_d=r_d, r_f=0.02, sigma_s=sigma,
                                sigma_fx=0.0, rho=0.0, fx_rate=1.0)
        # With rho=0 and sigma_fx=0, quanto-adjusted rate = r_d
        bs_call = float(bs_price_numpy(S, K, T, r_d, sigma))
        assert abs(quanto - bs_call) < 0.01

    def test_positive_prices(self):
        call = quanto_option(100, 100, 1.0, 0.05, 0.02, 0.2, 0.1, -0.3, 1.0, 'call')
        put = quanto_option(100, 100, 1.0, 0.05, 0.02, 0.2, 0.1, -0.3, 1.0, 'put')
        assert call > 0
        assert put > 0

    def test_fx_rate_scaling(self):
        """Doubling fx_rate should double the price."""
        p1 = quanto_option(100, 100, 1.0, 0.05, 0.02, 0.2, 0.1, -0.3, 1.0)
        p2 = quanto_option(100, 100, 1.0, 0.05, 0.02, 0.2, 0.1, -0.3, 2.0)
        assert abs(p2 - 2.0 * p1) < 1e-10

    def test_negative_rho_increases_call(self):
        """Negative correlation should increase quanto call price."""
        c_zero = quanto_option(100, 100, 1.0, 0.05, 0.02, 0.2, 0.1, 0.0, 1.0)
        c_neg = quanto_option(100, 100, 1.0, 0.05, 0.02, 0.2, 0.1, -0.5, 1.0)
        assert c_neg > c_zero
