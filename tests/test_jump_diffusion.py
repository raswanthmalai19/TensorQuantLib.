"""Tests for jump-diffusion models (Merton + Kou)."""

from __future__ import annotations

import numpy as np

from tensorquantlib.finance.black_scholes import bs_price_numpy
from tensorquantlib.finance.jump_diffusion import (
    kou_jump_price_mc,
    merton_jump_price,
    merton_jump_price_mc,
)


class TestMertonJumpDiffusion:
    """Tests for Merton (1976) jump-diffusion."""

    def test_reduces_to_bs_with_no_jumps(self):
        """With lambda=0, should equal Black-Scholes."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        jd = merton_jump_price(S, K, T, r, sigma, lam=0.0, mu_j=0.0, sigma_j=0.0)
        bs = float(bs_price_numpy(S, K, T, r, sigma))
        assert abs(jd - bs) < 0.01

    def test_put_call_parity(self):
        """C - P = S - K*exp(-rT)."""
        S, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.2
        call = merton_jump_price(S, K, T, r, sigma, 1.0, -0.1, 0.2, "call")
        put = merton_jump_price(S, K, T, r, sigma, 1.0, -0.1, 0.2, "put")
        parity = S - K * np.exp(-r * T)
        assert abs((call - put) - parity) < 0.01

    def test_positive_prices(self):
        call = merton_jump_price(100, 100, 1.0, 0.05, 0.2, 1.0, -0.1, 0.2, "call")
        put = merton_jump_price(100, 100, 1.0, 0.05, 0.2, 1.0, -0.1, 0.2, "put")
        assert call > 0
        assert put > 0

    def test_jumps_increase_otm_prices(self):
        """Adding jumps should increase OTM option prices (fat tails)."""
        bs = float(bs_price_numpy(100, 130, 0.5, 0.05, 0.2))
        jd = merton_jump_price(100, 130, 0.5, 0.05, 0.2, 2.0, 0.0, 0.3, "call")
        assert jd > bs

    def test_mc_matches_analytic(self):
        """MC estimate should be close to analytic."""
        S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.2
        lam, mu_j, sigma_j = 1.0, -0.1, 0.15
        analytic = merton_jump_price(S, K, T, r, sigma, lam, mu_j, sigma_j)
        mc = merton_jump_price_mc(S, K, T, r, sigma, lam, mu_j, sigma_j, n_paths=200_000, seed=42)
        assert abs(mc - analytic) / analytic < 0.05  # within 5%


class TestKouJumpDiffusion:
    """Tests for Kou (2002) double-exponential jump-diffusion."""

    def test_positive_price(self):
        price = kou_jump_price_mc(
            100, 100, 1.0, 0.05, 0.2, lam=1.0, p=0.5, eta1=10.0, eta2=5.0, n_paths=50_000, seed=42
        )
        assert price > 0

    def test_put_positive(self):
        price = kou_jump_price_mc(
            100,
            100,
            1.0,
            0.05,
            0.2,
            lam=1.0,
            p=0.5,
            eta1=10.0,
            eta2=5.0,
            option_type="put",
            n_paths=50_000,
            seed=42,
        )
        assert price > 0

    def test_upward_bias_increases_call(self):
        """More upward jumps (higher p) should increase OTM call price."""
        c_low = kou_jump_price_mc(
            100, 120, 0.5, 0.05, 0.2, lam=2.0, p=0.2, eta1=5.0, eta2=5.0, n_paths=200_000, seed=42
        )
        c_high = kou_jump_price_mc(
            100, 120, 0.5, 0.05, 0.2, lam=2.0, p=0.8, eta1=5.0, eta2=5.0, n_paths=200_000, seed=42
        )
        assert c_high > c_low

    def test_higher_intensity_fatter_tails(self):
        """Higher jump intensity → higher OTM call price."""
        c_low = kou_jump_price_mc(
            100, 130, 0.5, 0.05, 0.2, lam=0.5, p=0.5, eta1=10.0, eta2=5.0, n_paths=100_000, seed=42
        )
        c_high = kou_jump_price_mc(
            100, 130, 0.5, 0.05, 0.2, lam=5.0, p=0.5, eta1=10.0, eta2=5.0, n_paths=100_000, seed=42
        )
        assert c_high > c_low
