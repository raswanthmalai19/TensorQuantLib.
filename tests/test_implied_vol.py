"""Tests for implied volatility solver."""
from __future__ import annotations

import numpy as np
import pytest

from tensorquantlib.finance.black_scholes import bs_price_numpy
from tensorquantlib.finance.implied_vol import (
    implied_vol,
    implied_vol_batch,
    implied_vol_nr,
    iv_surface,
)


class TestImpliedVol:
    """Test Brent-method implied volatility solver."""

    @pytest.mark.parametrize("S,K,T,r,sigma", [
        (100, 100, 1.0, 0.05, 0.20),
        (100,  90, 0.5, 0.03, 0.15),
        (100, 110, 2.0, 0.01, 0.30),
        (80,  80, 0.25, 0.05, 0.40),  # ATM short-dated
    ])
    def test_round_trip_call(self, S, K, T, r, sigma):
        price = float(bs_price_numpy(S, K, T, r, sigma))
        iv = implied_vol(price, S, K, T, r)
        assert abs(iv - sigma) < 1e-6

    @pytest.mark.parametrize("S,K,T,r,sigma", [
        (100, 100, 1.0, 0.05, 0.20),
        (100, 110, 0.5, 0.03, 0.25),
    ])
    def test_round_trip_put(self, S, K, T, r, sigma):
        price = float(bs_price_numpy(S, K, T, r, sigma, option_type="put"))
        iv = implied_vol(price, S, K, T, r, option_type="put")
        assert abs(iv - sigma) < 1e-6

    def test_below_intrinsic_raises(self):
        with pytest.raises(ValueError, match="intrinsic"):
            implied_vol(-1.0, 100, 100, 1.0, 0.05)

    def test_newton_raphson_round_trip(self):
        price = float(bs_price_numpy(100, 100, 1.0, 0.05, 0.20))
        iv = implied_vol_nr(price, 100, 100, 1.0, 0.05)
        assert abs(iv - 0.20) < 1e-6

    def test_batch_returns_correct_shape(self):
        sigmas = np.array([0.15, 0.20, 0.25])
        prices = np.array([float(bs_price_numpy(100, 100, 1.0, 0.05, s)) for s in sigmas])
        ivs = implied_vol_batch(prices, S=100, K=100, T=1.0, r=0.05)
        assert ivs.shape == (3,)
        np.testing.assert_allclose(ivs, sigmas, atol=1e-5)

    def test_iv_surface(self):
        K = np.array([90.0, 100.0, 110.0])
        T = np.array([0.5, 1.0])
        prices = np.array([[float(bs_price_numpy(100, k, t, 0.05, 0.20)) for t in T] for k in K])
        surf = iv_surface(prices, 100.0, K, T, 0.05)
        assert surf.shape == (3, 2)
        np.testing.assert_allclose(surf, 0.20, atol=1e-4)

    def test_batch_nan_for_bad_price(self):
        ivs = implied_vol_batch([-999.0, 10.0], S=100, K=100, T=1.0, r=0.05)
        assert np.isnan(ivs[0])
        assert not np.isnan(ivs[1])


class TestNewtonRaphson:
    def test_deep_itm_call(self):
        sigma = 0.25
        price = float(bs_price_numpy(100, 50, 1.0, 0.05, sigma))
        iv = implied_vol_nr(price, 100, 50, 1.0, 0.05)
        assert abs(iv - sigma) < 1e-5

    def test_deep_otm_put(self):
        sigma = 0.30
        price = float(bs_price_numpy(100, 150, 1.0, 0.05, sigma, option_type="put"))
        iv = implied_vol_nr(price, 100, 150, 1.0, 0.05, option_type="put")
        assert abs(iv - sigma) < 1e-4
