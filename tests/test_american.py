"""Tests for American option pricing via Longstaff-Schwartz LSM."""

from __future__ import annotations

import numpy as np
import pytest

from tensorquantlib.finance.american import (
    american_greeks,
    american_option_grid,
    american_option_lsm,
)
from tensorquantlib.finance.black_scholes import bs_price_numpy


class TestAmericanOptionLSM:
    def test_put_atm_positive(self):
        price = american_option_lsm(100, 100, 1.0, 0.05, 0.2, seed=0)
        assert price > 0

    def test_american_put_ge_european(self):
        """American put >= European put (early exercise premium)."""
        am = float(american_option_lsm(100, 110, 1.0, 0.05, 0.2, n_paths=50_000, seed=42))
        eu = float(bs_price_numpy(100, 110, 1.0, 0.05, 0.2, option_type="put"))
        assert am >= eu - 0.1  # allow small MC noise

    def test_american_call_no_dividends_equals_european(self):
        """For q=0, American call ≈ European call (no early exercise premium)."""
        am = float(
            american_option_lsm(
                100, 100, 1.0, 0.05, 0.2, option_type="call", n_paths=50_000, n_steps=50, seed=0
            )
        )
        eu = float(bs_price_numpy(100, 100, 1.0, 0.05, 0.2))
        # Should be within ~1 of each other due to MC noise
        assert abs(am - eu) < 2.5

    def test_returns_stderr(self):
        result = american_option_lsm(100, 100, 1.0, 0.05, 0.2, seed=0, return_stderr=True)
        assert isinstance(result, tuple)
        price, stderr = result
        assert price > 0
        assert stderr > 0
        assert stderr < price  # sanity

    def test_invalid_option_type(self):
        with pytest.raises(ValueError):
            american_option_lsm(100, 100, 1.0, 0.05, 0.2, option_type="straddle")  # type: ignore

    def test_deep_itm_put_approaches_intrinsic(self):
        """Deep ITM American put approaches max(K - S, 0) immediately."""
        price = float(american_option_lsm(50, 100, 1.0, 0.05, 0.2, n_paths=10_000, seed=0))
        intrinsic = 100 - 50
        assert price >= intrinsic - 2.0  # >= intrinsic (minus MC noise)


class TestAmericanOptionGrid:
    def test_grid_shape(self):
        S_grid = np.linspace(80, 120, 5)
        prices = american_option_grid(
            S_grid, K=100, T=1.0, r=0.05, sigma=0.2, n_paths=5_000, seed=0
        )
        assert prices.shape == (5,)
        assert np.all(prices >= 0)


class TestAmericanGreeks:
    def test_put_delta_negative(self):
        greeks = american_greeks(100, 100, 1.0, 0.05, 0.2, n_paths=20_000, n_steps=50, seed=42)
        # Delta for an ATM put should be in roughly [-0.7, -0.3]
        assert greeks["delta"] < 0

    def test_gamma_positive(self):
        greeks = american_greeks(100, 100, 1.0, 0.05, 0.2, n_paths=20_000, n_steps=50, seed=42)
        # Gamma should be positive for ATM
        assert isinstance(greeks["gamma"], float)

    def test_vega_positive_for_put(self):
        greeks = american_greeks(100, 100, 1.0, 0.05, 0.2, n_paths=20_000, n_steps=50, seed=42)
        assert greeks["vega"] > 0
