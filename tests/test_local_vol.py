"""Tests for local volatility model (Dupire)."""

from __future__ import annotations

import numpy as np
import pytest

from tensorquantlib.finance.black_scholes import bs_price_numpy
from tensorquantlib.finance.local_vol import dupire_local_vol, local_vol_mc


class TestDupireLocalVol:
    """Tests for Dupire local volatility extraction."""

    @pytest.fixture
    def flat_vol_surface(self):
        """Create a flat IV surface (constant vol)."""
        S = 100.0
        sigma_flat = 0.2
        strikes = np.linspace(80, 120, 15)
        expiries = np.linspace(0.1, 2.0, 10)
        iv_surface = np.full((len(strikes), len(expiries)), sigma_flat)
        return strikes, expiries, iv_surface, S, sigma_flat

    def test_flat_vol_recovers_constant(self, flat_vol_surface):
        """Flat IV surface → local vol should be approximately constant."""
        strikes, expiries, iv_surface, S, sigma_flat = flat_vol_surface
        lv = dupire_local_vol(strikes, expiries, iv_surface, S, r=0.05)
        # Interior points should be close to flat vol
        interior = lv[3:-3, 2:-2]
        assert np.all(np.abs(interior - sigma_flat) < 0.05)

    def test_local_vol_positive(self, flat_vol_surface):
        """Local vol should always be positive."""
        strikes, expiries, iv_surface, S, _ = flat_vol_surface
        lv = dupire_local_vol(strikes, expiries, iv_surface, S, r=0.05)
        assert np.all(lv > 0)

    def test_local_vol_shape(self, flat_vol_surface):
        """Output shape matches input."""
        strikes, expiries, iv_surface, S, _ = flat_vol_surface
        lv = dupire_local_vol(strikes, expiries, iv_surface, S, r=0.05)
        assert lv.shape == iv_surface.shape

    def test_smile_produces_varying_local_vol(self):
        """A vol smile should produce a non-constant local vol."""
        S = 100.0
        strikes = np.linspace(80, 120, 20)
        expiries = np.linspace(0.25, 2.0, 8)
        # Create a smile: higher vol for OTM
        iv_surface = np.zeros((len(strikes), len(expiries)))
        for i, K in enumerate(strikes):
            moneyness = np.log(K / S)
            iv_surface[i, :] = 0.2 + 0.1 * moneyness**2
        lv = dupire_local_vol(strikes, expiries, iv_surface, S, r=0.05)
        # Local vol should vary
        assert np.std(lv) > 0.005


class TestLocalVolMC:
    """Tests for local vol Monte Carlo pricing."""

    def test_flat_local_vol_matches_bs(self):
        """Flat local vol = flat BS → should match BS price."""
        S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.2
        strikes = np.linspace(70, 130, 20)
        expiries = np.linspace(0.05, 1.0, 10)
        lv_surface = np.full((len(strikes), len(expiries)), sigma)

        mc_price = local_vol_mc(
            S, K, T, r, strikes, expiries, lv_surface, n_paths=100_000, n_steps=100, seed=42
        )
        bs = float(bs_price_numpy(S, K, T, r, sigma))
        assert abs(mc_price - bs) / bs < 0.05  # within 5%

    def test_positive_prices(self):
        S, K, T, r = 100.0, 100.0, 0.5, 0.05
        strikes = np.linspace(70, 130, 15)
        expiries = np.linspace(0.05, 1.0, 8)
        lv_surface = np.full((len(strikes), len(expiries)), 0.2)

        call = local_vol_mc(
            S, K, T, r, strikes, expiries, lv_surface, option_type="call", n_paths=50_000, seed=42
        )
        put = local_vol_mc(
            S, K, T, r, strikes, expiries, lv_surface, option_type="put", n_paths=50_000, seed=42
        )
        assert call > 0
        assert put > 0
