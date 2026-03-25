"""Tests for TT-accelerated pricing surrogates."""

import numpy as np
import pytest

from tensorquantlib.tt.pricing import (
    american_surrogate,
    exotic_surrogate,
    heston_surrogate,
    jump_diffusion_surrogate,
)


class TestJumpDiffusionSurrogate:
    """Jump-diffusion surrogate (analytic — fastest to build)."""

    def test_builds_and_evaluates(self):
        surr = jump_diffusion_surrogate(
            S_range=(90, 110),
            K_range=(95, 105),
            T_range=(0.5, 1.5),
            n_points=10,
            eps=1e-3,
        )
        price = surr.evaluate([100, 100, 1.0])
        assert price > 0

    def test_compression(self):
        surr = jump_diffusion_surrogate(
            S_range=(90, 110),
            K_range=(95, 105),
            T_range=(0.5, 1.5),
            n_points=12,
            eps=1e-3,
        )
        info = surr.summary()
        assert info["compression_ratio"] > 1.0

    def test_accuracy_vs_direct(self):
        from tensorquantlib.finance.jump_diffusion import merton_jump_price

        surr = jump_diffusion_surrogate(
            S_range=(90, 110),
            K_range=(95, 105),
            T_range=(0.5, 1.5),
            n_points=15,
            eps=1e-5,
        )
        # Evaluate at a grid point (should be very accurate)
        S, K, T = 100, 100, 1.0
        direct = merton_jump_price(S, K, T, 0.05, 0.2, 1.0, -0.05, 0.1)
        approx = surr.evaluate([S, K, T])
        assert abs(approx - direct) / direct < 0.02  # <2% error

    def test_put_type(self):
        surr = jump_diffusion_surrogate(
            S_range=(90, 110),
            K_range=(95, 105),
            T_range=(0.5, 1.5),
            n_points=10,
            eps=1e-3,
            option_type="put",
        )
        price = surr.evaluate([100, 100, 1.0])
        assert price > 0

    def test_batch_evaluate(self):
        surr = jump_diffusion_surrogate(
            S_range=(90, 110),
            K_range=(95, 105),
            T_range=(0.5, 1.5),
            n_points=10,
            eps=1e-3,
        )
        spots = np.array([[95, 100, 0.75], [100, 100, 1.0], [105, 100, 1.25]])
        prices = surr.evaluate(spots)
        assert prices.shape == (3,)
        assert np.all(prices > 0)

    def test_greeks(self):
        surr = jump_diffusion_surrogate(
            S_range=(85, 115),
            K_range=(95, 105),
            T_range=(0.5, 1.5),
            n_points=12,
            eps=1e-4,
        )
        g = surr.greeks([100, 100, 1.0])
        assert g["price"] > 0
        # Delta w.r.t. S should be positive for a call
        assert g["delta"][0] > 0


class TestExoticSurrogate:
    """Exotic option surrogate (Asian — MC-based, moderate speed)."""

    def test_asian_builds(self):
        surr = exotic_surrogate(
            exotic_type="asian",
            S_range=(90, 110),
            K_range=(95, 105),
            T_range=(0.5, 1.5),
            n_points=6,
            eps=1e-3,
            n_paths=10_000,
        )
        price = surr.evaluate([100, 100, 1.0])
        assert price > 0

    def test_barrier_builds(self):
        surr = exotic_surrogate(
            exotic_type="barrier_up_out",
            S_range=(90, 110),
            K_range=(95, 105),
            T_range=(0.5, 1.5),
            n_points=6,
            eps=1e-3,
            n_paths=10_000,
            barrier=130.0,
        )
        price = surr.evaluate([100, 100, 1.0])
        assert price >= 0

    def test_lookback_builds(self):
        surr = exotic_surrogate(
            exotic_type="lookback_fixed",
            S_range=(90, 110),
            K_range=(95, 105),
            T_range=(0.5, 1.5),
            n_points=6,
            eps=1e-3,
            n_paths=10_000,
        )
        price = surr.evaluate([100, 100, 1.0])
        assert price > 0

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Unknown exotic_type"):
            exotic_surrogate(exotic_type="invalid_type", n_points=3)


class TestAmericanSurrogate:
    """American option surrogate — LSM MC."""

    def test_builds_and_evaluates(self):
        surr = american_surrogate(
            S_range=(90, 110),
            K_range=(95, 105),
            T_range=(0.5, 1.5),
            n_points=6,
            eps=1e-3,
            n_paths=10_000,
            n_steps=50,
        )
        price = surr.evaluate([100, 100, 1.0])
        assert price > 0

    def test_put_otm_low_value(self):
        surr = american_surrogate(
            S_range=(90, 110),
            K_range=(95, 105),
            T_range=(0.5, 1.5),
            n_points=6,
            eps=1e-3,
            n_paths=10_000,
            n_steps=50,
        )
        # Deep ITM put should be more expensive
        p_itm = surr.evaluate([92, 105, 1.0])
        p_otm = surr.evaluate([108, 95, 1.0])
        assert p_itm > p_otm


class TestHestonSurrogate:
    """Heston MC surrogate — slowest, use small grid."""

    def test_builds_and_evaluates(self):
        surr = heston_surrogate(
            S_range=(95, 105),
            K_range=(95, 105),
            T_range=(0.5, 1.5),
            n_points=5,
            eps=1e-3,
            n_mc_paths=5_000,
        )
        price = surr.evaluate([100, 100, 1.0])
        assert price > 0

    def test_higher_spot_higher_call(self):
        surr = heston_surrogate(
            S_range=(90, 110),
            K_range=(95, 105),
            T_range=(0.5, 1.5),
            n_points=5,
            eps=1e-3,
            n_mc_paths=5_000,
        )
        p_lo = surr.evaluate([92, 100, 1.0])
        p_hi = surr.evaluate([108, 100, 1.0])
        assert p_hi > p_lo
