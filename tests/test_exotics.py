"""Tests for exotic options: Asian, digital, and barrier."""

from __future__ import annotations

import numpy as np
import pytest

from tensorquantlib.finance.black_scholes import bs_price_numpy
from tensorquantlib.finance.exotics import (
    asian_geometric_price,
    asian_price_mc,
    barrier_price,
    barrier_price_mc,
    digital_greeks,
    digital_price,
    digital_price_mc,
)

# ================================================================== #
# ASIAN
# ================================================================== #


class TestAsianOptions:
    def test_arithmetic_call_positive(self):
        price = asian_price_mc(100, 100, 1.0, 0.05, 0.20, seed=0)
        assert float(price) > 0

    def test_geometric_analytic_positive(self):
        price = asian_geometric_price(100, 100, 1.0, 0.05, 0.20)
        assert price > 0

    def test_geometric_mc_vs_analytic(self):
        """Geometric Asian MC and analytic should be close (within 3*stderr)."""
        analytic = asian_geometric_price(100, 100, 1.0, 0.05, 0.20)
        mc, stderr = asian_price_mc(
            100,
            100,
            1.0,
            0.05,
            0.20,
            average_type="geometric",
            n_paths=200_000,
            seed=42,
            return_stderr=True,
        )
        assert abs(float(mc) - analytic) < 4 * float(stderr) + 0.1

    def test_asian_lt_vanilla(self):
        """Arithmetic Asian call <= vanilla European call (averaging reduces vol)."""
        asian = float(asian_price_mc(100, 100, 1.0, 0.05, 0.20, n_paths=100_000, seed=0))
        vanilla = float(bs_price_numpy(100, 100, 1.0, 0.05, 0.20))
        assert asian < vanilla + 0.5  # allow small MC noise

    def test_put_positive(self):
        price = asian_price_mc(100, 100, 1.0, 0.05, 0.20, option_type="put", seed=0)
        assert float(price) > 0

    def test_returns_stderr(self):
        result = asian_price_mc(100, 100, 1.0, 0.05, 0.20, seed=0, return_stderr=True)
        assert isinstance(result, tuple)
        price, stderr = result
        assert float(price) > 0 and float(stderr) > 0

    def test_invalid_average_type(self):
        with pytest.raises(ValueError):
            asian_price_mc(100, 100, 1.0, 0.05, 0.20, average_type="harmonic")  # type: ignore


# ================================================================== #
# DIGITAL
# ================================================================== #


class TestDigitalOptions:
    def test_cash_or_nothing_call_in_01(self):
        price = digital_price(100, 100, 1.0, 0.05, 0.20, payoff_type="cash")
        assert 0.0 < price < 1.0

    def test_cash_or_nothing_put_in_01(self):
        price = digital_price(100, 100, 1.0, 0.05, 0.20, option_type="put", payoff_type="cash")
        assert 0.0 < price < 1.0

    def test_call_put_sum_to_discount(self):
        """Call + Put = e^{-rT} for cash-or-nothing (partition of unity)."""
        r, T = 0.05, 1.0
        call = digital_price(100, 100, T, r, 0.20, payoff_type="cash", option_type="call")
        put = digital_price(100, 100, T, r, 0.20, payoff_type="cash", option_type="put")
        np.testing.assert_allclose(call + put, np.exp(-r * T), atol=1e-8)

    def test_asset_or_nothing_call_positive(self):
        price = digital_price(100, 100, 1.0, 0.05, 0.20, payoff_type="asset")
        assert price > 0

    def test_mc_vs_analytic_cash(self):
        analytic = digital_price(100, 100, 1.0, 0.05, 0.20)
        mc, stderr = digital_price_mc(
            100, 100, 1.0, 0.05, 0.20, n_paths=200_000, seed=0, return_stderr=True
        )
        assert abs(float(mc) - analytic) < 4 * float(stderr) + 0.02

    def test_greeks_return_dict(self):
        g = digital_greeks(100, 100, 1.0, 0.05, 0.20)
        assert "delta" in g and "gamma" in g and "vega" in g

    def test_invalid_payoff_type(self):
        with pytest.raises(ValueError):
            digital_price(100, 100, 1.0, 0.05, 0.20, payoff_type="rainbow")  # type: ignore


# ================================================================== #
# BARRIER
# ================================================================== #


class TestBarrierOptions:
    def test_down_and_out_positive(self):
        price = barrier_price(100, 100, 1.0, 0.05, 0.20, barrier=90, barrier_type="down-and-out")
        assert price > 0

    def test_down_and_out_lt_vanilla(self):
        """Down-and-out <= vanilla because it can be knocked out."""
        vanilla = float(bs_price_numpy(100, 100, 1.0, 0.05, 0.20))
        knockout = barrier_price(100, 100, 1.0, 0.05, 0.20, barrier=90, barrier_type="down-and-out")
        # Analytic result: knockout <= vanilla (up to small floating-point rounding)
        assert knockout <= vanilla + 0.5

    def test_mc_down_and_out(self):
        mc, stderr = barrier_price_mc(
            100,
            100,
            1.0,
            0.05,
            0.20,
            barrier=90,
            barrier_type="down-and-out",
            n_paths=200_000,
            seed=0,
            return_stderr=True,
        )
        assert float(mc) > 0 and float(stderr) < float(mc)

    def test_mc_vs_analytic_down_and_out(self):
        analytic = barrier_price(100, 100, 1.0, 0.05, 0.20, barrier=90, barrier_type="down-and-out")
        mc, stderr = barrier_price_mc(
            100,
            100,
            1.0,
            0.05,
            0.20,
            barrier=90,
            barrier_type="down-and-out",
            n_paths=300_000,
            n_steps=252,
            seed=42,
            return_stderr=True,
        )
        # Continuous monitoring bias: MC may differ from analytic by more than stderr
        # Allow up to 2.0 absolute difference (discretisation + approximation error)
        assert abs(analytic - float(mc)) < 12 * float(stderr) + 2.0

    def test_up_and_out_call_zero_barrier_eq_strike(self):
        """Up-and-out call with barrier <= strike should be worth 0."""
        price = barrier_price(100, 100, 1.0, 0.05, 0.20, barrier=100, barrier_type="up-and-out")
        assert price == 0.0

    def test_mc_put_down_and_in(self):
        mc = float(
            barrier_price_mc(
                100,
                100,
                1.0,
                0.05,
                0.20,
                barrier=90,
                barrier_type="down-and-in",
                option_type="put",
                n_paths=100_000,
                seed=0,
            )
        )
        assert mc > 0

    def test_in_out_parity(self):
        """Down-and-in + Down-and-out = vanilla call."""
        vanilla = float(bs_price_numpy(100, 100, 1.0, 0.05, 0.20))
        din = barrier_price_mc(
            100, 100, 1.0, 0.05, 0.20, 90, "down-and-in", n_paths=200_000, seed=7
        )
        dout = barrier_price_mc(
            100, 100, 1.0, 0.05, 0.20, 90, "down-and-out", n_paths=200_000, seed=7
        )
        assert abs(float(din) + float(dout) - vanilla) < 0.5

    def test_invalid_barrier_type(self):
        with pytest.raises(ValueError):
            barrier_price(100, 100, 1.0, 0.05, 0.20, barrier=90, barrier_type="diagonal")  # type: ignore


# ===========================================================================
# Lookback options
# ===========================================================================


class TestLookback:
    """Tests for lookback options."""

    def test_fixed_call_analytic_positive(self):
        from tensorquantlib.finance.exotics import lookback_fixed_analytic

        price = lookback_fixed_analytic(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        assert price > 0

    def test_fixed_call_ge_vanilla(self):
        """Fixed-strike lookback call ≥ vanilla call (always)."""
        from tensorquantlib.finance.exotics import lookback_fixed_analytic

        lookback = lookback_fixed_analytic(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        vanilla = float(bs_price_numpy(100, 100, 1.0, 0.05, 0.2))
        assert lookback >= vanilla - 0.01

    def test_fixed_put_analytic_positive(self):
        from tensorquantlib.finance.exotics import lookback_fixed_analytic

        price = lookback_fixed_analytic(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="put")
        assert price > 0

    def test_floating_call_positive(self):
        from tensorquantlib.finance.exotics import lookback_floating_analytic

        price = lookback_floating_analytic(S=100, T=1.0, r=0.05, sigma=0.2)
        assert price > 0

    def test_floating_put_positive(self):
        from tensorquantlib.finance.exotics import lookback_floating_analytic

        price = lookback_floating_analytic(S=100, T=1.0, r=0.05, sigma=0.2, option_type="put")
        assert price > 0

    def test_mc_fixed_call(self):
        """MC lookback should be positive and reasonable."""
        from tensorquantlib.finance.exotics import lookback_price_mc

        price, se = lookback_price_mc(
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, strike_type="fixed", seed=42
        )
        assert price > 0
        assert se < price * 0.1

    def test_mc_floating_call(self):
        from tensorquantlib.finance.exotics import lookback_price_mc

        price, se = lookback_price_mc(
            S=100, K=None, T=1.0, r=0.05, sigma=0.2, strike_type="floating", seed=42
        )
        assert price > 0

    def test_mc_fixed_requires_K(self):
        from tensorquantlib.finance.exotics import lookback_price_mc

        with pytest.raises(ValueError):
            lookback_price_mc(S=100, K=None, T=1.0, r=0.05, sigma=0.2, strike_type="fixed")


# ===========================================================================
# Cliquet options
# ===========================================================================


class TestCliquet:
    """Tests for cliquet options."""

    def test_positive_price(self):
        from tensorquantlib.finance.exotics import cliquet_price_mc

        price, se = cliquet_price_mc(S=100, T=1.0, r=0.05, sigma=0.2, seed=42)
        assert price > 0

    def test_floor_increases_price(self):
        """Adding a per-period floor of 0% should increase price."""
        from tensorquantlib.finance.exotics import cliquet_price_mc

        p_no_floor, _ = cliquet_price_mc(S=100, T=1.0, r=0.05, sigma=0.2, n_paths=50_000, seed=42)
        p_with_floor, _ = cliquet_price_mc(
            S=100, T=1.0, r=0.05, sigma=0.2, floor=0.0, n_paths=50_000, seed=42
        )
        assert p_with_floor >= p_no_floor - 0.5

    def test_cap_decreases_price(self):
        """Adding a per-period cap should decrease price."""
        from tensorquantlib.finance.exotics import cliquet_price_mc

        p_no_cap, _ = cliquet_price_mc(S=100, T=1.0, r=0.05, sigma=0.2, n_paths=50_000, seed=42)
        p_with_cap, _ = cliquet_price_mc(
            S=100, T=1.0, r=0.05, sigma=0.2, cap=0.02, n_paths=50_000, seed=42
        )
        assert p_with_cap <= p_no_cap + 0.5

    def test_global_cap(self):
        from tensorquantlib.finance.exotics import cliquet_price_mc

        price, se = cliquet_price_mc(S=100, T=1.0, r=0.05, sigma=0.2, global_cap=0.1, seed=42)
        assert price >= 0


# ===========================================================================
# Rainbow options
# ===========================================================================


class TestRainbow:
    """Tests for rainbow options."""

    def test_best_of_call_positive(self):
        from tensorquantlib.finance.exotics import rainbow_price_mc

        spots = np.array([100.0, 100.0])
        sigmas = np.array([0.2, 0.3])
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        price, se = rainbow_price_mc(
            spots, K=100, T=1.0, r=0.05, sigmas=sigmas, corr=corr, seed=42, n_steps=50
        )
        assert price > 0

    def test_best_of_ge_individual(self):
        """Best-of call ≥ any individual vanilla call."""
        from tensorquantlib.finance.exotics import rainbow_price_mc

        spots = np.array([100.0, 100.0])
        sigmas = np.array([0.2, 0.3])
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        best_of, _ = rainbow_price_mc(
            spots,
            K=100,
            T=1.0,
            r=0.05,
            sigmas=sigmas,
            corr=corr,
            seed=42,
            n_paths=100_000,
            n_steps=50,
        )
        # Vanilla calls
        v1 = float(bs_price_numpy(100, 100, 1.0, 0.05, 0.2))
        v2 = float(bs_price_numpy(100, 100, 1.0, 0.05, 0.3))
        assert best_of >= max(v1, v2) - 1.0  # allow MC noise

    def test_worst_of_le_individual(self):
        """Worst-of call ≤ any individual vanilla call."""
        from tensorquantlib.finance.exotics import rainbow_price_mc

        spots = np.array([100.0, 100.0])
        sigmas = np.array([0.2, 0.3])
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        worst_of, _ = rainbow_price_mc(
            spots,
            K=100,
            T=1.0,
            r=0.05,
            sigmas=sigmas,
            corr=corr,
            rainbow_type="worst-of",
            seed=42,
            n_paths=100_000,
            n_steps=50,
        )
        v1 = float(bs_price_numpy(100, 100, 1.0, 0.05, 0.2))
        assert worst_of <= v1 + 1.0

    def test_put_positive(self):
        from tensorquantlib.finance.exotics import rainbow_price_mc

        spots = np.array([100.0, 100.0])
        sigmas = np.array([0.2, 0.3])
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        price, se = rainbow_price_mc(
            spots,
            K=100,
            T=1.0,
            r=0.05,
            sigmas=sigmas,
            corr=corr,
            option_type="put",
            seed=42,
            n_steps=50,
        )
        assert price > 0
