"""Tests for interest rate derivatives (caps, floors, swaptions)."""

from __future__ import annotations

import numpy as np
import pytest

from tensorquantlib.finance.ir_derivatives import (
    black76_caplet,
    black76_floorlet,
    cap_price,
    floor_price,
    swap_rate,
    swaption_parity,
    swaption_price,
)


class TestCapletFloorlet:
    """Tests for Black76 caplet/floorlet pricing."""

    def test_caplet_positive(self):
        price = black76_caplet(forward=0.05, strike=0.04, T=1.0, sigma=0.2, df=0.95)
        assert price > 0

    def test_floorlet_positive(self):
        price = black76_floorlet(forward=0.03, strike=0.04, T=1.0, sigma=0.2, df=0.95)
        assert price > 0

    def test_put_call_parity(self):
        """Caplet - Floorlet = df * tau * (F - K)."""
        F, K, T, sigma, df, tau = 0.05, 0.04, 1.0, 0.2, 0.95, 0.25
        cap = black76_caplet(F, K, T, sigma, df, tau=tau)
        flr = black76_floorlet(F, K, T, sigma, df, tau=tau)
        expected = df * tau * (F - K)
        assert abs((cap - flr) - expected) < 1e-10

    def test_atm_caplet_equals_floorlet(self):
        """At the money (F=K), caplet = floorlet."""
        F = K = 0.05
        cap = black76_caplet(F, K, 1.0, 0.2, 0.95)
        flr = black76_floorlet(F, K, 1.0, 0.2, 0.95)
        assert abs(cap - flr) < 1e-10

    def test_higher_vol_higher_price(self):
        c1 = black76_caplet(0.05, 0.05, 1.0, 0.1, 0.95)
        c2 = black76_caplet(0.05, 0.05, 1.0, 0.3, 0.95)
        assert c2 > c1


class TestCapFloor:
    """Tests for cap and floor pricing."""

    @pytest.fixture
    def cap_params(self):
        forwards = np.array([0.04, 0.045, 0.05, 0.05])
        expiries = np.array([0.25, 0.5, 0.75, 1.0])
        dfs = np.array([0.99, 0.978, 0.966, 0.955])
        return forwards, expiries, dfs

    def test_cap_positive(self, cap_params):
        forwards, expiries, dfs = cap_params
        price = cap_price(forwards, strike=0.04, expiries=expiries, sigma=0.2, dfs=dfs)
        assert price > 0

    def test_floor_positive(self, cap_params):
        forwards, expiries, dfs = cap_params
        price = floor_price(forwards, strike=0.06, expiries=expiries, sigma=0.2, dfs=dfs)
        assert price > 0

    def test_cap_floor_parity(self, cap_params):
        """Cap - Floor = sum(df * tau * (F - K))."""
        forwards, expiries, dfs = cap_params
        K = 0.045
        tau = 0.25
        c = cap_price(forwards, K, expiries, 0.2, dfs, tau=tau)
        f = floor_price(forwards, K, expiries, 0.2, dfs, tau=tau)
        expected = np.sum(dfs * tau * (forwards - K))
        assert abs((c - f) - expected) < 1e-10


class TestSwapRate:
    """Tests for par swap rate calculation."""

    def test_flat_curve(self):
        """Flat discount curve → swap rate = yield."""
        r = 0.05
        dfs = np.exp(-r * np.arange(0, 5.5, 0.5))
        sr = swap_rate(dfs, tau=0.5)
        # Should be approximately r
        assert abs(sr - r) < 0.001

    def test_positive(self):
        dfs = np.array([1.0, 0.98, 0.96, 0.94, 0.92])
        sr = swap_rate(dfs)
        assert sr > 0


class TestSwaption:
    """Tests for Black76 swaption pricing."""

    def test_payer_positive(self):
        price = swaption_price(swap_r=0.05, strike=0.04, T_option=1.0, sigma=0.2, annuity=4.0)
        assert price > 0

    def test_receiver_positive(self):
        price = swaption_price(
            swap_r=0.04, strike=0.05, T_option=1.0, sigma=0.2, annuity=4.0, payer=False
        )
        assert price > 0

    def test_swaption_parity(self):
        """Payer - Receiver = (S - K) * A * N."""
        S, K, T, sigma, A = 0.05, 0.04, 1.0, 0.2, 4.0
        payer = swaption_price(S, K, T, sigma, A, payer=True)
        receiver = swaption_price(S, K, T, sigma, A, payer=False)
        residual = swaption_parity(payer, receiver, S, K, A)
        assert abs(residual) < 1e-10

    def test_higher_vol_higher_price(self):
        p1 = swaption_price(0.05, 0.05, 1.0, 0.1, 4.0)
        p2 = swaption_price(0.05, 0.05, 1.0, 0.3, 4.0)
        assert p2 > p1

    def test_atm_payer_equals_receiver(self):
        """ATM payer = ATM receiver."""
        p = swaption_price(0.05, 0.05, 1.0, 0.2, 4.0, payer=True)
        r = swaption_price(0.05, 0.05, 1.0, 0.2, 4.0, payer=False)
        assert abs(p - r) < 1e-10
