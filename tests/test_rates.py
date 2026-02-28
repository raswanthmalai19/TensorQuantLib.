"""Tests for interest rate models (Vasicek, CIR, Nelson-Siegel)."""
from __future__ import annotations

import numpy as np
import pytest

from tensorquantlib.finance.rates import (
    vasicek_bond_price,
    vasicek_yield,
    vasicek_option_price,
    vasicek_simulate,
    cir_bond_price,
    cir_yield,
    cir_simulate,
    feller_condition,
    nelson_siegel,
    nelson_siegel_calibrate,
    bootstrap_yield_curve,
)


class TestVasicek:
    """Tests for the Vasicek short rate model."""

    def test_bond_price_positive(self):
        P = vasicek_bond_price(r0=0.05, kappa=0.5, theta=0.05, sigma=0.01, T=1.0)
        assert 0 < P < 1

    def test_sigma_zero_deterministic(self):
        """With sigma=0, bond price = exp(integral of deterministic rate)."""
        r0, kappa, theta, T = 0.05, 0.5, 0.05, 1.0
        P = vasicek_bond_price(r0, kappa, theta, sigma=0.0, T=T)
        # For r0=theta, rate is constant, so P = exp(-r0*T)
        expected = np.exp(-r0 * T)
        assert abs(P - expected) < 1e-10

    def test_yield_positive(self):
        y = vasicek_yield(r0=0.05, kappa=0.5, theta=0.05, sigma=0.01, T=1.0)
        assert y > 0

    def test_bond_price_decreasing_in_T(self):
        """Longer maturity bonds should be cheaper."""
        P1 = vasicek_bond_price(0.05, 0.5, 0.05, 0.01, 1.0)
        P5 = vasicek_bond_price(0.05, 0.5, 0.05, 0.01, 5.0)
        assert P5 < P1

    def test_option_put_call_parity(self):
        """Put-call parity: C - P = P(T_bond) - K * P(T_option)."""
        r0, kappa, theta, sigma = 0.05, 0.5, 0.05, 0.01
        T_opt, T_bond, K = 1.0, 2.0, 0.95
        call = vasicek_option_price(r0, kappa, theta, sigma, T_opt, T_bond, K, 'call')
        put = vasicek_option_price(r0, kappa, theta, sigma, T_opt, T_bond, K, 'put')
        P_bond = vasicek_bond_price(r0, kappa, theta, sigma, T_bond)
        P_opt = vasicek_bond_price(r0, kappa, theta, sigma, T_opt)
        parity = P_bond - K * P_opt
        assert abs((call - put) - parity) < 1e-10

    def test_option_positive(self):
        call = vasicek_option_price(0.05, 0.5, 0.05, 0.01, 1.0, 2.0, 0.95, 'call')
        assert call > 0

    def test_simulate_shape(self):
        paths = vasicek_simulate(0.05, 0.5, 0.05, 0.01, 1.0, n_steps=100, n_paths=500, seed=42)
        assert paths.shape == (101, 500)

    def test_simulate_mean_reversion(self):
        """Simulated mean should converge to theta."""
        paths = vasicek_simulate(0.10, 2.0, 0.05, 0.01, 5.0,
                                  n_steps=500, n_paths=10_000, seed=42)
        terminal_mean = np.mean(paths[-1])
        assert abs(terminal_mean - 0.05) < 0.01

    def test_mc_bond_vs_analytic(self):
        """MC bond price should approximate analytic."""
        r0, kappa, theta, sigma, T = 0.05, 0.5, 0.05, 0.01, 1.0
        paths = vasicek_simulate(r0, kappa, theta, sigma, T,
                                  n_steps=252, n_paths=50_000, seed=42)
        dt = T / 252
        # Numerical integration of rate paths
        integrals = np.sum(paths[:-1], axis=0) * dt
        mc_price = float(np.mean(np.exp(-integrals)))
        analytic_price = vasicek_bond_price(r0, kappa, theta, sigma, T)
        assert abs(mc_price - analytic_price) < 0.005


class TestCIR:
    """Tests for the CIR short rate model."""

    def test_bond_price_positive(self):
        P = cir_bond_price(r0=0.05, kappa=0.5, theta=0.05, sigma=0.1, T=1.0)
        assert 0 < P < 1

    def test_feller_satisfied(self):
        assert feller_condition(kappa=0.5, theta=0.05, sigma=0.1)

    def test_feller_violated(self):
        assert not feller_condition(kappa=0.1, theta=0.01, sigma=0.5)

    def test_yield_positive(self):
        y = cir_yield(0.05, 0.5, 0.05, 0.1, 1.0)
        assert y > 0

    def test_simulate_non_negative(self):
        """CIR paths should remain non-negative."""
        paths = cir_simulate(0.05, 0.5, 0.05, 0.1, 1.0,
                              n_steps=252, n_paths=1000, seed=42)
        assert np.all(paths >= 0)

    def test_simulate_shape(self):
        paths = cir_simulate(0.05, 0.5, 0.05, 0.1, 1.0, n_steps=100, n_paths=500, seed=42)
        assert paths.shape == (101, 500)

    def test_mc_bond_vs_analytic(self):
        """MC bond price should approximate analytic for CIR."""
        r0, kappa, theta, sigma, T = 0.05, 0.5, 0.05, 0.1, 1.0
        paths = cir_simulate(r0, kappa, theta, sigma, T,
                              n_steps=252, n_paths=50_000, seed=42)
        dt = T / 252
        integrals = np.sum(paths[:-1], axis=0) * dt
        mc_price = float(np.mean(np.exp(-integrals)))
        analytic_price = cir_bond_price(r0, kappa, theta, sigma, T)
        assert abs(mc_price - analytic_price) < 0.005

    def test_sigma_zero_equals_vasicek(self):
        """With vol-of-vol proportionality, CIR and Vasicek should agree when sigma approx 0."""
        r0, kappa, theta, T = 0.05, 0.5, 0.05, 1.0
        # At r0=theta, CIR with tiny sigma should match Vasicek with tiny sigma
        P_cir = cir_bond_price(r0, kappa, theta, sigma=0.001, T=T)
        P_vas = vasicek_bond_price(r0, kappa, theta, sigma=0.001, T=T)
        assert abs(P_cir - P_vas) < 0.001


class TestNelsonSiegel:
    """Tests for Nelson-Siegel yield curve model."""

    def test_flat_curve(self):
        """beta1=beta2=0 gives flat curve at beta0."""
        T = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        y = nelson_siegel(T, beta0=0.05, beta1=0.0, beta2=0.0, tau=1.0)
        np.testing.assert_allclose(y, 0.05, atol=1e-10)

    def test_short_rate(self):
        """At T→0, y → beta0 + beta1."""
        y = nelson_siegel(0.001, beta0=0.05, beta1=-0.02, beta2=0.01, tau=1.0)
        assert abs(y - 0.03) < 0.001

    def test_long_rate(self):
        """At T→∞, y → beta0."""
        y = nelson_siegel(100.0, beta0=0.05, beta1=-0.02, beta2=0.01, tau=1.0)
        assert abs(y - 0.05) < 0.001

    def test_vectorized(self):
        T = np.array([0.5, 1.0, 5.0])
        y = nelson_siegel(T, 0.05, -0.02, 0.01, 1.0)
        assert y.shape == (3,)

    def test_calibration_round_trip(self):
        """Generate NS yields, calibrate back."""
        T = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
        true_yields = nelson_siegel(T, 0.05, -0.02, 0.01, 2.0)
        result = nelson_siegel_calibrate(T, true_yields)
        assert result['rmse'] < 0.0001

    def test_scalar_input(self):
        y = nelson_siegel(1.0, 0.05, -0.02, 0.01, 1.0)
        assert isinstance(y, float)


class TestBootstrap:
    """Tests for yield curve bootstrap."""

    def test_basic_bootstrap(self):
        """Bootstrap from known prices."""
        maturities = np.array([1.0, 2.0, 5.0])
        # Prices from flat 5% yield
        prices = np.exp(-0.05 * maturities)
        yields = bootstrap_yield_curve(maturities, prices)
        np.testing.assert_allclose(yields, 0.05, atol=1e-10)

    def test_positive_yields(self):
        maturities = np.array([0.5, 1.0, 2.0])
        prices = np.array([0.975, 0.95, 0.90])
        yields = bootstrap_yield_curve(maturities, prices)
        assert np.all(yields > 0)
