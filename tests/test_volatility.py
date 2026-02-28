"""Tests for volatility surface models (SABR + SVI)."""
from __future__ import annotations

import numpy as np
import pytest

from tensorquantlib.finance.volatility import (
    sabr_implied_vol,
    sabr_calibrate,
    svi_raw,
    svi_implied_vol,
    svi_calibrate,
    svi_surface,
)


class TestSABR:
    """Tests for the SABR implied volatility model."""

    def test_atm_limit(self):
        """ATM vol should approximate alpha * F^(beta-1) for small T."""
        F, K, T = 100.0, 100.0, 0.001
        alpha, beta, rho, nu = 0.3, 0.5, 0.0, 0.4
        vol = sabr_implied_vol(F, K, T, alpha, beta, rho, nu)
        atm_approx = alpha / F ** (1.0 - beta)
        assert abs(vol - atm_approx) < 0.01

    def test_atm_equals_array(self):
        """ATM as scalar and array should agree."""
        F, T = 100.0, 1.0
        alpha, beta, rho, nu = 0.3, 0.5, -0.3, 0.4
        vol_scalar = sabr_implied_vol(F, 100.0, T, alpha, beta, rho, nu)
        vol_array = sabr_implied_vol(F, np.array([100.0]), T, alpha, beta, rho, nu)
        assert abs(vol_scalar - vol_array[0]) < 1e-12

    def test_smile_symmetry_lognormal(self):
        """For beta=1, rho=0 the smile should be symmetric in log-strike."""
        F, T = 100.0, 1.0
        alpha, beta, rho, nu = 0.2, 1.0, 0.0, 0.3
        K_low = F * np.exp(-0.1)
        K_high = F * np.exp(0.1)
        vol_low = sabr_implied_vol(F, K_low, T, alpha, beta, rho, nu)
        vol_high = sabr_implied_vol(F, K_high, T, alpha, beta, rho, nu)
        assert abs(vol_low - vol_high) < 0.005

    def test_vectorized_strikes(self):
        """Should work with arrays of strikes."""
        F, T = 100.0, 1.0
        alpha, beta, rho, nu = 0.3, 0.5, -0.3, 0.4
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        vols = sabr_implied_vol(F, strikes, T, alpha, beta, rho, nu)
        assert vols.shape == (5,)
        assert np.all(vols > 0)

    def test_negative_rho_skew(self):
        """Negative rho should produce downward skew (low strikes have higher vol)."""
        F, T = 100.0, 1.0
        alpha, beta, rho, nu = 0.3, 0.5, -0.5, 0.4
        vol_low = sabr_implied_vol(F, 80.0, T, alpha, beta, rho, nu)
        vol_high = sabr_implied_vol(F, 120.0, T, alpha, beta, rho, nu)
        assert vol_low > vol_high

    def test_zero_nu_flat_smile(self):
        """With nu=0, the smile should be approximately flat near ATM."""
        F, T = 100.0, 1.0
        alpha, beta, rho, nu = 0.2, 0.5, 0.0, 0.0
        vols = sabr_implied_vol(F, np.array([95.0, 100.0, 105.0]), T, alpha, beta, rho, nu)
        # Vols should be very close to each other
        assert np.std(vols) < 0.005

    def test_calibration_round_trip(self):
        """Generate SABR vols, calibrate back, check params recover."""
        F, T = 100.0, 1.0
        true_alpha, true_beta, true_rho, true_nu = 0.25, 0.5, -0.3, 0.35
        strikes = np.linspace(70, 130, 15)
        true_vols = sabr_implied_vol(F, strikes, T, true_alpha, true_beta, true_rho, true_nu)

        result = sabr_calibrate(
            true_vols, F, strikes, T, beta=true_beta,
            initial_guess=(0.2, -0.2, 0.3),
        )
        assert result['rmse'] < 0.001
        assert abs(result['alpha'] - true_alpha) < 0.02
        assert abs(result['rho'] - true_rho) < 0.05
        assert abs(result['nu'] - true_nu) < 0.05

    def test_positive_vols(self):
        """All implied vols should be positive for reasonable inputs."""
        F, T = 100.0, 1.0
        alpha, beta, rho, nu = 0.3, 0.5, -0.3, 0.4
        strikes = np.linspace(60, 140, 50)
        vols = sabr_implied_vol(F, strikes, T, alpha, beta, rho, nu)
        assert np.all(vols > 0)


class TestSVI:
    """Tests for SVI raw parameterization."""

    def test_minimum_variance(self):
        """SVI total variance has minimum a + b*sigma*sqrt(1-rho^2) at k = m - rho*sigma/sqrt(1-rho^2)."""
        a, b, rho, m, sigma = 0.04, 0.1, 0.0, 0.0, 0.1
        # For rho=0, minimum is exactly at k=m
        w_at_m = svi_raw(m, a, b, rho, m, sigma)
        expected_min = a + b * sigma * np.sqrt(1.0 - rho ** 2)
        assert abs(w_at_m - expected_min) < 1e-10

    def test_linear_wings(self):
        """For large |k|, SVI should be approximately linear."""
        a, b, rho, m, sigma = 0.04, 0.1, -0.3, 0.0, 0.1
        k_large = np.array([5.0, 10.0, 15.0])
        w = svi_raw(k_large, a, b, rho, m, sigma)
        # Check approximately linear: finite differences should be similar
        dw = np.diff(w) / np.diff(k_large)
        assert np.std(dw) < 0.001

    def test_no_arbitrage_condition(self):
        """Check that a + b*sigma*sqrt(1-rho^2) >= 0 (no negative variance)."""
        a, b, rho, m, sigma = 0.04, 0.1, -0.3, 0.0, 0.1
        min_var = a + b * sigma * np.sqrt(1.0 - rho ** 2)
        assert min_var >= 0

    def test_svi_implied_vol(self):
        """SVI implied vol should be sqrt(w/T)."""
        k, T = 0.0, 1.0
        a, b, rho, m, sigma = 0.04, 0.1, -0.3, 0.0, 0.1
        vol = svi_implied_vol(k, T, a, b, rho, m, sigma)
        w = svi_raw(k, a, b, rho, m, sigma)
        assert abs(vol - np.sqrt(w / T)) < 1e-10

    def test_vectorized(self):
        """SVI should work with arrays of k."""
        k = np.linspace(-0.5, 0.5, 20)
        a, b, rho, m, sigma = 0.04, 0.1, -0.3, 0.0, 0.1
        w = svi_raw(k, a, b, rho, m, sigma)
        assert w.shape == (20,)
        assert np.all(w > 0)

    def test_calibration_round_trip(self):
        """Generate SVI vols, calibrate back, check RMSE is small."""
        F, T = 100.0, 1.0
        true_a, true_b, true_rho, true_m, true_sigma = 0.04, 0.08, -0.25, 0.02, 0.15
        strikes = np.linspace(70, 130, 20)
        k = np.log(strikes / F)
        true_vols = svi_implied_vol(k, T, true_a, true_b, true_rho, true_m, true_sigma)

        result = svi_calibrate(true_vols, strikes, F, T)
        assert result['rmse'] < 0.001

    def test_surface_shape(self):
        """SVI surface should have correct shape."""
        strikes = np.linspace(80, 120, 10)
        expiries = np.array([0.25, 0.5, 1.0])
        F = 100.0
        params = [
            {'a': 0.04, 'b': 0.1, 'rho': -0.3, 'm': 0.0, 'sigma': 0.1},
            {'a': 0.04, 'b': 0.08, 'rho': -0.25, 'm': 0.0, 'sigma': 0.12},
            {'a': 0.04, 'b': 0.06, 'rho': -0.2, 'm': 0.0, 'sigma': 0.15},
        ]
        surface = svi_surface(strikes, expiries, F, params)
        assert surface.shape == (3, 10)
        assert np.all(surface > 0)

    def test_surface_term_structure(self):
        """Longer expiries should generally have lower ATM vol (for same total var)."""
        strikes = np.array([100.0])
        expiries = np.array([0.25, 1.0])
        F = 100.0
        # Same total variance params
        params = [
            {'a': 0.04, 'b': 0.1, 'rho': -0.3, 'm': 0.0, 'sigma': 0.1},
            {'a': 0.04, 'b': 0.1, 'rho': -0.3, 'm': 0.0, 'sigma': 0.1},
        ]
        surface = svi_surface(strikes, expiries, F, params)
        # vol = sqrt(w/T), same w means longer T has lower vol
        assert surface[0, 0] > surface[1, 0]
