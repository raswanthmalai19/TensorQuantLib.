"""Tests for basket option Monte Carlo pricing and grid construction."""

import numpy as np
import pytest
from tensorquantlib.finance.basket import (
    simulate_basket,
    build_pricing_grid,
    build_pricing_grid_analytic,
)


class TestSimulateBasket:
    """Test Monte Carlo basket option pricer."""

    def setup_method(self):
        """Standard 3-asset basket parameters."""
        self.d = 3
        self.S0 = np.array([100.0, 100.0, 100.0])
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.sigma = np.array([0.2, 0.2, 0.2])
        self.corr = np.array([
            [1.0, 0.3, 0.3],
            [0.3, 1.0, 0.3],
            [0.3, 0.3, 1.0],
        ])
        self.weights = np.array([1/3, 1/3, 1/3])

    def test_price_positive(self):
        price, stderr = simulate_basket(
            self.S0, self.K, self.T, self.r, self.sigma,
            self.corr, self.weights, n_paths=50_000, seed=42,
        )
        assert price > 0

    def test_stderr_small(self):
        """Standard error should be < 1% of price with 100K paths."""
        price, stderr = simulate_basket(
            self.S0, self.K, self.T, self.r, self.sigma,
            self.corr, self.weights, n_paths=100_000, seed=42,
        )
        assert stderr / price < 0.01

    def test_put_price_positive(self):
        price, _ = simulate_basket(
            self.S0, self.K, self.T, self.r, self.sigma,
            self.corr, self.weights, n_paths=50_000, seed=42,
            option_type="put",
        )
        assert price > 0

    def test_reproducibility(self):
        """Same seed → same price."""
        p1, _ = simulate_basket(
            self.S0, self.K, self.T, self.r, self.sigma,
            self.corr, self.weights, n_paths=10_000, seed=123,
        )
        p2, _ = simulate_basket(
            self.S0, self.K, self.T, self.r, self.sigma,
            self.corr, self.weights, n_paths=10_000, seed=123,
        )
        assert p1 == p2

    def test_deep_itm_call(self):
        """Deep ITM basket call ≈ forward - K (discounted)."""
        S0_high = np.array([200.0, 200.0, 200.0])
        price, _ = simulate_basket(
            S0_high, self.K, self.T, self.r, self.sigma,
            self.corr, self.weights, n_paths=50_000, seed=42,
        )
        # Forward ≈ 200 * exp(rT), basket = same since equal weights
        forward_approx = 200.0 * np.exp(self.r * self.T)
        intrinsic = np.exp(-self.r * self.T) * (forward_approx - self.K)
        # Price should be close to intrinsic (within 5%)
        assert abs(price - intrinsic) / intrinsic < 0.05

    def test_2asset(self):
        """Works for 2-asset basket."""
        S0 = np.array([100.0, 100.0])
        sigma = np.array([0.2, 0.25])
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        weights = np.array([0.5, 0.5])
        price, stderr = simulate_basket(
            S0, 100.0, 1.0, 0.05, sigma, corr, weights,
            n_paths=50_000, seed=42,
        )
        assert price > 0
        assert stderr > 0


class TestBuildPricingGrid:
    """Test grid construction for TT compression (small grids only)."""

    def test_2d_grid_shape(self):
        """2-asset grid should have shape (n, n)."""
        S0_ranges = [(80.0, 120.0), (80.0, 120.0)]
        sigma = np.array([0.2, 0.2])
        corr = np.eye(2)
        weights = np.array([0.5, 0.5])

        grid, axes = build_pricing_grid(
            S0_ranges, K=100.0, T=1.0, r=0.05,
            sigma=sigma, corr=corr, weights=weights,
            n_points=5, n_mc_paths=500, seed=42,
        )
        assert grid.shape == (5, 5)
        assert len(axes) == 2
        assert len(axes[0]) == 5

    def test_grid_values_positive(self):
        """Prices in the grid should be non-negative."""
        S0_ranges = [(80.0, 120.0), (80.0, 120.0)]
        sigma = np.array([0.2, 0.2])
        corr = np.eye(2)
        weights = np.array([0.5, 0.5])

        grid, _ = build_pricing_grid(
            S0_ranges, K=100.0, T=1.0, r=0.05,
            sigma=sigma, corr=corr, weights=weights,
            n_points=3, n_mc_paths=1000, seed=42,
        )
        assert np.all(grid >= 0)


class TestBuildPricingGridAnalytic:
    """Test the fast analytic grid builder."""

    def test_3d_grid_shape(self):
        S0_ranges = [(80.0, 120.0)] * 3
        sigma = np.array([0.2, 0.2, 0.2])
        weights = np.array([1/3, 1/3, 1/3])

        grid, axes = build_pricing_grid_analytic(
            S0_ranges, K=100.0, T=1.0, r=0.05,
            sigma=sigma, weights=weights, n_points=10,
        )
        assert grid.shape == (10, 10, 10)
        assert len(axes) == 3

    def test_values_non_negative(self):
        S0_ranges = [(80.0, 120.0)] * 3
        sigma = np.array([0.2, 0.2, 0.2])
        weights = np.array([1/3, 1/3, 1/3])

        grid, _ = build_pricing_grid_analytic(
            S0_ranges, K=100.0, T=1.0, r=0.05,
            sigma=sigma, weights=weights, n_points=10,
        )
        assert np.all(grid >= 0)

    def test_monotonic_in_spot(self):
        """Call price should increase as all spots increase."""
        S0_ranges = [(80.0, 120.0)] * 2
        sigma = np.array([0.2, 0.2])
        weights = np.array([0.5, 0.5])

        grid, _ = build_pricing_grid_analytic(
            S0_ranges, K=100.0, T=1.0, r=0.05,
            sigma=sigma, weights=weights, n_points=20,
        )
        # Holding asset 2 fixed at midpoint, price should increase with asset 1
        mid = 10
        col = grid[:, mid]
        # After the strike region, should be monotonically increasing
        diffs = np.diff(col)
        assert np.all(diffs >= -1e-10)  # non-decreasing
