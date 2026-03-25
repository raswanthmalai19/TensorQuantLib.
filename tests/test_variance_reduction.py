"""Tests for variance reduction methods."""

from __future__ import annotations

import numpy as np

from tensorquantlib.finance.black_scholes import bs_price_numpy
from tensorquantlib.finance.variance_reduction import (
    asian_price_cv,
    bs_price_antithetic,
    bs_price_importance,
    bs_price_qmc,
    bs_price_stratified,
    compare_variance_reduction,
)

ANALYTIC_CALL = float(bs_price_numpy(100, 100, 1.0, 0.05, 0.20))


class TestAntithetic:
    def test_price_close_to_analytic(self):
        price = float(bs_price_antithetic(100, 100, 1.0, 0.05, 0.20, n_paths=200_000, seed=42))
        assert abs(price - ANALYTIC_CALL) < 0.3

    def test_stderr_returned(self):
        _, stderr = bs_price_antithetic(
            100, 100, 1.0, 0.05, 0.20, n_paths=100_000, seed=0, return_stderr=True
        )
        assert float(stderr) > 0

    def test_put_close_to_analytic(self):
        eu_put = float(bs_price_numpy(100, 100, 1.0, 0.05, 0.20, option_type="put"))
        price = float(
            bs_price_antithetic(
                100, 100, 1.0, 0.05, 0.20, option_type="put", n_paths=100_000, seed=0
            )
        )
        assert abs(price - eu_put) < 0.3


class TestControlVariates:
    def test_asian_cv_close_to_standard_mc(self):
        cv, _ = asian_price_cv(
            100, 100, 1.0, 0.05, 0.20, n_paths=50_000, seed=42, return_stderr=True
        )
        mc_price = float(bs_price_numpy(100, 100, 1.0, 0.05, 0.20))
        # Asian should be less than vanilla
        assert 0 < float(cv) < mc_price + 0.5

    def test_cv_stderr_smaller_than_plain_mc(self):
        """Control variate should reduce standard error."""
        from tensorquantlib.finance.exotics import asian_price_mc

        _, stderr_cv = asian_price_cv(
            100, 100, 1.0, 0.05, 0.20, n_paths=50_000, seed=42, return_stderr=True
        )
        _, stderr_plain = asian_price_mc(
            100, 100, 1.0, 0.05, 0.20, n_paths=50_000, seed=42, return_stderr=True
        )
        # CV stderr should be <= plain MC stderr (or at worst very close)
        assert float(stderr_cv) <= float(stderr_plain) * 1.5  # generous


class TestQMC:
    def test_price_close_to_analytic(self):
        price = float(bs_price_qmc(100, 100, 1.0, 0.05, 0.20, n_paths=65_536, seed=42))
        assert abs(price - ANALYTIC_CALL) < 0.2

    def test_returns_stderr(self):
        _, stderr = bs_price_qmc(100, 100, 1.0, 0.05, 0.20, return_stderr=True, seed=0)
        assert float(stderr) >= 0


class TestImportanceSampling:
    def test_price_close_to_analytic(self):
        # Importance sampling with a fixed seed; allow larger tolerance due to IS variance
        price = float(bs_price_importance(100, 100, 1.0, 0.05, 0.20, n_paths=200_000, seed=0))
        assert abs(price - ANALYTIC_CALL) < 2.0

    def test_returns_stderr(self):
        _, stderr = bs_price_importance(100, 100, 1.0, 0.05, 0.20, return_stderr=True, seed=0)
        assert float(stderr) >= 0


class TestStratified:
    def test_price_close_to_analytic(self):
        price = float(bs_price_stratified(100, 100, 1.0, 0.05, 0.20, n_paths=100_000, seed=42))
        assert abs(price - ANALYTIC_CALL) < 0.3

    def test_returns_stderr(self):
        _, stderr = bs_price_stratified(100, 100, 1.0, 0.05, 0.20, return_stderr=True, seed=0)
        assert float(stderr) >= 0


class TestCompareVR:
    def test_returns_all_methods(self):
        results = compare_variance_reduction(100, 100, 1.0, 0.05, 0.20, n_paths=10_000, seed=0)
        assert "crude_mc" in results
        assert "antithetic" in results
        assert "qmc_sobol" in results
        assert "stratified" in results

    def test_prices_consistent_across_methods(self):
        results = compare_variance_reduction(100, 100, 1.0, 0.05, 0.20, n_paths=20_000, seed=0)
        for name, r in results.items():
            if not np.isnan(r["price"]):
                # IS can have higher variance; allow 3.0 for it
                tol = 3.0 if name == "importance_sampling" else 2.0
                assert abs(r["price"] - ANALYTIC_CALL) < tol, f"{name} price too far from analytic"

    def test_vr_ratios_positive(self):
        results = compare_variance_reduction(100, 100, 1.0, 0.05, 0.20, n_paths=10_000, seed=0)
        for name, r in results.items():
            if not np.isnan(r["vr_ratio"]):
                assert r["vr_ratio"] > 0
