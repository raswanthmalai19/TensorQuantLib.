"""Tests for TT Surrogate engine — tensorquantlib.tt.surrogate."""

import numpy as np

from tensorquantlib.tt.surrogate import TTSurrogate

# ── from_grid ────────────────────────────────────────────────────────────────


class TestFromGrid:
    """Tests for TTSurrogate.from_grid constructor."""

    def test_basic_construction(self):
        """Build surrogate from a pre-computed grid."""
        axes = [np.linspace(80, 120, 10), np.linspace(80, 120, 10)]
        grid = np.outer(axes[0], axes[1])  # smooth rank-1 tensor

        surr = TTSurrogate.from_grid(grid, axes, eps=1e-6)
        assert surr.n_assets == 2
        assert surr.compress_time > 0

    def test_reconstruction_accuracy(self):
        """Surrogate at grid points matches original grid values."""
        axes = [np.linspace(80, 120, 15), np.linspace(80, 120, 15)]
        grid = np.outer(axes[0], axes[1])

        surr = TTSurrogate.from_grid(grid, axes, eps=1e-10)

        # Evaluate at grid points
        for i in range(0, 15, 3):
            for j in range(0, 15, 3):
                val = surr.evaluate([axes[0][i], axes[1][j]])
                np.testing.assert_allclose(val, grid[i, j], rtol=1e-6)


# ── from_basket_analytic ─────────────────────────────────────────────────────


class TestFromBasketAnalytic:
    """Tests for the analytic basket surrogate."""

    def test_2asset_construction(self):
        """Build a 2-asset analytic surrogate."""
        surr = TTSurrogate.from_basket_analytic(
            S0_ranges=[(80, 120), (80, 120)],
            K=100,
            T=1.0,
            r=0.05,
            sigma=[0.2, 0.2],
            weights=[0.5, 0.5],
            n_points=15,
            eps=1e-4,
        )
        assert surr.n_assets == 2
        assert surr.build_time > 0

    def test_3asset_construction(self):
        """Build a 3-asset analytic surrogate."""
        surr = TTSurrogate.from_basket_analytic(
            S0_ranges=[(80, 120)] * 3,
            K=100,
            T=1.0,
            r=0.05,
            sigma=[0.2] * 3,
            weights=[1 / 3] * 3,
            n_points=10,
            eps=1e-3,
        )
        assert surr.n_assets == 3

    def test_evaluation_at_center(self):
        """Price at ATM spot should be positive."""
        surr = TTSurrogate.from_basket_analytic(
            S0_ranges=[(80, 120), (80, 120)],
            K=100,
            T=1.0,
            r=0.05,
            sigma=[0.2, 0.2],
            weights=[0.5, 0.5],
            n_points=20,
            eps=1e-4,
        )
        price = surr.evaluate([100.0, 100.0])
        assert price > 0, f"ATM price should be positive, got {price}"

    def test_evaluation_deep_itm(self):
        """Deep ITM basket should have high price."""
        surr = TTSurrogate.from_basket_analytic(
            S0_ranges=[(80, 120), (80, 120)],
            K=100,
            T=1.0,
            r=0.05,
            sigma=[0.2, 0.2],
            weights=[0.5, 0.5],
            n_points=20,
            eps=1e-4,
        )
        price_itm = surr.evaluate([118.0, 118.0])
        price_otm = surr.evaluate([82.0, 82.0])

        assert price_itm > price_otm

    def test_batch_evaluation(self):
        """Batch evaluation matches single-point evaluation."""
        surr = TTSurrogate.from_basket_analytic(
            S0_ranges=[(80, 120), (80, 120)],
            K=100,
            T=1.0,
            r=0.05,
            sigma=[0.2, 0.2],
            weights=[0.5, 0.5],
            n_points=15,
            eps=1e-4,
        )
        spots = np.array(
            [
                [90.0, 95.0],
                [100.0, 100.0],
                [110.0, 115.0],
            ]
        )
        batch_prices = surr.evaluate(spots)
        single_prices = np.array([surr.evaluate(s) for s in spots])

        np.testing.assert_allclose(batch_prices, single_prices, rtol=1e-10)


# ── greeks ───────────────────────────────────────────────────────────────────


class TestGreeks:
    """Tests for Greek computation through the surrogate."""

    def test_delta_positive_call(self):
        """Delta of a call basket should be positive."""
        surr = TTSurrogate.from_basket_analytic(
            S0_ranges=[(80, 120), (80, 120)],
            K=100,
            T=1.0,
            r=0.05,
            sigma=[0.2, 0.2],
            weights=[0.5, 0.5],
            n_points=20,
            eps=1e-4,
        )
        g = surr.greeks([100.0, 100.0])

        assert g["price"] > 0
        for k in range(2):
            assert g["delta"][k] > 0, f"Delta[{k}] should be positive for call"

    def test_gamma_positive_call(self):
        """Gamma of a call should be positive (convexity)."""
        surr = TTSurrogate.from_basket_analytic(
            S0_ranges=[(80, 120), (80, 120)],
            K=100,
            T=1.0,
            r=0.05,
            sigma=[0.2, 0.2],
            weights=[0.5, 0.5],
            n_points=20,
            eps=1e-4,
        )
        g = surr.greeks([100.0, 100.0])

        for k in range(2):
            assert g["gamma"][k] >= -1e-8, (
                f"Gamma[{k}] should be non-negative (got {g['gamma'][k]})"
            )

    def test_delta_increases_with_spot(self):
        """Delta should increase as we go deeper ITM."""
        surr = TTSurrogate.from_basket_analytic(
            S0_ranges=[(70, 130), (70, 130)],
            K=100,
            T=1.0,
            r=0.05,
            sigma=[0.2, 0.2],
            weights=[0.5, 0.5],
            n_points=25,
            eps=1e-4,
        )
        g_itm = surr.greeks([115.0, 115.0])
        g_otm = surr.greeks([85.0, 85.0])

        for k in range(2):
            assert g_itm["delta"][k] > g_otm["delta"][k]


# ── summary / diagnostics ───────────────────────────────────────────────────


class TestSummary:
    """Test diagnostic output."""

    def test_summary_keys(self):
        """Summary dict contains expected keys."""
        surr = TTSurrogate.from_basket_analytic(
            S0_ranges=[(80, 120), (80, 120)],
            K=100,
            T=1.0,
            r=0.05,
            sigma=[0.2, 0.2],
            weights=[0.5, 0.5],
            n_points=10,
            eps=1e-3,
        )
        s = surr.summary()

        required_keys = [
            "n_assets",
            "grid_shape",
            "tt_ranks",
            "max_rank",
            "tt_memory_bytes",
            "tt_memory_KB",
            "eps",
            "build_time_s",
            "compress_time_s",
            "full_memory_bytes",
            "compression_ratio",
        ]
        for key in required_keys:
            assert key in s, f"Missing key: {key}"

    def test_compression_ratio_gt_1(self):
        """Smooth pricing grid should compress well."""
        surr = TTSurrogate.from_basket_analytic(
            S0_ranges=[(80, 120)] * 3,
            K=100,
            T=1.0,
            r=0.05,
            sigma=[0.2] * 3,
            weights=[1 / 3] * 3,
            n_points=20,
            eps=1e-3,
        )
        s = surr.summary()
        assert s["compression_ratio"] > 1.0

    def test_print_summary(self, capsys):
        """print_summary runs without error."""
        surr = TTSurrogate.from_basket_analytic(
            S0_ranges=[(80, 120), (80, 120)],
            K=100,
            T=1.0,
            r=0.05,
            sigma=[0.2, 0.2],
            weights=[0.5, 0.5],
            n_points=10,
            eps=1e-3,
        )
        surr.print_summary()
        captured = capsys.readouterr()
        assert "TT-Surrogate Summary" in captured.out
        assert "Compression" in captured.out


# ── edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases for surrogate evaluation."""

    def test_spot_at_boundary(self):
        """Evaluation at grid boundary doesn't crash."""
        surr = TTSurrogate.from_basket_analytic(
            S0_ranges=[(80, 120), (80, 120)],
            K=100,
            T=1.0,
            r=0.05,
            sigma=[0.2, 0.2],
            weights=[0.5, 0.5],
            n_points=10,
            eps=1e-3,
        )
        # Exactly at edges
        p1 = surr.evaluate([80.0, 80.0])
        p2 = surr.evaluate([120.0, 120.0])
        assert np.isfinite(p1) and np.isfinite(p2)

    def test_spot_outside_grid_clamped(self):
        """Spots outside the grid are clamped (no crash)."""
        surr = TTSurrogate.from_basket_analytic(
            S0_ranges=[(80, 120), (80, 120)],
            K=100,
            T=1.0,
            r=0.05,
            sigma=[0.2, 0.2],
            weights=[0.5, 0.5],
            n_points=10,
            eps=1e-3,
        )
        # Outside both ends
        p = surr.evaluate([50.0, 150.0])
        assert np.isfinite(p)
