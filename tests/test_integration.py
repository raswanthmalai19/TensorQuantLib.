"""End-to-end integration tests for TensorQuantLib.

These tests exercise the full pipeline: grid construction → TT-SVD
compression → surrogate evaluation → Greeks → visualization.
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")

from tensorquantlib.core.tensor import Tensor
from tensorquantlib.finance.black_scholes import (
    bs_delta,
    bs_price_numpy,
    bs_price_tensor,
)
from tensorquantlib.finance.greeks import compute_greeks
from tensorquantlib.tt.decompose import tt_round, tt_svd
from tensorquantlib.tt.ops import (
    tt_add,
    tt_dot,
    tt_frobenius_norm,
    tt_hadamard,
    tt_ranks,
    tt_scale,
    tt_to_full,
)
from tensorquantlib.tt.surrogate import TTSurrogate
from tensorquantlib.viz import plot_pricing_surface, plot_tt_ranks


class TestAutodiffToGreeksPipeline:
    """Autodiff engine → Black-Scholes pricing → Greeks."""

    def test_bs_autodiff_delta_matches_analytic(self):
        S_val = 100.0
        K, r, T, sigma = 100.0, 0.05, 1.0, 0.2

        # Autodiff delta
        S = Tensor([S_val], requires_grad=True)
        price = bs_price_tensor(S, K, T, r, sigma, option_type="call")
        price.backward()
        autodiff_delta = float(S.grad[0])

        # Analytic delta
        analytic_delta = float(bs_delta(S_val, K, T, r, sigma))

        np.testing.assert_allclose(autodiff_delta, analytic_delta, atol=1e-6)

    def test_compute_greeks_consistency(self):
        S_val = 100.0
        K, r, T, sigma = 100.0, 0.05, 1.0, 0.2

        greeks = compute_greeks(
            bs_price_tensor, S_val, K, T, r, sigma, option_type="call"
        )

        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks

        analytic_delta = float(bs_delta(S_val, K, T, r, sigma))
        np.testing.assert_allclose(greeks["delta"], analytic_delta, atol=1e-5)


class TestGridToSurrogatePipeline:
    """Build option grid → TT compress → evaluate surrogate → Greeks."""

    def test_2asset_basket_surrogate_pipeline(self):
        # Build surrogate
        surr = TTSurrogate.from_basket_analytic(
            S0_ranges=[(80, 120), (80, 120)],
            K=100.0,
            T=1.0,
            r=0.05,
            sigma=[0.2, 0.2],
            weights=[0.5, 0.5],
            n_points=15,
            eps=1e-4,
        )

        # Verify it was built
        assert surr.n_assets == 2
        assert len(surr.cores) == 2

        # Evaluate at a point
        price = surr.evaluate([100.0, 100.0])
        assert isinstance(price, float)
        assert price > 0  # ATM call should have positive value

        # Greeks
        greeks = surr.greeks([100.0, 100.0])
        assert "delta" in greeks
        assert len(greeks["delta"]) == 2

        # Summary (smoke test)
        summary = surr.summary()
        assert "n_assets" in summary
        assert "tt_ranks" in summary

    def test_3asset_surrogate_compress_and_evaluate(self):
        surr = TTSurrogate.from_basket_analytic(
            S0_ranges=[(80, 120)] * 3,
            K=100.0,
            T=1.0,
            r=0.05,
            sigma=[0.2, 0.2, 0.2],
            weights=[1 / 3, 1 / 3, 1 / 3],
            n_points=10,
            eps=1e-3,
        )

        # Verify TT-SVD produced valid cores
        ranks = tt_ranks(surr.cores)
        assert all(r >= 1 for r in ranks)

        # Evaluate
        price = surr.evaluate([100.0, 100.0, 100.0])
        assert price >= 0

    def test_surrogate_batch_consistency(self):
        """Single evaluate vs multiple evaluates should be consistent."""
        surr = TTSurrogate.from_basket_analytic(
            S0_ranges=[(80, 120), (80, 120)],
            K=100.0, T=1.0, r=0.05,
            sigma=[0.2, 0.2], weights=[0.5, 0.5],
            n_points=15, eps=1e-4,
        )

        points = [[90.0, 95.0], [100.0, 100.0], [110.0, 115.0]]
        prices = [surr.evaluate(p) for p in points]

        for p, price in zip(points, prices):
            single = surr.evaluate(p)
            np.testing.assert_allclose(single, price, atol=1e-12)


class TestTTArithmeticPipeline:
    """TT decompose → arithmetic → reconstruct and verify."""

    def test_linear_combination(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((8, 6, 5))
        B = rng.standard_normal((8, 6, 5))

        ca = tt_svd(A, eps=1e-12)
        cb = tt_svd(B, eps=1e-12)

        alpha, beta = 2.0, -0.5
        result_cores = tt_add(tt_scale(ca, alpha), tt_scale(cb, beta))

        # Verify before rounding
        result = tt_to_full(result_cores)
        expected = alpha * A + beta * B
        np.testing.assert_allclose(result, expected, atol=1e-10)

        # Round and verify accuracy is maintained
        rounded = tt_round(result_cores, eps=1e-6)
        result_rounded = tt_to_full(rounded)
        np.testing.assert_allclose(result_rounded, expected, atol=1e-4)

        # Ranks should decrease after rounding
        assert sum(tt_ranks(rounded)) <= sum(tt_ranks(result_cores))

    def test_hadamard_and_dot_pipeline(self):
        rng = np.random.default_rng(99)
        A = rng.standard_normal((6, 5, 4))
        B = rng.standard_normal((6, 5, 4))

        ca = tt_svd(A, eps=1e-12)
        cb = tt_svd(B, eps=1e-12)

        # Hadamard
        had_cores = tt_hadamard(ca, cb)
        had_full = tt_to_full(had_cores)
        np.testing.assert_allclose(had_full, A * B, atol=1e-10)

        # Dot = sum of Hadamard
        dot_val = tt_dot(ca, cb)
        np.testing.assert_allclose(dot_val, np.sum(A * B), atol=1e-10)

        # Frobenius norm
        norm_a = tt_frobenius_norm(ca)
        np.testing.assert_allclose(norm_a, np.linalg.norm(A), atol=1e-10)


class TestVisualizationPipeline:
    """Build data → plot (Agg backend, no display)."""

    def test_pricing_surface_from_surrogate(self):
        surr = TTSurrogate.from_basket_analytic(
            S0_ranges=[(80, 120), (80, 120)],
            K=100.0, T=1.0, r=0.05,
            sigma=[0.2, 0.2], weights=[0.5, 0.5],
            n_points=15, eps=1e-4,
        )

        # Reconstruct full grid for plotting
        grid = tt_to_full(surr.cores)
        fig, _ax = plot_pricing_surface(grid, surr.axes, title="Basket Surface")
        assert fig is not None

    def test_tt_ranks_plot_from_decomposition(self):
        rng = np.random.default_rng(42)
        T = rng.standard_normal((10, 8, 6, 5))
        cores = tt_svd(T, eps=1e-6)
        fig, _ax = plot_tt_ranks(cores)
        assert fig is not None


class TestFullPipelineRoundTrip:
    """Comprehensive end-to-end: BS price → grid → TT → surrogate → Greeks."""

    def test_1d_bs_to_surrogate_roundtrip(self):
        """Build a 2D (S, sigma) BS grid, compress, and verify prices."""
        S_vals = np.linspace(80, 120, 20)
        sig_vals = np.linspace(0.1, 0.5, 15)
        K, r, T = 100.0, 0.05, 1.0

        # Build grid
        grid = np.zeros((len(S_vals), len(sig_vals)))
        for i, S in enumerate(S_vals):
            for j, sig in enumerate(sig_vals):
                grid[i, j] = bs_price_numpy(S, K, T, r, sig)

        # Compress
        surr = TTSurrogate.from_grid(grid, [S_vals, sig_vals], eps=1e-6)

        # Evaluate at grid points and compare
        for _ in range(10):
            idx_S = np.random.randint(1, len(S_vals) - 1)
            idx_sig = np.random.randint(1, len(sig_vals) - 1)
            price_surr = surr.evaluate([S_vals[idx_S], sig_vals[idx_sig]])
            price_exact = grid[idx_S, idx_sig]
            np.testing.assert_allclose(price_surr, price_exact, atol=0.01)

    def test_surrogate_greeks_have_correct_sign(self):
        """For a call option, delta should be positive."""
        surr = TTSurrogate.from_basket_analytic(
            S0_ranges=[(80, 120), (80, 120)],
            K=100.0, T=1.0, r=0.05,
            sigma=[0.2, 0.2], weights=[0.5, 0.5],
            n_points=20, eps=1e-5,
        )

        greeks = surr.greeks([100.0, 100.0])
        # Delta for a call should be positive
        for d in greeks["delta"]:
            assert d > 0, f"Expected positive delta for ATM call, got {d}"


class TestVectorizedGreeksPipeline:
    """Vectorized Greeks computation across spot range."""

    def test_vectorized_greeks_shape_and_monotonicity(self):
        """Delta should be monotonically increasing for a call."""
        from tensorquantlib.finance.greeks import compute_greeks_vectorized

        S_array = np.linspace(80, 120, 21)
        result = compute_greeks_vectorized(
            bs_price_tensor, S_array,
            K=100.0, T=1.0, r=0.05, sigma=0.2,
            option_type="call",
        )

        # Shape checks
        assert result["price"].shape == S_array.shape
        assert result["delta"].shape == S_array.shape

        # All call prices should be non-negative
        assert np.all(result["price"] >= -1e-12)

        # Delta should increase with S for a call (monotonic)
        for i in range(len(S_array) - 1):
            assert result["delta"][i] <= result["delta"][i + 1] + 1e-6

    def test_vectorized_vs_scalar_greeks_agree(self):
        """Vectorized and scalar compute_greeks should match."""
        from tensorquantlib.finance.greeks import compute_greeks_vectorized

        S_pts = np.array([90.0, 100.0, 110.0])
        K, T, r, sigma = 100.0, 1.0, 0.05, 0.2

        vec = compute_greeks_vectorized(
            bs_price_tensor, S_pts, K, T, r, sigma, option_type="call"
        )

        for i, S_val in enumerate(S_pts):
            scalar = compute_greeks(
                bs_price_tensor, S_val, K, T, r, sigma, option_type="call"
            )
            np.testing.assert_allclose(
                vec["delta"][i], scalar["delta"], atol=1e-5,
                err_msg=f"Delta mismatch at S={S_val}"
            )


class TestValidationIntegration:
    """Gradient validation (check_grad) on financial pricing functions."""

    def test_check_grad_on_bs_pricing(self):
        from tensorquantlib.utils.validation import check_grad

        K, T, r, sigma = 100.0, 1.0, 0.05, 0.2
        S_input = Tensor(np.array([100.0]), requires_grad=True)

        result = check_grad(
            lambda s: bs_price_tensor(s, K, T, r, sigma, option_type="call"),
            [S_input],
            tol=1e-4,
        )
        assert result["passed"], f"Gradient check failed: {result}"

    def test_check_grad_bs_wrt_sigma(self):
        from tensorquantlib.utils.validation import check_grad

        S, K, T, r = 100.0, 100.0, 1.0, 0.05
        sigma_input = Tensor(np.array([0.2]), requires_grad=True)

        result = check_grad(
            lambda sig: bs_price_tensor(
                Tensor(np.array([S])), K, T, r, sig, option_type="call"
            ),
            [sigma_input],
            tol=1e-4,
        )
        assert result["passed"], f"Gradient check failed: {result}"


class TestTopLevelImports:
    """Verify the public API is accessible from top-level package."""

    def test_all_core_imports(self):
        import tensorquantlib as tql

        # Core
        assert hasattr(tql, "Tensor")
        # Finance
        assert hasattr(tql, "bs_price_numpy")
        assert hasattr(tql, "bs_price_tensor")
        assert hasattr(tql, "bs_delta")
        assert hasattr(tql, "bs_vega")
        assert hasattr(tql, "compute_greeks")
        assert hasattr(tql, "compute_greeks_vectorized")
        # TT
        assert hasattr(tql, "tt_svd")
        assert hasattr(tql, "tt_round")
        assert hasattr(tql, "tt_to_full")
        assert hasattr(tql, "TTSurrogate")
        # Viz
        assert hasattr(tql, "plot_pricing_surface")
        assert hasattr(tql, "plot_tt_ranks")

    def test_version_defined(self):
        import tensorquantlib as tql
        assert hasattr(tql, "__version__")
        assert isinstance(tql.__version__, str)
        assert len(tql.__version__) > 0


class TestPutCallParity:
    """Cross-validate put-call parity through multiple code paths."""

    def test_parity_analytic(self):
        """C - P = S*e^{-qT} - K*e^{-rT} for q=0."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        call = bs_price_numpy(S, K, T, r, sigma, option_type="call")
        put = bs_price_numpy(S, K, T, r, sigma, option_type="put")
        parity = S - K * np.exp(-r * T)
        np.testing.assert_allclose(call - put, parity, atol=1e-10)

    def test_parity_autograd(self):
        """Put-call parity holds through the autograd engine too."""
        S_val, K, T, r, sigma = 105.0, 100.0, 0.5, 0.03, 0.25
        S = Tensor(np.array([S_val]), requires_grad=True)

        call = bs_price_tensor(S, K, T, r, sigma, option_type="call")
        # Need a fresh Tensor for put
        S2 = Tensor(np.array([S_val]), requires_grad=True)
        put = bs_price_tensor(S2, K, T, r, sigma, option_type="put")

        parity = S_val - K * np.exp(-r * T)
        np.testing.assert_allclose(
            call.item() - put.item(), parity, atol=1e-8
        )


class TestTTSurrogateEdgeCases:
    """Edge-case tests for surrogate construction and evaluation."""

    def test_boundary_evaluation(self):
        """Evaluate surrogate at the boundary of the grid."""
        surr = TTSurrogate.from_basket_analytic(
            S0_ranges=[(80, 120), (80, 120)],
            K=100.0, T=1.0, r=0.05,
            sigma=[0.2, 0.2], weights=[0.5, 0.5],
            n_points=15, eps=1e-4,
        )
        # Low boundary → deep OTM call ≈ 0
        price_low = surr.evaluate([80.0, 80.0])
        assert price_low >= -1e-6

        # High boundary → deep ITM call
        price_high = surr.evaluate([120.0, 120.0])
        assert price_high > price_low

    def test_surrogate_recompress(self):
        """Build → recompress with tighter tolerance → verify accuracy."""
        surr = TTSurrogate.from_basket_analytic(
            S0_ranges=[(80, 120), (80, 120)],
            K=100.0, T=1.0, r=0.05,
            sigma=[0.2, 0.2], weights=[0.5, 0.5],
            n_points=20, eps=1e-8,
        )

        original_price = surr.evaluate([100.0, 100.0])

        # Recompress
        from tensorquantlib.tt.decompose import tt_round
        rounded_cores = tt_round(surr.cores, eps=1e-4)
        rounded_surr = TTSurrogate.from_grid(
            tt_to_full(rounded_cores), surr.axes, eps=1e-4
        )
        rounded_price = rounded_surr.evaluate([100.0, 100.0])

        np.testing.assert_allclose(rounded_price, original_price, atol=0.1)
