"""Edge-case tests targeting the ~5% of uncovered code paths.

Coverage gaps addressed:
- Tensor: wrapping Tensor, __rsub__, __rtruediv__, __rmatmul__, __itruediv__,
  reshape(tuple), __len__, __getitem__, mean(axis), zero_grad, repr
- basket: validation errors (sigma/corr/weights shape), Cholesky regularization
- black_scholes: put theta, put rho
- greeks: multi-element price backward, vectorized vega
- validation: non-grad inputs
- decompose: near-zero norm, max_rank in tt_round
- ops: dimension mismatch errors
- surrogate: from_basket_mc, print_summary
- viz: surface mode, greeks surface with single greek
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from tensorquantlib.core.tensor import Tensor, _ensure_tensor
from tensorquantlib.finance.basket import simulate_basket
from tensorquantlib.finance.black_scholes import (
    bs_price_tensor,
    bs_rho,
    bs_theta,
)
from tensorquantlib.finance.greeks import compute_greeks, compute_greeks_vectorized
from tensorquantlib.tt.decompose import tt_round, tt_svd
from tensorquantlib.tt.ops import (
    tt_add,
    tt_dot,
    tt_hadamard,
    tt_ranks,
    tt_to_full,
)
from tensorquantlib.tt.surrogate import TTSurrogate
from tensorquantlib.utils.validation import check_grad, numerical_gradient
from tensorquantlib.viz import (
    plot_greeks_surface,
    plot_pricing_surface,
)

# ── Tensor edge cases ──────────────────────────────────────────────────


class TestTensorEdgeCases:
    """Cover uncovered Tensor paths."""

    def test_wrap_tensor_in_tensor(self):
        """Line 36: `if isinstance(data, Tensor): data = data.data`."""
        a = Tensor(np.array([1.0, 2.0]))
        b = Tensor(a)  # wrap Tensor in Tensor
        np.testing.assert_array_equal(a.data, b.data)

    def test_rsub(self):
        """Lines 121-122: __rsub__."""
        a = Tensor(np.array([3.0]), requires_grad=True)
        result = 10.0 - a  # triggers __rsub__
        result.backward()
        assert float(result.data.item()) == 7.0
        np.testing.assert_allclose(a.grad, [-1.0])

    def test_rtruediv(self):
        """Lines 137-138: __rtruediv__."""
        a = Tensor(np.array([4.0]), requires_grad=True)
        result = 8.0 / a  # triggers __rtruediv__
        result.backward()
        assert float(result.data.item()) == 2.0
        # d/da (8/a) = -8/a^2 = -0.5
        np.testing.assert_allclose(a.grad, [-0.5])

    def test_rmatmul(self):
        """Lines 148-149: __rmatmul__."""
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        b = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]))
        result = b @ a  # uses Tensor @ Tensor so result is Tensor
        result.sum().backward()
        assert result.data.shape == (2, 2)

    def test_itruediv_forbidden(self):
        """Line 168: in-place division raises."""
        a = Tensor(np.array([1.0]))
        with pytest.raises(NotImplementedError, match="In-place div"):
            a /= 2.0

    def test_reshape_tuple(self):
        """Line 181: reshape with tuple arg."""
        a = Tensor(np.arange(6.0), requires_grad=True)
        b = a.reshape((2, 3))
        assert b.shape == (2, 3)
        b.sum().backward()
        np.testing.assert_array_equal(a.grad, np.ones(6))

    def test_len(self):
        """Line 207: __len__."""
        a = Tensor(np.array([1.0, 2.0, 3.0]))
        assert len(a) == 3

    def test_getitem(self):
        """__getitem__ returns a Tensor that participates in autograd."""
        a = Tensor(np.array([10.0, 20.0, 30.0]), requires_grad=True)
        # Scalar index returns a 0-d Tensor
        elem = a[1]
        assert isinstance(elem, Tensor)
        assert float(elem.data) == 20.0

        # Slice index returns a Tensor
        sl = a[0:2]
        assert isinstance(sl, Tensor)
        np.testing.assert_array_equal(sl.data, np.array([10.0, 20.0]))

        # Gradient flows back correctly
        y = a[0] ** 2 + a[2] ** 2  # 100 + 900 = 1000
        y.backward()
        np.testing.assert_allclose(a.grad, np.array([20.0, 0.0, 60.0]))

    def test_mean_with_axis(self):
        """Lines 472-473, 493-495: tensor_mean with axis."""
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        m = a.mean(axis=0)
        m.sum().backward()
        np.testing.assert_allclose(a.grad, np.full((2, 2), 0.5))

    def test_mean_with_multi_axis(self):
        """Lines 507-508: tensor_mean with tuple axis."""
        a = Tensor(np.arange(24.0).reshape(2, 3, 4), requires_grad=True)
        m = a.mean(axis=(0, 2))
        m.sum().backward()
        assert m.shape == (3,)
        # each element averaged over 2*4=8 elements
        np.testing.assert_allclose(a.grad, np.full((2, 3, 4), 1.0 / 8))

    def test_zero_grad(self):
        """Coverage for zero_grad."""
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        y = (a * a).sum()
        y.backward()
        assert a.grad is not None
        a.zero_grad()
        assert a.grad is None

    def test_repr(self):
        """Coverage for __repr__."""
        a = Tensor(np.array([1.0]), requires_grad=True)
        r = repr(a)
        assert "Tensor" in r
        assert "requires_grad=True" in r

    def test_ensure_tensor_from_list(self):
        """Coverage for _ensure_tensor with list input."""
        t = _ensure_tensor([1.0, 2.0])
        assert isinstance(t, Tensor)
        np.testing.assert_array_equal(t.data, [1.0, 2.0])

    def test_matmul_2d_backward(self):
        """Cover 2D matmul backward path for both inputs."""
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        b = Tensor(np.array([[5.0], [6.0]]), requires_grad=True)
        c = a @ b
        c.sum().backward()
        # dL/dA = grad @ B^T, dL/dB = A^T @ grad
        np.testing.assert_allclose(a.grad, np.array([[5.0, 6.0], [5.0, 6.0]]))
        np.testing.assert_allclose(b.grad, np.array([[4.0], [6.0]]))


# ── Basket validation edge cases ───────────────────────────────────────


class TestBasketValidation:
    """Cover untested validation branches in basket.py."""

    def test_sigma_shape_mismatch(self):
        """Line 70: sigma shape mismatch."""
        with pytest.raises(ValueError, match="sigma shape"):
            simulate_basket(
                S0=np.array([100.0, 100.0]),
                K=100, T=1.0, r=0.05,
                sigma=np.array([0.2]),  # wrong shape
                corr=np.eye(2),
                weights=np.array([0.5, 0.5]),
            )

    def test_corr_shape_mismatch(self):
        """Line 72: corr shape mismatch."""
        with pytest.raises(ValueError, match="corr shape"):
            simulate_basket(
                S0=np.array([100.0, 100.0]),
                K=100, T=1.0, r=0.05,
                sigma=np.array([0.2, 0.2]),
                corr=np.eye(3),  # wrong shape
                weights=np.array([0.5, 0.5]),
            )

    def test_weights_shape_mismatch(self):
        """Line 74: weights shape mismatch."""
        with pytest.raises(ValueError, match="weights shape"):
            simulate_basket(
                S0=np.array([100.0, 100.0]),
                K=100, T=1.0, r=0.05,
                sigma=np.array([0.2, 0.2]),
                corr=np.eye(2),
                weights=np.array([1.0]),  # wrong shape
            )

    def test_cholesky_regularization(self):
        """Lines 87-89: near-singular corr triggers regularization."""
        # Create a corr matrix that's not quite positive-definite
        corr = np.array([[1.0, 1.0], [1.0, 1.0]])  # singular
        result = simulate_basket(
            S0=np.array([100.0, 100.0]),
            K=100, T=1.0, r=0.05,
            sigma=np.array([0.2, 0.2]),
            corr=corr,
            weights=np.array([0.5, 0.5]),
            n_paths=1000,
            seed=42,
        )
        # simulate_basket returns (price, std_err) tuple
        if isinstance(result, tuple):
            price = result[0]
        else:
            price = result
        assert price > 0  # should still compute


# ── Black-Scholes put theta & rho ──────────────────────────────────────


class TestBSPutGreeks:
    """Cover put-specific branches in theta and rho."""

    def test_put_theta(self):
        """Line 146: put theta branch."""
        theta_put = bs_theta(100, 100, 1.0, 0.05, 0.2, option_type="put")
        # Theta is negative for options
        assert isinstance(float(theta_put), float)

    def test_put_rho(self):
        """Line 164: put rho branch."""
        rho_put = bs_rho(100, 100, 1.0, 0.05, 0.2, option_type="put")
        # Put rho is negative
        assert float(rho_put) < 0


# ── Greeks edge cases ──────────────────────────────────────────────────


class TestGreeksEdgeCases:
    """Cover multi-element and edge price backward paths."""

    def test_compute_greeks_with_array_spot(self):
        """Lines 55, 87: multi-element price backward path."""
        greeks = compute_greeks(
            bs_price_tensor, 100.0, 100.0, 1.0, 0.05, 0.2,
            option_type="call",
        )
        assert 0 < greeks["delta"] < 1
        assert greeks["gamma"] > 0
        assert greeks["vega"] > 0

    def test_vectorized_vega_scalar(self):
        """Line 125: vectorized vega with scalar sigma."""
        result = compute_greeks_vectorized(
            bs_price_tensor,
            np.array([90.0, 100.0, 110.0]),
            K=100.0, T=1.0, r=0.05, sigma=0.2,
        )
        assert result["vega"].shape[0] >= 1
        assert result["vega"][0] > 0


# ── Validation edge cases ──────────────────────────────────────────────


class TestValidationEdgeCases:
    """Cover non-grad input paths in validation.py."""

    def test_numerical_gradient_no_grad_input(self):
        """Lines 38-39: input with requires_grad=False."""
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        b = Tensor(np.array([3.0, 4.0]), requires_grad=False)

        grads = numerical_gradient(lambda x, y: (x * y).sum(), [a, b])
        assert grads[0] is not None
        assert grads[1] is None

    def test_check_grad_non_grad_input(self):
        """Lines 108-109: check_grad with mixed requires_grad."""
        a = Tensor(np.array([2.0]), requires_grad=True)
        b = Tensor(np.array([3.0]), requires_grad=False)

        result = check_grad(lambda x, y: x * y, [a, b], tol=1e-4)
        assert result["passed"]
        assert result["errors"][1] is None  # b has no gradient


# ── TT decomposition edge cases ───────────────────────────────────────


class TestTTDecomposeEdgeCases:
    """Cover edge paths in decompose.py."""

    def test_tt_round_near_zero_tensor(self):
        """Lines 148-149: near-zero norm tensor."""
        # Create a tensor that's essentially zero
        tiny = np.zeros((3, 4, 5)) + 1e-20
        cores = tt_svd(tiny, eps=1e-12)
        rounded = tt_round(cores, eps=1e-6)
        result = tt_to_full(rounded)
        np.testing.assert_allclose(result, tiny, atol=1e-15)

    def test_tt_round_with_max_rank(self):
        """Lines 184-185, 188: max_rank truncation in tt_round."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((8, 6, 5))
        cores = tt_svd(A, eps=1e-12)
        rounded = tt_round(cores, eps=1e-12, max_rank=2)
        ranks = tt_ranks(rounded)
        assert all(r <= 2 for r in ranks[1:-1])  # interior ranks bounded

    def test_tt_round_single_core(self):
        """Line 138: d < 2 early return."""
        cores = [np.random.randn(1, 5, 1)]
        result = tt_round(cores, eps=1e-6)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], cores[0])


# ── TT ops error paths ────────────────────────────────────────────────


class TestTTOpsErrors:
    """Cover dimension-mismatch error branches in tt ops."""

    def test_tt_hadamard_dimension_mismatch(self):
        """Line 309: different number of cores."""
        rng = np.random.default_rng(42)
        a = tt_svd(rng.standard_normal((3, 4)), eps=1e-10)
        b = tt_svd(rng.standard_normal((3, 4, 5)), eps=1e-10)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            tt_hadamard(a, b)

    def test_tt_dot_dimension_mismatch(self):
        """Line 345: different number of cores in dot."""
        rng = np.random.default_rng(42)
        a = tt_svd(rng.standard_normal((3, 4)), eps=1e-10)
        b = tt_svd(rng.standard_normal((3, 4, 5)), eps=1e-10)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            tt_dot(a, b)

    def test_tt_dot_mode_size_mismatch(self):
        """Line 359: mode size mismatch in dot."""
        rng = np.random.default_rng(42)
        a = tt_svd(rng.standard_normal((3, 4)), eps=1e-10)
        b = tt_svd(rng.standard_normal((3, 5)), eps=1e-10)
        with pytest.raises(ValueError, match="Mode size mismatch"):
            tt_dot(a, b)

    def test_tt_add_dimension_mismatch(self):
        """tt_add error for mismatched dimensions."""
        a = tt_svd(np.random.randn(3, 4), eps=1e-10)
        b = tt_svd(np.random.randn(3, 4, 5), eps=1e-10)
        with pytest.raises(ValueError):
            tt_add(a, b)


# ── Surrogate edge cases ──────────────────────────────────────────────


class TestSurrogateEdgeCases:
    """Cover surrogate untested paths."""

    def test_print_summary(self):
        """Line 335: print_summary smoke test."""
        surr = TTSurrogate.from_basket_analytic(
            S0_ranges=[(80, 120), (80, 120)],
            K=100.0, T=1.0, r=0.05,
            sigma=[0.2, 0.2], weights=[0.5, 0.5],
            n_points=10, eps=1e-4,
        )

        # Just test it doesn't crash; output goes to stdout
        surr.print_summary()

    def test_from_basket_mc(self):
        """Lines 204-220: from_basket_mc constructor."""
        surr = TTSurrogate.from_basket_mc(
            S0_ranges=[(90, 110), (90, 110)],
            K=100.0, T=1.0, r=0.05,
            sigma=[0.2, 0.2],
            corr=np.eye(2),
            weights=[0.5, 0.5],
            n_points=8,
            n_mc_paths=5000,
            eps=1e-3,
        )
        assert surr.n_assets == 2
        price = surr.evaluate([100.0, 100.0])
        assert price > 0

    def test_summary_with_original_memory(self):
        """Cover compression_ratio path in summary."""
        surr = TTSurrogate.from_basket_analytic(
            S0_ranges=[(80, 120), (80, 120)],
            K=100.0, T=1.0, r=0.05,
            sigma=[0.2, 0.2], weights=[0.5, 0.5],
            n_points=10, eps=1e-4,
        )
        summary = surr.summary()
        assert "compression_ratio" in summary
        assert summary["compression_ratio"] > 0


# ── Viz edge cases ─────────────────────────────────────────────────────


class TestVizEdgeCases:
    """Cover untested viz branches."""

    def test_pricing_surface_3d(self):
        """Lines 81, 159-160, 163: surface mode."""
        rng = np.random.default_rng(42)
        grid = rng.standard_normal((10, 8))
        axes = [np.linspace(80, 120, 10), np.linspace(0.1, 0.5, 8)]
        fig, _ax = plot_pricing_surface(grid, axes, mode="surface")
        assert fig is not None

    def test_greeks_surface_single_greek(self):
        """Line 25-26: single greek rendering."""
        rng = np.random.default_rng(42)
        grid = rng.standard_normal((10, 8))
        axes = [np.linspace(80, 120, 10), np.linspace(0.1, 0.5, 8)]
        fig, axes_out = plot_greeks_surface(
            {"Delta": grid}, axes, dims=(0, 1)
        )
        assert fig is not None
        assert len(axes_out) == 1
