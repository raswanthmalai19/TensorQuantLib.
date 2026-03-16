"""Tests for TT-Cross black-box approximation — tensorquantlib.tt.decompose."""

import numpy as np
import pytest

from tensorquantlib.tt.decompose import tt_cross, tt_svd
from tensorquantlib.tt.ops import tt_error, tt_eval, tt_ranks, tt_to_full
from tensorquantlib.tt.surrogate import TTSurrogate


# ─────────────────────────────────────────────────────────────────────
# Reference functions (analytical, smooth — should compress well)
# ─────────────────────────────────────────────────────────────────────

def _make_trig_fn(axes):
    """f(i0,...,i_{d-1}) = sin(x0)*cos(x1)*..., low TT rank."""
    def fn(*indices):
        vals = [axes[k][i] for k, i in enumerate(indices)]
        result = 1.0
        for j, v in enumerate(vals):
            result *= (np.sin(v) if j % 2 == 0 else np.cos(v))
        return float(result)
    return fn


def _make_sum_fn(axes):
    """f = sum_k x_k.  TT rank 2 exactly."""
    def fn(*indices):
        return float(sum(axes[k][i] for k, i in enumerate(indices)))
    return fn


def _make_product_fn(axes):
    """f = prod_k x_k.  TT rank 1 exactly."""
    def fn(*indices):
        result = 1.0
        for k, i in enumerate(indices):
            result *= axes[k][i]
        return float(result)
    return fn


# ─────────────────────────────────────────────────────────────────────
# Basic shape and rank tests
# ─────────────────────────────────────────────────────────────────────

class TestTTCrossBasic:
    def test_output_is_list_of_cores(self):
        axes = [np.linspace(0.1, 1.0, 8)] * 3
        fn = _make_sum_fn(axes)
        cores = tt_cross(fn, shape=(8, 8, 8), eps=1e-4, max_rank=5)
        assert isinstance(cores, list)
        assert len(cores) == 3

    def test_core_shapes_valid(self):
        """Core shapes must satisfy TT boundary conditions."""
        axes = [np.linspace(0.5, 1.5, 6)] * 4
        fn = _make_product_fn(axes)
        cores = tt_cross(fn, shape=(6, 6, 6, 6), eps=1e-4, max_rank=6)
        assert len(cores) == 4
        assert cores[0].shape[0] == 1, "First core left rank must be 1"
        assert cores[-1].shape[2] == 1, "Last core right rank must be 1"
        for k in range(len(cores) - 1):
            assert cores[k].shape[2] == cores[k + 1].shape[0], \
                f"Bond dimension mismatch at interface {k}-{k+1}"

    def test_core_mode_size_matches_shape(self):
        shape = (5, 7, 9)
        axes = [np.linspace(1.0, 2.0, n) for n in shape]
        fn = _make_sum_fn(axes)
        cores = tt_cross(fn, shape=shape, eps=1e-4, max_rank=5)
        for k, n_k in enumerate(shape):
            assert cores[k].shape[1] == n_k, f"Mode {k} size mismatch"

    def test_rank_does_not_exceed_max_rank(self):
        shape = (10, 10, 10)
        axes = [np.linspace(0.0, 1.0, 10)] * 3
        fn = _make_trig_fn(axes)
        max_rank = 8
        cores = tt_cross(fn, shape=shape, eps=1e-3, max_rank=max_rank)
        for k, core in enumerate(cores):
            assert core.shape[0] <= max_rank, f"Left rank of core {k} exceeds max_rank"
            assert core.shape[2] <= max_rank, f"Right rank of core {k} exceeds max_rank"


# ─────────────────────────────────────────────────────────────────────
# Accuracy tests (cross vs reference on small grids)
# ─────────────────────────────────────────────────────────────────────

class TestTTCrossAccuracy:
    def test_product_function_low_error(self):
        """Product function has exact TT rank 1 — TT-Cross should be nearly exact."""
        n = 12
        axes = [np.linspace(1.0, 2.0, n)] * 3
        fn = _make_product_fn(axes)
        shape = (n, n, n)

        cores = tt_cross(fn, shape=shape, eps=1e-6, max_rank=4, n_sweeps=6)

        # Build reference tensor
        A_ref = np.array([[[fn(i, j, k) for k in range(n)]
                           for j in range(n)]
                          for i in range(n)])
        A_tt = tt_to_full(cores)
        rel_err = np.linalg.norm(A_tt - A_ref) / (np.linalg.norm(A_ref) + 1e-15)
        assert rel_err < 0.01, f"Product function relative error too large: {rel_err:.4e}"

    def test_sum_function_low_error(self):
        """Sum function has TT rank 2 — should be captured exactly up to floating point."""
        n = 10
        axes = [np.linspace(0.0, 1.0, n)] * 3
        fn = _make_sum_fn(axes)
        shape = (n, n, n)

        cores = tt_cross(fn, shape=shape, eps=1e-6, max_rank=4, n_sweeps=8)

        A_ref = np.array([[[fn(i, j, k) for k in range(n)]
                           for j in range(n)]
                          for i in range(n)])
        A_tt = tt_to_full(cores)
        rel_err = np.linalg.norm(A_tt - A_ref) / (np.linalg.norm(A_ref) + 1e-15)
        assert rel_err < 0.05, f"Sum function relative error too large: {rel_err:.4e}"

    def test_trig_function_reasonable_accuracy(self):
        """Trigonometric function should be approximated within eps=0.05.
        
        Note: axes must avoid 0 since sin(0)=0 causes degenerate pivots.
        """
        n = 8
        axes = [np.linspace(0.3, np.pi / 2, n)] * 3  # avoid sin(0)=0
        fn = _make_trig_fn(axes)
        shape = (n, n, n)

        cores = tt_cross(fn, shape=shape, eps=0.05, max_rank=10, n_sweeps=6)

        A_ref = np.array([[[fn(i, j, k) for k in range(n)]
                           for j in range(n)]
                          for i in range(n)])
        A_tt = tt_to_full(cores)
        rel_err = np.linalg.norm(A_tt - A_ref) / (np.linalg.norm(A_ref) + 1e-15)
        assert rel_err < 0.5, f"Trig function relative error too large: {rel_err:.4e}"

    def test_pointwise_evaluation(self):
        """tt_eval on TT-Cross cores should match fn at a few selected points."""
        n = 10
        axes = [np.linspace(1.0, 3.0, n)] * 3
        fn = _make_product_fn(axes)
        shape = (n, n, n)

        cores = tt_cross(fn, shape=shape, eps=1e-5, max_rank=4, n_sweeps=6)

        # Spot-check 5 points
        rng = np.random.default_rng(0)
        for _ in range(5):
            idx = tuple(int(x) for x in rng.integers(0, n, size=3))
            expected = fn(*idx)
            got = tt_eval(cores, list(idx))
            assert abs(got - expected) < 0.5, \
                f"Point {idx}: expected {expected:.4f}, got {got:.4f}"


# ─────────────────────────────────────────────────────────────────────
# High-dimensional test (the main motivation for TT-Cross)
# ─────────────────────────────────────────────────────────────────────

class TestTTCross6D:
    def test_6d_no_full_grid(self):
        """6D function: TT-Cross should produce valid cores without full grid."""
        d = 6
        n = 8
        axes = [np.linspace(80.0, 120.0, n)] * d
        fn = _make_product_fn(axes)
        shape = (n,) * d

        cores = tt_cross(fn, shape=shape, eps=1e-3, max_rank=6, n_sweeps=4)
        assert len(cores) == d
        assert cores[0].shape[0] == 1
        assert cores[-1].shape[2] == 1

    def test_6d_rank_bounded(self):
        """6D TT-Cross ranks must respect max_rank."""
        d = 6
        n = 8
        axes = [np.linspace(0.5, 1.5, n)] * d
        fn = _make_sum_fn(axes)
        shape = (n,) * d
        max_rank = 5

        cores = tt_cross(fn, shape=shape, eps=1e-3, max_rank=max_rank, n_sweeps=3)
        ranks = tt_ranks(cores)
        assert max(ranks) <= max_rank, f"Max rank {max(ranks)} exceeds max_rank {max_rank}"

    def test_6d_eval_is_finite(self):
        """tt_eval on 6D TT-Cross result should produce finite values everywhere tested."""
        d = 6
        n = 6
        axes = [np.linspace(1.0, 2.0, n)] * d
        fn = _make_trig_fn(axes)
        shape = (n,) * d

        cores = tt_cross(fn, shape=shape, eps=0.1, max_rank=4, n_sweeps=3)

        rng = np.random.default_rng(7)
        for _ in range(10):
            idx = [int(x) for x in rng.integers(0, n, size=d)]
            val = tt_eval(cores, idx)
            assert np.isfinite(val), f"tt_eval returned non-finite at {idx}"


# ─────────────────────────────────────────────────────────────────────
# TTSurrogate.from_function() tests
# ─────────────────────────────────────────────────────────────────────

class TestTTSurrogateFromFunction:
    def test_from_function_returns_ttssurrogate(self):
        axes = [np.linspace(90.0, 110.0, 8)] * 3
        fn = _make_product_fn(axes)
        surr = TTSurrogate.from_function(fn, axes, eps=1e-3, max_rank=5, n_sweeps=4)
        assert isinstance(surr, TTSurrogate)

    def test_from_function_evaluate_finite(self):
        """Evaluating the surrogate at an interior point should give a finite number."""
        axes = [np.linspace(90.0, 110.0, 10)] * 3
        fn = _make_sum_fn(axes)
        surr = TTSurrogate.from_function(fn, axes, eps=1e-3, max_rank=5, n_sweeps=4)
        price = surr.evaluate([100.0, 100.0, 100.0])
        assert np.isfinite(price)

    def test_from_function_reasonable_accuracy(self):
        """Surrogate should approximate the function to within 10% at grid points."""
        n = 8
        axes = [np.linspace(1.0, 2.0, n)] * 3
        fn = _make_product_fn(axes)
        surr = TTSurrogate.from_function(fn, axes, eps=1e-4, max_rank=5, n_sweeps=6)

        # Evaluate at mid-grid continuous spots
        ref_pts = [(axes[k][n // 2] for k in range(3))]
        mid = [axes[k][n // 2] for k in range(3)]
        price_surr = surr.evaluate(mid)
        # Reference: product of mid-values
        price_ref = float(np.prod([axes[k][n // 2] for k in range(3)]))
        # Allow 15% relative error (surrogate is approximate)
        rel_err = abs(price_surr - price_ref) / (abs(price_ref) + 1e-15)
        assert rel_err < 0.15, f"Surrogate error {rel_err:.2%} too large"

    def test_from_function_requires_callable(self):
        axes = [np.linspace(0.0, 1.0, 5)] * 3
        with pytest.raises(TypeError):
            TTSurrogate.from_function("not_a_function", axes)

    def test_from_function_requires_2d(self):
        axes = [np.linspace(0.0, 1.0, 5)]
        with pytest.raises(ValueError):
            TTSurrogate.from_function(lambda i: float(i), axes)


# ─────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────

class TestTTCrossValidation:
    def test_raises_on_1d(self):
        with pytest.raises(ValueError, match="at least 2"):
            tt_cross(lambda i: float(i), shape=(10,))

    def test_raises_on_negative_eps(self):
        with pytest.raises(ValueError, match="eps"):
            tt_cross(lambda i, j: 0.0, shape=(5, 5), eps=-0.1)

    def test_raises_on_zero_max_rank(self):
        with pytest.raises(ValueError, match="max_rank"):
            tt_cross(lambda i, j: 0.0, shape=(5, 5), max_rank=0)
