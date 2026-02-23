"""Tests for TT arithmetic operations: add, scale, hadamard, dot, norm."""

import numpy as np
import pytest

from tensorquantlib.tt.decompose import tt_svd
from tensorquantlib.tt.ops import (
    tt_add,
    tt_dot,
    tt_frobenius_norm,
    tt_hadamard,
    tt_scale,
    tt_to_full,
)

# ── helpers ──────────────────────────────────────────────────────────────

def _random_tensor(*shape: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape)


def _make_tt(*shape: int, eps: float = 1e-12, seed: int = 42):
    """Return (full_tensor, tt_cores)."""
    T = _random_tensor(*shape, seed=seed)
    cores = tt_svd(T, eps=eps)
    return T, cores


# ── tt_add ───────────────────────────────────────────────────────────────

class TestTTAdd:
    def test_basic_2d(self):
        A, ca = _make_tt(6, 8, seed=0)
        B, cb = _make_tt(6, 8, seed=1)
        result = tt_to_full(tt_add(ca, cb))
        np.testing.assert_allclose(result, A + B, atol=1e-12)

    def test_basic_3d(self):
        A, ca = _make_tt(5, 6, 7, seed=10)
        B, cb = _make_tt(5, 6, 7, seed=11)
        result = tt_to_full(tt_add(ca, cb))
        np.testing.assert_allclose(result, A + B, atol=1e-12)

    def test_basic_4d(self):
        A, ca = _make_tt(4, 5, 3, 4, seed=20)
        B, cb = _make_tt(4, 5, 3, 4, seed=21)
        result = tt_to_full(tt_add(ca, cb))
        np.testing.assert_allclose(result, A + B, atol=1e-12)

    def test_add_to_self(self):
        A, ca = _make_tt(5, 6, seed=30)
        result = tt_to_full(tt_add(ca, ca))
        np.testing.assert_allclose(result, 2 * A, atol=1e-12)

    def test_dimension_mismatch_raises(self):
        _, ca = _make_tt(5, 6, seed=0)
        _, cb = _make_tt(5, 6, 7, seed=1)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            tt_add(ca, cb)

    def test_mode_size_mismatch_raises(self):
        _, ca = _make_tt(5, 6, seed=0)
        _, cb = _make_tt(5, 7, seed=1)
        with pytest.raises(ValueError, match="Mode size mismatch"):
            tt_add(ca, cb)


# ── tt_scale ─────────────────────────────────────────────────────────────

class TestTTScale:
    def test_scale_positive(self):
        A, ca = _make_tt(5, 6, 7, seed=0)
        result = tt_to_full(tt_scale(ca, 3.5))
        np.testing.assert_allclose(result, 3.5 * A, atol=1e-12)

    def test_scale_negative(self):
        A, ca = _make_tt(5, 6, seed=1)
        result = tt_to_full(tt_scale(ca, -2.0))
        np.testing.assert_allclose(result, -2.0 * A, atol=1e-12)

    def test_scale_zero(self):
        A, ca = _make_tt(4, 5, 3, seed=2)
        result = tt_to_full(tt_scale(ca, 0.0))
        np.testing.assert_allclose(result, np.zeros_like(A), atol=1e-15)

    def test_scale_one(self):
        A, ca = _make_tt(5, 6, seed=3)
        result = tt_to_full(tt_scale(ca, 1.0))
        np.testing.assert_allclose(result, A, atol=1e-12)

    def test_scale_does_not_modify_original(self):
        _, ca = _make_tt(5, 6, seed=4)
        original_first = ca[0].copy()
        tt_scale(ca, 99.0)
        np.testing.assert_array_equal(ca[0], original_first)


# ── tt_hadamard ──────────────────────────────────────────────────────────

class TestTTHadamard:
    def test_basic_2d(self):
        A, ca = _make_tt(6, 8, seed=0)
        B, cb = _make_tt(6, 8, seed=1)
        result = tt_to_full(tt_hadamard(ca, cb))
        np.testing.assert_allclose(result, A * B, atol=1e-10)

    def test_basic_3d(self):
        A, ca = _make_tt(5, 6, 4, seed=10)
        B, cb = _make_tt(5, 6, 4, seed=11)
        result = tt_to_full(tt_hadamard(ca, cb))
        np.testing.assert_allclose(result, A * B, atol=1e-10)

    def test_hadamard_with_self(self):
        A, ca = _make_tt(5, 6, seed=20)
        result = tt_to_full(tt_hadamard(ca, ca))
        np.testing.assert_allclose(result, A ** 2, atol=1e-10)

    def test_dimension_mismatch_raises(self):
        _, ca = _make_tt(5, 6, seed=0)
        _, cb = _make_tt(5, 6, 7, seed=1)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            tt_hadamard(ca, cb)


# ── tt_dot ───────────────────────────────────────────────────────────────

class TestTTDot:
    def test_basic_2d(self):
        A, ca = _make_tt(6, 8, seed=0)
        B, cb = _make_tt(6, 8, seed=1)
        result = tt_dot(ca, cb)
        expected = np.sum(A * B)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_basic_3d(self):
        A, ca = _make_tt(5, 6, 4, seed=10)
        B, cb = _make_tt(5, 6, 4, seed=11)
        result = tt_dot(ca, cb)
        expected = np.sum(A * B)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_dot_with_self_equals_squared_norm(self):
        A, ca = _make_tt(5, 6, 7, seed=20)
        result = tt_dot(ca, ca)
        expected = np.sum(A ** 2)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_dot_returns_scalar(self):
        _, ca = _make_tt(5, 6, seed=0)
        _, cb = _make_tt(5, 6, seed=1)
        result = tt_dot(ca, cb)
        assert isinstance(result, float)


# ── tt_frobenius_norm ────────────────────────────────────────────────────

class TestTTFrobeniusNorm:
    def test_matches_numpy(self):
        A, ca = _make_tt(5, 6, 7, seed=0)
        result = tt_frobenius_norm(ca)
        expected = np.linalg.norm(A)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_norm_of_zero(self):
        # Create a zero tensor via scaling
        _, ca = _make_tt(5, 6, seed=1)
        zero_cores = tt_scale(ca, 0.0)
        assert tt_frobenius_norm(zero_cores) == pytest.approx(0.0, abs=1e-15)

    def test_norm_scales_with_scalar(self):
        _A, ca = _make_tt(5, 6, seed=2)
        alpha = 3.7
        scaled_cores = tt_scale(ca, alpha)
        np.testing.assert_allclose(
            tt_frobenius_norm(scaled_cores),
            abs(alpha) * tt_frobenius_norm(ca),
            atol=1e-10,
        )


# ── Cross-operation consistency ──────────────────────────────────────────

class TestCrossOperations:
    def test_add_scale_linearity(self):
        """(alpha*A + beta*B) via TT arithmetic matches NumPy."""
        A, ca = _make_tt(5, 6, 4, seed=0)
        B, cb = _make_tt(5, 6, 4, seed=1)
        alpha, beta = 2.5, -1.3

        result_cores = tt_add(tt_scale(ca, alpha), tt_scale(cb, beta))
        result = tt_to_full(result_cores)
        expected = alpha * A + beta * B
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_dot_via_hadamard_and_sum(self):
        """tt_dot matches sum-of-hadamard."""
        _A, ca = _make_tt(5, 4, 3, seed=10)
        _B, cb = _make_tt(5, 4, 3, seed=11)

        dot_result = tt_dot(ca, cb)
        had_full = tt_to_full(tt_hadamard(ca, cb))
        np.testing.assert_allclose(dot_result, np.sum(had_full), atol=1e-10)

    def test_norm_via_dot(self):
        """Frobenius norm = sqrt(dot(A, A))."""
        _, ca = _make_tt(6, 5, 4, seed=20)
        norm_val = tt_frobenius_norm(ca)
        dot_val = tt_dot(ca, ca)
        np.testing.assert_allclose(norm_val ** 2, dot_val, atol=1e-10)
