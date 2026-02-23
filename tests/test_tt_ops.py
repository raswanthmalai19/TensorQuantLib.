"""Tests for TT operations — tensorquantlib.tt.ops."""

import numpy as np
import pytest

from tensorquantlib.tt.decompose import tt_svd
from tensorquantlib.tt.ops import (
    tt_compression_ratio,
    tt_error,
    tt_eval,
    tt_eval_batch,
    tt_memory,
    tt_ranks,
    tt_to_full,
)

# ── helpers ──────────────────────────────────────────────────────────────────

def _smooth_tensor(shape, seed=42):
    np.random.default_rng(seed)
    vecs = [np.sin(np.linspace(0, np.pi, n)) + 0.1 for n in shape]
    result = vecs[0]
    for v in vecs[1:]:
        result = np.outer(result, v).reshape(-1)
    return result.reshape(shape)


def _random_tensor(shape, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape)


# ── tt_eval ──────────────────────────────────────────────────────────────────

class TestTTEval:
    """Tests for single-element TT evaluation."""

    def test_eval_matches_full_3d(self):
        """tt_eval matches tt_to_full for a 3D tensor."""
        shape = (5, 6, 7)
        A = _random_tensor(shape)
        cores = tt_svd(A, eps=1e-14)

        # Spot-check several elements
        for idx in [(0, 0, 0), (1, 2, 3), (4, 5, 6), (2, 3, 4)]:
            val = tt_eval(cores, idx)
            np.testing.assert_allclose(val, A[idx], atol=1e-10)

    def test_eval_matches_full_4d(self):
        """tt_eval matches tt_to_full for a 4D tensor."""
        shape = (3, 4, 5, 6)
        A = _random_tensor(shape)
        cores = tt_svd(A, eps=1e-14)

        for idx in [(0, 0, 0, 0), (2, 3, 4, 5), (1, 2, 3, 3)]:
            val = tt_eval(cores, idx)
            np.testing.assert_allclose(val, A[idx], atol=1e-10)

    def test_eval_wrong_dimension_raises(self):
        """tt_eval with wrong number of indices raises AssertionError."""
        shape = (4, 5, 6)
        A = _random_tensor(shape)
        cores = tt_svd(A, eps=1e-14)

        with pytest.raises(AssertionError):
            tt_eval(cores, (0, 0))  # too few

        with pytest.raises(AssertionError):
            tt_eval(cores, (0, 0, 0, 0))  # too many


# ── tt_eval_batch ────────────────────────────────────────────────────────────

class TestTTEvalBatch:
    """Tests for batch TT evaluation."""

    def test_batch_matches_single(self):
        """Batch evaluation matches single-point evaluation."""
        shape = (5, 6, 7)
        A = _random_tensor(shape)
        cores = tt_svd(A, eps=1e-14)

        indices = np.array([
            [0, 0, 0],
            [1, 2, 3],
            [4, 5, 6],
            [2, 3, 4],
        ])

        batch_vals = tt_eval_batch(cores, indices)
        single_vals = np.array([tt_eval(cores, tuple(idx)) for idx in indices])

        np.testing.assert_allclose(batch_vals, single_vals, atol=1e-12)

    def test_batch_matches_full_tensor(self):
        """Batch evaluation matches full tensor values."""
        shape = (4, 5, 6)
        A = _random_tensor(shape)
        cores = tt_svd(A, eps=1e-14)

        # Evaluate all elements
        all_indices = np.array(np.meshgrid(
            *[np.arange(n) for n in shape], indexing="ij"
        )).reshape(len(shape), -1).T

        batch_vals = tt_eval_batch(cores, all_indices)
        np.testing.assert_allclose(batch_vals, A.ravel(), atol=1e-10)

    def test_batch_large(self):
        """Batch evaluation on a larger set of random indices."""
        shape = (8, 8, 8)
        A = _random_tensor(shape)
        cores = tt_svd(A, eps=1e-14)

        rng = np.random.default_rng(123)
        n_points = 1000
        indices = np.column_stack([rng.integers(0, n, size=n_points) for n in shape])

        batch_vals = tt_eval_batch(cores, indices)
        expected = np.array([A[tuple(idx)] for idx in indices])

        np.testing.assert_allclose(batch_vals, expected, atol=1e-10)


# ── tt_to_full ───────────────────────────────────────────────────────────────

class TestTTToFull:
    """Tests for full reconstruction."""

    def test_reconstruction_shape(self):
        """Reconstructed tensor has the correct shape."""
        shape = (4, 5, 6)
        A = _random_tensor(shape)
        cores = tt_svd(A, eps=1e-14)

        A_rec = tt_to_full(cores)
        assert A_rec.shape == shape

    def test_reconstruction_values(self):
        """Reconstructed tensor matches original (tight tolerance)."""
        shape = (3, 4, 5, 6)
        A = _random_tensor(shape)
        cores = tt_svd(A, eps=1e-14)

        A_rec = tt_to_full(cores)
        np.testing.assert_allclose(A_rec, A, atol=1e-10)


# ── tt_ranks ─────────────────────────────────────────────────────────────────

class TestTTRanks:
    """Tests for TT-rank extraction."""

    def test_boundary_ranks(self):
        """First and last ranks are always 1."""
        shape = (5, 6, 7)
        A = _random_tensor(shape)
        cores = tt_svd(A, eps=1e-14)
        ranks = tt_ranks(cores)

        assert ranks[0] == 1
        assert ranks[-1] == 1

    def test_rank_count(self):
        """Number of ranks = d + 1."""
        shape = (3, 4, 5, 6)
        A = _random_tensor(shape)
        cores = tt_svd(A, eps=1e-14)
        ranks = tt_ranks(cores)

        assert len(ranks) == len(shape) + 1


# ── tt_memory ────────────────────────────────────────────────────────────────

class TestTTMemory:
    """Tests for TT memory computation."""

    def test_memory_positive(self):
        """Memory usage is positive."""
        shape = (4, 5, 6)
        A = _random_tensor(shape)
        cores = tt_svd(A, eps=1e-10)

        mem = tt_memory(cores)
        assert mem > 0

    def test_memory_less_than_full(self):
        """Compressed smooth tensor uses less memory than full."""
        shape = (10, 10, 10)
        A = _smooth_tensor(shape)
        cores = tt_svd(A, eps=1e-8)

        tt_mem = tt_memory(cores)
        full_mem = A.nbytes

        assert tt_mem < full_mem


# ── tt_error ─────────────────────────────────────────────────────────────────

class TestTTError:
    """Tests for reconstruction error computation."""

    def test_exact_zero_error(self):
        """Exact decomposition has ~zero error."""
        shape = (4, 5, 6)
        A = _random_tensor(shape)
        cores = tt_svd(A, eps=1e-14)

        err = tt_error(cores, A)
        assert err < 1e-10

    def test_truncated_error_bounded(self):
        """Truncated decomposition has bounded error."""
        shape = (8, 8, 8)
        A = _random_tensor(shape)
        eps = 0.05
        cores = tt_svd(A, eps=eps)

        err = tt_error(cores, A)
        assert err <= eps * 1.1

    def test_zero_tensor_error(self):
        """Error for zero tensor is 0."""
        A = np.zeros((4, 5, 6))
        cores = tt_svd(A, eps=1e-10)

        err = tt_error(cores, A)
        assert err == 0.0


# ── tt_compression_ratio ─────────────────────────────────────────────────────

class TestTTCompressionRatio:
    """Tests for compression ratio computation."""

    def test_smooth_tensor_high_compression(self):
        """Smooth tensor achieves high compression ratio."""
        shape = (10, 10, 10)
        A = _smooth_tensor(shape)
        cores = tt_svd(A, eps=1e-8)

        ratio = tt_compression_ratio(cores, shape)
        assert ratio > 1.0, f"Expected compression, got ratio {ratio}"

    def test_ratio_formula(self):
        """Compression ratio = full_bytes / tt_bytes."""
        shape = (5, 6, 7)
        A = _random_tensor(shape)
        cores = tt_svd(A, eps=1e-10)

        ratio = tt_compression_ratio(cores, shape)
        expected = (np.prod(shape) * 8) / tt_memory(cores)

        np.testing.assert_allclose(ratio, expected)
