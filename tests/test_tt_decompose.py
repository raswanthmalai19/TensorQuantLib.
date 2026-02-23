"""Tests for TT-SVD decomposition — tensorquantlib.tt.decompose."""

import numpy as np

from tensorquantlib.tt.decompose import _tt_norm, tt_round, tt_svd
from tensorquantlib.tt.ops import tt_error, tt_ranks, tt_to_full

# ── helpers ──────────────────────────────────────────────────────────────────

def _smooth_tensor(shape, seed=42):
    """Create a smooth low-rank tensor (should compress well)."""
    np.random.default_rng(seed)
    len(shape)
    # Outer product of smooth vectors → rank-1 tensor
    vecs = [np.sin(np.linspace(0, np.pi, n)) + 0.1 for n in shape]
    result = vecs[0]
    for v in vecs[1:]:
        result = np.outer(result, v).reshape(-1)
    return result.reshape(shape)


def _random_tensor(shape, seed=42):
    """Create a random tensor (harder to compress)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape)


# ── tt_svd basics ────────────────────────────────────────────────────────────

class TestTTSVDBasic:
    """Basic TT-SVD decomposition tests."""

    def test_3d_decomposition_shapes(self):
        """TT-SVD of 3D tensor produces correct core shapes."""
        shape = (4, 5, 6)
        A = _random_tensor(shape)
        cores = tt_svd(A, eps=1e-10)

        assert len(cores) == 3
        # First core: (1, n0, r0)
        assert cores[0].shape[0] == 1
        assert cores[0].shape[1] == shape[0]
        # Last core: (r_{d-2}, n_{d-1}, 1)
        assert cores[-1].shape[1] == shape[-1]
        assert cores[-1].shape[2] == 1
        # Bond dimensions match
        for k in range(len(cores) - 1):
            assert cores[k].shape[2] == cores[k + 1].shape[0]

    def test_4d_decomposition_shapes(self):
        """TT-SVD of 4D tensor produces correct core shapes."""
        shape = (3, 4, 5, 6)
        A = _random_tensor(shape)
        cores = tt_svd(A, eps=1e-10)

        assert len(cores) == 4
        assert cores[0].shape[0] == 1
        assert cores[-1].shape[2] == 1
        for k in range(len(cores) - 1):
            assert cores[k].shape[2] == cores[k + 1].shape[0]

    def test_exact_reconstruction_3d(self):
        """With eps≈0 the full tensor is recovered exactly."""
        shape = (4, 5, 6)
        A = _random_tensor(shape)
        cores = tt_svd(A, eps=1e-14)
        A_rec = tt_to_full(cores)

        np.testing.assert_allclose(A_rec, A, atol=1e-10)

    def test_exact_reconstruction_4d(self):
        """With eps≈0 a 4D tensor is recovered exactly."""
        shape = (3, 4, 5, 6)
        A = _random_tensor(shape)
        cores = tt_svd(A, eps=1e-14)
        A_rec = tt_to_full(cores)

        np.testing.assert_allclose(A_rec, A, atol=1e-10)

    def test_2d_decomposition(self):
        """TT-SVD works on a matrix (2D)."""
        shape = (5, 7)
        A = _random_tensor(shape)
        cores = tt_svd(A, eps=1e-14)
        A_rec = tt_to_full(cores)

        assert len(cores) == 2
        np.testing.assert_allclose(A_rec, A, atol=1e-10)


# ── tt_svd compression ──────────────────────────────────────────────────────

class TestTTSVDCompression:
    """Test compression quality and rank truncation."""

    def test_smooth_tensor_low_ranks(self):
        """Smooth (low-rank) tensor should compress to low TT-ranks."""
        shape = (10, 10, 10)
        A = _smooth_tensor(shape)
        cores = tt_svd(A, eps=1e-8)
        ranks = tt_ranks(cores)

        # Rank-1 tensor → all internal ranks should be ≤ 3
        for r in ranks[1:-1]:
            assert r <= 3, f"Rank {r} too large for a smooth tensor"

    def test_compression_within_tolerance(self):
        """Reconstruction error respects the prescribed tolerance."""
        shape = (8, 8, 8, 8)
        A = _random_tensor(shape)
        eps = 0.01
        cores = tt_svd(A, eps=eps)
        err = tt_error(cores, A)

        assert err <= eps * 1.1, f"Error {err:.6f} exceeds tolerance {eps}"

    def test_max_rank_cap(self):
        """max_rank caps all TT-ranks."""
        shape = (10, 10, 10)
        A = _random_tensor(shape)
        max_r = 3
        cores = tt_svd(A, eps=1e-14, max_rank=max_r)
        ranks = tt_ranks(cores)

        for r in ranks:
            assert r <= max_r, f"Rank {r} exceeds max_rank {max_r}"

    def test_different_tolerances(self):
        """Tighter tolerance → higher ranks (or equal)."""
        shape = (8, 8, 8)
        A = _random_tensor(shape)

        cores_loose = tt_svd(A, eps=0.1)
        cores_tight = tt_svd(A, eps=1e-10)

        ranks_loose = tt_ranks(cores_loose)
        ranks_tight = tt_ranks(cores_tight)

        # Max rank of tight should be >= loose
        assert max(ranks_tight) >= max(ranks_loose)


# ── tt_round ─────────────────────────────────────────────────────────────────

class TestTTRound:
    """Tests for TT-rounding (re-compression)."""

    def test_round_preserves_accuracy(self):
        """Rounding with tight tolerance preserves the tensor."""
        shape = (6, 7, 8)
        A = _random_tensor(shape)
        cores = tt_svd(A, eps=1e-14)
        cores_rounded = tt_round(cores, eps=1e-12)

        A_rec = tt_to_full(cores_rounded)
        np.testing.assert_allclose(A_rec, A, atol=1e-8)

    def test_round_reduces_ranks(self):
        """Rounding with loose tolerance should reduce ranks."""
        shape = (8, 8, 8)
        A = _random_tensor(shape)
        cores = tt_svd(A, eps=1e-14)
        cores_rounded = tt_round(cores, eps=0.1)

        ranks_orig = tt_ranks(cores)
        ranks_rounded = tt_ranks(cores_rounded)

        assert max(ranks_rounded) <= max(ranks_orig)

    def test_round_smooth_tensor(self):
        """Rounding a smooth tensor to rank-1."""
        shape = (10, 10, 10)
        A = _smooth_tensor(shape)
        cores = tt_svd(A, eps=1e-14)
        cores_rounded = tt_round(cores, eps=1e-6)

        ranks = tt_ranks(cores_rounded)
        # Should remain very low rank
        for r in ranks[1:-1]:
            assert r <= 3


# ── _tt_norm ─────────────────────────────────────────────────────────────────

class TestTTNorm:
    """Tests for TT-format Frobenius norm computation."""

    def test_norm_matches_full(self):
        """TT-norm matches np.linalg.norm of the full tensor."""
        shape = (5, 6, 7)
        A = _random_tensor(shape)
        cores = tt_svd(A, eps=1e-14)

        tt_n = _tt_norm(cores)
        full_n = np.linalg.norm(A)

        np.testing.assert_allclose(tt_n, full_n, rtol=1e-10)

    def test_norm_smooth_tensor(self):
        """TT-norm correct for a smooth tensor."""
        shape = (8, 8, 8)
        A = _smooth_tensor(shape)
        cores = tt_svd(A, eps=1e-14)

        np.testing.assert_allclose(_tt_norm(cores), np.linalg.norm(A), rtol=1e-10)


# ── edge cases ───────────────────────────────────────────────────────────────

class TestTTSVDEdgeCases:
    """Edge cases for TT-SVD."""

    def test_zero_tensor(self):
        """TT-SVD of all-zeros tensor produces all-zero cores."""
        shape = (4, 5, 6)
        A = np.zeros(shape)
        cores = tt_svd(A, eps=1e-10)
        A_rec = tt_to_full(cores)

        np.testing.assert_array_equal(A_rec, A)

    def test_ones_tensor(self):
        """TT-SVD of all-ones tensor → rank-1 decomposition."""
        shape = (4, 5, 6)
        A = np.ones(shape)
        cores = tt_svd(A, eps=1e-10)

        ranks = tt_ranks(cores)
        for r in ranks:
            assert r == 1
        np.testing.assert_allclose(tt_to_full(cores), A, atol=1e-12)

    def test_5d_tensor(self):
        """TT-SVD works on a 5D tensor."""
        shape = (3, 4, 3, 4, 3)
        A = _smooth_tensor(shape)
        cores = tt_svd(A, eps=1e-8)

        assert len(cores) == 5
        err = tt_error(cores, A)
        assert err < 1e-6
