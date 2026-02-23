"""
Tensor-Train operations — evaluation, reconstruction, arithmetic, diagnostics.

Provides:
    - tt_eval: Evaluate a single element from TT cores
    - tt_eval_batch: Vectorized multi-point evaluation
    - tt_to_full: Reconstruct full tensor from TT cores (validation only)
    - tt_add: Add two TT tensors (rank-additive)
    - tt_scale: Multiply a TT tensor by a scalar
    - tt_hadamard: Element-wise product of two TT tensors
    - tt_dot: Inner product of two TT tensors (no reconstruction)
    - tt_ranks: Get TT-ranks
    - tt_memory: Compute memory usage of TT format
    - tt_error: Compute relative reconstruction error
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Union


def tt_eval(cores: List[np.ndarray], indices: Tuple[int, ...]) -> float:
    """Evaluate a single element of a tensor in TT format.

    Computes A[i1, i2, ..., id] by contracting core slices:
        result = G1[:, i1, :] @ G2[:, i2, :] @ ... @ Gd[:, id, :]

    Complexity: O(d * r^2) where r is the max rank.

    Args:
        cores: List of TT-cores. cores[k].shape = (r_{k-1}, n_k, r_k).
        indices: Tuple of indices (i1, i2, ..., id).

    Returns:
        Scalar value at the given index.
    """
    assert len(indices) == len(cores), (
        f"Expected {len(cores)} indices, got {len(indices)}"
    )

    # Start with the first core slice: shape (1, r_0) → flatten to (r_0,)
    result = cores[0][:, indices[0], :]  # shape (r_0_left, r_0_right) = (1, r_0)

    for k in range(1, len(cores)):
        # cores[k][:, indices[k], :] has shape (r_{k-1}, r_k)
        slice_k = cores[k][:, indices[k], :]
        result = result @ slice_k  # matrix multiply: (1, r_{k-1}) @ (r_{k-1}, r_k) = (1, r_k)

    # Final result should be shape (1, 1) → scalar
    return float(result.item())


def tt_eval_batch(
    cores: List[np.ndarray],
    indices_array: np.ndarray,
) -> np.ndarray:
    """Evaluate multiple elements of a tensor in TT format.

    Vectorized version of tt_eval for batch evaluation.

    Args:
        cores: List of TT-cores.
        indices_array: Array of shape (n_points, d) where each row is an index tuple.

    Returns:
        Array of shape (n_points,) with values at the given indices.
    """
    n_points = indices_array.shape[0]
    d = len(cores)
    assert indices_array.shape[1] == d

    # Initialize with first core slices for all points
    # cores[0] shape: (1, n_0, r_0)
    # For each point p, extract cores[0][0, indices_array[p, 0], :] → shape (r_0,)
    idx_0 = indices_array[:, 0]  # shape (n_points,)
    result = cores[0][0, idx_0, :]  # shape (n_points, r_0)

    for k in range(1, d):
        idx_k = indices_array[:, k]  # shape (n_points,)
        # cores[k] shape: (r_{k-1}, n_k, r_k)
        # For each point p, we need cores[k][:, idx_k[p], :] → (r_{k-1}, r_k)
        # Then result[p] = result[p] @ cores[k][:, idx_k[p], :]
        slices = cores[k][:, idx_k, :]  # shape (r_{k-1}, n_points, r_k)
        slices = slices.transpose(1, 0, 2)  # shape (n_points, r_{k-1}, r_k)

        # Batch matrix multiply: (n_points, 1, r_{k-1}) @ (n_points, r_{k-1}, r_k)
        result = np.einsum("ni,nij->nj", result, slices)

    # result shape: (n_points, 1) → squeeze
    return result.squeeze(-1)


def tt_to_full(cores: List[np.ndarray]) -> np.ndarray:
    """Reconstruct the full tensor from TT cores.

    WARNING: Only use for validation on small tensors.
    Memory usage is O(n1 * n2 * ... * nd).

    Args:
        cores: List of TT-cores.

    Returns:
        Full tensor of shape (n1, n2, ..., nd).
    """
    d = len(cores)
    shape = tuple(c.shape[1] for c in cores)

    # Build by sequential contraction
    # Start with first core: (1, n_0, r_0) → squeeze to (n_0, r_0)
    result = cores[0].squeeze(0)  # shape (n_0, r_0)

    for k in range(1, d):
        # result shape: (n_0 * n_1 * ... * n_{k-1}, r_{k-1})
        # cores[k] shape: (r_{k-1}, n_k, r_k)

        r_prev = result.shape[-1]
        n_k = cores[k].shape[1]
        r_k = cores[k].shape[2]

        # Contract: result @ cores[k] reshaped
        # Reshape cores[k]: (r_{k-1}, n_k * r_k)
        core_mat = cores[k].reshape(r_prev, n_k * r_k)

        # result @ core_mat: (prod_prev, n_k * r_k)
        result = result @ core_mat

        # Reshape to separate n_k: (prod_prev * n_k, r_k)
        result = result.reshape(-1, r_k)

    # Final: (prod_all, 1) → reshape to original shape
    return result.reshape(shape)


def tt_ranks(cores: List[np.ndarray]) -> List[int]:
    """Get the TT-ranks (bond dimensions between cores).

    Args:
        cores: List of TT-cores.

    Returns:
        List of ranks [r_0, r_1, ..., r_d] where r_0 = r_d = 1.
    """
    ranks = [cores[0].shape[0]]  # r_0 = 1
    for c in cores:
        ranks.append(c.shape[2])
    return ranks


def tt_memory(cores: List[np.ndarray]) -> int:
    """Compute total memory usage of TT cores in bytes.

    Args:
        cores: List of TT-cores.

    Returns:
        Total bytes (assuming float64).
    """
    return sum(c.nbytes for c in cores)


def tt_error(
    cores: List[np.ndarray],
    original: np.ndarray,
) -> float:
    """Compute relative Frobenius reconstruction error.

    error = ||A - A_TT||_F / ||A||_F

    WARNING: Requires full tensor reconstruction — only for small tensors.

    Args:
        cores: List of TT-cores.
        original: Original tensor (same shape).

    Returns:
        Relative error (scalar).
    """
    reconstructed = tt_to_full(cores)
    norm_orig = np.linalg.norm(original)
    if norm_orig < 1e-15:
        return 0.0
    return float(np.linalg.norm(original - reconstructed) / norm_orig)


def tt_compression_ratio(
    cores: List[np.ndarray],
    original_shape: Tuple[int, ...],
) -> float:
    """Compute compression ratio: full_size / tt_size.

    Args:
        cores: List of TT-cores.
        original_shape: Shape of the original tensor.

    Returns:
        Compression ratio (> 1 means TT is smaller).
    """
    full_bytes = int(np.prod(original_shape)) * 8  # float64
    tt_bytes = tt_memory(cores)
    if tt_bytes == 0:
        return float("inf")
    return full_bytes / tt_bytes


# ====================================================================== #
# TT Arithmetic
# ====================================================================== #

def tt_add(
    cores_a: List[np.ndarray],
    cores_b: List[np.ndarray],
) -> List[np.ndarray]:
    """Add two TT tensors: C = A + B.

    The result has TT-ranks that are the *sum* of the input ranks.
    Use tt_round() afterwards to compress back down.

    Both tensors must have the same mode sizes (n_k).

    Args:
        cores_a: TT-cores of tensor A.
        cores_b: TT-cores of tensor B.

    Returns:
        TT-cores of A + B.
    """
    d = len(cores_a)
    if len(cores_b) != d:
        raise ValueError(
            f"Dimension mismatch: A has {d} cores, B has {len(cores_b)}"
        )

    result = []
    for k in range(d):
        ra_l, na, ra_r = cores_a[k].shape
        rb_l, nb, rb_r = cores_b[k].shape

        if na != nb:
            raise ValueError(
                f"Mode size mismatch at core {k}: A has {na}, B has {nb}"
            )

        if k == 0:
            # First core: concatenate along right rank → (1, n, ra_r + rb_r)
            new_core = np.concatenate([cores_a[k], cores_b[k]], axis=2)
        elif k == d - 1:
            # Last core: concatenate along left rank → (ra_l + rb_l, n, 1)
            new_core = np.concatenate([cores_a[k], cores_b[k]], axis=0)
        else:
            # Interior: block-diagonal → (ra_l + rb_l, n, ra_r + rb_r)
            new_core = np.zeros((ra_l + rb_l, na, ra_r + rb_r))
            new_core[:ra_l, :, :ra_r] = cores_a[k]
            new_core[ra_l:, :, ra_r:] = cores_b[k]

        result.append(new_core)

    return result


def tt_scale(
    cores: List[np.ndarray],
    alpha: float,
) -> List[np.ndarray]:
    """Multiply a TT tensor by a scalar: B = alpha * A.

    Only modifies the first core (ranks unchanged).

    Args:
        cores: TT-cores of tensor A.
        alpha: Scalar multiplier.

    Returns:
        TT-cores of alpha * A.
    """
    result = [c.copy() for c in cores]
    result[0] = alpha * result[0]
    return result


def tt_hadamard(
    cores_a: List[np.ndarray],
    cores_b: List[np.ndarray],
) -> List[np.ndarray]:
    """Element-wise (Hadamard) product of two TT tensors: C = A ⊙ B.

    The result has TT-ranks that are the *product* of the input ranks.
    Use tt_round() afterwards to compress.

    Args:
        cores_a: TT-cores of tensor A.
        cores_b: TT-cores of tensor B.

    Returns:
        TT-cores of A ⊙ B.
    """
    d = len(cores_a)
    if len(cores_b) != d:
        raise ValueError(
            f"Dimension mismatch: A has {d} cores, B has {len(cores_b)}"
        )

    result = []
    for k in range(d):
        ra_l, na, ra_r = cores_a[k].shape
        rb_l, nb, rb_r = cores_b[k].shape

        if na != nb:
            raise ValueError(
                f"Mode size mismatch at core {k}: A has {na}, B has {nb}"
            )

        # Kronecker product along rank dimensions for each mode index
        # Result core shape: (ra_l * rb_l, n, ra_r * rb_r)
        new_core = np.zeros((ra_l * rb_l, na, ra_r * rb_r))
        for i in range(na):
            # cores_a[k][:, i, :] is (ra_l, ra_r)
            # cores_b[k][:, i, :] is (rb_l, rb_r)
            # Kronecker product → (ra_l * rb_l, ra_r * rb_r)
            new_core[:, i, :] = np.kron(cores_a[k][:, i, :], cores_b[k][:, i, :])

        result.append(new_core)

    return result


def tt_dot(
    cores_a: List[np.ndarray],
    cores_b: List[np.ndarray],
) -> float:
    """Inner product of two TT tensors: <A, B> = sum(A ⊙ B).

    Computed efficiently without full reconstruction via
    sequential transfer-matrix contraction. O(d * r_a^2 * r_b^2 * n).

    Args:
        cores_a: TT-cores of tensor A.
        cores_b: TT-cores of tensor B.

    Returns:
        Scalar inner product.
    """
    d = len(cores_a)
    if len(cores_b) != d:
        raise ValueError(
            f"Dimension mismatch: A has {d} cores, B has {len(cores_b)}"
        )

    # Transfer matrix: shape (ra_l * rb_l,) initially (1,1) → scalar 1
    ra_l0 = cores_a[0].shape[0]
    rb_l0 = cores_b[0].shape[0]
    Z = np.ones((ra_l0, rb_l0))  # (1, 1)

    for k in range(d):
        ra_l, n_k, ra_r = cores_a[k].shape
        rb_l, nb, rb_r = cores_b[k].shape

        if n_k != nb:
            raise ValueError(
                f"Mode size mismatch at core {k}: A has {n_k}, B has {nb}"
            )

        # Contract: Z_new[ia, ib] = sum_j sum_{ia', ib'} Z[ia', ib'] * Ga[ia',j,ia] * Gb[ib',j,ib]
        # Efficient: loop over mode index j, accumulate
        Z_new = np.zeros((ra_r, rb_r))
        for j in range(n_k):
            # Ga_j = cores_a[k][:, j, :]  shape (ra_l, ra_r)
            # Gb_j = cores_b[k][:, j, :]  shape (rb_l, rb_r)
            # contribution = Ga_j^T @ Z @ Gb_j
            Z_new += cores_a[k][:, j, :].T @ Z @ cores_b[k][:, j, :]
        Z = Z_new

    return float(Z.item())


def tt_frobenius_norm(cores: List[np.ndarray]) -> float:
    """Frobenius norm of a TT tensor: ||A||_F = sqrt(<A, A>).

    Computed without reconstruction.

    Args:
        cores: TT-cores.

    Returns:
        Frobenius norm (scalar).
    """
    return float(np.sqrt(max(0.0, tt_dot(cores, cores))))
