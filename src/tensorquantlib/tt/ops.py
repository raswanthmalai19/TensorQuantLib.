"""
Tensor-Train operations — evaluation, reconstruction, diagnostics.

Provides:
    - tt_eval: Evaluate a single element from TT cores
    - tt_eval_batch: Vectorized multi-point evaluation
    - tt_to_full: Reconstruct full tensor from TT cores (validation only)
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
