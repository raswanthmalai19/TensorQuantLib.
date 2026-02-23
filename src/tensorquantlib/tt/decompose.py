"""
Tensor-Train decomposition algorithms.

Implements:
    - tt_svd: TT-SVD decomposition (Oseledets, 2011)
    - tt_round: TT-rounding via orthogonalization + truncated SVD
"""

from __future__ import annotations

import numpy as np


def tt_svd(
    tensor: np.ndarray,
    eps: float = 1e-6,
    max_rank: int | None = None,
) -> list[np.ndarray]:
    """Tensor-Train SVD decomposition.

    Decomposes a d-dimensional tensor A of shape (n1, n2, ..., nd) into
    a list of TT-cores [G1, G2, ..., Gd] where
    G_k has shape (r_{k-1}, n_k, r_k) and r_0 = r_d = 1.

    The reconstruction satisfies
    ``||A - A_TT||_F <= eps * ||A||_F``.

    Algorithm: Sequential left-to-right unfolding with truncated SVD.
    Per-step truncation threshold: ``delta = eps * ||A||_F / sqrt(d-1)``.

    Args:
        tensor: Input tensor, shape (n1, n2, ..., nd).
        eps: Relative truncation tolerance.
        max_rank: Maximum TT-rank (optional safety cap).

    Returns:
        List of TT-cores. cores[k].shape = (r_{k-1}, n_k, r_k).
    """
    d = tensor.ndim
    if d < 2:
        raise ValueError(f"Tensor must have at least 2 dimensions, got {d}")
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    if max_rank is not None and max_rank < 1:
        raise ValueError(f"max_rank must be >= 1, got {max_rank}")

    shape = tensor.shape
    norm_A = np.linalg.norm(tensor)

    # Handle zero tensor
    if norm_A < 1e-15:
        cores = []
        for k in range(d):
            cores.append(np.zeros((1, shape[k], 1)))
        return cores

    # Per-step truncation threshold (guarantees total error <= eps * ||A||_F)
    delta = eps * norm_A / np.sqrt(d - 1)

    cores = []
    C = tensor.copy().astype(np.float64)
    r_prev = 1

    for k in range(d - 1):
        n_k = shape[k]
        # Reshape C into 2D matrix: (r_prev * n_k) x (remaining dimensions)
        C = C.reshape(r_prev * n_k, -1)

        # Economy SVD
        U, S, Vt = np.linalg.svd(C, full_matrices=False)

        # Rank selection: find smallest r_k such that
        # sqrt(sum(S[r_k:]^2)) <= delta
        # Use reverse cumsum for numerical stability
        S_sq = S ** 2
        tail_norms_sq = np.cumsum(S_sq[::-1])[::-1]  # tail_norms_sq[i] = sum(S[i:]^2)

        # Find rank: smallest r such that tail_norms_sq[r] <= delta^2
        # tail_norms_sq has length len(S), and we want the smallest r >= 1
        # such that tail_norms_sq[r] <= delta^2 (where tail_norms_sq[len(S)] = 0)
        delta_sq = delta ** 2
        r_k = len(S)  # default: keep all
        for i in range(1, len(S)):
            if tail_norms_sq[i] <= delta_sq:
                r_k = i
                break

        # Apply max_rank cap
        if max_rank is not None:
            r_k = min(r_k, max_rank)

        # Ensure at least rank 1
        r_k = max(r_k, 1)

        # Truncate
        U_trunc = U[:, :r_k]
        S_trunc = S[:r_k]
        Vt_trunc = Vt[:r_k, :]

        # Store core: reshape U into (r_prev, n_k, r_k)
        cores.append(U_trunc.reshape(r_prev, n_k, r_k))

        # Prepare next iteration: C = diag(S) @ Vt
        C = np.diag(S_trunc) @ Vt_trunc
        r_prev = r_k

    # Last core: reshape remaining matrix into (r_prev, n_d, 1)
    n_d = shape[-1]
    cores.append(C.reshape(r_prev, n_d, 1))

    return cores


def tt_round(
    cores: list[np.ndarray],
    eps: float = 1e-6,
    max_rank: int | None = None,
) -> list[np.ndarray]:
    """Reduce TT-ranks via orthogonalization + truncated SVD sweep.

    Two-pass algorithm:
    1. Right-to-left QR sweep (right-orthogonalize)
    2. Left-to-right SVD sweep with truncation

    This is used after TT arithmetic (e.g., tt_add) which inflates ranks.

    Args:
        cores: List of TT-cores.
        eps: Relative truncation tolerance.
        max_rank: Maximum allowed rank.

    Returns:
        New list of TT-cores with reduced ranks.
    """
    d = len(cores)
    if d < 2:
        return [c.copy() for c in cores]

    # Work with copies
    cores = [c.copy() for c in cores]

    # Compute norm for truncation threshold
    # Reconstruct isn't feasible for large tensors, so estimate from cores
    # We use the Frobenius norm through the TT structure
    # For simplicity, do full reconstruction if small, otherwise use core norms
    norm_est = _tt_norm(cores)
    if norm_est < 1e-15:
        return cores

    delta = eps * norm_est / np.sqrt(d - 1)

    # ---- Pass 1: Right-to-left QR sweep ----
    for k in range(d - 1, 0, -1):
        r_left, n_k, r_right = cores[k].shape
        # Reshape core to (r_left, n_k * r_right) then transpose → (n_k * r_right, r_left)
        M = cores[k].reshape(r_left, n_k * r_right).T
        Q, R = np.linalg.qr(M)
        # Q: (n_k * r_right, new_r), R: (new_r, r_left)
        new_r = Q.shape[1]
        cores[k] = Q.T.reshape(new_r, n_k, r_right)
        # Absorb R into the previous core: contract on right bond dimension
        r_left_prev, n_prev, _ = cores[k - 1].shape
        # cores[k-1]: (r_left_prev, n_prev, r_left), R: (new_r, r_left)
        # new_core[k-1][i,j,l] = sum_m cores[k-1][i,j,m] * R[l,m]
        new_prev = cores[k - 1].reshape(r_left_prev * n_prev, r_left) @ R.T
        cores[k - 1] = new_prev.reshape(r_left_prev, n_prev, new_r)

    # ---- Pass 2: Left-to-right SVD sweep with truncation ----
    for k in range(d - 1):
        r_left, n_k, r_right = cores[k].shape
        M = cores[k].reshape(r_left * n_k, r_right)

        U, S, Vt = np.linalg.svd(M, full_matrices=False)

        # Rank truncation
        S_sq = S ** 2
        tail_norms_sq = np.cumsum(S_sq[::-1])[::-1]
        delta_sq = delta ** 2

        r_new = len(S)
        for i in range(1, len(S)):
            if tail_norms_sq[i] <= delta_sq:
                r_new = i
                break

        if max_rank is not None:
            r_new = min(r_new, max_rank)
        r_new = max(r_new, 1)

        U_trunc = U[:, :r_new]
        S_trunc = S[:r_new]
        Vt_trunc = Vt[:r_new, :]

        cores[k] = U_trunc.reshape(r_left, n_k, r_new)

        # Absorb S*Vt into next core
        SV = np.diag(S_trunc) @ Vt_trunc  # (r_new, r_right)
        _r_left_next, _n_next, _r_right_next = cores[k + 1].shape
        # cores[k+1] was (r_right, n_next, r_right_next), multiply from left
        cores[k + 1] = np.einsum("ij,jkl->ikl", SV, cores[k + 1])

    return cores


def _tt_norm(cores: list[np.ndarray]) -> float:
    """Compute the Frobenius norm of a tensor in TT format.

    Uses the transfer matrix approach: ||A||_F^2 = <A, A>_TT.
    Complexity: O(d * n * r^4) where r is the max rank.
    """
    d = len(cores)
    # Initialize: contract first core with itself
    # cores[0] shape: (1, n_0, r_0)
    G = cores[0]
    # <G, G> along mode n_0: sum over n_0 of G[:, i, :] ⊗ G[:, i, :]
    # Result shape: (r_0, r_0) — but since r_left=1 for first core, it's (r_0, r_0)
    r_0 = G.shape[2]
    Z = np.zeros((r_0, r_0))
    for i in range(G.shape[1]):
        Z += G[0, i, :].reshape(-1, 1) @ G[0, i, :].reshape(1, -1)

    for k in range(1, d):
        G = cores[k]
        _r_left, n_k, r_right = G.shape
        Z_new = np.zeros((r_right, r_right))
        for i in range(n_k):
            # G[:, i, :] is (r_left, r_right)
            slice_k = G[:, i, :]  # (r_left, r_right)
            # Z is (r_left, r_left) from previous step
            # Contribution: slice_k^T @ Z @ slice_k → (r_right, r_right)
            Z_new += slice_k.T @ Z @ slice_k
        Z = Z_new

    # Z is now (1, 1) — the squared norm
    return float(np.sqrt(float(Z.item())))
