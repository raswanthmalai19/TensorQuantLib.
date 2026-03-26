"""
Tensor-Train decomposition algorithms.

Implements:
    - tt_svd:   TT-SVD decomposition (Oseledets, 2011)
    - tt_round: TT-rounding via orthogonalization + truncated SVD
    - tt_cross: Black-box TT-Cross approximation (Oseledets & Tyrtyshnikov, 2010)
              Builds a TT decomposition without forming the full tensor,
              making 6+ asset problems feasible.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.linalg import qr


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
        S_sq = S**2
        tail_norms_sq = np.cumsum(S_sq[::-1])[::-1]  # tail_norms_sq[i] = sum(S[i:]^2)

        # Find rank: smallest r such that tail_norms_sq[r] <= delta^2
        # tail_norms_sq has length len(S), and we want the smallest r >= 1
        # such that tail_norms_sq[r] <= delta^2 (where tail_norms_sq[len(S)] = 0)
        delta_sq = delta**2
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
        S_sq = S**2
        tail_norms_sq = np.cumsum(S_sq[::-1])[::-1]
        delta_sq = delta**2

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


# ======================================================================
# TT-Cross (black-box approximation — no full tensor needed)
# ======================================================================


def _maxvol_greedy(A: np.ndarray, r: int, rng: np.random.Generator) -> np.ndarray:
    """Approximate maximum-volume row subset of A (n × k, n ≥ k).

    Returns r row indices forming an approximate maximum-volume
    (r × k) submatrix of A.  Uses greedy pivoting based on QR.

    Algorithm
    ---------
    1. Find first r pivots via QR with column pivoting on A^T.
    2. Iteratively swap rows to increase the determinant of the
       selected submatrix until convergence (maxvol criterion).
    """
    n, k_cols = A.shape
    r = min(r, n, k_cols)
    if r == 0:
        return np.array([], dtype=int)

    # Initial pivot rows from QR
    _, _, piv = qr(A.T, pivoting=True, mode="economic")
    idx = piv[:r].copy()

    # Iterative improvement: swap rows to increase abs(det)
    # B = A @ inv(A[idx, :]) — each row B[i] represents how much
    # row i is "outside" the current selection
    sub = A[idx, :]  # (r, k_cols)
    try:
        B = np.linalg.lstsq(sub.T, A.T, rcond=None)[0].T  # (n, r)
    except np.linalg.LinAlgError:
        return idx

    max_iter = min(100, n)
    tol = 1.0 + 1e-4
    for _ in range(max_iter):
        i_best, j_best = np.unravel_index(np.argmax(np.abs(B)), B.shape)
        if abs(B[i_best, j_best]) <= tol:
            break
        # Swap row i_best into position j_best
        idx[j_best] = i_best
        sub = A[idx, :]
        try:
            B = np.linalg.lstsq(sub.T, A.T, rcond=None)[0].T
        except np.linalg.LinAlgError:
            break

    return idx


def _eval_fiber(
    fn: Callable[..., float],
    left_idx: np.ndarray,  # shape (r_l, k)  — left multi-indices
    k: int,  # current mode position (0-based)
    n_k: int,  # size of mode k
    right_idx: np.ndarray,  # shape (r_r, d-k-1)  — right multi-indices
    d: int,
) -> np.ndarray:
    """Evaluate fn on all (left × {0..n_k-1} × right) index combinations.

    Returns
    -------
    np.ndarray of shape ``(r_l * n_k, r_r)``
        C[il * n_k + ik, ir] = fn(*left_idx[il], ik, *right_idx[ir])
    """
    r_l = left_idx.shape[0]
    r_r = right_idx.shape[0] if right_idx.ndim > 0 and right_idx.size > 0 else 1
    C = np.zeros((r_l * n_k, r_r))
    for il in range(r_l):
        left_part = left_idx[il].tolist() if k > 0 else []
        for ik in range(n_k):
            row = il * n_k + ik
            if k == d - 1:
                # Last mode: no right indices
                C[row, 0] = fn(*left_part, ik)
            else:
                for ir in range(r_r):
                    right_part = right_idx[ir].tolist() if (d - k - 1) > 0 else []
                    C[row, ir] = fn(*left_part, ik, *right_part)
    return C


def _eval_interface(
    fn: Callable[..., float],
    left_idx: np.ndarray,  # shape (r_l, k+1) — left pivots at next boundary
    right_idx: np.ndarray,  # shape (r_r, d-k-1) — right pivots at current boundary
    d: int,
) -> np.ndarray:
    """Evaluate fn on all (left × right) combinations.

    Returns
    -------
    np.ndarray of shape ``(r_l, r_r)``
        Z[il, ir] = fn(*left_idx[il], *right_idx[ir])
    """
    r_l = left_idx.shape[0]
    r_r = right_idx.shape[0] if right_idx.ndim > 0 and right_idx.size > 0 else 1
    n_right_dims = right_idx.shape[1] if right_idx.ndim > 1 else 0
    Z = np.zeros((r_l, r_r))
    for il in range(r_l):
        for ir in range(r_r):
            idx = list(left_idx[il]) + (list(right_idx[ir]) if n_right_dims > 0 else [])
            Z[il, ir] = fn(*idx)
    return Z


def tt_cross(
    fn: Callable[..., float],
    shape: tuple[int, ...],
    eps: float = 1e-4,
    max_rank: int = 20,
    n_sweeps: int = 8,
    seed: int = 42,
) -> list[np.ndarray]:
    """TT-Cross black-box approximation (Oseledets & Tyrtyshnikov, 2010).

    Constructs a Tensor-Train decomposition of a *d*-dimensional function
    **without forming the full tensor**.  Only queries ``fn`` at a
    carefully selected set of index combinations — O(d · r² · n) evaluations
    instead of O(n^d) for TT-SVD.

    This makes 6+ asset problems feasible:

    * 6 assets, 15 pts/axis, rank 10 → ~54,000 evaluations
    * vs. 15^6 = 11,390,625 for full-grid TT-SVD

    Algorithm
    ---------
    1. **Initialise** right index sets J_k randomly.
    2. **Left-to-right sweep**: for each core k, evaluate the cross
       C_k = f(I_k × {0..n_k-1} × J_k) and select new left pivots I_{k+1}
       via greedy maxvol on the QR factor of C_k.
    3. **Build TT-cores** using the cross-interpolation formula:
       Core_k = C_k @ pinv(Z_k) where Z_k = f(I_{k+1} ++ J_k) is the
       (r_k × r_k) interface matrix.
    4. **Alternating sweeps** refine accuracy.

    Parameters
    ----------
    fn : callable
        Function accepting ``d`` integer arguments (grid indices)
        and returning a float::

            fn(i_0, i_1, ..., i_{d-1}) -> float

        Use ``functools.partial`` or a lambda to curry other parameters.
    shape : tuple of int
        Mode sizes ``(n_0, n_1, ..., n_{d-1})``.  These are *index* sizes.
        To convert continuous axes to indices, wrap ``fn`` accordingly.
    eps : float
        Target relative accuracy.  Controls rank selection via the
        tolerance passed to the SVD truncation after each cross.
    max_rank : int
        Hard upper bound on TT-ranks.
    n_sweeps : int
        Number of left-to-right + right-to-left alternating sweeps.
        ``n_sweeps=1`` gives a single L→R pass (fast, lower accuracy).
        ``n_sweeps=4`` is sufficient for smooth pricing surfaces.
    seed : int
        Random seed for initialising right index sets.

    Returns
    -------
    list of np.ndarray
        TT-cores[k].shape = ``(r_{k-1}, n_k, r_k)`` with
        ``r_0 = r_d = 1``.

    Examples
    --------
    Compress a 6-asset basket payoff without forming the 15^6 grid::

        import numpy as np
        from tensorquantlib.tt.decompose import tt_cross

        # Suppose price_lookup(i0, i1, i2, i3, i4, i5) evaluates the
        # basket option price at the i-th point on each asset's price axis.
        axes = [np.linspace(80, 120, 15)] * 6
        def price_lookup(*indices):
            spots = [axes[k][i] for k, i in enumerate(indices)]
            return basket_mc(spots, ...)   # your existing pricer

        cores = tt_cross(price_lookup, shape=(15,)*6, max_rank=15, n_sweeps=6)

    Notes
    -----
    After calling ``tt_cross``, wrap the result in a ``TTSurrogate``::

        from tensorquantlib.tt.surrogate import TTSurrogate
        surr = TTSurrogate(cores=cores, axes=axes, eps=eps)
    """
    d = len(shape)
    if d < 2:
        raise ValueError(f"TT-Cross requires at least 2 dimensions, got {d}")
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    if max_rank < 1:
        raise ValueError(f"max_rank must be >= 1, got {max_rank}")

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Step 1: Initialise right index sets J[k], shape (r_init, d-k-1)
    # J[k] stores right multi-indices used when building core k.
    # ------------------------------------------------------------------
    r_init = min(2, max_rank)

    # J[k] — right pivots at interface k → k+1
    # Each row of J[k] is a (d-k-1)-dimensional multi-index.
    J: list[np.ndarray] = []
    for k in range(d - 1):
        n_right = d - k - 1
        if n_right > 0:
            rows = np.stack(
                [rng.integers(0, shape[k + 1 + j], size=r_init) for j in range(n_right)],
                axis=1,
            )
        else:
            rows = np.zeros((r_init, 0), dtype=int)
        J.append(rows)

    # ------------------------------------------------------------------
    # Step 2: Left-to-right sweep to build left pivot sets left_pivots[k]
    # left_pivots[k] — left pivots at interface k-1 → k, shape (r_k, k)
    # left_pivots[0] is a single "empty" index — the left boundary has rank 1.
    # ------------------------------------------------------------------
    left_pivots: list[np.ndarray] = [np.zeros((1, 0), dtype=int)]

    for sweep in range(n_sweeps):
        # ---- Left-to-right ----
        for k in range(d - 1):
            r_l = left_pivots[k].shape[0]
            r_r = J[k].shape[0]
            n_k = shape[k]

            # Evaluate cross C: shape (r_l * n_k, r_r)
            C = _eval_fiber(fn, left_pivots[k], k, n_k, J[k], d)

            # QR + maxvol to select r_new pivot rows
            r_candidate = min(max_rank, r_l * n_k, max(r_r, 1))
            Q_mat, _ = qr(C, mode="economic")
            Q_r = Q_mat[:, :r_candidate]
            pivot_rows = _maxvol_greedy(Q_r, r_candidate, rng)

            # Decode rows back to (il, ik) pairs
            r_new = len(pivot_rows)
            new_I = np.zeros((r_new, k + 1), dtype=int)
            for j, row in enumerate(pivot_rows):
                il_dec = int(row) // n_k
                ik_dec = int(row) % n_k
                if k > 0 and il_dec < left_pivots[k].shape[0]:
                    new_I[j, :k] = left_pivots[k][il_dec, :]
                new_I[j, k] = ik_dec

            if sweep == 0:
                left_pivots.append(new_I)
            else:
                left_pivots[k + 1] = new_I

        # ---- Right-to-left (refine J) ----
        for k in range(d - 2, -1, -1):
            r_l = left_pivots[k].shape[0]
            r_r = J[k].shape[0]
            n_k = shape[k]

            C = _eval_fiber(fn, left_pivots[k], k, n_k, J[k], d)

            # Select new right pivots from column pivoting of C^T
            r_candidate = min(max_rank, r_l * n_k, max(r_r, 1))
            _, _, piv_col = qr(C.T, pivoting=True, mode="economic")
            pivot_rows = piv_col[:r_candidate]

            r_new = len(pivot_rows)
            n_right = d - k - 1
            new_J = np.zeros((r_new, n_right), dtype=int)
            for j, row in enumerate(pivot_rows):
                il_dec = int(row) // n_k
                ik_dec = int(row) % n_k
                # Right multi-index = (ik_dec, J[k][il_dec])
                if n_right == 1:
                    new_J[j, 0] = ik_dec
                elif n_right > 1 and il_dec < left_pivots[k + 1].shape[0]:
                    # current right pivot is the k+1 index combined with J[k+1]
                    new_J[j, 0] = ik_dec
                    if k + 1 < len(J) and il_dec < J[k + 1].shape[0]:
                        new_J[j, 1:] = J[k + 1][il_dec % J[k + 1].shape[0], :]

            J[k] = new_J

    # ------------------------------------------------------------------
    # Step 3: Build final TT-cores using the cross-interpolation formula.
    # Core_k = C_k @ pinv(Z_k)  reshaped to (r_{k-1}, n_k, r_k)
    # ------------------------------------------------------------------
    cores: list[np.ndarray] = []

    for k in range(d):
        n_k = shape[k]
        r_l = left_pivots[k].shape[0]

        if k < d - 1:
            r_r = J[k].shape[0]
            # Fiber: (r_l * n_k, r_r)
            C = _eval_fiber(fn, left_pivots[k], k, n_k, J[k], d)

            # Interface matrix Z: (|left_pivots[k+1]|, r_r)
            Z = _eval_interface(fn, left_pivots[k + 1], J[k], d)

            # Core = C @ pinv(Z): shape (r_l * n_k, r_next)
            # pinv handles rank-deficient Z gracefully
            Z_pinv = np.linalg.pinv(Z)  # (r_r, r_next)
            core_mat = C @ Z_pinv  # (r_l * n_k, r_next)

            # Truncate numerical noise via SVD
            U, s, Vt = np.linalg.svd(core_mat, full_matrices=False)
            # Keep singular values above eps * max
            thresh = eps * s[0] if s[0] > 0 else eps
            r_trunc = max(1, int(np.sum(s > thresh)))
            r_trunc = min(r_trunc, max_rank)
            core_mat = (U[:, :r_trunc] * s[:r_trunc]) @ Vt[:r_trunc, :]

            # Adjust r_next to r_trunc
            r_out = core_mat.shape[1]
            cores.append(core_mat.reshape(r_l, n_k, r_out))

        else:
            # Last core: fiber only, no interface
            right_dummy = np.zeros((1, 0), dtype=int)
            C = _eval_fiber(fn, left_pivots[k], k, n_k, right_dummy, d)  # (r_l * n_k, 1)
            cores.append(C.reshape(r_l, n_k, 1))

    return cores
