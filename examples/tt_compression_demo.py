#!/usr/bin/env python3
"""
Example: Tensor-Train Compression
===================================

Demonstrates TT-SVD decomposition, rounding, and arithmetic:
- Compressing a smooth multi-dimensional function
- Inspecting TT-ranks and memory savings
- TT arithmetic: addition, scaling, Hadamard product, inner product
- Reconstruction error at varying tolerances

Usage:
    python3 examples/tt_compression_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from tensorquantlib.tt.decompose import tt_svd, tt_round
from tensorquantlib.tt.ops import (
    tt_to_full, tt_ranks, tt_memory, tt_error,
    tt_compression_ratio, tt_add, tt_scale,
    tt_hadamard, tt_dot, tt_frobenius_norm,
)


def main() -> None:
    print("=" * 60)
    print("  TensorQuantLib — TT Compression Demo")
    print("=" * 60)

    # ── 1. Build a smooth test tensor ──────────────────
    print("\n1. Build a 4D test tensor (smooth function)")
    n = 20
    x = [np.linspace(0, 1, n) for _ in range(4)]
    grid = np.meshgrid(*x, indexing="ij")
    # Smooth function: exp(-(x1^2 + x2^2 + x3^2 + x4^2))
    A = np.exp(-(grid[0] ** 2 + grid[1] ** 2 + grid[2] ** 2 + grid[3] ** 2))
    print(f"   Shape: {A.shape}")
    print(f"   Full tensor size: {A.nbytes / 1024:.1f} KB")
    print(f"   Frobenius norm: {np.linalg.norm(A):.6f}")

    # ── 2. TT-SVD compression ──────────────────────────
    print("\n2. TT-SVD compression (eps=1e-6)")
    cores = tt_svd(A, eps=1e-6)
    ranks = tt_ranks(cores)
    mem = tt_memory(cores)
    ratio = tt_compression_ratio(cores, A)
    error = tt_error(cores, A)
    print(f"   TT-ranks: {ranks}")
    print(f"   TT memory: {mem / 1024:.2f} KB")
    print(f"   Compression ratio: {ratio:.1f}×")
    print(f"   Relative error: {error:.2e}")

    # ── 3. Varying tolerance ──────────────────────────
    print("\n3. Compression vs. tolerance")
    print(f"   {'eps':>10s}  {'Max Rank':>8s}  {'Memory KB':>10s}  {'Ratio':>8s}  {'Error':>10s}")
    print(f"   {'─' * 10}  {'─' * 8}  {'─' * 10}  {'─' * 8}  {'─' * 10}")
    for eps in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]:
        c = tt_svd(A, eps=eps)
        r = tt_ranks(c)
        m = tt_memory(c)
        cr = tt_compression_ratio(c, A)
        e = tt_error(c, A)
        print(f"   {eps:10.0e}  {max(r):8d}  {m / 1024:10.2f}  {cr:8.1f}  {e:10.2e}")

    # ── 4. TT rounding ────────────────────────────────
    print("\n4. TT rounding (recompress to lower rank)")
    cores_fine = tt_svd(A, eps=1e-10)
    print(f"   Before rounding: ranks = {tt_ranks(cores_fine)}")
    cores_rounded = tt_round(cores_fine, eps=1e-4)
    print(f"   After rounding:  ranks = {tt_ranks(cores_rounded)}")
    print(f"   Rounding error: {tt_error(cores_rounded, A):.2e}")

    # ── 5. TT arithmetic ──────────────────────────────
    print("\n5. TT arithmetic")
    # Build a second tensor
    B = np.sin(grid[0]) * np.cos(grid[1]) * np.exp(-grid[2]) * (1 + grid[3])
    cores_b = tt_svd(B, eps=1e-6)

    # Addition
    cores_sum = tt_add(cores, cores_b)
    print(f"   A ranks: {tt_ranks(cores)}")
    print(f"   B ranks: {tt_ranks(cores_b)}")
    print(f"   A+B ranks: {tt_ranks(cores_sum)}")
    sum_full = tt_to_full(cores_sum)
    sum_err = np.linalg.norm(sum_full - (A + B)) / np.linalg.norm(A + B)
    print(f"   A+B error: {sum_err:.2e}")

    # Scaling
    cores_scaled = tt_scale(cores, 2.5)
    print(f"\n   2.5*A error: {tt_error(cores_scaled, 2.5 * A):.2e}")

    # Hadamard product
    cores_had = tt_hadamard(cores, cores_b)
    had_full = tt_to_full(cores_had)
    had_err = np.linalg.norm(had_full - A * B) / np.linalg.norm(A * B)
    print(f"   A⊙B error: {had_err:.2e}")

    # Inner product
    dot_tt = tt_dot(cores, cores_b)
    dot_exact = np.sum(A * B)
    print(f"\n   <A, B> TT:    {dot_tt:.6f}")
    print(f"   <A, B> exact: {dot_exact:.6f}")
    print(f"   Difference:   {abs(dot_tt - dot_exact):.2e}")

    # Frobenius norm
    norm_tt = tt_frobenius_norm(cores)
    norm_exact = np.linalg.norm(A)
    print(f"\n   ||A||_F TT:    {norm_tt:.6f}")
    print(f"   ||A||_F exact: {norm_exact:.6f}")

    print("\n✓ All TT compression examples completed successfully.")


if __name__ == "__main__":
    main()
