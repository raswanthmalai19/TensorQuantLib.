#!/usr/bin/env python3
"""
Benchmark: TT-Surrogate vs Brute-Force Grid Evaluation
========================================================

Measures:
1. Grid build time (analytic)
2. TT-SVD compression time
3. TT evaluation speed vs direct grid lookup
4. Memory usage comparison
5. Accuracy vs tolerance trade-off

Usage:
    python3 benchmarks/bench_tt_vs_mc.py
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorquantlib.tt.surrogate import TTSurrogate
from tensorquantlib.tt.decompose import tt_svd
from tensorquantlib.tt.ops import tt_eval_batch, tt_to_full, tt_ranks, tt_memory, tt_error
from tensorquantlib.finance.basket import build_pricing_grid_analytic


def benchmark_compression_vs_tolerance():
    """Test compression quality at different tolerances."""
    print("\n" + "=" * 70)
    print("  Benchmark 1: Compression vs Tolerance (3-asset, 30 pts/axis)")
    print("=" * 70)

    S0_ranges = [(80, 120)] * 3
    K, T, r = 100, 1.0, 0.05
    sigma = [0.2, 0.25, 0.3]
    weights = [1 / 3] * 3
    n_points = 30

    # Build grid once
    grid, axes = build_pricing_grid_analytic(
        S0_ranges, K, T, r, sigma, weights, n_points
    )
    full_bytes = grid.nbytes

    print(f"\n  Full grid: {grid.shape} = {np.prod(grid.shape):,} entries, "
          f"{full_bytes / 1024:.1f} KB\n")

    print(f"  {'Tolerance':>12s}  {'Max Rank':>8s}  {'TT KB':>8s}  "
          f"{'Ratio':>8s}  {'Error':>10s}  {'SVD ms':>8s}")
    print(f"  {'-' * 12}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 10}  {'-' * 8}")

    for eps in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10]:
        t0 = time.perf_counter()
        cores = tt_svd(grid, eps=eps)
        dt = (time.perf_counter() - t0) * 1000

        ranks = tt_ranks(cores)
        tt_bytes = tt_memory(cores)
        err = tt_error(cores, grid)
        ratio = full_bytes / tt_bytes if tt_bytes > 0 else float("inf")

        print(f"  {eps:12.0e}  {max(ranks):8d}  {tt_bytes / 1024:8.1f}  "
              f"{ratio:8.1f}×  {err:10.2e}  {dt:8.1f}")


def benchmark_evaluation_speed():
    """Compare TT evaluation speed vs direct grid indexing."""
    print("\n" + "=" * 70)
    print("  Benchmark 2: Evaluation Speed — TT vs Direct Grid")
    print("=" * 70)

    for n_assets in [2, 3, 4]:
        S0_ranges = [(80, 120)] * n_assets
        K, T, r = 100, 1.0, 0.05
        sigma = [0.2 + 0.02 * i for i in range(n_assets)]
        weights = [1.0 / n_assets] * n_assets
        n_points = 20

        # Build and compress
        grid, axes = build_pricing_grid_analytic(
            S0_ranges, K, T, r, sigma, weights, n_points
        )
        cores = tt_svd(grid, eps=1e-4)

        # Generate random evaluation points (as indices)
        rng = np.random.default_rng(42)
        n_evals = 50_000
        indices = np.column_stack([rng.integers(0, n_points, n_evals) for _ in range(n_assets)])

        # Direct grid lookup
        t0 = time.perf_counter()
        direct_vals = np.array([grid[tuple(idx)] for idx in indices])
        t_direct = time.perf_counter() - t0

        # TT evaluation
        t0 = time.perf_counter()
        tt_vals = tt_eval_batch(cores, indices)
        t_tt = time.perf_counter() - t0

        speedup = t_direct / t_tt if t_tt > 0 else float("inf")
        max_err = np.max(np.abs(direct_vals - tt_vals))

        print(f"\n  {n_assets}-asset ({n_points}^{n_assets} grid):")
        print(f"    Direct:  {t_direct:.4f}s for {n_evals:,} evals "
              f"({n_evals / t_direct:.0f}/s)")
        print(f"    TT:      {t_tt:.4f}s for {n_evals:,} evals "
              f"({n_evals / t_tt:.0f}/s)")
        print(f"    Speedup: {speedup:.1f}×")
        print(f"    Max error: {max_err:.2e}")


def benchmark_surrogate_e2e():
    """End-to-end surrogate benchmark including interpolation."""
    print("\n" + "=" * 70)
    print("  Benchmark 3: End-to-End Surrogate (with interpolation)")
    print("=" * 70)

    for n_assets in [2, 3]:
        surr = TTSurrogate.from_basket_analytic(
            S0_ranges=[(80, 120)] * n_assets,
            K=100, T=1.0, r=0.05,
            sigma=[0.2 + 0.02 * i for i in range(n_assets)],
            weights=[1.0 / n_assets] * n_assets,
            n_points=30,
            eps=1e-4,
        )

        # Evaluate batch of random spots
        rng = np.random.default_rng(42)
        n_evals = 10_000
        spots = rng.uniform(80, 120, size=(n_evals, n_assets))

        t0 = time.perf_counter()
        prices = surr.evaluate(spots)
        t_eval = time.perf_counter() - t0

        # Single Greeks evaluation
        spot_atm = [100.0] * n_assets
        t0 = time.perf_counter()
        n_greek_calls = 100
        for _ in range(n_greek_calls):
            g = surr.greeks(spot_atm)
        t_greeks = (time.perf_counter() - t0) / n_greek_calls

        s = surr.summary()
        print(f"\n  {n_assets}-asset surrogate (30 pts/axis, eps=1e-4):")
        print(f"    Build:         {s['build_time_s']:.3f}s")
        print(f"    Compress:      {s['compress_time_s']:.3f}s")
        print(f"    Max rank:      {s['max_rank']}")
        print(f"    Compression:   {s.get('compression_ratio', 'N/A'):.1f}×")
        print(f"    {n_evals:,} evals:   {t_eval:.4f}s ({n_evals / t_eval:.0f}/s)")
        print(f"    Greeks (Δ+Γ):  {t_greeks * 1000:.2f}ms per call")


def benchmark_memory_scaling():
    """Show how memory scales with number of assets."""
    print("\n" + "=" * 70)
    print("  Benchmark 4: Memory Scaling vs Number of Assets")
    print("=" * 70)

    n_points = 15
    print(f"\n  Grid: {n_points} pts/axis, eps=1e-3\n")
    print(f"  {'Assets':>6s}  {'Full Entries':>12s}  {'Full MB':>8s}  "
          f"{'TT KB':>8s}  {'Ratio':>8s}  {'Max Rank':>8s}")
    print(f"  {'-' * 6}  {'-' * 12}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}")

    for n_assets in [2, 3, 4, 5]:
        full_entries = n_points ** n_assets
        full_mb = full_entries * 8 / (1024 ** 2)

        surr = TTSurrogate.from_basket_analytic(
            S0_ranges=[(80, 120)] * n_assets,
            K=100, T=1.0, r=0.05,
            sigma=[0.2 + 0.02 * i for i in range(n_assets)],
            weights=[1.0 / n_assets] * n_assets,
            n_points=n_points,
            eps=1e-3,
        )
        s = surr.summary()
        tt_kb = s["tt_memory_KB"]
        ratio = full_mb * 1024 / tt_kb if tt_kb > 0 else float("inf")

        print(f"  {n_assets:6d}  {full_entries:12,}  {full_mb:8.2f}  "
              f"{tt_kb:8.1f}  {ratio:8.0f}×  {s['max_rank']:8d}")


def main():
    print("\n" + "=" * 70)
    print("  TensorQuantLib — Benchmark Suite")
    print("=" * 70)

    benchmark_compression_vs_tolerance()
    benchmark_evaluation_speed()
    benchmark_surrogate_e2e()
    benchmark_memory_scaling()

    print("\n" + "=" * 70)
    print("  All benchmarks complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
