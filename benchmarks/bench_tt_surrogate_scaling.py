"""
Benchmark 2 — TT-Surrogate Scalability
=========================================
Tests the Tensor-Train surrogate pricer scaling from 2 to 5 assets.

For every asset count d ∈ {2, 3, 4, 5} it:
  1. Builds a pricing grid via analytic moment-matching (log-normal Gentle 1993).
  2. Compresses it with TT-SVD (calls LAPACK dgesdd internally → Apple Accelerate).
  3. Evaluates 300 random test points and compares against the analytic reference.
  4. Reports: grid size, TT size, compression ratio, ranks, build/compress/eval times.

For d=5 an additional TT-Cross build is run — this never forms the full 3.2 M-entry
grid and shows how the library handles high-dimensional problems.

Why this stresses the M1:
  - TT-SVD calls np.linalg.svd in a loop → Accelerate LAPACK (dgesdd).
  - Building the analytic grid calls np.meshgrid + vectorised BS formulas on ~25 M
    float64 values for d=5 → all dispatched through Accelerate BLAS/Veclib.
  - evaluate() uses np.searchsorted + vectorised indexing on large arrays.

Run from the repo root:
    python benchmarks/bench_tt_surrogate_scaling.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "src")

from tensorquantlib.tt.surrogate import TTSurrogate
from tensorquantlib.tt.ops import tt_ranks, tt_memory
from tensorquantlib.finance.basket import build_pricing_grid_analytic

# ── Common option parameters ─────────────────────────────────────────────────
K       = 100.0
T       = 1.0
r       = 0.05
EPS     = 1e-3   # TT-SVD tolerance
N_PTS   = 20     # grid points per axis
N_TEST  = 300    # random test points per d
S_LO    = 80.0   # spot lower bound per asset
S_HI    = 120.0  # spot upper bound per asset
RNG_SEED = 42

DIV  = "─" * 72
DIV2 = "═" * 72


def fmt_ms(t: float) -> str:
    return f"{t * 1_000:.1f} ms"


def fmt_mb(n_bytes: int) -> str:
    return f"{n_bytes / 1_048_576:.2f} MB"


print(DIV2)
print("BENCHMARK 2 — TT-Surrogate Scalability  (d = 2 … 5 assets)")
print(f"  Grid: {N_PTS} pts/axis  |  K={K}  T={T}  r={r}  eps={EPS}")
print(DIV2)

rng = np.random.default_rng(RNG_SEED)

# Header for the summary table (printed at the end)
header_cols = ["d", "grid_pts", "grid_MB", "TT_kB", "ratio", "max_rank",
               "build_ms", "compress_ms", "eval_ms/300", "MAE%"]
rows: list[list[str]] = []

# ── Per-dimension loop ───────────────────────────────────────────────────────
for d in range(2, 6):
    sigma   = [0.20] * d
    weights = [1.0 / d] * d
    ranges  = [(S_LO, S_HI)] * d

    print(f"\n{'─'*40}")
    print(f"  d = {d} assets  (grid shape: {N_PTS}^{d} = {N_PTS**d:,} points)")
    print(f"{'─'*40}")

    # ── Step 1: Build TT-Surrogate ──────────────────────────────────────────
    surr = TTSurrogate.from_basket_analytic(
        S0_ranges=ranges,
        K=K, T=T, r=r,
        sigma=sigma,
        weights=weights,
        n_points=N_PTS,
        eps=EPS,
    )

    ranks     = tt_ranks(surr.cores)
    mem_bytes = tt_memory(surr.cores)
    grid_bytes = N_PTS ** d * 8   # float64

    print(f"  Build time (grid)   : {fmt_ms(surr.build_time)}")
    print(f"  Compress time (SVD) : {fmt_ms(surr.compress_time)}")
    print(f"  Grid memory         : {fmt_mb(grid_bytes)}")
    print(f"  TT memory           : {mem_bytes / 1024:.1f} kB")
    print(f"  Compression ratio   : {grid_bytes / mem_bytes:.1f}×")
    print(f"  TT ranks            : {ranks}")

    # ── Step 2: Accuracy at N_TEST random test points ───────────────────────
    test_spots = rng.uniform(S_LO + 2, S_HI - 2, size=(N_TEST, d))

    # Analytic reference (same moment-matching formula used in grid build)
    # Build a tiny per-point reference from build_pricing_grid_analytic at size 2
    # — faster: just evaluate a new tiny grid around each test point.
    # FASTER shortcut: call build_pricing_grid_analytic at n_points=2 and
    #   evaluate corner prices … but that's complicated.
    # Instead: build a 1-point "grid" by evaluating the moment-matching formula
    # directly.  We replicate the logic from basket.py for speed.
    from scipy.stats import norm as _snorm

    def _analytic_basket(spots: np.ndarray) -> float:
        """Moment-matching analytic basket call price."""
        spots = np.asarray(spots, dtype=float)
        _sigma = np.asarray(sigma, dtype=float)
        _weights = np.asarray(weights, dtype=float)
        F  = spots * np.exp(r * T)
        E_B  = float(np.dot(_weights, F))
        # second moment (diagonal dominates with zero correlation assumption)
        E_B2 = float(
            np.sum(_weights[:, None] * _weights[None, :] *
                   F[:, None] * F[None, :] *
                   np.where(np.eye(d, dtype=bool),
                            np.exp(_sigma[None, :] ** 2 * T),
                            1.0))
        )
        sig2 = np.log(max(E_B2, 1e-300)) - 2.0 * np.log(max(E_B, 1e-300))
        sig_X = np.sqrt(max(sig2, 1e-24))
        sqrtT = np.sqrt(T)
        d1 = (np.log(max(E_B, 1e-300) / K) + 0.5 * sig2 * T) / (sig_X * sqrtT)
        d2 = d1 - sig_X * sqrtT
        disc = np.exp(-r * T)
        return max(disc * (E_B * _snorm.cdf(d1) - K * _snorm.cdf(d2)), 0.0)

    t_ref = time.perf_counter()
    ref_prices = np.array([_analytic_basket(test_spots[i]) for i in range(N_TEST)])
    t_ref = time.perf_counter() - t_ref

    t_eval = time.perf_counter()
    surr_prices = surr.evaluate(test_spots)
    t_eval = time.perf_counter() - t_eval

    # Relative MAE (only on non-trivial prices to avoid division issues)
    mask = ref_prices > 0.05
    if mask.sum() > 0:
        rel_err = np.mean(np.abs(surr_prices[mask] - ref_prices[mask]) / ref_prices[mask]) * 100
    else:
        rel_err = float("nan")
    abs_err = float(np.mean(np.abs(surr_prices - ref_prices)))

    print(f"\n  Accuracy vs analytic ref ({N_TEST} random points):")
    print(f"    Evaluate time (surr) : {fmt_ms(t_eval)}")
    print(f"    Evaluate time (ref)  : {fmt_ms(t_ref)}")
    print(f"    MAE (absolute)       : {abs_err:.4f}")
    print(f"    MAE (relative %)     : {rel_err:.2f}%")
    print(f"    Surrogate price range: [{surr_prices.min():.3f}, {surr_prices.max():.3f}]")
    print(f"    Analytic  price range: [{ref_prices.min():.3f}, {ref_prices.max():.3f}]")

    rows.append([
        str(d),
        f"{N_PTS**d:,}",
        f"{fmt_mb(grid_bytes)}",
        f"{mem_bytes / 1024:.1f}",
        f"{grid_bytes / mem_bytes:.1f}×",
        str(max(ranks)),
        f"{surr.build_time * 1000:.1f}",
        f"{surr.compress_time * 1000:.1f}",
        f"{t_eval * 1000:.1f}",
        f"{rel_err:.2f}%",
    ])

# ── TT-Cross on d=5 (no full grid) ───────────────────────────────────────────
print(f"\n{'═'*40}")
print("  TT-Cross (d=5, no full grid) via from_function")
print(f"{'═'*40}")

d5 = 5
sigma5   = [0.20] * d5
weights5 = [1.0 / d5] * d5
axes5    = [np.linspace(S_LO, S_HI, N_PTS)] * d5

from scipy.stats import norm as _snorm5

def _basket_pricer_fn(*indices):
    """Basket pricer indexed on the grid axes (called by TT-Cross)."""
    spots = np.array([axes5[k][i] for k, i in enumerate(indices)], dtype=float)
    _sigma = np.asarray(sigma5, dtype=float)
    _w     = np.asarray(weights5, dtype=float)
    F      = spots * np.exp(r * T)
    E_B   = float(np.dot(_w, F))
    E_B2  = float(
        np.sum(_w[:, None] * _w[None, :] *
               F[:, None] * F[None, :] *
               np.where(np.eye(d5, dtype=bool),
                        np.exp(_sigma[None, :] ** 2 * T), 1.0))
    )
    sig2  = np.log(max(E_B2, 1e-300)) - 2.0 * np.log(max(E_B, 1e-300))
    sig_X = np.sqrt(max(sig2, 1e-24))
    sqrtT = np.sqrt(T)
    d1    = (np.log(max(E_B, 1e-300) / K) + 0.5 * sig2 * T) / (sig_X * sqrtT)
    d2    = d1 - sig_X * sqrtT
    disc  = np.exp(-r * T)
    return max(disc * (E_B * _snorm5.cdf(d1) - K * _snorm5.cdf(d2)), 0.0)

t0_cross = time.perf_counter()
surr_cross = TTSurrogate.from_function(
    fn=_basket_pricer_fn,
    axes=axes5,
    eps=EPS,
    max_rank=12,
    n_sweeps=5,
    seed=42,
)
t_cross = time.perf_counter() - t0_cross

# Full grid size for comparison
full_grid_bytes = N_PTS ** d5 * 8
ranks_cross = tt_ranks(surr_cross.cores)
mem_cross   = tt_memory(surr_cross.cores)

# Quick accuracy check at 100 test pts
test5 = rng.uniform(S_LO + 2, S_HI - 2, size=(100, d5))
ref5  = np.array([
    _basket_pricer_fn(*[
        int(np.searchsorted(axes5[k], test5[i, k])) for k in range(d5)
    ])
    for i in range(100)
])
surr5 = surr_cross.evaluate(test5)
mask5 = ref5 > 0.05
rel5  = float(np.mean(np.abs(surr5[mask5] - ref5[mask5]) / ref5[mask5]) * 100) if mask5.sum() > 0 else float("nan")

print(f"  TT-Cross build time      : {fmt_ms(t_cross)}")
print(f"  TT-Cross ranks           : {ranks_cross}")
print(f"  TT-Cross memory          : {mem_cross / 1024:.1f} kB")
print(f"  Full-grid memory (never  ")
print(f"    allocated)             : {fmt_mb(full_grid_bytes)}  (NOT built)")
print(f"  Accuracy vs ref (100pts) : {rel5:.2f}% rel MAE")

# ── Summary Table ─────────────────────────────────────────────────────────────
print(f"\n{DIV}")
print("BENCHMARK 2 SUMMARY — TT-Surrogate Scaling")
col_w = [4, 10, 9, 9, 8, 10, 10, 13, 13, 8]
header = [h.ljust(w) for h, w in zip(header_cols, col_w)]
print("  " + " | ".join(header))
print("  " + "-+-".join("-" * w for w in col_w))
for row in rows:
    print("  " + " | ".join(v.ljust(w) for v, w in zip(row, col_w)))

print(f"\n  TT-Cross d=5: build {fmt_ms(t_cross)} | ranks {ranks_cross} | "
      f"memory {mem_cross / 1024:.0f} kB | rel MAE {rel5:.2f}%")

print(f"\n  KEY INSIGHT:")
print(f"  Full 5-asset grid = {N_PTS**5:,} pts × 8 B = {fmt_mb(N_PTS**5 * 8)}")
print(f"  TT-SVD d=5 memory = {tt_memory(surr.cores) / 1024:.0f} kB  ({N_PTS**5 * 8 / tt_memory(surr.cores):.0f}× smaller)")
print(f"  TT-Cross never builds the full grid (only samples O(d·r²·n) points)")
print(DIV)
