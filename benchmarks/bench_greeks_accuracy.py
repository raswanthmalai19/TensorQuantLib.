"""
Benchmark 1 — Greeks Accuracy & Speed
========================================
Tests the accuracy of autograd-computed Greeks against analytic Black-Scholes
formulas, and measures throughput on a 10,000-option surface grid.

Grid: 100 spot prices × 100 volatilities = 10,000 options

Machine target: Apple M1 with NumPy linked to Apple Accelerate (BLAS/LAPACK).
All vectorised NumPy ops (norm.cdf, norm.pdf, matmul, erf) dispatch through
Accelerate automatically — no explicit setup required.

Run from the repo root:
    python -m benchmarks.bench_greeks_accuracy
or
    python benchmarks/bench_greeks_accuracy.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "src")

from tensorquantlib.finance.black_scholes import (
    bs_delta,
    bs_gamma,
    bs_price_numpy,
    bs_price_tensor,
    bs_vega,
)
from tensorquantlib.finance.greeks import compute_greeks, compute_greeks_vectorized
from tensorquantlib.core.second_order import second_order_greeks

# ── Fixed option parameters ──────────────────────────────────────────────────
K = 100.0   # ATM strike
T = 1.0     # 1 year
r = 0.05    # risk-free rate
q = 0.0     # no dividends

# ── Grid sizes ─────────────────────────────────────────────────────────────
N_S     = 100   # spot points per vol slice
N_SIGMA = 100   # vol levels
N_SO    = 200   # second-order Greeks batch
N_FULL  =  50   # full 5-greek compute points

S_grid     = np.linspace(60.0, 140.0, N_S)
sigma_grid = np.linspace(0.05, 0.60, N_SIGMA)

DIV = "─" * 68


def fmt_ms(t_sec: float) -> str:
    return f"{t_sec * 1_000:.1f} ms"


def fmt_us(t_sec: float) -> str:
    return f"{t_sec * 1_000_000:.1f} µs"


def section(n: int, title: str) -> None:
    print(f"\n[{n}] {title}")


# ── Check Accelerate is active ───────────────────────────────────────────────
try:
    np_config = np.__config__.blas_opt_info   # type: ignore[attr-defined]
    np_blas = "Accelerate" if "accelerate" in str(np_config).lower() else "System"
except AttributeError:
    np_blas = "unknown (NumPy ≥ 2.0 API diff)"

print(DIV)
print("BENCHMARK 1 — Greeks Accuracy & Speed")
print(f"  {N_S}×{N_SIGMA} option grid  |  K={K}  T={T}  r={r}")
print(f"  NumPy BLAS: {np_blas}")
print(DIV)

# ============================================================================
# Section 1 — Analytic Baseline
# Pure vectorised NumPy: dispatch through Accelerate BLAS/Veclib
# ============================================================================
section(1, "Analytic BS Greeks — vectorised NumPy (Apple Accelerate)")

t0 = time.perf_counter()
an_price = np.zeros((N_S, N_SIGMA))
an_delta = np.zeros((N_S, N_SIGMA))
an_gamma = np.zeros((N_S, N_SIGMA))
an_vega  = np.zeros((N_S, N_SIGMA))

for j, sig in enumerate(sigma_grid):
    an_price[:, j] = bs_price_numpy(S_grid, K, T, r, sig)
    an_delta[:, j] = bs_delta(S_grid, K, T, r, sig)
    an_gamma[:, j] = bs_gamma(S_grid, K, T, r, sig)
    an_vega[:, j]  = bs_vega(S_grid, K, T, r, sig)

t_analytic = time.perf_counter() - t0
n_options = N_S * N_SIGMA

print(f"  Options priced         : {n_options:,}")
print(f"  Wall time              : {fmt_ms(t_analytic)}")
print(f"  Throughput (price+3Δ)  : {4 * n_options / t_analytic:,.0f} evaluations/sec")
print(f"  Price  range : [{an_price.min():.3f}, {an_price.max():.3f}]")
print(f"  Delta  range : [{an_delta.min():.4f}, {an_delta.max():.4f}]")
print(f"  Gamma  range : [{an_gamma.min():.6f}, {an_gamma.max():.6f}]")
print(f"  Vega   range : [{an_vega.min():.4f}, {an_vega.max():.4f}]")

# ============================================================================
# Section 2 — Autograd Delta Accuracy
# compute_greeks_vectorized builds the Tensor graph and calls .backward()
# verifying the autograd engine matches analytic formulas element-wise.
# ============================================================================
section(2, "Autograd Delta accuracy — compute_greeks_vectorized")

t0 = time.perf_counter()
ag_delta = np.zeros((N_S, N_SIGMA))

for j, sig in enumerate(sigma_grid):
    g = compute_greeks_vectorized(bs_price_tensor, S_grid, K, T, r, sig, q, "call")
    ag_delta[:, j] = g["delta"]

t_autograd = time.perf_counter() - t0

mae_delta = float(np.mean(np.abs(ag_delta - an_delta)))
max_delta = float(np.max(np.abs(ag_delta - an_delta)))
rel_err   = float(np.mean(np.abs(ag_delta - an_delta) / np.maximum(np.abs(an_delta), 1e-12)))

print(f"  Wall time              : {fmt_ms(t_autograd)}")
print(f"  Throughput (delta)     : {n_options / t_autograd:,.0f} Greeks/sec")
print(f"  Autograd vs analytic delta:")
print(f"    MAE                  : {mae_delta:.2e}")
print(f"    Max absolute error   : {max_delta:.2e}")
print(f"    Mean relative error  : {rel_err:.2e}")
print(f"  Speed ratio (autograd / analytic) : {t_autograd / t_analytic:.1f}×  "
      f"[autograd builds compute graph]")

pass_fail = "PASS ✓" if mae_delta < 1e-6 else "FAIL ✗"
print(f"  Accuracy test (MAE < 1e-6): {pass_fail}")

# ============================================================================
# Section 3 — Second-Order Greeks Accuracy
# second_order_greeks uses 5 forward evaluations (hybrid semi-analytic).
# Gamma is compared against the analytic formula; Vanna and Volga are shown.
# ============================================================================
section(3, f"Second-Order Greeks — Gamma/Vanna/Volga ({N_SO} random points)")

rng = np.random.default_rng(42)
so_S   = rng.uniform(70.0, 130.0, N_SO)
so_sig = rng.uniform(0.10, 0.50,  N_SO)

t0 = time.perf_counter()
so_gamma = np.empty(N_SO)
so_vanna = np.empty(N_SO)
so_volga = np.empty(N_SO)

for i in range(N_SO):
    so = second_order_greeks(bs_price_tensor, so_S[i], K, T, r, so_sig[i], q, "call")
    so_gamma[i] = so["gamma"]
    so_vanna[i] = so["vanna"]
    so_volga[i] = so["volga"]

t_so = time.perf_counter() - t0

# Gamma vs analytic
ref_gamma = np.array([bs_gamma(so_S[i], K, T, r, so_sig[i]) for i in range(N_SO)])
mae_gamma = float(np.mean(np.abs(so_gamma - ref_gamma)))
max_gamma = float(np.max(np.abs(so_gamma - ref_gamma)))

print(f"  Wall time              : {fmt_ms(t_so)}  ({fmt_us(t_so / N_SO)} / point)")
print(f"  Throughput             : {N_SO / t_so:,.0f} points/sec")
print(f"  Gamma vs analytic:")
print(f"    MAE                  : {mae_gamma:.2e}")
print(f"    Max absolute error   : {max_gamma:.2e}")
print(f"  Vanna range : [{so_vanna.min():.4f}, {so_vanna.max():.4f}]")
print(f"  Volga range : [{so_volga.min():.4f}, {so_volga.max():.4f}]")

gpass = "PASS ✓" if mae_gamma < 2e-3 else "FAIL ✗"
print(f"  Accuracy test (MAE < 2e-3): {gpass}  [finite-diff hybrid, ~0.3% of gamma range]")

# ============================================================================
# Section 4 — Full 5-Greek Compute (Delta/Vega/Gamma/Vanna/Volga via compute_greeks)
# compute_greeks = first-order autograd + second_order_greeks in one call.
# ============================================================================
section(4, f"Full 5-Greek compute — Delta+Vega+Gamma+Vanna+Volga ({N_FULL} points)")

full_S   = rng.uniform(80.0, 120.0, N_FULL)
full_sig = rng.uniform(0.10, 0.40,  N_FULL)

# Warm-up to exclude JIT/import overhead
_ = compute_greeks(bs_price_tensor, full_S[0], K, T, r, full_sig[0])

t0 = time.perf_counter()
results_full = [
    compute_greeks(bs_price_tensor, float(full_S[i]), K, T, r, float(full_sig[i]), q, "call")
    for i in range(N_FULL)
]
t_full = time.perf_counter() - t0

deltas_full = np.array([g["delta"] for g in results_full])
gammas_full = np.array([g["gamma"] for g in results_full])

# Cross-check vs analytic
ref_delta_full = bs_delta(full_S, K, T, r, full_sig[0])  # approximate (fixed sig)
mae_delta_full = float(np.mean(np.abs(
    deltas_full - np.array([bs_delta(float(full_S[i]), K, T, r, float(full_sig[i])) for i in range(N_FULL)])
)))
mae_gamma_full = float(np.mean(np.abs(
    gammas_full - np.array([bs_gamma(float(full_S[i]), K, T, r, float(full_sig[i])) for i in range(N_FULL)])
)))

print(f"  Wall time              : {fmt_ms(t_full)}")
print(f"  Per point              : {fmt_us(t_full / N_FULL)}")
print(f"  Throughput             : {N_FULL / t_full:,.1f} full-greek-sets/sec")
print(f"  Delta MAE vs analytic  : {mae_delta_full:.2e}")
print(f"  Gamma MAE vs analytic  : {mae_gamma_full:.2e}")

# ============================================================================
# Summary Table
# ============================================================================
print(f"\n{DIV}")
print("BENCHMARK 1 SUMMARY")
print(f"{'Metric':<40} {'Value':>26}")
print(f"{'─'*40} {'─'*26}")
print(f"{'Options in grid':<40} {n_options:>26,}")
print(f"{'Analytic BS wall time':<40} {fmt_ms(t_analytic):>26}")
print(f"{'Autograd delta wall time':<40} {fmt_ms(t_autograd):>26}")
print(f"{'Second-order wall time (200 pts)':<40} {fmt_ms(t_so):>26}")
print(f"{'Full-5-greek wall time (50 pts)':<40} {fmt_ms(t_full):>26}")
print(f"{'Autograd vs analytic: delta MAE':<40} {mae_delta:>26.2e}")
print(f"{'Second-order: gamma MAE':<40} {mae_gamma:>26.2e}")
print(f"{'Autograd overhead vs analytic':<40} {t_autograd / t_analytic:>25.1f}×")
print(DIV)
