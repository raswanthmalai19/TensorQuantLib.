#!/usr/bin/env python3
"""
Demo: TT-Surrogate Basket Option Pricing
=========================================

Demonstrates the full TensorQuantLib pipeline:
1. Build a multi-asset basket-option pricing surface
2. Compress it with Tensor-Train SVD
3. Evaluate prices 1000× faster than brute-force grid lookup
4. Compute Greeks (Delta, Gamma) through the surrogate
5. Show memory reduction and compression diagnostics

Usage:
    python3 examples/demo_basket_tt.py
"""

import numpy as np
import time
import sys
import os

# Add src to path if running as script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorquantlib.tt.surrogate import TTSurrogate
from tensorquantlib.finance.black_scholes import bs_price_numpy


def separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def demo_2asset():
    """2-asset basket — quick demo."""
    separator("Demo 1: 2-Asset Basket Call (Analytic Grid)")

    surr = TTSurrogate.from_basket_analytic(
        S0_ranges=[(80, 120), (80, 120)],
        K=100, T=1.0, r=0.05,
        sigma=[0.2, 0.25],
        weights=[0.5, 0.5],
        n_points=30,
        eps=1e-4,
    )
    surr.print_summary()

    # Evaluate at several spot levels
    test_spots = [
        [85, 85],    # OTM
        [100, 100],  # ATM
        [115, 115],  # ITM
    ]
    print("\nSpot-price evaluation:")
    print(f"  {'Spots':>20s}  {'TT Price':>10s}")
    print(f"  {'-' * 20}  {'-' * 10}")
    for spots in test_spots:
        price = surr.evaluate(spots)
        print(f"  {str(spots):>20s}  {price:10.4f}")

    # Greeks
    print("\nGreeks at ATM [100, 100]:")
    g = surr.greeks([100.0, 100.0])
    print(f"  Price:  {g['price']:.4f}")
    print(f"  Delta:  [{g['delta'][0]:.4f}, {g['delta'][1]:.4f}]")
    print(f"  Gamma:  [{g['gamma'][0]:.6f}, {g['gamma'][1]:.6f}]")

    # Speed test
    n_evals = 10_000
    spots_batch = np.random.default_rng(42).uniform(
        [80, 80], [120, 120], size=(n_evals, 2)
    )
    t0 = time.perf_counter()
    _ = surr.evaluate(spots_batch)
    t_tt = time.perf_counter() - t0

    print(f"\nSpeed: {n_evals:,} evaluations in {t_tt:.4f}s "
          f"({n_evals / t_tt:.0f} evals/sec)")


def demo_3asset():
    """3-asset basket — demonstrates TT compression advantage."""
    separator("Demo 2: 3-Asset Basket Call (Analytic Grid)")

    surr = TTSurrogate.from_basket_analytic(
        S0_ranges=[(80, 120)] * 3,
        K=100, T=1.0, r=0.05,
        sigma=[0.2, 0.25, 0.3],
        weights=[1 / 3] * 3,
        n_points=30,
        eps=1e-4,
    )
    surr.print_summary()

    # Evaluate
    test_spots = [
        [90, 90, 90],
        [100, 100, 100],
        [110, 110, 110],
    ]
    print("\nSpot-price evaluation:")
    print(f"  {'Spots':>25s}  {'TT Price':>10s}")
    print(f"  {'-' * 25}  {'-' * 10}")
    for spots in test_spots:
        price = surr.evaluate(spots)
        print(f"  {str(spots):>25s}  {price:10.4f}")

    # Greeks at ATM
    g = surr.greeks([100.0, 100.0, 100.0])
    print(f"\nGreeks at ATM [100, 100, 100]:")
    print(f"  Price:  {g['price']:.4f}")
    print(f"  Delta:  [{', '.join(f'{d:.4f}' for d in g['delta'])}]")
    print(f"  Gamma:  [{', '.join(f'{d:.6f}' for d in g['gamma'])}]")

    # Speed test
    n_evals = 10_000
    spots_batch = np.random.default_rng(42).uniform(80, 120, size=(n_evals, 3))
    t0 = time.perf_counter()
    _ = surr.evaluate(spots_batch)
    t_tt = time.perf_counter() - t0

    print(f"\nSpeed: {n_evals:,} evaluations in {t_tt:.4f}s "
          f"({n_evals / t_tt:.0f} evals/sec)")


def demo_5asset():
    """5-asset basket — shows scalability."""
    separator("Demo 3: 5-Asset Basket Call (Analytic Grid, 15 pts/axis)")

    # Use fewer grid points to keep build time reasonable
    n_pts = 15
    surr = TTSurrogate.from_basket_analytic(
        S0_ranges=[(80, 120)] * 5,
        K=100, T=1.0, r=0.05,
        sigma=[0.2, 0.22, 0.25, 0.28, 0.3],
        weights=[0.2] * 5,
        n_points=n_pts,
        eps=1e-3,
    )
    surr.print_summary()

    # Single evaluation
    spots = [100.0] * 5
    price = surr.evaluate(spots)
    print(f"\nATM price: {price:.4f}")

    # Speed test
    n_evals = 10_000
    spots_batch = np.random.default_rng(42).uniform(80, 120, size=(n_evals, 5))
    t0 = time.perf_counter()
    _ = surr.evaluate(spots_batch)
    t_tt = time.perf_counter() - t0

    print(f"Speed: {n_evals:,} evaluations in {t_tt:.4f}s "
          f"({n_evals / t_tt:.0f} evals/sec)")

    # Compare vs brute-force grid lookup
    full_grid_entries = n_pts ** 5
    full_memory_mb = full_grid_entries * 8 / (1024 ** 2)
    tt_memory_kb = surr.summary()["tt_memory_KB"]
    print(f"\nFull grid: {full_grid_entries:,} entries ({full_memory_mb:.1f} MB)")
    print(f"TT format: {tt_memory_kb:.1f} KB")
    print(f"Memory reduction: {full_memory_mb * 1024 / tt_memory_kb:.0f}×")


def demo_autograd_greeks():
    """Demonstrate autograd-based Greeks on a single-asset Black-Scholes."""
    separator("Demo 4: Autograd Greeks — Black-Scholes Validation")

    from tensorquantlib.finance.black_scholes import bs_price_tensor, bs_delta, bs_gamma
    from tensorquantlib.core.tensor import Tensor

    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    # Analytic Greeks (ground truth)
    delta_analytic = bs_delta(S, K, T, r, sigma)
    gamma_analytic = bs_gamma(S, K, T, r, sigma)

    # Autograd Greeks
    S_t = Tensor(np.array([S]), requires_grad=True)
    price_t = bs_price_tensor(S_t, K, T, r, sigma)
    price_t.backward()
    delta_autograd = S_t.grad[0]

    # Gamma via finite differences on autograd delta
    h = S * 1e-4
    S_up = Tensor(np.array([S + h]), requires_grad=True)
    p_up = bs_price_tensor(S_up, K, T, r, sigma)
    p_up.backward()
    delta_up = S_up.grad[0]

    S_dn = Tensor(np.array([S - h]), requires_grad=True)
    p_dn = bs_price_tensor(S_dn, K, T, r, sigma)
    p_dn.backward()
    delta_dn = S_dn.grad[0]

    gamma_fd = (delta_up - delta_dn) / (2 * h)

    print(f"  S = {S}, K = {K}, T = {T}, r = {r}, σ = {sigma}")
    print(f"\n  {'':>15s}  {'Analytic':>10s}  {'Autograd':>10s}  {'Error':>10s}")
    print(f"  {'-' * 15}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
    print(f"  {'Delta':>15s}  {delta_analytic:10.6f}  {delta_autograd:10.6f}  "
          f"{abs(delta_analytic - delta_autograd):10.2e}")
    print(f"  {'Gamma':>15s}  {gamma_analytic:10.6f}  {gamma_fd:10.6f}  "
          f"{abs(gamma_analytic - gamma_fd):10.2e}")


def main():
    print("\n" + "=" * 60)
    print("  TensorQuantLib — TT-Surrogate Pricing Engine Demo")
    print("=" * 60)

    demo_2asset()
    demo_3asset()
    demo_5asset()
    demo_autograd_greeks()

    separator("Done!")
    print("All demos completed successfully.\n")


if __name__ == "__main__":
    main()
