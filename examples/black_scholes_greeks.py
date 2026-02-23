#!/usr/bin/env python3
"""
Example: Black-Scholes Pricing & Greeks
========================================

Demonstrates:
- Analytic BS pricing (NumPy)
- Autograd-based Greeks (Tensor)
- Comparing analytic vs. autograd Greeks
- Vectorized Greeks across spot prices

Usage:
    python3 examples/black_scholes_greeks.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from tensorquantlib.finance.black_scholes import (
    bs_price_numpy, bs_price_tensor,
    bs_delta, bs_gamma, bs_vega, bs_theta, bs_rho,
)
from tensorquantlib.finance.greeks import compute_greeks, compute_greeks_vectorized


def main() -> None:
    print("=" * 60)
    print("  TensorQuantLib — Black-Scholes Greeks")
    print("=" * 60)

    # Option parameters
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    # ── 1. Analytic price ──────────────────────────────
    print("\n1. Analytic Black-Scholes price")
    call_price = bs_price_numpy(S, K, T, r, sigma)
    put_price = bs_price_numpy(S, K, T, r, sigma, option_type="put")
    print(f"   Call: {float(call_price):.4f}")
    print(f"   Put:  {float(put_price):.4f}")
    print(f"   Put-Call parity check: C - P = {float(call_price - put_price):.4f}"
          f"  (expected {S - K * np.exp(-r * T):.4f})")

    # ── 2. Analytic Greeks ──────────────────────────────
    print("\n2. Analytic Greeks (call)")
    print(f"   Delta: {float(bs_delta(S, K, T, r, sigma)):+.6f}")
    print(f"   Gamma: {float(bs_gamma(S, K, T, r, sigma)):+.6f}")
    print(f"   Vega:  {float(bs_vega(S, K, T, r, sigma)):+.6f}")
    print(f"   Theta: {float(bs_theta(S, K, T, r, sigma)):+.6f}")
    print(f"   Rho:   {float(bs_rho(S, K, T, r, sigma)):+.6f}")

    # ── 3. Autograd Greeks ──────────────────────────────
    print("\n3. Autograd Greeks (via Tensor backprop)")
    greeks = compute_greeks(bs_price_tensor, S, K, T, r, sigma)
    print(f"   Price: {greeks['price']:.4f}")
    print(f"   Delta: {greeks['delta']:+.6f}")
    print(f"   Vega:  {greeks['vega']:+.6f}")
    print(f"   Gamma: {greeks['gamma']:+.6f}")

    # ── 4. Comparison ──────────────────────────────────
    print("\n4. Analytic vs. Autograd comparison")
    delta_analytic = float(bs_delta(S, K, T, r, sigma))
    delta_autograd = greeks["delta"]
    print(f"   Delta analytic:  {delta_analytic:+.6f}")
    print(f"   Delta autograd:  {delta_autograd:+.6f}")
    print(f"   Difference:      {abs(delta_analytic - delta_autograd):.2e}")

    gamma_analytic = float(bs_gamma(S, K, T, r, sigma))
    gamma_autograd = greeks["gamma"]
    print(f"   Gamma analytic:  {gamma_analytic:+.6f}")
    print(f"   Gamma autograd:  {gamma_autograd:+.6f}")
    print(f"   Difference:      {abs(gamma_analytic - gamma_autograd):.2e}")

    # ── 5. Vectorized Greeks ────────────────────────────
    print("\n5. Vectorized Greeks across spot prices")
    S_array = np.linspace(80, 120, 9)
    vec = compute_greeks_vectorized(bs_price_tensor, S_array, K, T, r, sigma)
    print(f"   {'Spot':>6s}  {'Price':>8s}  {'Delta':>8s}")
    print(f"   {'─' * 6}  {'─' * 8}  {'─' * 8}")
    for i, s in enumerate(S_array):
        print(f"   {s:6.1f}  {vec['price'][i]:8.4f}  {vec['delta'][i]:+8.4f}")

    print("\n✓ All Black-Scholes examples completed successfully.")


if __name__ == "__main__":
    main()
