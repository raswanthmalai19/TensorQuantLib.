#!/usr/bin/env python3
"""
Example: Autograd Basics
========================

Demonstrates TensorQuantLib's reverse-mode autodiff engine:
- Creating Tensors with gradient tracking
- Forward computation
- Backward pass and gradient extraction
- Gradient checking with numerical differences

Usage:
    python3 examples/autograd_intro.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from tensorquantlib.core.tensor import Tensor
from tensorquantlib.utils.validation import check_grad


def main() -> None:
    print("=" * 50)
    print("  TensorQuantLib — Autograd Introduction")
    print("=" * 50)

    # ── 1. Scalar computation ──────────────────────────
    print("\n1. Scalar autodiff: f(x) = x^2 + 3x + 1")
    x = Tensor(np.array(2.0), requires_grad=True)
    f = x ** 2 + x * 3 + 1  # f(2)=11, df/dx=2x+3=7
    f.backward()
    print(f"   x = {x.data.item():.1f}")
    print(f"   f(x) = {f.data.item():.1f}")
    print(f"   df/dx = {x.grad.item():.1f}  (expected 7.0)")

    # ── 2. Vector computation ──────────────────────────
    print("\n2. Vector autodiff: f(x) = sum(x^2)")
    x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    f = (x ** 2).sum()  # f = 14, grad = [2, 4, 6]
    f.backward()
    print(f"   x = {x.data}")
    print(f"   f(x) = {f.data.item():.1f}")
    print(f"   ∇f = {x.grad}  (expected [2, 4, 6])")

    # ── 3. Matrix operations ──────────────────────────
    print("\n3. Matrix autodiff: f(A) = sum(A @ b)")
    A = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    b = Tensor(np.array([1.0, 1.0]))
    f = (A @ b).sum()  # f = (1+2)+(3+4) = 10
    f.backward()
    print(f"   A =\n{A.data}")
    print(f"   f(A) = {f.data.item():.1f}")
    print(f"   ∇f/∇A =\n{A.grad}")

    # ── 4. Chained ops with exp/log ──────────────────
    print("\n4. Chained: f(x) = log(exp(x) + 1)  (softplus)")
    x = Tensor(np.array(1.0), requires_grad=True)
    f = (x.exp() + 1).log()
    f.backward()
    # df/dx = exp(x) / (exp(x) + 1) = sigmoid(x) ≈ 0.7311
    print(f"   x = {x.data.item():.1f}")
    print(f"   f(x) = {f.data.item():.4f}")
    print(f"   df/dx = {x.grad.item():.4f}  (sigmoid ≈ 0.7311)")

    # ── 5. Gradient checking ──────────────────────────
    print("\n5. Gradient check (autograd vs. finite differences)")

    def test_fn(a: Tensor, b: Tensor) -> Tensor:
        return ((a * b).sum() + (a ** 2).sum()).reshape(())

    a = Tensor(np.random.randn(3, 2), requires_grad=True)
    b = Tensor(np.random.randn(3, 2), requires_grad=True)
    result = check_grad(test_fn, [a, b])
    print(f"   Max relative error: {result['max_error']:.2e}")
    print(f"   Passed: {result['passed']}")

    print("\n✓ All autograd examples completed successfully.")


if __name__ == "__main__":
    main()
