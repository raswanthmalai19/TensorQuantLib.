"""
Numerical gradient validation utilities.

Provides central-difference gradient checking to validate the autograd engine.
Used in tests to verify that backward() produces correct gradients.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from tensorquantlib.core.tensor import Tensor


def numerical_gradient(
    fn: Callable[..., Tensor],
    inputs: list[Tensor],
    eps: float = 1e-5,
) -> list[np.ndarray | None]:
    """Compute numerical gradients via central differences.

    For each input tensor with requires_grad=True, perturbs each element
    by ±eps and computes (f(x+eps) - f(x-eps)) / (2*eps).

    Args:
        fn: Function that takes Tensor inputs and returns a scalar Tensor.
        inputs: List of input Tensors.
        eps: Perturbation size for finite differences.

    Returns:
        List of gradient arrays, one per input. None for inputs
        where requires_grad=False.
    """
    grads: list[np.ndarray | None] = []
    for inp in inputs:
        if not inp.requires_grad:
            grads.append(None)
            continue

        grad = np.zeros_like(inp.data)
        it = np.nditer(inp.data, flags=["multi_index"], op_flags=[['readwrite']])
        while not it.finished:
            idx = it.multi_index
            old_val = inp.data[idx]

            # f(x + eps)
            inp.data[idx] = old_val + eps
            fxp = fn(*inputs).data.sum()

            # f(x - eps)
            inp.data[idx] = old_val - eps
            fxm = fn(*inputs).data.sum()

            # Central difference
            grad[idx] = (fxp - fxm) / (2 * eps)

            # Restore
            inp.data[idx] = old_val
            it.iternext()

        grads.append(grad)
    return grads


def check_grad(
    fn: Callable[..., Tensor],
    inputs: list[Tensor],
    eps: float = 1e-5,
    tol: float = 1e-5,
) -> dict[str, object]:
    """Compare autograd gradients with numerical gradients.

    Runs forward + backward through fn, then computes numerical gradients
    via central differences, and reports the maximum relative error.

    Args:
        fn: Function taking Tensor inputs, returning a scalar Tensor.
        inputs: List of input Tensors (those with requires_grad=True are checked).
        eps: Perturbation for finite differences.
        tol: Tolerance for pass/fail.

    Returns:
        Dict with keys:
            'max_error': float — maximum relative error across all inputs
            'errors': list of per-input max relative errors (None if not checked)
            'passed': bool — True if max_error < tol
    """
    # Zero existing gradients
    for inp in inputs:
        inp.zero_grad()

    # Autograd forward + backward
    out = fn(*inputs)
    # Sum if not scalar to get a scalar loss
    if out.data.size > 1:
        out = out.sum()
    out.backward()

    # Numerical gradients
    num_grads = numerical_gradient(fn, inputs, eps=eps)

    errors: list[float | None] = []
    max_error = 0.0

    for inp, ng in zip(inputs, num_grads):
        if ng is None:
            errors.append(None)
            continue

        ag = inp.grad if inp.grad is not None else np.zeros_like(inp.data)
        # Relative error: |ag - ng| / max(|ag|, |ng|, 1e-8)
        diff = np.abs(ag - ng)
        scale = np.maximum(np.abs(ag), np.abs(ng))
        scale = np.maximum(scale, 1e-8)
        rel_error = (diff / scale).max()

        errors.append(float(rel_error))
        max_error = max(max_error, float(rel_error))

    return {
        "max_error": max_error,
        "errors": errors,
        "passed": max_error < tol,
    }
