"""
Autograd-based Greeks computation.

Computes option sensitivities by running the pricing function through
the Tensor autograd engine and extracting gradients.

    - Delta  (dV/dS)       via first-order autograd
    - Vega   (dV/dσ)       via first-order autograd
    - Gamma  (d²V/dS²)     via second-order autograd (gamma_autograd)
    - Vanna  (d²V/dS dσ)   via second-order autograd (vanna_autograd)
    - Volga  (d²V/dσ²)     via second-order autograd (volga_autograd)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from tensorquantlib.core.tensor import Tensor
from tensorquantlib.core.second_order import (
    gamma_autograd,
    vanna_autograd,
    volga_autograd,
    second_order_greeks,
)


def compute_greeks(
    price_fn: Callable[..., Tensor],
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    include_second_order: bool = True,
) -> dict[str, float]:
    """Compute Delta, Vega, Gamma (and optionally Vanna, Volga) using autograd.

    First-order Greeks (Delta, Vega) come from a single backward pass.
    Second-order Greeks (Gamma, Vanna, Volga) use the hybrid semi-analytic
    method from ``tensorquantlib.core.second_order``: analytical gradients
    differentiated once more by central differences (~1e-10 accuracy).

    Args:
        price_fn: Pricing function accepting
                  ``(S, K, T, r, sigma, q, option_type)`` and returning a Tensor.
        S: Spot price.
        K: Strike price.
        T: Time to expiry.
        r: Risk-free rate.
        sigma: Volatility.
        q: Dividend yield.
        option_type: ``'call'`` or ``'put'``.
        include_second_order: If True (default), also compute Gamma, Vanna,
                              and Volga. Set False for speed when only
                              first-order Greeks are needed.

    Returns:
        Dict with keys ``'price'``, ``'delta'``, ``'vega'``, and (when
        ``include_second_order=True``) ``'gamma'``, ``'vanna'``, ``'volga'``.
    """
    # ---- Delta & Vega via first-order autograd ----
    S_t = Tensor(np.array(S, dtype=np.float64), requires_grad=True)
    sigma_t = Tensor(np.array(sigma, dtype=np.float64), requires_grad=True)

    price = price_fn(S_t, K, T, r, sigma_t, q, option_type)
    price_val = float(price.data.item()) if price.data.size == 1 else float(price.data.sum())

    if price.data.size > 1:
        price.sum().backward()
    else:
        price.backward()

    delta = float(S_t.grad.item()) if S_t.grad is not None and S_t.grad.size == 1 else (float(S_t.grad.sum()) if S_t.grad is not None else 0.0)
    vega = float(sigma_t.grad.item()) if sigma_t.grad is not None and sigma_t.grad.size == 1 else (float(sigma_t.grad.sum()) if sigma_t.grad is not None else 0.0)

    result: dict[str, float] = {
        "price": price_val,
        "delta": delta,
        "vega": vega,
    }

    if include_second_order:
        so = second_order_greeks(price_fn, S, K, T, r, sigma, q, option_type)
        result["gamma"] = so["gamma"]
        result["vanna"] = so["vanna"]
        result["volga"] = so["volga"]

    return result



def compute_greeks_vectorized(
    price_fn: Callable[..., Tensor],
    S_array: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
) -> dict[str, np.ndarray]:
    """Compute Greeks for a vector of spot prices.

    Args:
        price_fn: Pricing function.
        S_array: 1D array of spot prices.
        K, T, r, sigma, q, option_type: Option parameters.

    Returns:
        Dict with 'price', 'delta', 'vega' as arrays matching S_array.
    """
    S_t = Tensor(np.asarray(S_array, dtype=np.float64), requires_grad=True)
    sigma_t = Tensor(np.array(sigma, dtype=np.float64), requires_grad=True)

    price = price_fn(S_t, K, T, r, sigma_t, q, option_type)

    if price.data.size > 1:
        price.sum().backward()
    else:
        price.backward()

    return {
        "price": price.data.copy(),
        "delta": S_t.grad.copy() if S_t.grad is not None else np.zeros_like(S_array),
        "vega": np.full(
            len(S_array),
            float(sigma_t.grad.item()) if sigma_t.grad is not None else 0.0,
        ),
    }
