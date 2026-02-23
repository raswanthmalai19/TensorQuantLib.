"""
Autograd-based Greeks computation.

Computes option sensitivities by running the pricing function through
the Tensor autograd engine and extracting gradients.

    - Delta (dV/dS) via autograd
    - Vega (dV/dsigma) via autograd
    - Gamma (d²V/dS²) via finite-difference on autograd Delta
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Dict

from tensorquantlib.core.tensor import Tensor, _ensure_tensor


def compute_greeks(
    price_fn: Callable[..., Tensor],
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
) -> Dict[str, float]:
    """Compute Delta, Vega, and Gamma using autograd + finite differences.

    Args:
        price_fn: Pricing function (e.g., bs_price_tensor) that accepts
                  (S, K, T, r, sigma, q, option_type) and returns a Tensor.
        S: Spot price.
        K: Strike price.
        T: Time to expiry.
        r: Risk-free rate.
        sigma: Volatility.
        q: Dividend yield.
        option_type: 'call' or 'put'.

    Returns:
        Dict with keys 'price', 'delta', 'vega', 'gamma'.
    """
    # ---- Delta & Vega via autograd ----
    S_t = Tensor(np.array(S, dtype=np.float64), requires_grad=True)
    sigma_t = Tensor(np.array(sigma, dtype=np.float64), requires_grad=True)

    price = price_fn(S_t, K, T, r, sigma_t, q, option_type)
    price_val = float(price.data)

    # Backward pass
    if price.data.size > 1:
        price.sum().backward()
    else:
        price.backward()

    delta = float(S_t.grad) if S_t.grad is not None else 0.0
    vega = float(sigma_t.grad) if sigma_t.grad is not None else 0.0

    # ---- Gamma via finite-difference on Delta ----
    h = S * 1e-4
    gamma = _finite_diff_gamma(price_fn, S, K, T, r, sigma, q, option_type, h)

    return {
        "price": price_val,
        "delta": delta,
        "vega": vega,
        "gamma": gamma,
    }


def _finite_diff_gamma(
    price_fn: Callable[..., Tensor],
    S: float, K: float, T: float, r: float, sigma: float,
    q: float, option_type: str, h: float,
) -> float:
    """Compute Gamma = d²V/dS² via finite-difference on autograd Delta.

    Gamma ≈ (Delta(S+h) - Delta(S-h)) / (2*h)
    """
    def _delta_at(s_val: float) -> float:
        s = Tensor(np.array(s_val, dtype=np.float64), requires_grad=True)
        p = price_fn(s, K, T, r, sigma, q, option_type)
        if p.data.size > 1:
            p.sum().backward()
        else:
            p.backward()
        return float(s.grad) if s.grad is not None else 0.0

    delta_up = _delta_at(S + h)
    delta_down = _delta_at(S - h)
    return (delta_up - delta_down) / (2 * h)


def compute_greeks_vectorized(
    price_fn: Callable[..., Tensor],
    S_array: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
) -> Dict[str, np.ndarray]:
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
        "vega": np.atleast_1d(float(sigma_t.grad)) if sigma_t.grad is not None else np.zeros(1),
    }
