"""
Black-Scholes option pricing — both NumPy (analytic) and Tensor (autograd) versions.

Provides:
    - bs_price_numpy: Pure NumPy analytic pricing (ground truth)
    - bs_price_tensor: Tensor-based pricing that flows through autograd
    - bs_delta, bs_gamma, bs_vega, bs_theta, bs_rho: Analytic Greeks (NumPy)
"""

from __future__ import annotations

import numpy as np
from typing import Any, Tuple, Union
from scipy.stats import norm

from tensorquantlib.core.tensor import Tensor, tensor_norm_cdf, tensor_exp, tensor_log, tensor_sqrt


# ====================================================================== #
# Analytic Black-Scholes (NumPy) — ground truth
# ====================================================================== #

def _d1_d2(
    S: Union[float, np.ndarray],
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> Tuple[Any, Any]:
    """Compute d1 and d2 for Black-Scholes formula."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price_numpy(
    S: Union[float, np.ndarray],
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",

) -> Any:
    """Analytic Black-Scholes price (pure NumPy).

    Args:
        S: Spot price (scalar or array).
        K: Strike price.
        T: Time to expiry (years).
        r: Risk-free rate.
        sigma: Volatility.
        q: Continuous dividend yield (default 0).
        option_type: 'call' or 'put'.

    Returns:
        Option price (same shape as S).
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    S_arr = np.asarray(S, dtype=float)
    if np.any(S_arr <= 0):
        raise ValueError("Spot price S must be positive")
    if K <= 0:
        raise ValueError("Strike K must be positive")
    if T <= 0:
        raise ValueError("Time to expiry T must be positive")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive")

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


# ====================================================================== #
# Analytic Greeks (NumPy) — for validation
# ====================================================================== #

def bs_delta(
    S: Union[float, np.ndarray],
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",

) -> Any:
    """Analytic Delta: dV/dS."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    else:
        return -np.exp(-q * T) * norm.cdf(-d1)


def bs_gamma(
    S: Union[float, np.ndarray],
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,

) -> Any:
    """Analytic Gamma: d²V/dS² (same for call and put)."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(
    S: Union[float, np.ndarray],
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,

) -> Any:
    """Analytic Vega: dV/dsigma (same for call and put)."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def bs_theta(
    S: Union[float, np.ndarray],
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",

) -> Any:
    """Analytic Theta: -dV/dT (time decay per year)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    term1 = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if option_type == "call":
        return term1 - r * K * np.exp(-r * T) * norm.cdf(d2) + q * S * np.exp(-q * T) * norm.cdf(d1)
    else:
        return term1 + r * K * np.exp(-r * T) * norm.cdf(-d2) - q * S * np.exp(-q * T) * norm.cdf(-d1)


def bs_rho(
    S: Union[float, np.ndarray],
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",

) -> Any:
    """Analytic Rho: dV/dr."""
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)


# ====================================================================== #
# Tensor-based Black-Scholes (flows through autograd)
# ====================================================================== #

def bs_price_tensor(
    S: Union[float, Tensor],
    K: Union[float, Tensor],
    T: Union[float, Tensor],
    r: Union[float, Tensor],
    sigma: Union[float, Tensor],
    q: float = 0.0,
    option_type: str = "call",
) -> Tensor:
    """Black-Scholes price using Tensor operations (supports autograd).

    All inputs can be Tensor objects. The computation graph is built
    so that calling .backward() on the result will produce gradients
    (Delta = dPrice/dS, Vega = dPrice/dsigma, etc.).

    Args:
        S: Spot price (Tensor or float).
        K: Strike price (Tensor or float).
        T: Time to expiry (Tensor or float).
        r: Risk-free rate (Tensor or float).
        sigma: Volatility (Tensor or float).
        q: Dividend yield (float, default 0).
        option_type: 'call' or 'put'.

    Returns:
        Option price as a Tensor.
    """
    from tensorquantlib.core.tensor import _ensure_tensor

    S = _ensure_tensor(S)
    K = _ensure_tensor(K)
    T = _ensure_tensor(T)
    r = _ensure_tensor(r)
    sigma = _ensure_tensor(sigma)
    q_t = _ensure_tensor(q)

    sqrt_T = T.sqrt()
    d1 = ((S / K).log() + (r - q_t + sigma * sigma * _ensure_tensor(0.5)) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    discount = (r * T * _ensure_tensor(-1.0)).exp()  # exp(-rT)
    div_discount = (q_t * T * _ensure_tensor(-1.0)).exp()  # exp(-qT)

    if option_type == "call":
        price = S * div_discount * tensor_norm_cdf(d1) - K * discount * tensor_norm_cdf(d2)
    else:
        price = K * discount * tensor_norm_cdf(d2 * _ensure_tensor(-1.0)) - S * div_discount * tensor_norm_cdf(d1 * _ensure_tensor(-1.0))

    return price
