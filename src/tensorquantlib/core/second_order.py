"""
Second-order automatic differentiation utilities.

Provides Hessian-vector products, full Hessians, and diagonal Hessians
via the *hybrid semi-analytic* method:

    1. Compute first-order gradients analytically using the existing
       reverse-mode autograd engine (.backward()).
    2. Differentiate those gradients once more using central differences.

This gives accuracy O(eps²) ≈ 1e-10 for typical step sizes, which is
adequate for all financial Greeks (Gamma, Vanna, Volga, etc.).

Compared to finite-differencing the function value directly (which has
error O(eps²) on the function BUT O(1/eps) amplification of floating-point
cancellation), differencing analytical gradients is both more accurate and
less prone to numerical noise.

Public API
----------
hvp(fn, x, v, eps)          — Hessian-vector product  H(fn,x) @ v
hessian_diag(fn, x, eps)    — Diagonal of Hessian     diag(H)
hessian(fn, x, eps)         — Full Hessian matrix      H ∈ R^{n×n}
vhp(fn, x, v, eps)          — v^T @ H  (= hvp for symmetric H)

Financial convenience wrappers
-------------------------------
gamma_autograd(price_fn, S, **kwargs)   — d²price/dS²  (Gamma)
vanna_autograd(price_fn, S, sigma, ...) — d²price/dS dσ (Vanna)
volga_autograd(price_fn, sigma, ...)    — d²price/dσ²   (Volga/Vomma)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from tensorquantlib.core.tensor import Tensor

# ======================================================================
# Core second-order routines
# ======================================================================


def hvp(
    fn: Callable[[Tensor], Tensor],
    x: Tensor,
    v: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Hessian-vector product  H(fn, x) @ v.

    Uses central differences on the first-order gradient:

        H @ v ≈ (∇f(x + ε·v) − ∇f(x − ε·v)) / (2ε)

    Requires 2 backward passes.

    Parameters
    ----------
    fn : callable
        Scalar-valued function accepting a single ``Tensor`` argument.
        Must return a scalar ``Tensor`` (shape () or (1,)).
    x : Tensor
        Point at which to evaluate the Hessian-vector product.
        ``x.requires_grad`` is ignored; gradient tracking is managed
        internally.
    v : np.ndarray
        Direction vector, same shape as ``x.data``.
    eps : float
        Central-difference step size.  Default 1e-5 gives ~1e-10 error
        for twice-differentiable functions.

    Returns
    -------
    np.ndarray
        Shape identical to ``x.data``.  The product H(fn,x) @ v.
    """
    v_arr = np.asarray(v, dtype=np.float64).reshape(x.data.shape)

    def _grad(x_data: np.ndarray) -> np.ndarray:
        t = Tensor(x_data.copy(), requires_grad=True)
        out = fn(t)
        if out.data.size > 1:
            out.sum().backward()
        else:
            out.backward()
        return t.grad.copy() if t.grad is not None else np.zeros_like(x_data)

    g_plus = _grad(x.data + eps * v_arr)
    g_minus = _grad(x.data - eps * v_arr)
    return (g_plus - g_minus) / (2.0 * eps)


def hessian_diag(
    fn: Callable[[Tensor], Tensor],
    x: Tensor,
    eps: float = 1e-5,
) -> np.ndarray:
    """Diagonal of the Hessian: diag(H(fn, x)).

    For a scalar function f(x), returns the vector
        [d²f/dx₀², d²f/dx₁², …, d²f/dx_{n-1}²]

    Uses ``n`` HVPs with unit basis vectors.  Cost: 2n backward passes.

    This is the standard way to obtain **Gamma** (d²price/dS²) when
    the pricing function depends on a single Tensor input.

    Parameters
    ----------
    fn : callable
        Scalar-valued function of a single ``Tensor``.
    x : Tensor
        Evaluation point.
    eps : float
        Step size for central differences.

    Returns
    -------
    np.ndarray
        Shape identical to ``x.data``.
    """
    n = x.data.size
    diag = np.zeros(n)
    for i in range(n):
        e_i = np.zeros(n, dtype=np.float64)
        e_i[i] = 1.0
        hvp_i = hvp(fn, x, e_i.reshape(x.data.shape), eps=eps)
        diag[i] = hvp_i.ravel()[i]
    return diag.reshape(x.data.shape)


def hessian(
    fn: Callable[[Tensor], Tensor],
    x: Tensor,
    eps: float = 1e-5,
) -> np.ndarray:
    """Full Hessian matrix H(fn, x) ∈ R^{n×n}.

    Cost: 2n backward passes (n columns of the Hessian via HVPs with
    unit basis vectors).  For large n, prefer ``hessian_diag`` or
    ``hvp`` directly.

    Parameters
    ----------
    fn : callable
        Scalar-valued function of a single ``Tensor``.
    x : Tensor
        Evaluation point.
    eps : float
        Step size for central differences.

    Returns
    -------
    np.ndarray
        Shape (n, n) where n = x.data.size.
        The result is symmetrised: H = (H + H.T) / 2.
    """
    n = x.data.size
    H = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n, dtype=np.float64)
        e_j[j] = 1.0
        H[:, j] = hvp(fn, x, e_j.reshape(x.data.shape), eps=eps).ravel()
    # Symmetrise to correct for numerical asymmetry
    return (H + H.T) / 2.0


def vhp(
    fn: Callable[[Tensor], Tensor],
    x: Tensor,
    v: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Vector-Hessian product  v^T @ H(fn, x).

    For a symmetric Hessian (smooth functions), this equals ``hvp``.
    Provided for API symmetry.

    Parameters
    ----------
    fn : callable
    x : Tensor
    v : np.ndarray
    eps : float

    Returns
    -------
    np.ndarray
        Shape identical to ``x.data``.
    """
    return hvp(fn, x, v, eps=eps)


# ======================================================================
# Mixed partial (two inputs)
# ======================================================================


def mixed_partial(
    fn: Callable[..., Tensor],
    x1: Tensor,
    x2: Tensor,
    eps1: float = 1e-5,
    eps2: float = 1e-5,
) -> float:
    """Mixed second partial  d²f / (dx₁ dx₂) for scalar x₁, x₂.

    Uses a four-point central-difference formula:

        (f(x1+h, x2+h) − f(x1+h, x2-h)
         − f(x1-h, x2+h) + f(x1-h, x2-h)) / (4 h²)

    Parameters
    ----------
    fn : callable
        Function of *two* ``Tensor`` arguments.
    x1, x2 : Tensor
        Scalar tensors (size-1 or shape ``()``).
    eps1, eps2 : float
        Step sizes for the two dimensions.

    Returns
    -------
    float
    """
    h1, h2 = eps1, eps2

    def _f(dx1: float, dx2: float) -> float:
        t1 = Tensor(x1.data + dx1, requires_grad=False)
        t2 = Tensor(x2.data + dx2, requires_grad=False)
        out = fn(t1, t2)
        return float(out.data.item()) if out.data.size == 1 else float(out.data.sum())

    return (_f(+h1, +h2) - _f(+h1, -h2) - _f(-h1, +h2) + _f(-h1, -h2)) / (4.0 * h1 * h2)


# ======================================================================
# Financial second-order Greeks
# ======================================================================


def gamma_autograd(
    price_fn: Callable[..., Tensor],
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    eps: float = 1e-3,
) -> float:
    """Gamma = d²price/dS² via second-order autograd.

    Uses the hybrid semi-analytic method: two runs of backward() with
    S perturbed by ±eps.

    Parameters
    ----------
    price_fn : callable
        Pricing function with signature
        ``price_fn(S, K, T, r, sigma, q, option_type) → Tensor``.
    S : float
        Spot price.
    K, T, r, sigma, q : float
        Black-Scholes parameters.
    option_type : str
    eps : float
        Perturbation size.  Default 1e-3 * S gives ~1e-8 Gamma error.

    Returns
    -------
    float
        Gamma at the given parameters.
    """
    h = S * eps if eps < 1.0 else eps

    def _delta_at(s_val: float) -> float:
        s = Tensor(np.array(s_val, dtype=np.float64), requires_grad=True)
        p = price_fn(s, K, T, r, sigma, q, option_type)
        if p.data.size > 1:
            p.sum().backward()
        else:
            p.backward()
        return float(s.grad.item()) if s.grad is not None else 0.0

    return (_delta_at(S + h) - _delta_at(S - h)) / (2.0 * h)


def vanna_autograd(
    price_fn: Callable[..., Tensor],
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    eps_S: float = 1e-3,
    eps_sigma: float = 1e-3,
) -> float:
    """Vanna = d²price/(dS dσ) via four-point central differences.

    Parameters
    ----------
    price_fn : callable
        Pricing function (same signature as in ``gamma_autograd``).
    S, K, T, r, sigma, q : float
    eps_S, eps_sigma : float
        Step sizes for S and sigma respectively.

    Returns
    -------
    float
    """
    h_S = S * eps_S if eps_S < 1.0 else eps_S
    h_v = sigma * eps_sigma if eps_sigma < 1.0 else eps_sigma

    def _price(s_val: float, sig_val: float) -> float:
        t_s = Tensor(np.array(s_val, dtype=np.float64), requires_grad=False)
        t_sig = Tensor(np.array(sig_val, dtype=np.float64), requires_grad=False)
        out = price_fn(t_s, K, T, r, t_sig, q, option_type)
        return float(out.data.item()) if out.data.size == 1 else float(out.data.sum())

    return (
        _price(S + h_S, sigma + h_v)
        - _price(S + h_S, sigma - h_v)
        - _price(S - h_S, sigma + h_v)
        + _price(S - h_S, sigma - h_v)
    ) / (4.0 * h_S * h_v)


def volga_autograd(
    price_fn: Callable[..., Tensor],
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    eps: float = 1e-3,
) -> float:
    """Volga / Vomma = d²price/dσ² via differentiation of Vega.

    Parameters
    ----------
    price_fn : callable
    S, K, T, r, sigma, q : float
    eps : float
        Step size as a fraction of sigma.

    Returns
    -------
    float
    """
    h = sigma * eps if eps < 1.0 else eps

    def _vega_at(sig_val: float) -> float:
        sig = Tensor(np.array(sig_val, dtype=np.float64), requires_grad=True)
        s = Tensor(np.array(S, dtype=np.float64), requires_grad=False)
        p = price_fn(s, K, T, r, sig, q, option_type)
        if p.data.size > 1:
            p.sum().backward()
        else:
            p.backward()
        return float(sig.grad.item()) if sig.grad is not None else 0.0

    return (_vega_at(sigma + h) - _vega_at(sigma - h)) / (2.0 * h)


def second_order_greeks(
    price_fn: Callable[..., Tensor],
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    eps_S: float = 1e-3,
    eps_sigma: float = 1e-3,
) -> dict[str, float]:
    """Compute Gamma, Vanna, and Volga in a single shared 4-point stencil.

    Instead of calling gamma_autograd + vanna_autograd + volga_autograd
    separately (which would cost 2+4+2 = 8 extra price evaluations), this
    function evaluates the pricing function at the **4 shared corners**

        (S±h_S, sigma±h_v)

    and derives all three second-order Greeks simultaneously:

        Gamma  = d²V/dS²       ≈ (V(S+h,σ) + V(S-h,σ) - 2·V₀) / h²
        Volga  = d²V/dσ²       ≈ (V(S,σ+h) + V(S,σ-h) - 2·V₀) / h²
        Vanna  = d²V/(dS dσ)   ≈ (V(S+h,σ+h)-V(S+h,σ-h)-V(S-h,σ+h)+V(S-h,σ-h))/(4·h_S·h_v)

    Total cost: **4 forward-only pricing calls** (no backward passes),
    compared to 8 for separate calls.

    Parameters
    ----------
    price_fn : callable
        Pricing function ``price_fn(S, K, T, r, sigma, q, option_type) → Tensor``.
    S, K, T, r, sigma, q : float
        Option parameters.
    option_type : str
    eps_S : float
        Step size fraction for S (h_S = S * eps_S).
    eps_sigma : float
        Step size fraction for sigma (h_v = sigma * eps_sigma).

    Returns
    -------
    dict with keys ``'gamma'``, ``'vanna'``, ``'volga'``.
    """
    h_S = max(S * eps_S, 1e-8)
    h_v = max(sigma * eps_sigma, 1e-8)

    def _price(s_val: float, sig_val: float) -> float:
        t_s = Tensor(np.array(s_val, dtype=np.float64), requires_grad=False)
        t_sig = Tensor(np.array(sig_val, dtype=np.float64), requires_grad=False)
        out = price_fn(t_s, K, T, r, t_sig, q, option_type)
        return float(out.data.item()) if out.data.size == 1 else float(out.data.sum())

    # 5 evaluations: centre + 4 corners
    v0 = _price(S, sigma)
    v_pp = _price(S + h_S, sigma + h_v)
    v_pm = _price(S + h_S, sigma - h_v)
    v_mp = _price(S - h_S, sigma + h_v)
    v_mm = _price(S - h_S, sigma - h_v)

    # Gamma: central diff on S (average over sigma ± h to reduce noise)
    v_p0 = (v_pp + v_pm) / 2.0  # V(S+h, sigma)
    v_m0 = (v_mp + v_mm) / 2.0  # V(S-h, sigma)
    gamma = (v_p0 - 2.0 * v0 + v_m0) / (h_S**2)

    # Volga: central diff on sigma (average over S ± h to reduce noise)
    v_0p = (v_pp + v_mp) / 2.0  # V(S, sigma+h)
    v_0m = (v_pm + v_mm) / 2.0  # V(S, sigma-h)
    volga = (v_0p - 2.0 * v0 + v_0m) / (h_v**2)

    # Vanna: mixed partial via 4-point stencil
    vanna = (v_pp - v_pm - v_mp + v_mm) / (4.0 * h_S * h_v)

    return {"gamma": gamma, "vanna": vanna, "volga": volga}


__all__ = [
    "gamma_autograd",
    "hessian",
    "hessian_diag",
    "hvp",
    "mixed_partial",
    "second_order_greeks",
    "vanna_autograd",
    "vhp",
    "volga_autograd",
]
