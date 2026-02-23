"""
Basket option pricing via Monte Carlo simulation.

Provides:
    - simulate_basket: Multi-asset GBM simulation for basket option pricing
    - build_pricing_grid: Construct a multi-dimensional pricing tensor
                          for Tensor-Train compression
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


def simulate_basket(
    S0: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: np.ndarray,
    corr: np.ndarray,
    weights: np.ndarray,
    n_paths: int = 100_000,
    q: Optional[np.ndarray] = None,
    option_type: str = "call",
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Price a basket option via Monte Carlo simulation.

    Uses correlated geometric Brownian motion (GBM) under the
    risk-neutral measure to simulate terminal asset prices.

    Payoff_call = max(sum(w_i * S_i(T)) - K, 0)
    Price = exp(-rT) * E[Payoff]

    Args:
        S0: Initial spot prices, shape (d,).
        K: Strike price.
        T: Time to expiry (years).
        r: Risk-free rate.
        sigma: Volatilities, shape (d,).
        corr: Correlation matrix, shape (d, d).
        weights: Basket weights, shape (d,).
        n_paths: Number of Monte Carlo paths.
        q: Dividend yields, shape (d,). Defaults to zero.
        option_type: 'call' or 'put'.
        seed: Random seed for reproducibility.

    Returns:
        (price, stderr): Discounted expected payoff and its standard error.
    """
    d = len(S0)
    if q is None:
        q = np.zeros(d)

    rng = np.random.default_rng(seed)

    # Cholesky decomposition of correlation matrix
    # Regularize if needed for numerical stability
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        corr_reg = corr + 1e-8 * np.eye(d)
        L = np.linalg.cholesky(corr_reg)

    # Generate correlated standard normals: (n_paths, d)
    Z = rng.standard_normal((n_paths, d))
    Z_corr = Z @ L.T

    # Simulate terminal prices under risk-neutral measure
    # S_i(T) = S_i(0) * exp((r - q_i - sigma_i^2/2)*T + sigma_i*sqrt(T)*Z_i)
    drift = (r - q - 0.5 * sigma ** 2) * T         # shape (d,)
    diffusion = sigma * np.sqrt(T) * Z_corr          # shape (n_paths, d)
    S_T = S0 * np.exp(drift + diffusion)              # shape (n_paths, d)

    # Basket value at maturity
    basket = S_T @ weights                             # shape (n_paths,)

    # Payoff
    if option_type == "call":
        payoff = np.maximum(basket - K, 0.0)
    else:
        payoff = np.maximum(K - basket, 0.0)

    # Discounted price and standard error
    discount = np.exp(-r * T)
    price = discount * np.mean(payoff)
    stderr = discount * np.std(payoff) / np.sqrt(n_paths)

    return float(price), float(stderr)


def _price_at_spots(
    spots: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: np.ndarray,
    corr: np.ndarray,
    weights: np.ndarray,
    n_mc_paths: int,
    q: Optional[np.ndarray],
    option_type: str,
    seed: int,
) -> float:
    """Price a single basket option for given spot values."""
    price, _ = simulate_basket(
        S0=spots, K=K, T=T, r=r, sigma=sigma, corr=corr,
        weights=weights, n_paths=n_mc_paths, q=q,
        option_type=option_type, seed=seed,
    )
    return price


def build_pricing_grid(
    S0_ranges: list[Tuple[float, float]],
    K: float,
    T: float,
    r: float,
    sigma: np.ndarray,
    corr: np.ndarray,
    weights: np.ndarray,
    n_points: int = 30,
    n_mc_paths: int = 1_000,
    q: Optional[np.ndarray] = None,
    option_type: str = "call",
    seed: int = 42,
) -> Tuple[np.ndarray, list[np.ndarray]]:
    """Build a multi-dimensional pricing tensor for TT compression.

    For each combination of spot prices on the grid, runs a Monte Carlo
    simulation to compute the basket option price. Returns a d-dimensional
    tensor of shape (n_points, n_points, ..., n_points).

    Args:
        S0_ranges: List of (min, max) tuples for each asset's spot range.
        K: Strike price.
        T: Time to expiry.
        r: Risk-free rate.
        sigma: Volatilities, shape (d,).
        corr: Correlation matrix, shape (d, d).
        weights: Basket weights, shape (d,).
        n_points: Number of grid points per asset.
        n_mc_paths: MC paths per grid point (use fewer for speed).
        q: Dividend yields.
        option_type: 'call' or 'put'.
        seed: Base random seed.

    Returns:
        (grid_tensor, axes): The pricing tensor and list of 1D grid axes.
    """
    d = len(S0_ranges)
    if q is None:
        q = np.zeros(d)

    # Create grid axes for each asset
    axes = [np.linspace(lo, hi, n_points) for lo, hi in S0_ranges]

    # Create meshgrid indices
    grid_shape = tuple([n_points] * d)
    grid = np.zeros(grid_shape)

    # Iterate over all grid points
    # Use np.ndindex for clean iteration over d-dimensional grid
    total_points = n_points ** d
    for flat_idx, multi_idx in enumerate(np.ndindex(*grid_shape)):
        # Construct spot vector for this grid point
        spots = np.array([axes[i][multi_idx[i]] for i in range(d)])
        # Use a deterministic seed per grid point for reproducibility
        point_seed = seed + flat_idx
        grid[multi_idx] = _price_at_spots(
            spots, K, T, r, sigma, corr, weights,
            n_mc_paths, q, option_type, point_seed,
        )

    return grid, axes


def build_pricing_grid_analytic(
    S0_ranges: list[Tuple[float, float]],
    K: float,
    T: float,
    r: float,
    sigma: np.ndarray,
    weights: np.ndarray,
    n_points: int = 30,
) -> Tuple[np.ndarray, list[np.ndarray]]:
    """Build a pricing grid using a fast analytic approximation.

    Instead of MC (slow), uses the discounted expected payoff of a basket
    under a simplifying assumption (independent lognormals). This is much
    faster and produces a smooth tensor ideal for TT compression demos.

    The payoff is: max(sum(w_i * S_i) - K, 0), priced as:
    Price ≈ exp(-rT) * max(sum(w_i * S_i * exp((r - 0.5*sigma_i^2)*T)) - K, 0)

    This is the "intrinsic forward value" — not a perfect price but produces
    a smooth, low-rank tensor that demonstrates TT compression well.

    Args:
        S0_ranges: List of (min, max) tuples for each asset.
        K: Strike.
        T: Time to expiry.
        r: Risk-free rate.
        sigma: Volatilities, shape (d,).
        weights: Basket weights, shape (d,).
        n_points: Grid points per axis.

    Returns:
        (grid_tensor, axes): Pricing tensor and grid axes.
    """
    d = len(S0_ranges)
    axes = [np.linspace(lo, hi, n_points) for lo, hi in S0_ranges]

    # Build meshgrid
    grids = np.meshgrid(*axes, indexing="ij")  # list of d arrays, each shape (n,...,n)

    # Forward value of each asset: S_i * exp((r - 0.5*sigma_i^2)*T)
    discount = np.exp(-r * T)
    basket_forward = np.zeros(grids[0].shape)
    for i in range(d):
        fwd_factor = np.exp((r - 0.5 * sigma[i] ** 2) * T)
        basket_forward += weights[i] * grids[i] * fwd_factor

    # Intrinsic value (discounted)
    grid = discount * np.maximum(basket_forward - K, 0.0)

    return grid, axes
