"""Financial pricing engines — Black-Scholes, basket options, Monte Carlo."""

from .black_scholes import (
    bs_price_numpy,
    bs_price_tensor,
    bs_delta,
    bs_gamma,
    bs_vega,
    bs_theta,
    bs_rho,
)
from .greeks import compute_greeks, compute_greeks_vectorized
from .basket import (
    simulate_basket,
    build_pricing_grid,
    build_pricing_grid_analytic,
)

__all__ = [
    "bs_price_numpy",
    "bs_price_tensor",
    "bs_delta",
    "bs_gamma",
    "bs_vega",
    "bs_theta",
    "bs_rho",
    "compute_greeks",
    "compute_greeks_vectorized",
    "simulate_basket",
    "build_pricing_grid",
    "build_pricing_grid_analytic",
]
