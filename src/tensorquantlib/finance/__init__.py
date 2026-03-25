"""Financial pricing engines — Black-Scholes, basket options, Monte Carlo."""

from .basket import (
    build_pricing_grid,
    build_pricing_grid_analytic,
    simulate_basket,
)
from .black_scholes import (
    bs_delta,
    bs_gamma,
    bs_price_numpy,
    bs_price_tensor,
    bs_rho,
    bs_theta,
    bs_vega,
)
from .greeks import compute_greeks, compute_greeks_vectorized

# Lazy submodule access — these are importable via tensorquantlib.finance.<module>
# but not eagerly imported to avoid import overhead.
__all__ = [
    "american",
    "basket",
    "black_scholes",
    "bs_delta",
    "bs_gamma",
    "bs_price_numpy",
    "bs_price_tensor",
    "bs_rho",
    "bs_theta",
    "bs_vega",
    "build_pricing_grid",
    "build_pricing_grid_analytic",
    "compute_greeks",
    "compute_greeks_vectorized",
    "credit",
    "exotics",
    "fx",
    "greeks",
    "heston",
    "implied_vol",
    "ir_derivatives",
    "jump_diffusion",
    "local_vol",
    "rates",
    "risk",
    "simulate_basket",
    "variance_reduction",
    "volatility",
]
