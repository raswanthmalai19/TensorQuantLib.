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
    "simulate_basket",
    # Submodules (importable via tensorquantlib.finance.<name>)
    "american",
    "basket",
    "black_scholes",
    "credit",
    "exotics",
    "fx",
    "greeks",
    "heston",
    "implied_vol",
    "rates",
    "risk",
    "variance_reduction",
    "volatility",
    # New models (v0.3.0)
    "jump_diffusion",
    "local_vol",
    "ir_derivatives",
]
