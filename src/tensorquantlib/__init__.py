"""TensorQuantLib — Tensor-Train surrogate pricing engine with autodiff."""

__version__ = "0.1.0"
__author__ = "TensorQuantLib Contributors"

from tensorquantlib.core.tensor import Tensor
from tensorquantlib.tt.surrogate import TTSurrogate
from tensorquantlib.tt.decompose import tt_svd, tt_round
from tensorquantlib.tt.ops import (
    tt_eval,
    tt_eval_batch,
    tt_to_full,
    tt_ranks,
    tt_memory,
    tt_error,
    tt_compression_ratio,
    tt_add,
    tt_scale,
    tt_hadamard,
    tt_dot,
    tt_frobenius_norm,
)
from tensorquantlib.finance.black_scholes import (
    bs_price_numpy,
    bs_price_tensor,
    bs_delta,
    bs_gamma,
    bs_vega,
    bs_theta,
    bs_rho,
)
from tensorquantlib.finance.basket import (
    simulate_basket,
    build_pricing_grid,
    build_pricing_grid_analytic,
)
from tensorquantlib.finance.greeks import compute_greeks, compute_greeks_vectorized
from tensorquantlib.viz import plot_pricing_surface, plot_greeks_surface, plot_tt_ranks

__all__ = [
    # Core
    "Tensor",
    # TT compression
    "TTSurrogate",
    "tt_svd",
    "tt_round",
    "tt_eval",
    "tt_eval_batch",
    "tt_to_full",
    "tt_ranks",
    "tt_memory",
    "tt_error",
    "tt_compression_ratio",
    "tt_add",
    "tt_scale",
    "tt_hadamard",
    "tt_dot",
    "tt_frobenius_norm",
    # Finance
    "bs_price_numpy",
    "bs_price_tensor",
    "bs_delta",
    "bs_gamma",
    "bs_vega",
    "bs_theta",
    "bs_rho",
    "simulate_basket",
    "build_pricing_grid",
    "build_pricing_grid_analytic",
    "compute_greeks",
    "compute_greeks_vectorized",
    # Visualization
    "plot_pricing_surface",
    "plot_greeks_surface",
    "plot_tt_ranks",
]
