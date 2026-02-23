"""TensorQuantLib — Tensor-Train surrogate pricing engine with autodiff."""

__version__ = "0.1.0"
__author__ = "TensorQuantLib Contributors"

from tensorquantlib.core.tensor import Tensor
from tensorquantlib.finance.basket import (
    build_pricing_grid,
    build_pricing_grid_analytic,
    simulate_basket,
)
from tensorquantlib.finance.black_scholes import (
    bs_delta,
    bs_gamma,
    bs_price_numpy,
    bs_price_tensor,
    bs_rho,
    bs_theta,
    bs_vega,
)
from tensorquantlib.finance.greeks import compute_greeks, compute_greeks_vectorized
from tensorquantlib.tt.decompose import tt_round, tt_svd
from tensorquantlib.tt.ops import (
    tt_add,
    tt_compression_ratio,
    tt_dot,
    tt_error,
    tt_eval,
    tt_eval_batch,
    tt_frobenius_norm,
    tt_hadamard,
    tt_memory,
    tt_ranks,
    tt_scale,
    tt_to_full,
)
from tensorquantlib.tt.surrogate import TTSurrogate
from tensorquantlib.viz import plot_greeks_surface, plot_pricing_surface, plot_tt_ranks

__all__ = [
    # TT compression
    "TTSurrogate",
    # Core
    "Tensor",
    "bs_delta",
    "bs_gamma",
    # Finance
    "bs_price_numpy",
    "bs_price_tensor",
    "bs_rho",
    "bs_theta",
    "bs_vega",
    "build_pricing_grid",
    "build_pricing_grid_analytic",
    "compute_greeks",
    "compute_greeks_vectorized",
    "plot_greeks_surface",
    # Visualization
    "plot_pricing_surface",
    "plot_tt_ranks",
    "simulate_basket",
    "tt_add",
    "tt_compression_ratio",
    "tt_dot",
    "tt_error",
    "tt_eval",
    "tt_eval_batch",
    "tt_frobenius_norm",
    "tt_hadamard",
    "tt_memory",
    "tt_ranks",
    "tt_round",
    "tt_scale",
    "tt_svd",
    "tt_to_full",
]
