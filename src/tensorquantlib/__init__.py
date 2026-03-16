"""TensorQuantLib — Tensor-Train surrogate pricing engine with autodiff."""

__version__ = "0.3.0"
__author__ = "TensorQuantLib Contributors"

# ── Core autograd ───────────────────────────────────────────────────
from tensorquantlib.core.tensor import (
    Tensor,
    tensor_sin,
    tensor_cos,
    tensor_tanh,
    tensor_abs,
    tensor_clip,
    tensor_where,
    tensor_softmax,
)

# ── Second-order autodiff ───────────────────────────────────────────
from tensorquantlib.core.second_order import (
    hvp,
    hessian,
    hessian_diag,
    vhp,
    mixed_partial,
    gamma_autograd,
    vanna_autograd,
    volga_autograd,
    second_order_greeks,
)

# ── Finance — Black-Scholes ─────────────────────────────────────────
from tensorquantlib.finance.black_scholes import (
    bs_delta,
    bs_gamma,
    bs_price_numpy,
    bs_price_tensor,
    bs_rho,
    bs_theta,
    bs_vega,
)

# ── Finance — Implied Volatility ────────────────────────────────────
from tensorquantlib.finance.implied_vol import (
    implied_vol,
    implied_vol_batch,
    implied_vol_nr,
    iv_surface,
)

# ── Finance — Heston Model ──────────────────────────────────────────
from tensorquantlib.finance.heston import (
    HestonParams,
    HestonCalibrator,
    heston_price,
    heston_price_mc,
    heston_greeks,
)

# ── Finance — American Options (LSM) ───────────────────────────────
from tensorquantlib.finance.american import (
    american_option_lsm,
    american_option_grid,
    american_greeks,
)

# ── Finance — Exotic Options ────────────────────────────────────────
from tensorquantlib.finance.exotics import (
    asian_price_mc,
    asian_geometric_price,
    digital_price,
    digital_price_mc,
    digital_greeks,
    barrier_price,
    barrier_price_mc,
    lookback_fixed_analytic,
    lookback_floating_analytic,
    lookback_price_mc,
    cliquet_price_mc,
    rainbow_price_mc,
)

# ── Finance — Volatility Surface Models ─────────────────────────────
from tensorquantlib.finance.volatility import (
    sabr_implied_vol,
    sabr_calibrate,
    svi_raw,
    svi_implied_vol,
    svi_calibrate,
    svi_surface,
)

# ── Finance — Interest Rate Models ──────────────────────────────────
from tensorquantlib.finance.rates import (
    vasicek_bond_price,
    vasicek_yield,
    vasicek_option_price,
    vasicek_simulate,
    cir_bond_price,
    cir_yield,
    cir_simulate,
    feller_condition,
    nelson_siegel,
    nelson_siegel_calibrate,
    bootstrap_yield_curve,
)

# ── Finance — FX Options ────────────────────────────────────────────
from tensorquantlib.finance.fx import (
    garman_kohlhagen,
    gk_greeks,
    fx_forward,
    quanto_option,
)

# ── Finance — Credit Risk ───────────────────────────────────────────
from tensorquantlib.finance.credit import (
    merton_default_prob,
    merton_credit_spread,
    survival_probability,
    hazard_rate_from_spread,
    cds_spread,
    cds_price,
)

# ── Finance — Variance Reduction ────────────────────────────────────
from tensorquantlib.finance.variance_reduction import (
    bs_price_antithetic,
    asian_price_cv,
    bs_price_qmc,
    bs_price_importance,
    bs_price_stratified,
    compare_variance_reduction,
)

# ── Finance — Risk Metrics ──────────────────────────────────────────
from tensorquantlib.finance.risk import (
    PortfolioRisk,
    OptionPosition,
    var_parametric,
    var_historical,
    var_mc,
    cvar,
    scenario_analysis,
    greeks_portfolio,
)

# ── Finance — Jump-Diffusion Models ─────────────────────────────────
from tensorquantlib.finance.jump_diffusion import (
    merton_jump_price,
    merton_jump_price_mc,
    kou_jump_price_mc,
)

# ── Finance — Local Volatility ──────────────────────────────────────
from tensorquantlib.finance.local_vol import (
    dupire_local_vol,
    local_vol_mc,
)

# ── Finance — IR Derivatives (Black76) ──────────────────────────────
from tensorquantlib.finance.ir_derivatives import (
    black76_caplet,
    black76_floorlet,
    cap_price,
    floor_price,
    swap_rate,
    swaption_price,
    swaption_parity,
)

# ── Finance — Basket & Greeks ───────────────────────────────────────
from tensorquantlib.finance.basket import (
    build_pricing_grid,
    build_pricing_grid_analytic,
    simulate_basket,
)
from tensorquantlib.finance.greeks import compute_greeks, compute_greeks_vectorized

# ── TT Compression ──────────────────────────────────────────────────
from tensorquantlib.tt.decompose import tt_round, tt_svd, tt_cross
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
from tensorquantlib.tt.pricing import (
    heston_surrogate,
    american_surrogate,
    exotic_surrogate,
    jump_diffusion_surrogate,
)

# ── Visualization ────────────────────────────────────────────────────
from tensorquantlib.viz import plot_greeks_surface, plot_pricing_surface, plot_tt_ranks

__all__ = [
    # Core
    "Tensor",
    "tensor_sin", "tensor_cos", "tensor_tanh", "tensor_abs",
    "tensor_clip", "tensor_where", "tensor_softmax",
    # Black-Scholes
    "bs_price_numpy", "bs_price_tensor",
    "bs_delta", "bs_gamma", "bs_vega", "bs_theta", "bs_rho",
    # Implied volatility
    "implied_vol", "implied_vol_batch", "implied_vol_nr", "iv_surface",
    # Heston
    "HestonParams", "HestonCalibrator",
    "heston_price", "heston_price_mc", "heston_greeks",
    # American options
    "american_option_lsm", "american_option_grid", "american_greeks",
    # Exotics
    "asian_price_mc", "asian_geometric_price",
    "digital_price", "digital_price_mc", "digital_greeks",
    "barrier_price", "barrier_price_mc",
    "lookback_fixed_analytic", "lookback_floating_analytic", "lookback_price_mc",
    "cliquet_price_mc", "rainbow_price_mc",
    # Volatility surface
    "sabr_implied_vol", "sabr_calibrate",
    "svi_raw", "svi_implied_vol", "svi_calibrate", "svi_surface",
    # Interest rates
    "vasicek_bond_price", "vasicek_yield", "vasicek_option_price", "vasicek_simulate",
    "cir_bond_price", "cir_yield", "cir_simulate", "feller_condition",
    "nelson_siegel", "nelson_siegel_calibrate", "bootstrap_yield_curve",
    # FX options
    "garman_kohlhagen", "gk_greeks", "fx_forward", "quanto_option",
    # Credit risk
    "merton_default_prob", "merton_credit_spread",
    "survival_probability", "hazard_rate_from_spread",
    "cds_spread", "cds_price",
    # Variance reduction
    "bs_price_antithetic", "asian_price_cv", "bs_price_qmc",
    "bs_price_importance", "bs_price_stratified", "compare_variance_reduction",
    # Risk
    "PortfolioRisk", "OptionPosition",
    "var_parametric", "var_historical", "var_mc", "cvar",
    "scenario_analysis", "greeks_portfolio",
    # Jump-diffusion
    "merton_jump_price", "merton_jump_price_mc", "kou_jump_price_mc",
    # Local volatility
    "dupire_local_vol", "local_vol_mc",
    # IR derivatives
    "black76_caplet", "black76_floorlet", "cap_price", "floor_price",
    "swap_rate", "swaption_price", "swaption_parity",
    # Basket & Greeks
    "simulate_basket", "build_pricing_grid", "build_pricing_grid_analytic",
    "compute_greeks", "compute_greeks_vectorized",
    # Second-order autodiff
    "hvp", "hessian", "hessian_diag", "vhp", "mixed_partial",
    "gamma_autograd", "vanna_autograd", "volga_autograd", "second_order_greeks",
    # TT compression
    "TTSurrogate",
    "tt_svd", "tt_round", "tt_cross",
    "tt_eval", "tt_eval_batch", "tt_to_full",
    "tt_ranks", "tt_memory", "tt_error", "tt_compression_ratio",
    "tt_add", "tt_scale", "tt_hadamard", "tt_dot", "tt_frobenius_norm",
    # TT-accelerated pricers
    "heston_surrogate", "american_surrogate",
    "exotic_surrogate", "jump_diffusion_surrogate",
    # Visualization
    "plot_pricing_surface", "plot_greeks_surface", "plot_tt_ranks",
]
