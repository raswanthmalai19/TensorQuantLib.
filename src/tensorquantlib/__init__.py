"""TensorQuantLib — Tensor-Train surrogate pricing engine with autodiff."""

__version__ = "0.3.0"
__author__ = "TensorQuantLib Contributors"

# ── Core autograd ───────────────────────────────────────────────────
# ── Backtesting ──────────────────────────────────────────────────────
from tensorquantlib.backtest import (
    EQUITY_COMM,
    EQUITY_SLIP,
    FX_COMM,
    ILLIQUID_SLIP,
    ZERO_COST,
    BacktestEngine,
    BacktestResult,
    CommissionModel,
    DeltaGammaHedgeStrategy,
    DeltaHedgeStrategy,
    GammaScalpingStrategy,
    SlippageModel,
    StraddleStrategy,
    Strategy,
    Trade,
    annualized_return,
    calmar_ratio,
    hedge_efficiency,
    hedge_pnl_attribution,
    information_ratio,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    turnover,
    win_rate,
)

# ── Second-order autodiff ───────────────────────────────────────────
from tensorquantlib.core.second_order import (
    gamma_autograd,
    hessian,
    hessian_diag,
    hvp,
    mixed_partial,
    second_order_greeks,
    vanna_autograd,
    vhp,
    volga_autograd,
)
from tensorquantlib.core.tensor import (
    Tensor,
    tensor_abs,
    tensor_clip,
    tensor_cos,
    tensor_sin,
    tensor_softmax,
    tensor_tanh,
    tensor_where,
)

# ── Finance — American Options (LSM) ───────────────────────────────
from tensorquantlib.finance.american import (
    american_greeks,
    american_option_grid,
    american_option_lsm,
)

# ── Finance — Basket & Greeks ───────────────────────────────────────
from tensorquantlib.finance.basket import (
    build_pricing_grid,
    build_pricing_grid_analytic,
    simulate_basket,
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

# ── Finance — Credit Risk ───────────────────────────────────────────
from tensorquantlib.finance.credit import (
    cds_price,
    cds_spread,
    hazard_rate_from_spread,
    merton_credit_spread,
    merton_default_prob,
    survival_probability,
)

# ── Finance — Exotic Options ────────────────────────────────────────
from tensorquantlib.finance.exotics import (
    asian_geometric_price,
    asian_price_mc,
    barrier_price,
    barrier_price_mc,
    cliquet_price_mc,
    digital_greeks,
    digital_price,
    digital_price_mc,
    lookback_fixed_analytic,
    lookback_floating_analytic,
    lookback_price_mc,
    rainbow_price_mc,
)

# ── Finance — FX Options ────────────────────────────────────────────
from tensorquantlib.finance.fx import (
    fx_forward,
    garman_kohlhagen,
    gk_greeks,
    quanto_option,
)
from tensorquantlib.finance.greeks import compute_greeks, compute_greeks_vectorized

# ── Finance — Heston Model ──────────────────────────────────────────
from tensorquantlib.finance.heston import (
    HestonCalibrator,
    HestonParams,
    heston_greeks,
    heston_price,
    heston_price_mc,
)

# ── Finance — Implied Volatility ────────────────────────────────────
from tensorquantlib.finance.implied_vol import (
    implied_vol,
    implied_vol_batch,
    implied_vol_nr,
    iv_surface,
)

# ── Finance — IR Derivatives (Black76) ──────────────────────────────
from tensorquantlib.finance.ir_derivatives import (
    black76_caplet,
    black76_floorlet,
    cap_price,
    floor_price,
    swap_rate,
    swaption_parity,
    swaption_price,
)

# ── Finance — Jump-Diffusion Models ─────────────────────────────────
from tensorquantlib.finance.jump_diffusion import (
    kou_jump_price_mc,
    merton_jump_price,
    merton_jump_price_mc,
)

# ── Finance — Local Volatility ──────────────────────────────────────
from tensorquantlib.finance.local_vol import (
    dupire_local_vol,
    local_vol_mc,
)

# ── Finance — Interest Rate Models ──────────────────────────────────
from tensorquantlib.finance.rates import (
    bootstrap_yield_curve,
    cir_bond_price,
    cir_simulate,
    cir_yield,
    feller_condition,
    nelson_siegel,
    nelson_siegel_calibrate,
    vasicek_bond_price,
    vasicek_option_price,
    vasicek_simulate,
    vasicek_yield,
)

# ── Finance — Risk Metrics ──────────────────────────────────────────
from tensorquantlib.finance.risk import (
    OptionPosition,
    PortfolioRisk,
    cvar,
    greeks_portfolio,
    scenario_analysis,
    var_historical,
    var_mc,
    var_parametric,
)

# ── Finance — Variance Reduction ────────────────────────────────────
from tensorquantlib.finance.variance_reduction import (
    asian_price_cv,
    bs_price_antithetic,
    bs_price_importance,
    bs_price_qmc,
    bs_price_stratified,
    compare_variance_reduction,
)

# ── Finance — Volatility Surface Models ─────────────────────────────
from tensorquantlib.finance.volatility import (
    sabr_calibrate,
    sabr_implied_vol,
    svi_calibrate,
    svi_implied_vol,
    svi_raw,
    svi_surface,
)

# ── TT Compression ──────────────────────────────────────────────────
from tensorquantlib.tt.decompose import tt_cross, tt_round, tt_svd
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
from tensorquantlib.tt.pricing import (
    american_surrogate,
    exotic_surrogate,
    heston_surrogate,
    jump_diffusion_surrogate,
)
from tensorquantlib.tt.surrogate import TTSurrogate

# ── Visualization ────────────────────────────────────────────────────
from tensorquantlib.viz import plot_greeks_surface, plot_pricing_surface, plot_tt_ranks

__all__ = [
    # Core
    "Tensor",
    "tensor_abs",
    "tensor_clip",
    "tensor_cos",
    "tensor_sin",
    "tensor_softmax",
    "tensor_tanh",
    "tensor_where",
    # Black-Scholes
    "bs_delta",
    "bs_gamma",
    "bs_price_numpy",
    "bs_price_tensor",
    "bs_rho",
    "bs_theta",
    "bs_vega",
    # Implied volatility
    "implied_vol",
    "implied_vol_batch",
    "implied_vol_nr",
    "iv_surface",
    # Heston
    "HestonCalibrator",
    "HestonParams",
    "heston_greeks",
    "heston_price",
    "heston_price_mc",
    # American options
    "american_greeks",
    "american_option_grid",
    "american_option_lsm",
    # Exotics
    "asian_geometric_price",
    "asian_price_mc",
    "barrier_price",
    "barrier_price_mc",
    "cliquet_price_mc",
    "digital_greeks",
    "digital_price",
    "digital_price_mc",
    "lookback_fixed_analytic",
    "lookback_floating_analytic",
    "lookback_price_mc",
    "rainbow_price_mc",
    # Volatility surface
    "sabr_calibrate",
    "sabr_implied_vol",
    "svi_calibrate",
    "svi_implied_vol",
    "svi_raw",
    "svi_surface",
    # Interest rates
    "bootstrap_yield_curve",
    "cir_bond_price",
    "cir_simulate",
    "cir_yield",
    "feller_condition",
    "nelson_siegel",
    "nelson_siegel_calibrate",
    "vasicek_bond_price",
    "vasicek_option_price",
    "vasicek_simulate",
    "vasicek_yield",
    # FX options
    "fx_forward",
    "garman_kohlhagen",
    "gk_greeks",
    "quanto_option",
    # Credit risk
    "cds_price",
    "cds_spread",
    "hazard_rate_from_spread",
    "merton_credit_spread",
    "merton_default_prob",
    "survival_probability",
    # Variance reduction
    "asian_price_cv",
    "bs_price_antithetic",
    "bs_price_importance",
    "bs_price_qmc",
    "bs_price_stratified",
    "compare_variance_reduction",
    # Risk
    "OptionPosition",
    "PortfolioRisk",
    "cvar",
    "greeks_portfolio",
    "scenario_analysis",
    "var_historical",
    "var_mc",
    "var_parametric",
    # Jump-diffusion
    "kou_jump_price_mc",
    "merton_jump_price",
    "merton_jump_price_mc",
    # Local volatility
    "dupire_local_vol",
    "local_vol_mc",
    # IR derivatives
    "black76_caplet",
    "black76_floorlet",
    "cap_price",
    "floor_price",
    "swap_rate",
    "swaption_parity",
    "swaption_price",
    # Basket & Greeks
    "build_pricing_grid",
    "build_pricing_grid_analytic",
    "compute_greeks",
    "compute_greeks_vectorized",
    "simulate_basket",
    # Second-order autodiff
    "gamma_autograd",
    "hessian",
    "hessian_diag",
    "hvp",
    "mixed_partial",
    "second_order_greeks",
    "vanna_autograd",
    "vhp",
    "volga_autograd",
    # TT compression
    "TTSurrogate",
    "tt_add",
    "tt_compression_ratio",
    "tt_cross",
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
    # TT-accelerated pricers
    "american_surrogate",
    "exotic_surrogate",
    "heston_surrogate",
    "jump_diffusion_surrogate",
    # Visualization
    "plot_greeks_surface",
    "plot_pricing_surface",
    "plot_tt_ranks",
    # Backtesting — engine
    "BacktestEngine",
    "BacktestResult",
    "COMMISSIONMODEL",
    "EQUITY_COMM",
    "EQUITY_SLIP",
    "FX_COMM",
    "ILLIQUID_SLIP",
    "SlippageModel",
    "ZERO_COST",
    # Backtesting — strategies
    "DeltaGammaHedgeStrategy",
    "DeltaHedgeStrategy",
    "GammaScalpingStrategy",
    "Strategy",
    "StraddleStrategy",
    "Trade",
    # Backtesting — metrics
    "annualized_return",
    "calmar_ratio",
    "hedge_efficiency",
    "hedge_pnl_attribution",
    "information_ratio",
    "max_drawdown",
    "profit_factor",
    "sharpe_ratio",
    "sortino_ratio",
    "turnover",
    "win_rate",
]
