# Changelog

All notable changes to TensorQuantLib are documented here.

## [0.3.0] — 2025-02-XX

### Added

**Volatility Surface Models**
- `sabr_implied_vol` — Hagan (2002) SABR approximation for implied volatility
- `sabr_calibrate` — calibrate SABR (alpha, rho, nu) to market smile
- `svi_raw` — SVI raw parameterization (Gatheral 2004)
- `svi_implied_vol`, `svi_calibrate`, `svi_surface` — SVI fitting and surface construction

**Extended Exotic Options**
- `lookback_fixed_analytic` — Goldman-Sosin-Gatto (1979) fixed-strike lookback
- `lookback_floating_analytic` — floating-strike lookback (analytic)
- `lookback_price_mc` — Monte Carlo lookback (fixed and floating)
- `cliquet_price_mc` — ratchet cliquet with per-period and global caps/floors
- `rainbow_price_mc` — best-of / worst-of multi-asset options with correlated GBM

**Interest Rate Models**
- `vasicek_bond_price`, `vasicek_yield`, `vasicek_option_price` — Vasicek closed-form
- `vasicek_simulate` — Vasicek path simulation
- `cir_bond_price`, `cir_yield`, `cir_simulate` — CIR model
- `feller_condition` — Feller condition check for CIR
- `nelson_siegel`, `nelson_siegel_calibrate` — Nelson-Siegel yield curve
- `bootstrap_yield_curve` — simple yield curve bootstrap from bond prices

**FX Options**
- `garman_kohlhagen` — Garman-Kohlhagen FX option pricing
- `gk_greeks` — GK delta, gamma, vega, theta, rho_d, rho_f
- `fx_forward` — covered interest rate parity
- `quanto_option` — quanto-adjusted Black-Scholes

**Credit Risk Models**
- `merton_default_prob`, `merton_credit_spread` — Merton (1974) structural model
- `cds_spread`, `cds_price` — CDS par spread and mark-to-market
- `survival_probability`, `hazard_rate_from_spread` — hazard rate utilities

**Market Data Integration**
- `get_stock_price`, `get_historical_prices` — stock data via yfinance
- `get_options_chain` — options chain retrieval
- `historical_volatility` — realised vol computation
- `get_risk_free_rate` — Treasury rate proxy

**Backtesting Framework**
- `BacktestEngine`, `BacktestResult` — strategy simulation engine
- `Strategy`, `DeltaHedgeStrategy`, `StraddleStrategy` — strategy base + implementations
- `sharpe_ratio`, `max_drawdown`, `sortino_ratio` — performance metrics
- `win_rate`, `profit_factor` — trade-level statistics

### Changed
- Test count increased from 353 to 450+
- Added `volatility`, `rates`, `fx`, `credit` to top-level exports
- Added `data` and `backtest` subpackages

## [0.2.0] — 2025-02-XX

### Added

**New Tensor Operations**
- `tensor_sin`, `tensor_cos`, `tensor_tanh` — trigonometric/hyperbolic ops with autograd
- `tensor_abs` — absolute value with sign-based gradient
- `tensor_clip` — element-wise clipping with pass-through gradient
- `tensor_where` — conditional selection with gradient routing
- `tensor_softmax` — numerically stable softmax with full gradient

**Implied Volatility**
- `implied_vol_brent` — robust IV solver via Brent's method
- `implied_vol_newton` — fast IV solver via Newton-Raphson
- `implied_vol_batch` — vectorised IV for arrays of option prices
- `build_iv_surface` — construct implied volatility surface from market data

**Heston Stochastic Volatility Model**
- `heston_price` — semi-analytic pricing via Gil-Pelaez characteristic function inversion
- `heston_price_mc` — Monte Carlo pricing with QE (Andersen 2008) and Euler schemes
- `heston_calibrate` — calibrate Heston params to market prices via least-squares
- `heston_greeks` — Delta, Vega, Theta, Rho via finite-difference bumps

**American Options**
- `american_option_lsm` — Longstaff-Schwartz least-squares Monte Carlo
- `american_option_grid` — early-exercise grid for visualization
- `american_option_greeks` — Greeks via finite-difference bumps

**Exotic Options**
- `asian_option_price` — arithmetic & geometric Asian options (MC)
- `digital_option_price` — cash-or-nothing & asset-or-nothing digitals (analytic)
- `barrier_option_price` — all 8 barrier types (analytic Reiner-Rubinstein)

**Variance Reduction**
- `mc_antithetic` — antithetic variates
- `mc_control_variate` — control variate with optimal beta
- `mc_quasi_monte_carlo` — Sobol quasi-random sampling
- `mc_importance_sampling` — shifted-mean importance sampling
- `mc_stratified` — stratified sampling
- `compare_variance_reduction` — side-by-side comparison of all methods

**Risk Metrics**
- `var_parametric`, `var_historical`, `var_monte_carlo` — Value at Risk (3 methods)
- `cvar` — Conditional Value at Risk (Expected Shortfall)
- `scenario_analysis` — P&L under user-defined scenarios
- `OptionPosition`, `PortfolioRisk` — portfolio-level risk aggregation

**CLI**
- `python -m tensorquantlib` — command-line interface with info/price/greeks/demo commands

### Fixed
- Heston characteristic function: corrected P1 integrand shift from `-0.5j` to `-1j`
- Importance sampling: fixed bias from missing likelihood ratio in payoff weighting
- NumPy deprecation: replaced `float(array)` with `.item()` throughout test suite

### Changed
- Heston MC default scheme changed from Euler to QE (Andersen 2008) for lower bias
- Test count increased from 243 to 353

## [0.1.1] — 2025-01-XX

### Fixed
- NumPy deprecation warnings: replaced all `float(array)` calls with `.item()` pattern
- Fixed incorrect API examples in `docs/quickstart.rst` and `docs/index.rst`
- Fixed `bs_price_tensor` parameter order in documentation

### Added
- 36 new edge-case tests — coverage increased from 95% to 98% (243 total tests)
- `LIMITATIONS.md` — comprehensive known limitations documentation
- `DEPLOYMENT.md` — Docker, PyPI, and cloud deployment guide
- `Dockerfile` with production and dev targets
- `notebooks/tutorial.ipynb` — interactive Jupyter tutorial
- `benchmarks/RESULTS.md` — benchmark results with analysis
- `docs/limitations.rst` — limitations page in Sphinx docs
- TestPyPI step in publish workflow

### Changed
- Updated README with badges, correct test count (243), architecture tree (added `viz/`)
- Enhanced `publish.yml` with build → TestPyPI → PyPI pipeline
- Updated `CHANGELOG.md` with all v0.1.1 changes

## [0.1.0] — 2026-02-24

### Added

**Core Engine**
- `Tensor` class with reverse-mode automatic differentiation
- 16 differentiable operations: add, sub, mul, div, neg, matmul, pow, exp, log, sqrt, sum, mean, reshape, transpose, maximum, norm_cdf
- Gradient accumulation for fan-out nodes (correct handling of shared nodes)
- `numerical_gradient` and `check_grad` utilities for gradient validation

**Financial Engine**
- `bs_price_numpy` — analytic Black-Scholes pricing (NumPy, ground truth)
- `bs_price_tensor` — Tensor-based Black-Scholes (autograd-compatible)
- Analytic Greeks: `bs_delta`, `bs_gamma`, `bs_vega`, `bs_theta`, `bs_rho`
- `compute_greeks` — autograd Delta + Vega + finite-diff Gamma
- `simulate_basket` — Monte Carlo basket option pricing with correlated GBM
- `build_pricing_grid` — MC-based pricing grid construction
- `build_pricing_grid_analytic` — fast analytic approximation pricing grid

**TT Compression Engine**
- `tt_svd` — Oseledets (2011) TT-SVD with adaptive rank truncation
- `tt_round` — TT-rounding via QR + SVD two-pass sweep
- `tt_eval` / `tt_eval_batch` — O(d·r²) single and batch evaluation
- `tt_to_full` — full tensor reconstruction (for validation)
- `tt_ranks`, `tt_memory`, `tt_error`, `tt_compression_ratio` — diagnostics

**TT Surrogate**
- `TTSurrogate` — full pipeline: grid → TT-SVD → multi-linear interpolation
- Constructors: `from_grid`, `from_basket_analytic`, `from_basket_mc`
- `evaluate` / `evaluate_tensor` — fast pricing with optional autograd
- `greeks` — Delta and Gamma via finite differences
- `summary` / `print_summary` — compression diagnostics

**Tests**
- 133 tests, 100% passing across all modules

**Examples & Benchmarks**
- `examples/demo_basket_tt.py` — end-to-end 2/3/5-asset demo
- `benchmarks/bench_tt_vs_mc.py` — compression, speed, memory scaling benchmarks
