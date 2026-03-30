# TensorQuantLib

[![PyPI](https://img.shields.io/pypi/v/tensorquantlib.svg)](https://pypi.org/project/tensorquantlib/)
[![CI](https://github.com/raswanthmalai19/TensorQuantLib./actions/workflows/ci.yml/badge.svg)](https://github.com/raswanthmalai19/TensorQuantLib./actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Coverage: 98%](https://img.shields.io/badge/coverage-98%25-brightgreen.svg)](https://github.com/raswanthmalai19/TensorQuantLib.)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)
[![Typed: mypy strict](https://img.shields.io/badge/typed-mypy%20strict-blue.svg)](https://mypy-lang.org/)
[![Tests: 698](https://img.shields.io/badge/tests-698%20passing-brightgreen.svg)](https://github.com/raswanthmalai19/TensorQuantLib./actions)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://raswanthmalai19.github.io/TensorQuantLib/)

**A comprehensive quantitative finance library with tensor-train compression, automatic differentiation, and stochastic models — built from scratch with NumPy and SciPy.**

TensorQuantLib provides a complete toolkit for derivatives pricing, risk management, and portfolio analysis. It implements:

- **Black-Scholes** with instant analytic Greeks
- **Heston stochastic volatility** via characteristic functions (10-200x faster than Monte Carlo)
- **American & Exotic options** via Longstaff-Schwartz and variance-reduced Monte Carlo
- **High-dimensional pricing** via Tensor-Train compression (100-1000x speedup for repeated surface evals)
- **Risk analytics** with Greeks via custom reverse-mode autodiff engine (first and second-order)

> **No PyTorch. No TensorFlow. No CUDA dependency.** Custom autograd, custom TT-SVD, custom stochastic models — built entirely with NumPy and SciPy. Runs on any platform, costs nothing to deploy.

---

## 📚 Documentation

**Complete API reference, tutorials, and guides:**

| Resource | Link | Purpose |
|----------|------|---------|
| **Main Site** | [raswanthmalai19.github.io/TensorQuantLib/](https://raswanthmalai19.github.io/TensorQuantLib/) | Full documentation with tutorials |
| **Quick Start** | [Getting Started in 5 min](https://raswanthmalai19.github.io/TensorQuantLib/quickstart.html) | Installation and first example |
| **API Reference** | [Complete API docs](https://raswanthmalai19.github.io/TensorQuantLib/api.html) | All functions and classes |
| **Theory** | [Math & algorithms](https://raswanthmalai19.github.io/TensorQuantLib/theory.html) | Mathematical foundations |
| **Performance** | [Benchmarks & tuning](https://raswanthmalai19.github.io/TensorQuantLib/performance.html) | Latency, optimization, production guide |
| **Limitations** | [What to know](https://raswanthmalai19.github.io/TensorQuantLib/limitations.html) | Current limitations and roadmap |

### Other Links
- **[PyPI Package](https://pypi.org/project/tensorquantlib/)** — `pip install tensorquantlib`
- **[GitHub Repository](https://github.com/raswanthmalai19/TensorQuantLib.)** — Source code and issues
- **[Changelog](https://raswanthmalai19.github.io/TensorQuantLib/changelog.html)** — Version history

---

## Features

| Module | Capability | Description |
|--------|-----------|-------------|
| **Core** | Autograd Engine | Reverse-mode AD — 23+ differentiable ops (sin, cos, tanh, softmax, clip, where, abs) |
| **Core** | Second-Order AD | Hessians, HVPs, mixed partials, Gamma/Vanna/Volga via autodiff |
| **Black-Scholes** | Analytic Pricing | BS call/put + full Greeks (Delta, Gamma, Vega, Theta, Rho) |
| **Implied Vol** | IV Solvers | Brent, Newton-Raphson, batch IV, volatility surface builder |
| **Heston** | Stochastic Vol | Semi-analytic (Gil-Pelaez CF), QE Monte Carlo, calibration |
| **American** | Early Exercise | Longstaff-Schwartz LSM, exercise boundary, Greeks |
| **Exotics** | Exotic Options | Asian (arithmetic/geometric), Digital (cash/asset), Barrier (8 types), Lookback, Cliquet, Rainbow |
| **Volatility** | Vol Surface Models | SABR (Hagan 2002), SVI (Gatheral 2004), calibration to market data |
| **Rates** | Interest Rate Models | Vasicek, CIR, Nelson-Siegel, yield curve bootstrapping |
| **FX** | FX Derivatives | Garman-Kohlhagen, FX Greeks, FX forwards, quanto options |
| **Credit** | Credit Risk | Merton structural model, CDS pricing, hazard rates, survival probabilities |
| **Jump-Diffusion** | Jump Models | Merton jump-diffusion, Kou double-exponential |
| **Local Vol** | Local Volatility | Dupire local vol, local vol Monte Carlo |
| **IR Derivatives** | Rate Products | Black-76 caps/floors, swaptions, swap rates, swaption parity |
| **Variance Reduction** | MC Efficiency | Antithetic, control variate, QMC (Sobol), importance sampling, stratified |
| **Risk** | Risk Metrics | VaR (parametric/historical/MC), CVaR, scenario analysis, portfolio Greeks |
| **Backtesting** | Strategy Testing | Backtesting engine, strategy framework, performance metrics (Sharpe, Sortino, max drawdown) |
| **TT Compression** | Tensor Trains | TT-SVD, TT-cross, rounding, arithmetic, surrogate pricing |
| **Basket Options** | Multi-Asset | Correlated GBM Monte Carlo, analytic moment-matching (Gentle 1993) |
| **Visualization** | Plots | Pricing surfaces, Greek surfaces, TT-rank charts |
| **Data** | Market Data | Yahoo Finance integration, historical prices, options chains |
| **CLI** | Command Line | `python -m tensorquantlib` — info, price, greeks, demo |

---

## Installation

### pip install from PyPI (recommended)

```bash
pip install tensorquantlib
```

### With optional dependencies

```bash
# Visualization support
pip install tensorquantlib[viz]

# Market data support
pip install tensorquantlib[data]

# Everything
pip install tensorquantlib[all]
```

### pip install from GitHub (latest development version)

```bash
pip install git+https://github.com/raswanthmalai19/TensorQuantLib..git
```

### Clone & install (development)

```bash
git clone https://github.com/raswanthmalai19/TensorQuantLib..git
cd TensorQuantLib.
pip install -e ".[dev]"
python -m pytest tests/ -q   # 698 passed
```

---

## Performance at a Glance

> Measured on Apple M1. Numbers scale linearly on multi-core servers.

| Workflow | Latency | Notes |
|---|---|---|
| `black_scholes` / `bs_greeks` | **< 5 µs** | Analytic; production vanilla book default |
| `barrier_price` | **< 5 µs** | Rubinstein-Reiner closed form |
| `garman_kohlhagen` | **< 5 µs** | FX vanilla |
| `vasicek_bond_price` | **< 1 µs** | Closed-form A(T)/B(T) |
| `implied_vol` | **< 1 ms** | Brent solver |
| `heston_price` (CF) | **~1 ms** | 100-pt Gaussian quadrature; **10-200x faster than Heston MC** |
| `TTSurrogate.evaluate` 3D | **1.5 µs** | After one-time 2 ms build; repeated evals |
| `TTSurrogate.evaluate` 5D | **~5 µs** | 42× memory compression vs full grid; **100-1000x faster than MC for repeated evals** |
| `heston_price_mc` | 200–500 ms | Validation only; use CF for live pricing |
| `HestonCalibrator.fit` | 5–15 s default → **< 0.5 s** optimised | See [Performance Guide](docs/performance.rst) |
| `american_option_lsm` | 100–500 ms | Longstaff-Schwartz Monte Carlo |
| `asian_price_mc` | 100–400 ms | Monte Carlo; use `asian_price_cv` for 10x variance reduction |

### Speed Gains Explained

- **Analytic pricers** (Black-Scholes, Vasicek): instant microsecond pricing
- **Heston via characteristic functions**: **10-200x faster than Heston Monte Carlo**
- **TT Surrogates for repeated pricing**: Build surface once, then query repeated evals at microsecond latency — **100-1000x faster than re-running Monte Carlo each time**
- **Monte Carlo methods** (American, Asian, exotic): Efficient but inherently slower; use for features that require it

**Rule of thumb**: for anything that needs a price in a hot loop, use the analytic
or CF pricer.  Build a `TTSurrogate` once at startup to replace any repeated MC pricer with
µs-latency evaluation.  See the full
[Performance & Production Guide](docs/performance.rst) for tuning knobs,
parallel calibration, memory profiling, and the production checklist.

---

## Quick Start

### Black-Scholes with Autodiff Greeks

```python
from tensorquantlib import Tensor, bs_price_tensor
import numpy as np

S = Tensor(np.array([100.0]), requires_grad=True)
price = bs_price_tensor(S, K=100, T=1.0, r=0.05, sigma=0.2)
price.backward()

print(f"Price: {price.item():.4f}")        # 10.4506
print(f"Delta: {S.grad[0]:.4f}")           # 0.6368
```

### Heston Stochastic Volatility

```python
from tensorquantlib import HestonParams, heston_price, heston_price_mc

params = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)

# Semi-analytic (characteristic function)
price = heston_price(S=100, K=100, T=1.0, r=0.05, params=params)

# Monte Carlo with QE scheme
mc_price, mc_se = heston_price_mc(
    S=100, K=100, T=1.0, r=0.05, params=params,
    n_paths=100_000, scheme='qe', return_stderr=True,
)
```

### American Options (Longstaff-Schwartz)

```python
from tensorquantlib import american_option_lsm

price, se = american_option_lsm(
    S=100, K=100, T=1.0, r=0.05, sigma=0.2,
    option_type='put', n_paths=100_000, n_steps=100,
)
```

### Implied Volatility

```python
from tensorquantlib import implied_vol, iv_surface
import numpy as np

iv = implied_vol(market_price=10.45, S=100, K=100, T=1.0, r=0.05)
# iv ≈ 0.20

# Build a full IV surface
strikes = np.linspace(80, 120, 9)
expiries = np.array([0.25, 0.5, 1.0])
surface = iv_surface(S=100, r=0.05, sigma=0.2, strikes=strikes, expiries=expiries)
```

### Exotic Options

```python
from tensorquantlib import asian_price_mc, barrier_price, digital_price

# Arithmetic Asian call
asian = asian_price_mc(S=100, K=100, T=1.0, r=0.05, sigma=0.2)

# Cash-or-nothing digital
digital = digital_price(S=100, K=100, T=1.0, r=0.05, sigma=0.2)

# Down-and-out barrier call
barrier = barrier_price(
    S=100, K=100, T=1.0, r=0.05, sigma=0.2,
    barrier=90, barrier_type='down-and-out',
)
```

### Volatility Surface Models (SABR & SVI)

```python
from tensorquantlib import sabr_implied_vol, svi_implied_vol
import numpy as np

# SABR implied volatility
vol = sabr_implied_vol(F=100, K=100, T=1.0, alpha=0.3, beta=0.5, rho=-0.3, nu=0.4)

# SVI parameterization (k = log-moneyness ln(K/F))
k = np.linspace(-0.2, 0.2, 50)  # log-moneyness grid
vols = svi_implied_vol(k, T=1.0, a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
```

### Interest Rate Models

```python
from tensorquantlib import vasicek_bond_price, cir_bond_price, nelson_siegel
import numpy as np

# Vasicek zero-coupon bond
bond = vasicek_bond_price(r0=0.05, kappa=0.3, theta=0.05, sigma=0.02, T=5.0)

# CIR bond price
cir = cir_bond_price(r0=0.05, kappa=0.5, theta=0.05, sigma=0.1, T=5.0)

# Nelson-Siegel yield curve
maturities = np.array([0.25, 0.5, 1, 2, 5, 10, 30])
yields = nelson_siegel(maturities, beta0=0.05, beta1=-0.02, beta2=0.03, tau=1.5)
```

### FX Options

```python
from tensorquantlib import garman_kohlhagen, fx_forward

# Garman-Kohlhagen European FX option
price = garman_kohlhagen(S=1.25, K=1.30, T=0.5, r_d=0.05, r_f=0.02, sigma=0.1)

# FX forward rate
fwd = fx_forward(S=1.25, r_d=0.05, r_f=0.02, T=1.0)
```

### Credit Risk

```python
from tensorquantlib import merton_default_prob, cds_spread

# Merton structural model — probability of default
pd = merton_default_prob(V=100, D=80, T=1.0, r=0.05, sigma_V=0.25)

# CDS spread (from constant hazard rate)
spread = cds_spread(hazard_rate=0.02, T=5.0, recovery=0.4, r=0.03)
```

### Second-Order Greeks (Autodiff)

```python
from tensorquantlib import second_order_greeks, bs_price_tensor

result = second_order_greeks(
    price_fn=bs_price_tensor,
    S=100, K=100, T=1.0, r=0.05, sigma=0.2,
)
# result = {'gamma': ..., 'vanna': ..., 'volga': ...}
```

### Risk Metrics

```python
from tensorquantlib import var_parametric, cvar
import numpy as np

returns = np.random.normal(0.0005, 0.02, 252)
var_95 = var_parametric(returns, confidence=0.95)
es_95 = cvar(returns, confidence=0.95)
```

### TT Compression Surrogate

```python
from tensorquantlib import TTSurrogate

surr = TTSurrogate.from_basket_analytic(
    S0_ranges=[(80, 120)] * 3,
    K=100, T=1.0, r=0.05,
    sigma=[0.20, 0.25, 0.30],
    weights=[1/3, 1/3, 1/3],
    n_points=30, eps=1e-4,
)

surr.print_summary()
price = surr.evaluate([100.0, 105.0, 95.0])
greeks = surr.greeks([100.0, 105.0, 95.0])
```

### CLI

```bash
python -m tensorquantlib info           # Library info
python -m tensorquantlib price          # Price an option
python -m tensorquantlib greeks         # Compute Greeks
python -m tensorquantlib demo           # Run quick demo
```

---

## Benchmark Results

### TT Compression — Memory Scaling

| Assets | Full Grid | TT Size | Compression Ratio |
|--------|-----------|---------|-------------------|
| 2 | 0.002 MB | 3.3 KB | 1x |
| 3 | 0.026 MB | 28 KB | 1x |
| 4 | 0.39 MB | 91 KB | **4x** |
| **5** | **5.79 MB** | **142 KB** | **42x** |

Full numbers: [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md)

---

## Architecture

```
tensorquantlib/
├── core/
│   ├── tensor.py            # Tensor class — reverse-mode autograd, 23+ ops
│   ├── second_order.py      # Hessians, HVPs, Gamma/Vanna/Volga autodiff
│   └── ops.py               # Public re-exports
├── finance/
│   ├── black_scholes.py     # Analytic + Tensor BS pricing, full Greeks
│   ├── implied_vol.py       # Brent/Newton IV, batch, surface builder
│   ├── heston.py            # Semi-analytic CF, QE MC, calibration, Greeks
│   ├── american.py          # Longstaff-Schwartz LSM, grid, Greeks
│   ├── exotics.py           # Asian, Digital, Barrier, Lookback, Cliquet, Rainbow
│   ├── volatility.py        # SABR & SVI vol surface models + calibration
│   ├── rates.py             # Vasicek, CIR, Nelson-Siegel, yield bootstrapping
│   ├── fx.py                # Garman-Kohlhagen FX options, forwards, quanto
│   ├── credit.py            # Merton structural model, CDS pricing
│   ├── jump_diffusion.py    # Merton & Kou jump-diffusion
│   ├── local_vol.py         # Dupire local vol, local vol MC
│   ├── ir_derivatives.py    # Black-76 caps/floors, swaptions
│   ├── variance_reduction.py  # Antithetic, CV, QMC, IS, stratified
│   ├── risk.py              # VaR, CVaR, scenario analysis, portfolio risk
│   ├── greeks.py            # Autograd-based Greeks
│   └── basket.py            # MC basket pricing, analytic grid construction
├── tt/
│   ├── decompose.py         # TT-SVD, TT-cross, TT-rounding
│   ├── ops.py               # tt_eval, tt_to_full, arithmetic (12 ops)
│   ├── surrogate.py         # TTSurrogate end-to-end pipeline
│   └── pricing.py           # Pre-built TT surrogates (Heston, American, exotic)
├── backtest/
│   ├── engine.py            # Backtesting engine with portfolio tracking
│   ├── strategy.py          # Strategy base class + built-in strategies
│   └── metrics.py           # Sharpe, Sortino, max drawdown, Calmar, etc.
├── data/
│   └── market.py            # Yahoo Finance integration, options chains
├── viz/
│   └── plots.py             # Pricing surfaces, Greek surfaces, rank plots
├── utils/
│   └── validation.py        # Numerical gradient checking
└── __main__.py              # CLI entry point
```

---

## Test Coverage

| Module | Tests | Description |
|--------|-------|-------------|
| Core autograd engine | 73 | Forward/backward pass, all 23+ ops, broadcasting |
| Second-order autodiff | 12 | Hessian, HVP, mixed partials, second-order Greeks |
| Black-Scholes + Greeks | 32 | Pricing, put-call parity, all 5 Greeks |
| Implied volatility | 18 | Brent, Newton, batch, surface, edge cases |
| Heston model | 22 | Semi-analytic, MC (QE/Euler), calibration, Greeks |
| American options | 15 | LSM, grid, Greeks, put-call comparison |
| Exotic options | 28 | Asian, digital, barrier, lookback, cliquet, rainbow |
| Volatility models | 14 | SABR, SVI, calibration, surface construction |
| Interest rates | 20 | Vasicek, CIR, Nelson-Siegel, bootstrapping |
| FX options | 12 | Garman-Kohlhagen, Greeks, forwards, quanto |
| Credit risk | 14 | Merton, CDS, hazard rates, survival probabilities |
| Jump-diffusion | 10 | Merton, Kou, analytic vs MC |
| Local volatility | 8 | Dupire, local vol MC |
| IR derivatives | 14 | Caps, floors, swaptions, swap rate |
| Variance reduction | 20 | Antithetic, CV, QMC, IS, stratified |
| Risk metrics | 18 | VaR (3 methods), CVaR, scenarios, portfolio |
| Backtesting | 12 | Engine, strategies, metrics |
| TT decomposition & ops | 34 | SVD, cross, rounding, arithmetic (12 ops) |
| TT surrogate & pricing | 20 | Surrogate construction, evaluation, Greeks |
| Visualization | 21 | Pricing surfaces, Greek surfaces, rank plots |
| Integration & edge cases | 57 | Cross-module, boundary conditions, numerics |
| **Total** | **588** | **98% line coverage** |

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=tensorquantlib --cov-report=term-missing
```

---

## API Reference

150+ public functions and classes exported from the top-level package:

```python
from tensorquantlib import (
    # Core autograd
    Tensor,
    tensor_sin, tensor_cos, tensor_tanh, tensor_abs,
    tensor_clip, tensor_where, tensor_softmax,

    # Second-order autodiff
    hvp, hessian, hessian_diag, vhp, mixed_partial,
    gamma_autograd, vanna_autograd, volga_autograd, second_order_greeks,

    # Black-Scholes
    bs_price_numpy, bs_price_tensor,
    bs_delta, bs_gamma, bs_vega, bs_theta, bs_rho,
    compute_greeks, compute_greeks_vectorized,

    # Implied volatility
    implied_vol, implied_vol_batch, implied_vol_nr, iv_surface,

    # Heston
    HestonParams, HestonCalibrator,
    heston_price, heston_price_mc, heston_greeks,

    # American options
    american_option_lsm, american_option_grid, american_greeks,

    # Exotics
    asian_price_mc, asian_geometric_price,
    digital_price, digital_price_mc, digital_greeks,
    barrier_price, barrier_price_mc,
    lookback_fixed_analytic, lookback_floating_analytic, lookback_price_mc,
    cliquet_price_mc, rainbow_price_mc,

    # Volatility surface
    sabr_implied_vol, sabr_calibrate,
    svi_raw, svi_implied_vol, svi_calibrate, svi_surface,

    # Interest rates
    vasicek_bond_price, vasicek_yield, vasicek_option_price, vasicek_simulate,
    cir_bond_price, cir_yield, cir_simulate, feller_condition,
    nelson_siegel, nelson_siegel_calibrate, bootstrap_yield_curve,

    # FX options
    garman_kohlhagen, gk_greeks, fx_forward, quanto_option,

    # Credit risk
    merton_default_prob, merton_credit_spread,
    survival_probability, hazard_rate_from_spread,
    cds_spread, cds_price,

    # Jump-diffusion
    merton_jump_price, merton_jump_price_mc, kou_jump_price_mc,

    # Local volatility
    dupire_local_vol, local_vol_mc,

    # IR derivatives (Black-76)
    black76_caplet, black76_floorlet, cap_price, floor_price,
    swap_rate, swaption_price, swaption_parity,

    # Variance reduction
    bs_price_antithetic, asian_price_cv, bs_price_qmc,
    bs_price_importance, bs_price_stratified, compare_variance_reduction,

    # Risk
    PortfolioRisk, OptionPosition,
    var_parametric, var_historical, var_mc, cvar,
    scenario_analysis, greeks_portfolio,

    # Basket
    simulate_basket, build_pricing_grid, build_pricing_grid_analytic,

    # TT compression
    TTSurrogate,
    tt_svd, tt_round, tt_cross,
    tt_eval, tt_eval_batch, tt_to_full,
    tt_ranks, tt_memory, tt_error, tt_compression_ratio,
    tt_add, tt_scale, tt_hadamard, tt_dot, tt_frobenius_norm,

    # TT-accelerated pricers
    heston_surrogate, american_surrogate,
    exotic_surrogate, jump_diffusion_surrogate,

    # Visualization
    plot_pricing_surface, plot_greeks_surface, plot_tt_ranks,
)
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | >= 1.24 | Core numerics |
| `scipy` | >= 1.10 | Special functions, optimization, integration |
| `matplotlib` | >= 3.7 | Visualization (optional `[viz]`) |
| `yfinance` | >= 0.2 | Market data (optional `[data]`) |

---

## References

- Oseledets, I.V. (2011). *Tensor-Train Decomposition*. SIAM J. Sci. Comput. 33(5).
- Heston, S.L. (1993). *A Closed-Form Solution for Options with Stochastic Volatility*. RFS 6(2).
- Andersen, L.B.G. (2008). *Efficient Simulation of the Heston Stochastic Volatility Model*. J. Comp. Fin.
- Longstaff, F.A. & Schwartz, E.S. (2001). *Valuing American Options by Simulation*. RFS.
- Hagan, P.S. et al. (2002). *Managing Smile Risk*. Wilmott Magazine. (SABR model)
- Gatheral, J. (2004). *A Parsimonious Arbitrage-Free Implied Volatility Parameterization*. (SVI)
- Vasicek, O.A. (1977). *An Equilibrium Characterization of the Term Structure*. J. Fin. Econ.
- Cox, J.C., Ingersoll, J.E. & Ross, S.A. (1985). *A Theory of the Term Structure of Interest Rates*. Econometrica.
- Merton, R.C. (1974). *On the Pricing of Corporate Debt*. J. Finance.
- Garman, M.B. & Kohlhagen, S.W. (1983). *Foreign Currency Option Values*. J. Int. Money & Finance.
- Black, F. (1976). *The Pricing of Commodity Contracts*. J. Fin. Econ.
- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*. J. Pol. Econ.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development setup, coding standards, and guidelines.

---

## License

MIT — see [`LICENSE`](LICENSE).

---

## Roadmap

- [x] Custom autograd engine (23+ differentiable ops)
- [x] Second-order autodiff (Hessian, HVP, Gamma/Vanna/Volga)
- [x] Black-Scholes pricing + full Greeks
- [x] TT-SVD compression + surrogate pricing
- [x] Heston stochastic volatility (analytic + MC + calibration)
- [x] American options (Longstaff-Schwartz LSM)
- [x] Exotic options (Asian, Digital, Barrier, Lookback, Cliquet, Rainbow)
- [x] Variance reduction techniques (5 methods + comparison)
- [x] Risk metrics (VaR, CVaR, scenario analysis)
- [x] Volatility surface models (SABR, SVI + calibration)
- [x] Interest rate models (Vasicek, CIR, Nelson-Siegel)
- [x] FX options (Garman-Kohlhagen, forwards, quanto)
- [x] Credit risk (Merton structural, CDS pricing)
- [x] Jump-diffusion models (Merton, Kou)
- [x] Local volatility (Dupire, MC)
- [x] IR derivatives (Black-76 caps/floors, swaptions)
- [x] Market data integration (Yahoo Finance)
- [x] Backtesting framework (engine, strategies, metrics)
- [x] CLI interface
- [x] PyPI release
- [ ] GPU acceleration via CuPy

---

*Built from scratch — no ML framework dependencies. Custom autograd + custom TT-SVD + custom stochastic models.*
