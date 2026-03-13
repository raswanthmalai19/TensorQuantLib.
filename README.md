# TensorQuantLib

[![CI](https://github.com/raswanthmalai19/TensorQuantLib/actions/workflows/ci.yml/badge.svg)](https://github.com/raswanthmalai19/TensorQuantLib/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Coverage: 98%](https://img.shields.io/badge/coverage-98%25-brightgreen.svg)](https://github.com/raswanthmalai19/TensorQuantLib)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)
[![Typed: mypy strict](https://img.shields.io/badge/typed-mypy%20strict-blue.svg)](https://mypy-lang.org/)
[![Tests: 353](https://img.shields.io/badge/tests-353%20passing-brightgreen.svg)](https://github.com/raswanthmalai19/TensorQuantLib/actions)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://raswanthmalai19.github.io/TensorQuantLib./)

**A comprehensive quantitative finance library with tensor-train compression, automatic differentiation, and stochastic models — built from scratch with NumPy and SciPy.**

TensorQuantLib provides a complete toolkit for derivatives pricing, risk management, and portfolio analysis. It compresses high-dimensional pricing surfaces using **Tensor-Train (TT) decomposition**, prices options with **Black-Scholes, Heston, and Monte Carlo engines**, and computes Greeks via a custom **reverse-mode autodiff engine**.

> **No PyTorch. No TensorFlow. No JAX.** Custom autograd, custom TT-SVD, custom stochastic models — built entirely with NumPy and SciPy.

---

## 📖 Documentation

Complete documentation with API reference, tutorials, and examples is available at:
**[https://raswanthmalai19.github.io/TensorQuantLib./](https://raswanthmalai19.github.io/TensorQuantLib./)**

### Quick Links
- [**API Reference**](https://raswanthmalai19.github.io/TensorQuantLib./api.html) — Full module documentation
- [**Quick Start**](https://raswanthmalai19.github.io/TensorQuantLib./quickstart.html) — Getting started guide
- [**Theory**](https://raswanthmalai19.github.io/TensorQuantLib./theory.html) — Mathematical foundations
- [**Examples**](https://raswanthmalai19.github.io/TensorQuantLib./quickstart.html#examples) — Code examples and tutorials
- [**Changelog**](https://raswanthmalai19.github.io/TensorQuantLib./changelog.html) — Version history

---

## Features

| Module | Feature | Description |
|--------|---------|-------------|
| **Core** | Autograd Engine | Reverse-mode AD — 23 differentiable ops (incl. sin, cos, tanh, softmax) |
| **Black-Scholes** | Analytic Pricing | BS pricing + full Greeks (Delta, Gamma, Vega, Theta, Rho) |
| **Implied Vol** | IV Solvers | Brent, Newton-Raphson, batch IV, volatility surface builder |
| **Heston** | Stochastic Vol | Semi-analytic (Gil-Pelaez CF), QE Monte Carlo, calibration |
| **American** | Early Exercise | Longstaff-Schwartz LSM, exercise boundary, Greeks |
| **Exotics** | Exotic Options | Asian (arith/geo), Digital (cash/asset), Barrier (8 types) |
| **Variance Reduction** | MC Efficiency | Antithetic, control variate, QMC (Sobol), importance sampling, stratified |
| **Risk** | Risk Metrics | VaR (3 methods), CVaR, scenario analysis, portfolio risk |
| **TT Compression** | Tensor Trains | TT-SVD, rounding, arithmetic, surrogate pricing |
| **Basket Options** | Multi-Asset | Correlated GBM Monte Carlo, grid construction |
| **Visualization** | Plots | Pricing surfaces, Greek surfaces, TT-rank charts |
| **CLI** | Command Line | `python -m tensorquantlib` — info, price, greeks, demo |

---

## Installation

### pip install (recommended)

```bash
pip install git+https://github.com/raswanthmalai19/TensorQuantLib.git
```

### With optional dependencies

```bash
# Visualization support
pip install "git+https://github.com/raswanthmalai19/TensorQuantLib.git#egg=tensorquantlib[viz]"

# Market data support
pip install "git+https://github.com/raswanthmalai19/TensorQuantLib.git#egg=tensorquantlib[data]"

# Everything
pip install "git+https://github.com/raswanthmalai19/TensorQuantLib.git#egg=tensorquantlib[all]"
```

### Clone & install (development)

```bash
git clone https://github.com/raswanthmalai19/TensorQuantLib.git
cd TensorQuantLib
pip install -e ".[dev]"
python -m pytest tests/ -q   # 353 passed
```

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
from tensorquantlib.finance.heston import heston_price, heston_price_mc

# Semi-analytic (characteristic function)
price = heston_price(S=100, K=100, T=1.0, r=0.05, v0=0.04,
                     kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)

# Monte Carlo with QE scheme
mc_price, mc_se = heston_price_mc(S=100, K=100, T=1.0, r=0.05, v0=0.04,
                                   kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7,
                                   n_paths=100_000, scheme='qe')
```

### American Options (Longstaff-Schwartz)

```python
from tensorquantlib.finance.american import american_option_lsm

price, se = american_option_lsm(S=100, K=100, T=1.0, r=0.05, sigma=0.2,
                                 option_type='put', n_paths=100_000, n_steps=100)
```

### Implied Volatility

```python
from tensorquantlib.finance.implied_vol import implied_vol_brent, build_iv_surface

iv = implied_vol_brent(market_price=10.45, S=100, K=100, T=1.0, r=0.05)
# iv ≈ 0.20
```

### Exotic Options

```python
from tensorquantlib.finance.exotics import asian_option_price, barrier_option_price

# Arithmetic Asian call
asian_price, se = asian_option_price(S=100, K=100, T=1.0, r=0.05, sigma=0.2,
                                      option_type='call', asian_type='arithmetic')

# Down-and-out call with barrier at 90
barrier_price = barrier_option_price(S=100, K=100, T=1.0, r=0.05, sigma=0.2,
                                      barrier=90, option_type='call',
                                      barrier_type='down-and-out')
```

### Risk Metrics

```python
from tensorquantlib.finance.risk import var_parametric, cvar
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

### Memory Scaling (15 pts/axis, eps=1e-3)

| Assets | Full Grid | TT Size | Compression |
|--------|-----------|---------|-------------|
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
│   ├── tensor.py          # Tensor class — reverse-mode autograd, 23 ops
│   └── ops.py             # Public re-exports
├── finance/
│   ├── black_scholes.py   # Analytic + Tensor BS pricing, full Greeks
│   ├── implied_vol.py     # Brent/Newton IV, batch, surface builder
│   ├── heston.py          # Semi-analytic CF, QE MC, calibration, Greeks
│   ├── american.py        # Longstaff-Schwartz LSM, grid, Greeks
│   ├── exotics.py         # Asian, Digital, Barrier options
│   ├── variance_reduction.py  # Antithetic, CV, QMC, IS, stratified
│   ├── risk.py            # VaR, CVaR, scenario analysis, portfolio risk
│   ├── greeks.py          # Autograd-based Greeks
│   └── basket.py          # MC basket pricing, grid construction
├── tt/
│   ├── decompose.py       # TT-SVD, TT-rounding
│   ├── ops.py             # tt_eval, tt_to_full, arithmetic
│   └── surrogate.py       # TTSurrogate end-to-end pipeline
├── viz/
│   └── plots.py           # Pricing surfaces, Greek surfaces, rank plots
├── utils/
│   └── validation.py      # Numerical gradient checking
└── __main__.py            # CLI entry point
```

---

## Test Coverage

| Module | Tests |
|--------|-------|
| Core autograd engine (incl. new ops) | 73 |
| Black-Scholes + Greeks | 32 |
| Implied volatility | 18 |
| Heston model | 22 |
| American options | 15 |
| Exotic options | 28 |
| Variance reduction | 20 |
| Risk metrics | 18 |
| TT decomposition & ops | 34 |
| TT surrogate | 15 |
| Integration & edge cases | 57 |
| Visualization | 21 |
| **Total** | **353 (98% coverage)** |

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=tensorquantlib --cov-report=term-missing
```

---

## API Reference

```python
from tensorquantlib import (
    # Core
    Tensor,
    # Black-Scholes
    bs_price_numpy, bs_price_tensor,
    bs_delta, bs_gamma, bs_vega, bs_theta, bs_rho,
    compute_greeks, compute_greeks_vectorized,
    # Implied Volatility
    implied_vol_brent, implied_vol_newton, implied_vol_batch, build_iv_surface,
    # Heston
    heston_price, heston_price_mc, heston_calibrate, heston_greeks,
    # American Options
    american_option_lsm, american_option_grid, american_option_greeks,
    # Exotic Options
    asian_option_price, digital_option_price, barrier_option_price,
    # Variance Reduction
    mc_antithetic, mc_control_variate, mc_quasi_monte_carlo,
    mc_importance_sampling, mc_stratified, compare_variance_reduction,
    # Risk
    var_parametric, var_historical, var_monte_carlo, cvar,
    scenario_analysis, OptionPosition, PortfolioRisk,
    # Basket Options
    simulate_basket, build_pricing_grid, build_pricing_grid_analytic,
    # TT Compression
    TTSurrogate,
    tt_svd, tt_round,
    tt_eval, tt_eval_batch, tt_to_full,
    tt_ranks, tt_memory, tt_error, tt_compression_ratio,
    tt_add, tt_scale, tt_hadamard, tt_dot, tt_frobenius_norm,
    # Visualization
    plot_pricing_surface, plot_greeks_surface, plot_tt_ranks,
)
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | >= 1.24 | Core numerics |
| `scipy` | >= 1.10 | Special functions, optimization |
| `matplotlib` | >= 3.7 | Visualization (optional `[viz]`) |
| `yfinance` | >= 0.2 | Market data (optional `[data]`) |

---

## References

- Oseledets, I.V. (2011). *Tensor-Train Decomposition*. SIAM J. Sci. Comput. 33(5).
- Heston, S.L. (1993). *A Closed-Form Solution for Options with Stochastic Volatility*.
- Andersen, L.B.G. (2008). *Efficient Simulation of the Heston Stochastic Volatility Model*.
- Longstaff, F.A. & Schwartz, E.S. (2001). *Valuing American Options by Simulation*.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*.

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development setup, coding standards, and guidelines.

---

## License

MIT — see [`LICENSE`](LICENSE).

---

## Roadmap

- [x] Custom autograd engine (23 ops)
- [x] Black-Scholes pricing + Greeks
- [x] TT-SVD compression + surrogate pricing
- [x] Heston stochastic volatility
- [x] American options (LSM)
- [x] Exotic options (Asian, Digital, Barrier)
- [x] Variance reduction techniques
- [x] Risk metrics (VaR, CVaR)
- [x] CLI interface
- [ ] Volatility surface models (SABR, SVI)
- [ ] Interest rate models (Vasicek, CIR)
- [ ] FX options (Garman-Kohlhagen, Quanto)
- [ ] Credit risk (Merton, CDS)
- [ ] Market data integration (yfinance)
- [ ] Backtesting framework
- [ ] PyPI release
- [ ] GPU acceleration via CuPy

---

*Built from scratch — no ML framework dependencies. Custom autograd + custom TT-SVD + custom interpolation.*
# TensorQuantLib.
