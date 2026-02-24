# TensorQuantLib

[![CI](https://github.com/raswanthmalai19/TensorQuantLib/actions/workflows/ci.yml/badge.svg)](https://github.com/raswanthmalai19/TensorQuantLib/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Coverage: 98%](https://img.shields.io/badge/coverage-98%25-brightgreen.svg)](https://github.com/raswanthmalai19/TensorQuantLib)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)
[![Typed: mypy strict](https://img.shields.io/badge/typed-mypy%20strict-blue.svg)](https://mypy-lang.org/)
[![Tests: 243](https://img.shields.io/badge/tests-243%20passing-brightgreen.svg)](https://github.com/raswanthmalai19/TensorQuantLib/actions)

**A Tensor-Train surrogate pricing engine with automatic differentiation for multi-asset options — runs on a laptop, prices like a cluster.**

TensorQuantLib solves a real quant finance problem: pricing multi-asset basket options fast. It compresses high-dimensional pricing surfaces using **Tensor-Train (TT) decomposition**, cutting memory by up to **42x** and evaluation time by orders of magnitude, while computing Greeks via a from-scratch **reverse-mode autodiff engine**.

> **No PyTorch. No TensorFlow. No JAX.** Custom autograd, custom TT-SVD, custom interpolation — built entirely with NumPy and SciPy.

---

## The Problem It Solves

Pricing a 5-asset basket option via Monte Carlo at 100,000 paths takes ~30 seconds per point. For a risk desk that needs 50,000 different spot combinations that is 17 days of compute.

**TensorQuantLib's approach:**

```
1. Build the pricing surface once on a structured grid    (~10 seconds)
2. Compress with TT-SVD: 5-asset 15-pt/axis -> 5.8 MB -> 142 KB (42x)
3. Evaluate any new point via multi-linear interpolation  (microseconds)
4. Compute Delta and Gamma via automatic differentiation  (exact)
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Custom Autograd Engine** | Reverse-mode AD — 16+ differentiable ops, tape-based graph |
| **Black-Scholes Pricing** | Analytic pricing + full Greeks (Delta, Gamma, Vega, Theta, Rho) |
| **Basket Option Monte Carlo** | Correlated GBM with Cholesky decomposition |
| **TT-SVD Compression** | Oseledets (2011) algorithm, adaptive rank truncation |
| **TT-Surrogate Pricing** | Multi-linear interpolation on compressed TT-cores |
| **TT Arithmetic** | Add, scale, Hadamard product, dot product, Frobenius norm |
| **Visualization** | Pricing surfaces, Greek surfaces, TT-rank bar charts |
| **243 tests, 98% coverage** | Tested edge cases, integration, gradients |

---

## Installation

### pip install (recommended — works in Google Colab, Jupyter, or anywhere)

```bash
pip install git+https://github.com/raswanthmalai19/TensorQuantLib..git
```

That's it. No cloning required.

---

### Google Colab

Paste this into the first cell of any Colab notebook:

```python
!pip install git+https://github.com/raswanthmalai19/TensorQuantLib..git

# Then import and use immediately:
import numpy as np
from tensorquantlib import TTSurrogate, Tensor, bs_price_tensor
```

---

### With visualization support

```bash
pip install "git+https://github.com/raswanthmalai19/TensorQuantLib..git#egg=tensorquantlib[viz]"
```

---

### Clone & install locally (for development)

```bash
git clone https://github.com/raswanthmalai19/TensorQuantLib..git
cd TensorQuantLib.

# Install with all dev dependencies
pip install -e ".[dev]"

# Verify everything works
python -m pytest tests/ -q
# 243 passed in ~2s
```

---

## Quick Start

### Build a Basket Option Surrogate

```python
import numpy as np
from tensorquantlib import TTSurrogate

surr = TTSurrogate.from_basket_analytic(
    S0_ranges=[(80, 120)] * 3,   # spot price range per asset
    K=100, T=1.0, r=0.05,
    sigma=[0.20, 0.25, 0.30],
    weights=[1/3, 1/3, 1/3],
    n_points=30,
    eps=1e-4,
)

surr.print_summary()
# Full grid:     27,000 entries (210.9 KB)
# TT memory:     28.0 KB  (7.5x smaller)
# Max TT-rank:   30
# Compress time: 2ms

price = surr.evaluate([100.0, 105.0, 95.0])
print(f"Price:  {price:.4f}")

greeks = surr.greeks([100.0, 105.0, 95.0])
print(f"Delta:  {greeks['delta']}")   # [0.3421, 0.3156, 0.2890]
print(f"Gamma:  {greeks['gamma']}")   # [0.0023, 0.0021, 0.0019]
```

### Autograd Greeks on Black-Scholes

```python
import numpy as np
from tensorquantlib import Tensor, bs_price_tensor

S = Tensor(np.array([100.0]), requires_grad=True)
price = bs_price_tensor(S, K=100, T=1.0, r=0.05, sigma=0.2)
price.backward()

print(f"Price:            {price.item():.6f}")    # 10.450584
print(f"Delta (autodiff): {S.grad[0]:.6f}")       # 0.636831 (exact)
```

### TT-SVD Compression

```python
import numpy as np
from tensorquantlib import tt_svd, tt_to_full, tt_ranks, tt_compression_ratio, tt_error

A = np.random.randn(20, 20, 20, 20)   # 128 KB

cores = tt_svd(A, eps=1e-4)

print(f"Ranks:        {tt_ranks(cores)}")
print(f"Compression:  {tt_compression_ratio(cores, A.shape):.1f}x")
print(f"Rel. error:   {tt_error(cores, A):.2e}")
```

### Monte Carlo Basket Pricing

```python
import numpy as np
from tensorquantlib import simulate_basket

price, stderr = simulate_basket(
    S0      = np.array([100.0, 100.0, 100.0]),
    K       = 100.0,
    T       = 1.0,
    r       = 0.05,
    sigma   = np.array([0.20, 0.25, 0.18]),
    corr    = np.array([[1.0, 0.3, 0.2],
                        [0.3, 1.0, 0.3],
                        [0.2, 0.3, 1.0]]),
    weights = np.array([1/3, 1/3, 1/3]),
    n_paths = 100_000,
)
print(f"Basket price: {price:.4f} +/- {stderr:.4f}")
```

---

## Use in Google Colab

```python
# Install directly from GitHub — no PyPI account needed
!pip install git+https://github.com/raswanthmalai19/TensorQuantLib.git

import tensorquantlib as tql
print(tql.__version__)  # 0.1.0

import numpy as np
from tensorquantlib import TTSurrogate

surr = TTSurrogate.from_basket_analytic(
    S0_ranges=[(80, 120)] * 3,
    K=100, T=1.0, r=0.05,
    sigma=[0.20, 0.25, 0.30],
    weights=[1/3, 1/3, 1/3],
    n_points=25, eps=1e-3,
)

price = surr.evaluate([100.0, 100.0, 100.0])
print(f"Price: {price:.4f}")
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

### Compression vs Accuracy (3-asset, 30 pts/axis)

| Tolerance | Max Rank | TT Size | Compression | Error |
|-----------|----------|---------|-------------|-------|
| 1e-01 | 2 | 1.9 KB | **112x** | 7.8e-02 |
| 1e-02 | 8 | 16.6 KB | **12.7x** | 7.6e-03 |
| 1e-03 | 23 | 123 KB | 1.7x | 9.4e-04 |
| 1e-04 | 30 | 225 KB | 0.9x | **4.7e-15** |

Full numbers: [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md). Run yourself:
```bash
python benchmarks/bench_tt_vs_mc.py
```

---

## Architecture

```
tensorquantlib/
├── core/
│   ├── tensor.py        # Tensor class — reverse-mode autograd, 16+ ops
│   └── ops.py           # Public re-exports
├── finance/
│   ├── black_scholes.py # Analytic + Tensor BS pricing, full Greeks
│   ├── greeks.py        # Autograd-based Greeks (Delta, Vega, Gamma)
│   └── basket.py        # Monte Carlo basket pricing, grid construction
├── tt/
│   ├── decompose.py     # TT-SVD algorithm, TT-rounding (QR + SVD)
│   ├── ops.py           # tt_eval, tt_to_full, tt_ranks, arithmetic
│   └── surrogate.py     # TTSurrogate — full end-to-end pipeline
├── viz/
│   └── plots.py         # Pricing surfaces, Greek surfaces, rank plots
└── utils/
    └── validation.py    # Numerical gradient checking
```

---

## How It Works

### Tensor-Train Decomposition

A d-dimensional pricing surface A in R^(n1 x ... x nd) is represented as:

    A(i1,...,id) = G1[i1] * G2[i2] * ... * Gd[id]

where each core Gk[ik] is an r_{k-1} x r_k matrix. The TT-ranks r_k are determined adaptively by the SVD truncation tolerance eps.

Storage scales as sum_k(r_{k-1} * n_k * r_k) — linear in d for bounded ranks, vs product_k(n_k) for the full tensor.

### Reverse-Mode Autodiff

The `Tensor` class builds a dynamic computation graph:
1. Every operation records inputs and a `_backward` closure  
2. `backward()` does a topological sort then a chain-rule accumulation sweep  
3. Gradients are un-broadcast where necessary

This gives exact first-order derivatives (Delta = dP/dS, Vega = dP/dsigma) for any pricing function built from the 16 supported ops.

---

## API Reference

```python
from tensorquantlib import (
    # Core
    Tensor,
    # Finance
    bs_price_numpy, bs_price_tensor,
    bs_delta, bs_gamma, bs_vega, bs_theta, bs_rho,
    simulate_basket, build_pricing_grid, build_pricing_grid_analytic,
    compute_greeks, compute_greeks_vectorized,
    # TT compression
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

## Test Coverage

| Module | Tests |
|--------|-------|
| Core autograd engine | 52 |
| Financial engine | 32 |
| TT decomposition & ops | 34 |
| TT surrogate | 15 |
| Edge cases & integration | 57 |
| Visualization | 16 |
| Examples & benchmarks | 37 |
| **Total** | **243 (98% coverage)** |

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=tensorquantlib --cov-report=term-missing
```

---

## Project Layout

```
TensorQuantLib/
├── src/tensorquantlib/    # Library source code
├── tests/                 # 243 pytest tests
├── benchmarks/            # Performance benchmarks + RESULTS.md
├── examples/              # 4 runnable demo scripts
├── notebooks/             # tutorial.ipynb
├── docs/                  # Sphinx documentation source
├── .github/workflows/     # CI + publish.yml (PyPI trusted publishing)
├── Dockerfile             # Production + dev container targets
├── DEPLOYMENT.md          # Docker, PyPI, AWS/GCP/Azure guide
├── LIMITATIONS.md         # Known limitations + roadmap
├── CONTRIBUTING.md        # Development setup and guidelines
└── pyproject.toml         # PEP 621 packaging
```

---

## Examples

```bash
python examples/demo_basket_tt.py        # Full basket option surrogate demo
python examples/autograd_intro.py        # Autograd engine walkthrough
python examples/black_scholes_greeks.py  # BS pricing and Greeks
python examples/tt_compression_demo.py  # TT compression showcase
jupyter notebook notebooks/tutorial.ipynb  # Interactive tutorial
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | >= 1.24 | Core numerics |
| `scipy` | >= 1.10 | Special functions (norm CDF) |
| `matplotlib` | >= 3.7 | Visualization (optional, `[viz]`) |

No PyTorch, TensorFlow, or JAX required.

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development setup, coding standards (Google-style docstrings, mypy strict, ruff), and how to add new TT operations or financial models.

---

## References

- Oseledets, I.V. (2011). *Tensor-Train Decomposition*. SIAM J. Sci. Comput. 33(5), 2295-2317.
- Baydin et al. (2018). *Automatic differentiation in machine learning: a survey*. JMLR 18(153).
- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*. JPE 81(3).
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.

---

## License

MIT — see [`LICENSE`](LICENSE).

---

## Roadmap

- [ ] PyPI release (`pip install tensorquantlib`)
- [ ] GPU acceleration via CuPy
- [ ] Second-order autodiff (Hessian-vector products)
- [ ] American option pricing (Longstaff-Schwartz)
- [ ] Stochastic volatility models (Heston)
- [ ] d > 5 assets via cross-approximation (TT-cross)

---

*Built from scratch — no ML framework dependencies. Custom autograd + custom TT-SVD + custom interpolation.*
# TensorQuantLib.
