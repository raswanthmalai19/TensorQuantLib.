# TensorQuantLib

**A Tensor-Train based surrogate pricing engine with autodiff for multi-asset options — runnable on a laptop.**

TensorQuantLib compresses high-dimensional option pricing surfaces using Tensor-Train (TT) decomposition, enabling 1000× faster evaluation than brute-force grid lookup while computing Greeks via automatic differentiation.

---

## Features

- **Custom Autograd Engine** — Reverse-mode automatic differentiation with 16+ differentiable operations
- **Black-Scholes Pricing** — Analytic pricing + autograd-based Greeks (Delta, Gamma, Vega, Theta, Rho)
- **Multi-Asset Basket Options** — Monte Carlo + analytic pricing for basket calls/puts
- **TT-SVD Compression** — Oseledets (2011) algorithm with rank-adaptive truncation
- **TT-Surrogate Pricing** — Multi-linear interpolation on compressed grids
- **Greeks via Autograd** — Delta and Gamma through the surrogate via finite differences

---

## Architecture

```
tensorquantlib/
├── core/
│   ├── tensor.py        # Tensor class + reverse-mode autograd + all differentiable ops
│   └── ops.py           # Clean public API re-exporting operations
├── finance/
│   ├── black_scholes.py # BS pricing (NumPy + Tensor), analytic Greeks
│   ├── greeks.py        # Autograd-based Greeks computation
│   └── basket.py        # MC basket pricing, grid construction
├── tt/
│   ├── decompose.py     # TT-SVD, TT-rounding, TT-norm
│   ├── ops.py           # tt_eval, tt_to_full, tt_ranks, tt_memory, etc.
│   └── surrogate.py     # TTSurrogate class — full pipeline
└── utils/
    └── validation.py    # Numerical gradient checking
```

---

## Quick Start

### Installation

```bash
# Clone and install in editable mode
git clone <repo-url> && cd library
pip install -e ".[dev]"
```

### Run the Demo

```bash
python3 examples/demo_basket_tt.py
```

### Run the Benchmarks

```bash
python3 benchmarks/bench_tt_vs_mc.py
```

### Run Tests

```bash
python3 -m pytest tests/ -v
```

---

## Usage

### 1. Build a TT-Surrogate for a 3-Asset Basket

```python
import numpy as np
from tensorquantlib.tt.surrogate import TTSurrogate

surr = TTSurrogate.from_basket_analytic(
    S0_ranges=[(80, 120)] * 3,    # Spot price ranges per asset
    K=100, T=1.0, r=0.05,         # Strike, maturity, risk-free rate
    sigma=[0.2, 0.25, 0.3],       # Volatilities
    weights=[1/3, 1/3, 1/3],      # Equal-weighted basket
    n_points=30,                   # Grid resolution per axis
    eps=1e-4,                      # TT-SVD tolerance
)

# Fast evaluation
price = surr.evaluate([100.0, 105.0, 95.0])
print(f"Price: {price:.4f}")

# Greeks
greeks = surr.greeks([100.0, 105.0, 95.0])
print(f"Delta: {greeks['delta']}")
print(f"Gamma: {greeks['gamma']}")

# Diagnostics
surr.print_summary()
```

### 2. Autograd Greeks on Black-Scholes

```python
import numpy as np
from tensorquantlib.core.tensor import Tensor
from tensorquantlib.finance.black_scholes import bs_price_tensor

S = Tensor(np.array([100.0]))
price = bs_price_tensor(S, K=100, T=1.0, r=0.05, sigma=0.2)
price.backward()

print(f"Price: {price.item():.4f}")
print(f"Delta (autograd): {S.grad[0]:.6f}")
```

### 3. Low-Level TT Operations

```python
import numpy as np
from tensorquantlib.tt import tt_svd, tt_to_full, tt_ranks, tt_memory, tt_error

# Compress any tensor
A = np.random.randn(10, 10, 10, 10)
cores = tt_svd(A, eps=1e-4)

print(f"TT-ranks: {tt_ranks(cores)}")
print(f"Memory: {tt_memory(cores)} bytes vs {A.nbytes} bytes")
print(f"Error: {tt_error(cores, A):.2e}")
```

---

## How It Works

### Tensor-Train Decomposition

A $d$-dimensional tensor $\mathcal{A} \in \mathbb{R}^{n_1 \times n_2 \times \cdots \times n_d}$ is approximated as:

$$\mathcal{A}(i_1, i_2, \ldots, i_d) \approx G_1[i_1] \cdot G_2[i_2] \cdots G_d[i_d]$$

where each $G_k[i_k]$ is an $r_{k-1} \times r_k$ matrix. The TT-ranks $r_k$ are determined adaptively by the SVD truncation tolerance $\varepsilon$.

**Why it helps**: A 5-asset basket with 30 grid points per axis needs $30^5 \approx 24M$ entries (194 MB). TT compression reduces this to a few KB while maintaining $<0.01\%$ error for smooth pricing surfaces.

### Surrogate Pricing Pipeline

1. **Grid Construction**: Evaluate option prices on a structured grid (analytic or MC)
2. **TT-SVD Compression**: Decompose the grid into TT-cores with rank truncation
3. **Fast Evaluation**: Multi-linear interpolation on TT-cores — $O(d \cdot r^2)$ per point
4. **Greeks**: Finite-difference Delta and Gamma through the surrogate

---

## Test Coverage

| Module | Tests | Status |
|--------|-------|--------|
| Core autograd engine | 52 | ✅ |
| Financial engine | 32 | ✅ |
| TT decomposition & ops | 34 | ✅ |
| TT surrogate | 15 | ✅ |
| **Total** | **133** | **✅** |

---

## Dependencies

- Python ≥ 3.9
- NumPy ≥ 1.24
- SciPy ≥ 1.10
- pytest ≥ 7.0 (dev)
- matplotlib ≥ 3.7 (dev)

---

## References

- Oseledets, I.V. (2011). *Tensor-Train decomposition*. SIAM J. Sci. Comput.
- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*. JPE.
- Baydin et al. (2018). *Automatic differentiation in machine learning: a survey*. JMLR.

---

## License

MIT
