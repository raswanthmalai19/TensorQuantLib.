# Known Limitations

This document covers the current limitations of TensorQuantLib so users can
make informed decisions about where and how to use the library.

---

## Dimensionality

| Assets | Grid Points | Memory (full) | TT Compression | Status |
|--------|-------------|---------------|----------------|--------|
| 2–3    | 30/axis     | < 1 MB        | 1–13×          | ✅ Excellent |
| 4      | 20/axis     | ~1 MB         | 4×             | ✅ Good |
| 5      | 15/axis     | ~6 MB         | 42×            | ⚠️ Usable |
| 6+     | 10/axis     | exponential   | varies         | ❌ Not tested |

- The library is designed for **d ≤ 5 assets**. Beyond 5 dimensions, TT ranks
  can grow rapidly for non-smooth payoffs, and grid construction time becomes
  the bottleneck.
- For 6+ assets, consider sparse-grid or quasi-Monte-Carlo approaches.

## Performance

- **Single-threaded**: All computations run on a single CPU thread. There is no
  multi-threading, GPU support, or distributed compute.
- **No JIT compilation**: Unlike PyTorch/JAX, the autograd engine does not
  compile or fuse operations. Each op is executed eagerly in Python.
- **TT evaluation overhead**: For small grids (< 10,000 entries), direct NumPy
  indexing can be faster than TT-core contraction. TT compression benefits
  appear at ≥4 dimensions.

## Autograd Engine

- **First-order only**: The autograd engine supports first-order derivatives
  (Delta, Vega). Second-order derivatives (Gamma) are computed via finite
  differences, not true second-order autodiff.
- **No in-place operations**: Operations like `+=` on Tensor objects are not
  tracked by the computation graph. Use `a = a + b` instead.
- **No complex numbers**: The Tensor class only supports real-valued (float64)
  data.
- **Limited op set**: 16 differentiable operations are supported. Missing ops
  include: `sin`, `cos`, `tanh`, `abs`, `where`, `concatenate`.

## Financial Models

- **European options only**: The library prices European-style options.
  American, Bermudan, Asian, and barrier options are not supported.
- **Constant parameters**: Volatility, rate, and correlation are assumed
  constant (no term structure, no stochastic volatility).
- **Correlation matrix**: Must be positive semi-definite. Near-singular
  correlation matrices are regularized automatically (diagonal jitter), which
  may silently alter the intended correlation structure.
- **Basket approximation**: `from_basket_analytic` uses a weighted
  Black-Scholes approximation, not a proper basket formula. For accurate basket
  prices, use `from_basket_mc`.

## TT Compression

- **Smooth payoffs only**: TT-SVD achieves high compression on smooth pricing
  surfaces. Discontinuous payoffs (e.g., digital options) or payoffs with kinks
  at the strike will show higher ranks and lower compression.
- **No adaptive grid**: Grid points are uniform per axis. Adaptive refinement
  near the strike would improve efficiency but is not implemented.
- **Fixed precision**: The `eps` tolerance controls relative error in the
  Frobenius norm, not pointwise error. Individual evaluation points may have
  larger errors than `eps`.

## Visualization

- **Matplotlib only**: Plotting requires matplotlib. No support for Plotly,
  Bokeh, or other interactive backends.
- **2D/3D only**: `plot_pricing_surface` and `plot_greeks_surface` work for
  1-asset or 2-asset surrogates. Higher-dimensional slicing is left to the
  user.

## Packaging & Deployment

- **Not yet on PyPI**: Install from source. PyPI publishing is configured via
  GitHub Actions but not yet triggered.
- **No conda package**: Only pip installation is supported.
- **Python ≥ 3.10**: Python 3.9 is not explicitly tested in CI despite
  `requires-python = ">=3.10"` in pyproject.toml.

---

## Roadmap

Potential improvements (contributions welcome):

1. GPU acceleration via CuPy/JAX backends
2. Second-order autodiff (Hessian-vector products)
3. American option support via Longstaff-Schwartz
4. Adaptive grid refinement near the strike
5. Stochastic volatility models (Heston)
6. Higher-dimensional support (d > 5) via cross-approximation
7. Streaming/online TT updates for live pricing
