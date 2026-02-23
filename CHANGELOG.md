# Changelog

All notable changes to TensorQuantLib are documented here.

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
