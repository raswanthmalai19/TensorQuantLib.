# TensorQuantLib Repository Inventory
**Analysis Date:** March 23, 2026  
**Focus:** Identify documentation duplicates, code redundancy, test file coverage, and unnecessary files

---

## Summary
- **Total Files Analyzed:** 100+
- **Markdown/Documentation Files:** 13 (root + docs + benchmarks)
- **Test Files:** 30
- **Example Files:** 4
- **Unnecessary Files Found:** 3 candidates for cleanup

---

## 1. DOCUMENTATION & MARKDOWN FILES

### Root Level Documentation (6 files)

| File | Size | Purpose | Status | Action |
|------|------|---------|--------|--------|
| [README.md](README.md) | 558 lines | **PRIMARY** main project documentation with badges, features, quick links | ✅ KEEP | Core documentation |
| [CHANGELOG.md](CHANGELOG.md) | 177 lines | Version history, features/fixes/improvements per release | ✅ KEEP | Source of truth for changelog |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 151 lines | Contribution guidelines, development setup, code standards | ✅ KEEP | Community contributions guide |
| [DEPLOYMENT.md](DEPLOYMENT.md) | 308 lines | Detailed deployment instructions, Docker, environment setup | ✅ KEEP | Deployment reference |
| [LIMITATIONS.md](LIMITATIONS.md) | 96 lines | Known limitations, dimensionality constraints, performance caps | ✅ KEEP | User expectations document |
| [TEST_REPORT.md](TEST_REPORT.md) | 182 lines | Test execution summary, coverage stats, test categories | ⚠️ EVALUATE | Auto-generated report (consider moving to CI/CD) |

### Sphinx Documentation (docs/ directory - 8 files)

| File | Purpose | Relationship to Root | Status | Action |
|------|---------|----------------------|--------|--------|
| [docs/index.rst](docs/index.rst) | Sphinx project index, TOC, landing page | Original | ✅ KEEP | Root documentation source |
| [docs/api.rst](docs/api.rst) | Auto-generated API reference from docstrings | Generated | ✅ KEEP | Built-in to Sphinx build |
| [docs/quickstart.rst](docs/quickstart.rst) | Getting started tutorial with examples | Original | ✅ KEEP | Tutorial documentation |
| [docs/theory.rst](docs/theory.rst) | Mathematical foundations, formulas, algorithms | Original | ✅ KEEP | Theory reference |
| [docs/performance.rst](docs/performance.rst) | Performance benchmarks, timing comparisons | Original (837 lines) | ✅ KEEP | Performance documentation |
| [docs/changelog.rst](docs/changelog.rst) | **STUB** - Redirects to `/CHANGELOG.md` | Points to root CHANGELOG.md | ⚠️ CONSIDER REMOVING | Unnecessary indirection |
| [docs/limitations.rst](docs/limitations.rst) | Mirror of `/LIMITATIONS.md` formatted for Sphinx | Duplicate | ✅ KEEP | Sphinx documentation needs both formats |
| [docs/_static/, _templates/] | Sphinx theme assets | Auto-generated | ✅ KEEP | Build infrastructure |

### Benchmarks Documentation (1 file)

| File | Size | Purpose | Status | Action |
|------|------|---------|--------|--------|
| [benchmarks/RESULTS.md](benchmarks/RESULTS.md) | 53 lines | Benchmark execution results and performance metrics | ✅ KEEP | Performance tracking documentation |

### Documentation Assessment
- ✅ **No major duplication issues** - root files and docs/ coexist appropriately
- ⚠️ **Minor issue:** `docs/changelog.rst` is a stub that could be removed (Sphinx can link directly to external markdown)
- **Format separation:** Markdown files are the source; RST files are Sphinx-formatted versions for documentation builds
- **Recommendation:** Keep dual formats for maintainability

---

## 2. CODE FILES - DUPLICATES & MODULES

### Core Module Structure (src/tensorquantlib/)

```
tensorquantlib/
├── core/              [Tensor operations, autodiff engine]
│   ├── tensor.py      - Tensor class, autograd implementation
│   ├── ops.py         - Operation re-exports (imports from tensor.py)
│   └── second_order.py - Second-order derivatives (Gamma, Vanna, Volga)
│
├── finance/           [Pricing engines and models]
│   ├── black_scholes.py - BS pricing, closed-form Greeks
│   ├── heston.py      - Heston model, vol surface
│   ├── american.py    - American option pricing (Binomial)
│   ├── exotic.py      - Exotic options (Asian, Barrier, Lookback, etc.)
│   ├── basket.py      - Multi-asset basket options
│   ├── greeks.py      - Greek computation via autograd
│   ├── implied_vol.py - Implied volatility calibration
│   ├── fx.py          - FX options (Garman-Kohlhagen)
│   ├── rates.py       - Interest rate instruments (bonds, swaps)
│   ├── ir_derivatives.py - IR derivative pricing
│   ├── credit.py      - Credit derivatives (CDS, spreads)
│   ├── jump_diffusion.py - Jump-diffusion models
│   ├── local_vol.py   - Local volatility models
│   ├── volatility.py  - Vol models (SABR, SVI)
│   ├── variance_reduction.py - Variance reduction (antithetic, stratified)
│   └── risk.py        - Risk metrics (VaR, CVaR, ES)
│
├── tt/                [Tensor-Train decomposition]
│   ├── decompose.py   - TT-SVD, TT-Cross algorithms
│   ├── ops.py         - TT operations (add, multiply, round, etc.)
│   ├── pricing.py     - TT-based pricing (surrogate surfaces)
│   └── surrogate.py   - TTSurrogate class for pricing
│
├── backtest/          [Portfolio backtesting]
│   ├── engine.py      - Backtest execution engine
│   ├── strategy.py    - Strategy base classes
│   └── metrics.py     - Performance metrics (Sharpe, Sortino, etc.)
│
├── data/              [Market data access]
│   └── market.py      - Market data API
│
├── viz/               [Visualization]
│   └── plots.py       - Price surface plots, Greeks surfaces
│
└── utils/             [Utilities]
    └── validation.py  - Input validation helpers
```

### Analysis: NO DUPLICATE MODULES
✅ **All modules serve unique purposes** - no redundant code files  
✅ **Clear separation of concerns** - finance, core, tt, backtest, data are all distinct  
✅ **Cohesive structure** - related functions grouped logically

---

## 3. TEST FILES

### Test File Inventory (30 files in /tests/)

#### Finance Module Tests (16 files)
| Test File | Target Module | Scope | Status |
|-----------|---------------|-------|--------|
| test_black_scholes.py | finance/black_scholes.py | BS pricing, closed-form Greeks | ✅ KEEP |
| test_heston.py | finance/heston.py | Heston pricing, calibration | ✅ KEEP |
| test_american.py | finance/american.py | American option pricing | ✅ KEEP |
| test_exotics.py | finance/exotics.py | Asian, Barrier, Lookback options | ✅ KEEP |
| test_basket.py | finance/basket.py | Multi-asset basket pricing | ✅ KEEP |
| test_greeks.py | finance/greeks.py | Greek computation | ✅ KEEP |
| test_implied_vol.py | finance/implied_vol.py | IV calibration | ✅ KEEP |
| test_fx.py | finance/fx.py | FX option pricing | ✅ KEEP |
| test_rates.py | finance/rates.py | Interest rate instruments | ✅ KEEP |
| test_ir_derivatives.py | finance/ir_derivatives.py | IR derivatives | ✅ KEEP |
| test_credit.py | finance/credit.py | Credit derivatives | ✅ KEEP |
| test_risk.py | finance/risk.py | Risk metrics | ✅ KEEP |
| test_volatility.py | finance/volatility.py | Vol models (SABR, SVI) | ✅ KEEP |
| test_jump_diffusion.py | finance/jump_diffusion.py | Jump-diffusion models | ✅ KEEP |
| test_local_vol.py | finance/local_vol.py | Local volatility | ✅ KEEP |
| test_variance_reduction.py | finance/variance_reduction.py | Variance reduction methods | ✅ KEEP |

#### Core Module Tests (5 files)
| Test File | Target Module | Scope | Status |
|-----------|---------------|-------|--------|
| test_tensor.py | core/tensor.py | Tensor class, autograd | ✅ KEEP |
| test_ops.py | core/ops.py | Core operations | ✅ KEEP |
| test_new_ops.py | core/ops.py | **NEW** ops (sin, cos, tanh, abs, clip, where, softmax) | ✅ KEEP |
| test_second_order.py | core/second_order.py | Second-order Greeks | ✅ KEEP |
| test_tt_ops.py | tt/ops.py | TT-specific operations | ✅ KEEP |

#### Tensor-Train Module Tests (5 files)
| Test File | Target Module | Scope | Status |
|-----------|---------------|-------|--------|
| test_tt_decompose.py | tt/decompose.py | TT-SVD, TT-Cross | ✅ KEEP |
| test_tt_cross.py | tt/decompose.py | TT-Cross algorithm detail | ⚠️ POSSIBLE OVERLAP |
| test_tt_arithmetic.py | tt/ops.py | TT arithmetic | ✅ KEEP |
| test_tt_pricing.py | tt/pricing.py | TT pricing | ✅ KEEP |
| test_surrogate.py | tt/surrogate.py | TTSurrogate class | ✅ KEEP |

#### Special/Aggregate Tests (4 files)
| Test File | Scope | Purpose | Status |
|-----------|-------|---------|--------|
| test_coverage_gaps.py | **CLI + Coverage gaps** | Tests `__main__.py` CLI and identifies coverage blindspots | ✅ KEEP |
| test_edge_cases.py | **Edge cases** | Targets ~5% uncovered code paths across modules | ✅ KEEP |
| test_validation.py | **Input validation** | Error handling and validation across library | ✅ KEEP |
| test_integration.py | **Full workflow** | End-to-end integration tests | ✅ KEEP |
| test_benchmark_validation.py | **Benchmarks** | Validates benchmark outputs | ✅ KEEP |
| test_data.py | data/market.py | Market data module | ✅ KEEP |
| test_backtest.py | backtest/ | Portfolio backtesting | ✅ KEEP |
| test_viz.py | viz/plots.py | Visualization | ✅ KEEP |

### Test Coverage Assessment
✅ **Comprehensive coverage** - 30 test files, 588 passing tests (per README)  
⚠️ **Potential overlap:** `test_tt_cross.py` and `test_tt_decompose.py` may have partial overlap
- `test_tt_decompose.py` - General TT decomposition tests
- `test_tt_cross.py` - TT-Cross specific edge cases and algorithm details
- **Recommendation:** Keep both (they test different aspects of similar functionality)

---

## 4. EXAMPLE & BENCHMARK FILES

### Examples (examples/ directory - 4 files)
| File | Topic | Purpose | Status |
|------|-------|---------|--------|
| [examples/autograd_intro.py](examples/autograd_intro.py) | **Autograd** | Demonstrates reverse-mode autodiff engine | ✅ KEEP |
| [examples/black_scholes_greeks.py](examples/black_scholes_greeks.py) | **BS Greeks** | Analytic vs autograd Greeks comparison | ✅ KEEP |
| [examples/demo_basket_tt.py](examples/demo_basket_tt.py) | **Basket + TT** | Multi-asset basket with TT surrogate | ✅ KEEP |
| [examples/tt_compression_demo.py](examples/tt_compression_demo.py) | **TT Compression** | TT-SVD, TT operations, reconstruction error | ✅ KEEP |

✅ **No duplicates** - Each example covers a distinct topic

### Benchmarks (benchmarks/ directory - 5 files)
| File | Focus | Purpose | Status |
|------|-------|---------|--------|
| [benchmarks/bench_greeks_accuracy.py](benchmarks/bench_greeks_accuracy.py) | **Greeks accuracy** | Compares analytic vs numerical Greeks | ✅ KEEP |
| [benchmarks/bench_portfolio_risk.py](benchmarks/bench_portfolio_risk.py) | **Risk metrics** | Portfolio VaR/CVaR performance | ✅ KEEP |
| [benchmarks/bench_tt_surrogate_scaling.py](benchmarks/bench_tt_surrogate_scaling.py) | **TT scaling** | TT surrogate speed (1000× speedup) | ✅ KEEP |
| [benchmarks/bench_tt_vs_mc.py](benchmarks/bench_tt_vs_mc.py) | **TT vs MC** | Tensor-Train vs Monte Carlo | ✅ KEEP |
| [benchmarks/RESULTS.md](benchmarks/RESULTS.md) | **Results summary** | Benchmark execution results | ✅ KEEP |

✅ **No duplicates** - Each benchmark tests different performance aspects

---

## 5. TOP-LEVEL DEBUG/UTILITY FILES

### Root-Level Python Scripts (3 files)

| File | Size | Purpose | Status | Recommendation |
|------|------|---------|--------|-----------------|
| [_verify_fixes.py](_verify_fixes.py) | 3.0 KB | **Verification script** - spot-checks recent audit fixes (ops exports, autograd, detach) | ⚠️ REVIEW | CLEANUP CANDIDATE |
| [debug_ttcross.py](debug_ttcross.py) | 1.6 KB | **Debug script** - reproduces TT-Cross issue with trig functions (SOLVED) | ⚠️ CLEANUP | **REMOVE** - Issue resolved |
| [test_doc_examples.py](test_doc_examples.py) | 2.6 KB | **Test script** - Validates that documentation code examples run correctly | ✅ KEEP | Integration testing |

### Shell Scripts (1 file)

| File | Purpose | Status | Recommendation |
|------|---------|--------|-----------------|
| [fix-history.sh](fix-history.sh) | Git history rewrite (replace 'claude' commits with 'raswanthmalai19') | ⏸️ ONE-TIME | REMOVE - Already executed |

---

## 6. AUTO-GENERATED & BUILD ARTIFACTS

### Cache & Build Directories (Safe to delete, auto-regenerate)
| Directory | Purpose | Status |
|-----------|---------|--------|
| `.pytest_cache/` | pytest cache | ✅ AUTO-GENERATED |
| `.mypy_cache/` | mypy type-checking cache | ✅ AUTO-GENERATED |
| `.ruff_cache/` | ruff linter cache | ✅ AUTO-GENERATED |
| `.benchmarks/` | benchmark result cache | ✅ AUTO-GENERATED |
| `dist/` | Build wheel & source dist (tensorquantlib-0.3.0.tar.gz, whl) | ✅ AUTO-GENERATED |
| `src/tensorquantlib.egg-info/` | Egg info metadata | ✅ AUTO-GENERATED |

### Version Control & Configuration
| File | Status |
|------|--------|
| `.git/` | Repo history | ✅ KEEP |
| `.github/workflows/` (3 files: ci.yml, pages.yml, publish.yml) | CI/CD workflows | ✅ KEEP |
| `.gitignore` | Git ignore rules | ✅ KEEP |
| `.pre-commit-config.yaml` | Pre-commit hooks | ✅ KEEP |
| `.venv/` | Python virtual environment | ✅ KEEP |

---

## 7. EMPTY & UNNECESSARY DIRECTORIES

| Directory | Status | Recommendation |
|-----------|--------|-----------------|
| [blog/](blog/) | **EMPTY** - No files | 🗑️ REMOVE | Delete the empty directory |

---

## 8. CONFIGURATION FILES

### Project Configuration (Single source of truth)
| File | Purpose | Status |
|------|---------|--------|
| [pyproject.toml](pyproject.toml) | Modern Python packaging config (PEP 518) | ✅ KEEP |

✅ **No duplicate setup files** - pyproject.toml is the only build config

### Pre-commit Configuration
| File | Purpose | Status |
|------|---------|--------|
| [.pre-commit-config.yaml](.pre-commit-config.yaml) | Pre-commit hooks (ruff, mypy, etc.) | ✅ KEEP |

---

## FINAL RECOMMENDATIONS

### 🗑️ REMOVE (Cleanup Candidates)

1. **[debug_ttcross.py](debug_ttcross.py)** (1.6 KB)
   - **Reason:** Debug script for resolved TT-Cross issue
   - **Status:** Issue already fixed; script no longer needed
   - **Risk:** None - issue is solved

2. **[fix-history.sh](fix-history.sh)** (1.1 KB)
   - **Reason:** One-time git history rewrite script
   - **Status:** Already executed (git history changed)
   - **Risk:** None - not used in active workflow

3. **[blog/](blog/)** (empty directory)
   - **Reason:** Empty placeholder
   - **Status:** Contains no files
   - **Risk:** None - placeholder only

### ⚠️ CONSIDER MOVING (Optional)

1. **[TEST_REPORT.md](TEST_REPORT.md)** (182 lines)
   - **Reason:** Auto-generated report of test results
   - **Current use:** Demonstrates test status
   - **Option 1:** Keep as-is (documentation)
   - **Option 2:** Move to CI/CD artifacts (generate on each run)
   - **Recommendation:** Keep (useful for reference)

2. **[docs/changelog.rst](docs/changelog.rst)** (5 lines, stub)
   - **Reason:** Sphinx stub that redirects to root CHANGELOG.md
   - **Current purpose:** Maintains Sphinx TOC
   - **Option:** Could be removed if Sphinx can link directly to external markdown
   - **Recommendation:** Keep (prevents broken Sphinx TOC)

3. **[_verify_fixes.py](_verify_fixes.py)** (3.0 KB)
   - **Reason:** Temporary verification script for audit fixes
   - **Status:** Audit completed; script main purpose served
   - **Option 1:** Keep (regression check)
   - **Option 2:** Integrate into test suite
   - **Option 3:** Remove (redundant with test suite)
   - **Recommendation:** Consider converting to proper test or removing

### ✅ KEEP (Necessary Files)

- All source code in `src/tensorquantlib/`
- All 30 test files (comprehensive coverage, no real duplicates)
- All 4 example files (all unique topics)
- All 8 Sphinx documentation files
- All root markdown documentation (README, CHANGELOG, CONTRIBUTING, etc.)
- All CI/CD workflows, config files, and version control

---

## DUPLICATION SUMMARY

| Category | Status |
|----------|--------|
| **Code Modules** | ✅ No duplicates (all unique purpose) |
| **Test Files** | ✅ No major duplicates (potential minor overlap in TT tests is justified) |
| **Examples** | ✅ No duplicates (all unique topics) |
| **Benchmarks** | ✅ No duplicates (all unique metrics) |
| **Documentation** | ⚠️ Intentional format duplication (Markdown + RST for Sphinx) |
| **Build Artifacts** | ✅ All auto-generated (safe to delete, regenerate on build) |

---

## ACTION ITEMS PRIORITY

### Immediate (Safe to do now)
- [ ] Remove `debug_ttcross.py` (resolved debug artifact)
- [ ] Remove `fix-history.sh` (one-time script already executed)
- [ ] Delete empty `blog/` directory

### Optional (Based on preference)
- [ ] Review if `_verify_fixes.py` should be kept or converted to permanent test
- [ ] Evaluate if `TEST_REPORT.md` should be moved to CI artifacts

### Do Not Remove
- Keep all source code modules
- Keep all 30 test files (no real redundancy)
- Keep documentation (root + Sphinx)
- Keep all CI/CD infrastructure

---

**Total Unnecessary Files:** 3 (debug_ttcross.py, fix-history.sh, blog/)  
**Total Lines Saved:** ~2-3 KB  
**Repository Health:** ✅ **Excellent** - Well-organized, no major redundancy
