Known Limitations
=================

This page documents current limitations so users can make informed decisions
about where and how to use TensorQuantLib.


Dimensionality
--------------

- Designed for **d ≤ 5 assets**. Beyond 5 dimensions, TT ranks can grow
  rapidly for non-smooth payoffs and grid construction time becomes the
  bottleneck.
- For 6+ assets, consider sparse-grid or quasi-Monte-Carlo approaches.

========  ===========  ========  ==============  ===========
Assets    Grid Points  Full MB   TT Compression  Status
========  ===========  ========  ==============  ===========
2–3       30/axis      < 1 MB    1–13×           ✅ Excellent
4         20/axis      ~1 MB     4×              ✅ Good
5         15/axis      ~6 MB     42×             ⚠️ Usable
6+        10/axis      grows     varies          ❌ Untested
========  ===========  ========  ==============  ===========


Performance
-----------

- **Single-threaded**: All computations run on a single CPU thread. No
  multi-threading, GPU, or distributed compute.
- **No JIT compilation**: Unlike PyTorch/JAX, operations execute eagerly in
  Python.
- **TT evaluation overhead**: For grids < 10,000 entries, direct NumPy
  indexing can be faster than TT-core contraction.


Autograd Engine
---------------

- **First-order only**: Supports first-order derivatives (Delta, Vega).
  Gamma is computed via finite differences.
- **No in-place ops**: ``+=`` is not tracked. Use ``a = a + b`` instead.
- **Real-valued only**: No complex number support.
- **16 ops**: Missing ``sin``, ``cos``, ``tanh``, ``abs``, ``where``,
  ``concatenate``.


Financial Models
----------------

**What IS Implemented:**

- **Black-Scholes**: closed-form pricing and all analytical Greeks
- **American options**: Longstaff-Schwartz LSM with polynomial regression
- **Asian options**: Monte Carlo with variance reduction techniques
- **Exotic options**: Barrier (8 types), Digital, Lookback, Cliquet, Rainbow
- **Heston stochastic volatility**: semi-analytic CF (fast) + QE Monte Carlo

**Important Note**: Tensor-Train speedup (100-1000x) does NOT apply to Monte Carlo methods (American, Asian, Exotic). TT acceleration works only for smooth analytic surfaces (BS, Heston CF).

**Model Limitations:**

- **Constant parameters**: No term structure or stochastic volatility (except Heston).
- **Basket approximation**: ``from_basket_analytic`` uses a weighted BS
  approximation. For accurate basket prices, use ``from_basket_mc``.


TT Compression
--------------

- **Smooth payoffs**: TT-SVD achieves high compression on smooth surfaces.
  Discontinuous payoffs show higher ranks.
- **Uniform grid**: No adaptive refinement near the strike.
- **Frobenius norm**: The ``eps`` tolerance controls relative error in
  Frobenius norm, not pointwise error.


Roadmap
-------

Completed ✅ (v0.3.0):

✅ Reverse-mode autodiff (23+ differentiable ops)
✅ American option support (Longstaff-Schwartz LSM)
✅ Stochastic volatility (Heston semi-analytic CF + QE)
✅ Exotic options (all types via MC + variance reduction)
✅ Second-order Greeks (Gamma, Vanna, Volga via autodiff)
✅ 698 test cases, 98% code coverage

Potential improvements (contributions welcome):

1. GPU acceleration via CuPy/JAX backends
2. Adaptive grid refinement near the strike
3. Higher-dimensional support (d > 5) via cross-approximation
4. Streaming/online TT updates for live pricing
5. Stochastic correlation models
