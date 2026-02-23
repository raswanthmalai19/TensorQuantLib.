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

- **European options only**: No American, Bermudan, Asian, or barrier support.
- **Constant parameters**: No term structure or stochastic volatility.
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

Potential improvements (contributions welcome):

1. GPU acceleration via CuPy/JAX backends
2. Second-order autodiff (Hessian-vector products)
3. American option support via Longstaff-Schwartz
4. Adaptive grid refinement near the strike
5. Stochastic volatility models (Heston)
6. Higher-dimensional support (d > 5) via cross-approximation
