TensorQuantLib Documentation
============================

**TensorQuantLib** is a Tensor-Train based surrogate pricing engine with
reverse-mode autodiff for multi-asset options, designed to run on a laptop.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   api
   theory
   changelog


Quick Overview
--------------

.. code-block:: python

   from tensorquantlib import TTSurrogate

   # Build a 3-asset basket option surrogate
   surr = TTSurrogate.from_basket_analytic(
       n_assets=3, n_points=30, strike=100.0
   )
   surr.print_summary()

   # Evaluate the surrogate — 1000× faster than grid lookup
   price = surr.evaluate([100.0, 100.0, 100.0])
   greeks = surr.greeks([100.0, 100.0, 100.0])


Features
--------

- **Reverse-mode autodiff**: Tensor class with 16+ differentiable ops
- **Black-Scholes engine**: pricing + all analytical Greeks
- **Basket option pricing**: Monte Carlo with correlated GBM
- **TT-SVD compression**: Oseledets (2011) algorithm with rounding
- **TT arithmetic**: add, scale, Hadamard, dot, Frobenius norm
- **TTSurrogate**: multi-linear interpolation on compressed grids
- **Visualization**: publication-quality plotting utilities


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
