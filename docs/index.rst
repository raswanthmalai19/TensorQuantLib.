TensorQuantLib Documentation
============================

**TensorQuantLib** is a comprehensive quantitative finance library with
tensor-train compression, automatic differentiation, and stochastic models —
built from scratch with NumPy and SciPy. 150+ public functions covering
derivatives pricing, risk management, and portfolio analysis.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   performance
   api
   theory
   limitations
   changelog


.. note::

   **Quant traders / fintech engineers**: see the :doc:`performance` guide for
   latency tables, Heston calibration speedups, TT-rank tuning, and the
   production configuration checklist.

   **Researchers**: the :doc:`theory` page has full mathematical derivations
   for every model.


Quick Overview
--------------

.. code-block:: python

   from tensorquantlib import TTSurrogate

   # Build a 3-asset basket option surrogate
   surr = TTSurrogate.from_basket_analytic(
       S0_ranges=[(80, 120)] * 3,
       K=100, T=1.0, r=0.05,
       sigma=[0.2, 0.25, 0.3],
       weights=[1/3, 1/3, 1/3],
       n_points=30,
   )
   surr.print_summary()

   # Evaluate the surrogate — ~5 microseconds per eval
   # (100-1000× faster than re-running Monte Carlo for repeated evals)
   price = surr.evaluate([100.0, 100.0, 100.0])
   greeks = surr.greeks([100.0, 100.0, 100.0])


Features
--------

- **Reverse-mode autodiff**: Tensor class with 23+ differentiable ops
- **Second-order autodiff**: Hessians, HVPs, Gamma/Vanna/Volga
- **Black-Scholes engine**: pricing + all analytical Greeks
- **Heston stochastic volatility**: semi-analytic CF, QE Monte Carlo, calibration
- **American options**: Longstaff-Schwartz LSM with exercise boundary
- **Exotic options**: Asian, Digital, Barrier (8 types), Lookback, Cliquet, Rainbow
- **Volatility surface**: SABR (Hagan 2002), SVI (Gatheral 2004) with calibration
- **Interest rates**: Vasicek, CIR, Nelson-Siegel, yield curve bootstrapping
- **FX options**: Garman-Kohlhagen, FX Greeks, forwards, quanto
- **Credit risk**: Merton structural model, CDS pricing, hazard rates
- **Jump-diffusion**: Merton jump-diffusion, Kou double-exponential
- **Local volatility**: Dupire local vol, local vol Monte Carlo
- **IR derivatives**: Black-76 caps/floors, swaptions, swap rate
- **Variance reduction**: antithetic, control variate, QMC, importance sampling, stratified
- **Risk metrics**: VaR (parametric/historical/MC), CVaR, scenario analysis
- **Backtesting**: engine, strategies, performance metrics (Sharpe, Sortino, Calmar)
- **TT compression**: TT-SVD, TT-cross, rounding, arithmetic, surrogate pricing
- **Basket options**: correlated GBM Monte Carlo, analytic moment-matching
- **Visualization**: pricing surfaces, Greek surfaces, TT-rank charts
- **CLI**: ``python -m tensorquantlib`` — info, price, greeks, demo


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
