Quickstart
==========

.. note::

   **Quant traders & fintech engineers** — jump straight to the
   :ref:`pricer-decision-guide` to pick the right function for your use
   case in under 60 seconds.  If you care only about throughput, latency,
   and production settings, read the
   :doc:`performance` guide first.

   **Researchers & PhD students** — the :doc:`theory` page has full
   mathematical derivations with LaTeX for every model.


Installation
------------

.. code-block:: bash

   pip install tensorquantlib

For development:

.. code-block:: bash

   git clone https://github.com/raswanthmalai19/TensorQuantLib.git
   cd TensorQuantLib
   pip install -e ".[dev]"


Your First Surrogate
--------------------

Build a 3-asset basket option pricing surrogate in seconds:

.. code-block:: python

   from tensorquantlib import TTSurrogate

   # Build the surrogate from an analytic approximation
   surr = TTSurrogate.from_basket_analytic(
       S0_ranges=[(80, 120)] * 3,    # Spot price ranges per asset
       K=100, T=1.0, r=0.05,         # Strike, maturity, risk-free rate
       sigma=[0.2, 0.25, 0.3],       # Volatilities
       weights=[1/3, 1/3, 1/3],      # Equal-weighted basket
       n_points=30,                   # Grid resolution per axis
       eps=1e-4,                      # TT-SVD tolerance
   )

   # Print compression diagnostics
   surr.print_summary()

   # Evaluate at a specific spot price vector
   price = surr.evaluate([100.0, 105.0, 95.0])
   print(f"Price: {price:.4f}")

   # Compute finite-difference Greeks
   greeks = surr.greeks([100.0, 105.0, 95.0])
   print(f"Deltas: {greeks['delta']}")
   print(f"Gammas: {greeks['gamma']}")


.. _pricer-decision-guide:

Pricer Decision Guide
---------------------

Choose your function based on what you actually need:

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 15 30

   * - You want…
     - Use
     - Latency
     - Accuracy
     - Notes
   * - Vanilla call/put price
     - ``black_scholes``
     - <10 µs
     - Exact
     - Production default for vanilla books
   * - Vanilla Greeks
     - ``bs_greeks``
     - <10 µs
     - Exact
     - Analytic Delta, Gamma, Theta, Vega, Rho
   * - Second-order Greeks (Gamma, Vanna, Volga)
     - ``second_order_greeks``
     - ~1 ms
     - Numerical
     - 4-point finite-diff bump on autodiff graph
   * - Implied volatility from price
     - ``implied_vol``
     - <1 ms
     - <1 bp error
     - Brent solver; safe for all moneyness
   * - IV surface from prices
     - ``iv_surface``
     - O(N) × 1 ms
     - —
     - Vectorised over (K, T) grid
   * - Heston price (production)
     - ``heston_price``
     - ~1 ms
     - Semi-analytic
     - Gil-Pelaez CF; 100-pt Gaussian quadrature
   * - Heston price (validation/stress)
     - ``heston_price_mc``
     - 200–500 ms
     - MC ± stderr
     - QE scheme, 100 K paths by default
   * - Heston calibration
     - ``HestonCalibrator.fit``
     - 5–15 s
     - RMSE ~0.001 IV
     - See :doc:`performance` for 5× speedup tips
   * - American option
     - ``american_option_lsm``
     - 100–500 ms
     - MC (LSM)
     - 50 K paths × 100 steps default
   * - Asian option
     - ``asian_price_mc``
     - 100–400 ms
     - MC
     - Use ``asian_price_cv`` for 10× lower variance
   * - Barrier option (analytic)
     - ``barrier_price``
     - <10 µs
     - Exact (GBM)
     - Rubinstein-Reiner closed form
   * - Basket option (N≥3 assets)
     - ``TTSurrogate``
     - <1 µs eval after build
     - <0.1% (eps=1e-3)
     - Build once, price millions
   * - SABR vol smile
     - ``sabr_implied_vol``
     - <10 µs
     - Hagan 2002 approx
     - For calibration use ``sabr_calibrate``
   * - SVI vol surface
     - ``svi_implied_vol``
     - <10 µs
     - Exact fit to SVI params
     - Log-moneyness parameterisation
   * - FX vanilla
     - ``garman_kohlhagen``
     - <10 µs
     - Exact
     - Pass ``r_f`` as foreign rate
   * - CDS spread
     - ``cds_spread``
     - <1 ms
     - Analytic
     - Pass ``hazard_rate``; survival probability built in
   * - Bond price (Vasicek)
     - ``vasicek_bond_price``
     - <1 µs
     - Exact
     - Closed-form A(T)/B(T)

.. tip::

   **Batch pricing rule of thumb**:
   Wrap any scalar pricer in ``np.vectorize`` or a list comprehension to price
   thousands of options without leaving Python.  For > 10 K options per second
   see :ref:`batch-pricing` in the performance guide.


Autodiff Greeks
---------------

Use the ``Tensor`` class for automatic differentiation:

.. code-block:: python

   import numpy as np
   from tensorquantlib import Tensor, bs_price_tensor

   S = Tensor(np.array([100.0]), requires_grad=True)
   K, T, r, sigma = 100.0, 1.0, 0.05, 0.2

   price = bs_price_tensor(S, K=K, T=T, r=r, sigma=sigma, option_type="call")
   price.backward()

   delta = S.grad  # dPrice/dS — exact, computed via autodiff
   print(f"Delta (autodiff): {delta}")


Second-Order Greeks
-------------------

Compute Gamma, Vanna, and Volga in a single efficient call:

.. code-block:: python

   from tensorquantlib import second_order_greeks, bs_price_tensor

   result = second_order_greeks(
       price_fn=bs_price_tensor,
       S=100, K=100, T=1.0, r=0.05, sigma=0.2,
   )
   print(f"Gamma: {result['gamma']:.6f}")
   print(f"Vanna: {result['vanna']:.6f}")
   print(f"Volga: {result['volga']:.6f}")


Heston Stochastic Volatility
-----------------------------

Price options under the Heston model using the ``HestonParams`` container:

.. code-block:: python

   from tensorquantlib import HestonParams, heston_price, heston_price_mc

   params = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)

   # Semi-analytic via characteristic function
   price = heston_price(S=100, K=100, T=1.0, r=0.05, params=params)
   print(f"Heston call (analytic): {price:.4f}")

   # Monte Carlo with QE scheme
   mc_price, mc_se = heston_price_mc(
       S=100, K=100, T=1.0, r=0.05, params=params,
       n_paths=100_000, scheme='qe', return_stderr=True,
   )
   print(f"Heston call (MC): {mc_price:.4f} ± {mc_se:.4f}")


American Options
----------------

Price American options using the Longstaff-Schwartz algorithm:

.. code-block:: python

   from tensorquantlib import american_option_lsm

   price, se = american_option_lsm(
       S=100, K=100, T=1.0, r=0.05, sigma=0.2,
       option_type='put', n_paths=100_000, n_steps=100,
   )
   print(f"American put: {price:.4f} ± {se:.4f}")


Implied Volatility
------------------

Extract implied volatility from market prices:

.. code-block:: python

   from tensorquantlib import implied_vol, iv_surface
   import numpy as np

   # Single option — Brent's method
   iv = implied_vol(market_price=10.45, S=100, K=100, T=1.0, r=0.05)
   print(f"Implied volatility: {iv:.4f}")   # ≈ 0.2000

   # Build a full IV surface
   strikes = np.linspace(80, 120, 9)
   expiries = np.array([0.25, 0.5, 1.0])
   surface = iv_surface(S=100, r=0.05, sigma=0.2, strikes=strikes, expiries=expiries)


Exotic Options
--------------

Price Asian, digital, barrier, lookback, and rainbow options:

.. code-block:: python

   from tensorquantlib import (
       asian_price_mc, digital_price, barrier_price,
       lookback_fixed_analytic, cliquet_price_mc,
   )

   # Arithmetic Asian call
   asian = asian_price_mc(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
   print(f"Asian call: {asian:.4f}")

   # Cash-or-nothing digital
   digital = digital_price(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
   print(f"Digital call: {digital:.4f}")

   # Down-and-out barrier call
   barrier = barrier_price(
       S=100, K=100, T=1.0, r=0.05, sigma=0.2,
       barrier=90, barrier_type='down-and-out',
   )
   print(f"Barrier call: {barrier:.4f}")

   # Fixed-strike lookback call (analytic)
   lookback = lookback_fixed_analytic(
       S=100, S_min=95, K=100, T=1.0, r=0.05, sigma=0.2,
   )
   print(f"Lookback call: {lookback:.4f}")


Volatility Surface Models
-------------------------

SABR and SVI parameterizations for the volatility smile:

.. code-block:: python

   from tensorquantlib import sabr_implied_vol, svi_implied_vol
   import numpy as np

   # SABR implied vol (Hagan 2002)
   vol = sabr_implied_vol(F=100, K=100, T=1.0, alpha=0.3, beta=0.5, rho=-0.3, nu=0.4)
   print(f"SABR ATM vol: {vol:.4f}")

   # SVI parameterization (k = log-moneyness ln(K/F))
   k = np.linspace(-0.2, 0.2, 50)
   vols = svi_implied_vol(k, T=1.0, a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
   print(f"SVI vols range: [{vols.min():.4f}, {vols.max():.4f}]")


Interest Rate Models
--------------------

Vasicek, CIR, and Nelson-Siegel yield curve models:

.. code-block:: python

   from tensorquantlib import vasicek_bond_price, cir_bond_price, nelson_siegel
   import numpy as np

   # Vasicek zero-coupon bond
   P = vasicek_bond_price(r0=0.05, kappa=0.3, theta=0.05, sigma=0.02, T=5.0)
   print(f"Vasicek bond: {P:.6f}")

   # CIR bond price
   P_cir = cir_bond_price(r0=0.05, kappa=0.5, theta=0.05, sigma=0.1, T=5.0)
   print(f"CIR bond: {P_cir:.6f}")

   # Nelson-Siegel yield curve
   maturities = np.array([0.25, 0.5, 1, 2, 5, 10, 30])
   yields = nelson_siegel(maturities, beta0=0.05, beta1=-0.02, beta2=0.03, tau=1.5)
   print(f"Yields: {yields}")


FX Options
----------

Garman-Kohlhagen FX option pricing:

.. code-block:: python

   from tensorquantlib import garman_kohlhagen, gk_greeks, fx_forward

   # European FX call
   price = garman_kohlhagen(S=1.25, K=1.30, T=0.5, r_d=0.05, r_f=0.02, sigma=0.1)
   print(f"FX call: {price:.6f}")

   # FX Greeks
   greeks = gk_greeks(S=1.25, K=1.30, T=0.5, r_d=0.05, r_f=0.02, sigma=0.1)
   print(f"FX delta: {greeks['delta']:.4f}")

   # FX forward rate
   fwd = fx_forward(S=1.25, r_d=0.05, r_f=0.02, T=1.0)
   print(f"1Y forward: {fwd:.4f}")


Credit Risk
-----------

Merton structural model and CDS pricing:

.. code-block:: python

   from tensorquantlib import merton_default_prob, merton_credit_spread, cds_spread

   # Probability of default
   pd = merton_default_prob(V=100, D=80, T=1.0, r=0.05, sigma_V=0.25)
   print(f"Default probability: {pd:.4f}")

   # Credit spread
   spread = merton_credit_spread(V=100, D=80, T=1.0, r=0.05, sigma_V=0.25)
   print(f"Credit spread: {spread * 10000:.1f} bps")

   # CDS spread from hazard rate
   cds = cds_spread(hazard_rate=0.02, T=5.0, recovery=0.4, r=0.03)
   print(f"CDS spread: {cds * 10000:.1f} bps")


Jump-Diffusion Models
---------------------

Merton and Kou jump-diffusion pricing:

.. code-block:: python

   from tensorquantlib import merton_jump_price, merton_jump_price_mc

   # Analytic (series expansion)
   price = merton_jump_price(
       S=100, K=100, T=1.0, r=0.05, sigma=0.2,
       lam=1.0, mu_j=-0.05, sigma_j=0.1,
   )
   print(f"Merton jump (analytic): {price:.4f}")

   # Monte Carlo
   mc = merton_jump_price_mc(
       S=100, K=100, T=1.0, r=0.05, sigma=0.2,
       lam=1.0, mu_j=-0.05, sigma_j=0.1, seed=42,
   )
   print(f"Merton jump (MC): {mc:.4f}")


IR Derivatives
--------------

Black-76 caps, floors, and swaptions:

.. code-block:: python

   from tensorquantlib import cap_price, swap_rate, swaption_price
   import numpy as np

   # Interest rate cap
   forwards = np.array([0.05, 0.055, 0.06, 0.065])
   expiries = np.array([0.25, 0.5, 0.75, 1.0])
   dfs = np.exp(-0.04 * expiries)
   cap = cap_price(forwards, strike=0.05, expiries=expiries, sigma=0.2, dfs=dfs)
   print(f"Cap price: {cap:.6f}")

   # Par swap rate
   sr = swap_rate(dfs=np.array([1.0, 0.98, 0.96, 0.94, 0.92]), tau=0.5)
   print(f"Swap rate: {sr:.4f}")


Variance Reduction
------------------

Compare Monte Carlo variance reduction techniques:

.. code-block:: python

   from tensorquantlib import compare_variance_reduction

   results = compare_variance_reduction(
       S=100, K=100, T=1.0, r=0.05, sigma=0.2, n_paths=50_000,
   )
   for name, (price, se) in results.items():
       print(f"{name:20s}  price={price:.4f}  stderr={se:.4f}")


Risk Metrics
------------

Compute Value at Risk and Expected Shortfall:

.. code-block:: python

   import numpy as np
   from tensorquantlib import var_parametric, cvar

   returns = np.random.normal(0.0005, 0.02, 252)
   var95 = var_parametric(returns, confidence=0.95)
   es95 = cvar(returns, confidence=0.95)
   print(f"VaR(95%): {var95:.4f}")
   print(f"CVaR(95%): {es95:.4f}")


TT Compression
--------------

Compress any multi-dimensional array with TT-SVD:

.. code-block:: python

   import numpy as np
   from tensorquantlib import tt_svd, tt_to_full, tt_compression_ratio

   # Create a large 4D tensor
   A = np.random.randn(20, 20, 20, 20)

   # Compress with tolerance
   cores = tt_svd(A, eps=1e-6)

   # Check compression
   ratio = tt_compression_ratio(cores, A.shape)
   print(f"Compression ratio: {ratio:.1f}×")

   # Reconstruct
   A_approx = tt_to_full(cores)
   error = np.linalg.norm(A - A_approx) / np.linalg.norm(A)
   print(f"Relative error: {error:.2e}")


CLI
---

Use TensorQuantLib from the command line:

.. code-block:: bash

   python -m tensorquantlib info      # Library version and capabilities
   python -m tensorquantlib price     # Interactive option pricer
   python -m tensorquantlib greeks    # Compute Greeks
   python -m tensorquantlib demo      # Quick demonstration
