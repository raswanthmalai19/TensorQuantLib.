Quickstart
==========

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


Heston Stochastic Volatility
-----------------------------

Price options under the Heston model using semi-analytic or Monte Carlo methods:

.. code-block:: python

   from tensorquantlib.finance.heston import heston_price, heston_price_mc

   # Semi-analytic via characteristic function
   price = heston_price(
       S=100, K=100, T=1.0, r=0.05,
       v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7,
       option_type='call',
   )
   print(f"Heston call (analytic): {price:.4f}")

   # Monte Carlo with QE scheme
   mc_price, mc_se = heston_price_mc(
       S=100, K=100, T=1.0, r=0.05,
       v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7,
       n_paths=100_000, scheme='qe',
   )
   print(f"Heston call (MC): {mc_price:.4f} ± {mc_se:.4f}")


American Options
----------------

Price American options using the Longstaff-Schwartz algorithm:

.. code-block:: python

   from tensorquantlib.finance.american import american_option_lsm

   price, se = american_option_lsm(
       S=100, K=100, T=1.0, r=0.05, sigma=0.2,
       option_type='put', n_paths=100_000, n_steps=100,
   )
   print(f"American put: {price:.4f} ± {se:.4f}")


Implied Volatility
------------------

Extract implied volatility from market prices:

.. code-block:: python

   from tensorquantlib.finance.implied_vol import implied_vol_brent

   iv = implied_vol_brent(
       market_price=10.45, S=100, K=100, T=1.0, r=0.05, option_type='call',
   )
   print(f"Implied volatility: {iv:.4f}")   # ≈ 0.2000


Exotic Options
--------------

Price Asian, digital, and barrier options:

.. code-block:: python

   from tensorquantlib.finance.exotics import (
       asian_option_price, digital_option_price, barrier_option_price,
   )

   # Arithmetic Asian call
   asian, se = asian_option_price(
       S=100, K=100, T=1.0, r=0.05, sigma=0.2,
       option_type='call', asian_type='arithmetic',
   )
   print(f"Asian call: {asian:.4f}")

   # Cash-or-nothing digital
   digital = digital_option_price(
       S=100, K=100, T=1.0, r=0.05, sigma=0.2,
       option_type='call', digital_type='cash',
   )
   print(f"Digital call: {digital:.4f}")

   # Down-and-out barrier call
   barrier = barrier_option_price(
       S=100, K=100, T=1.0, r=0.05, sigma=0.2,
       barrier=90, option_type='call', barrier_type='down-and-out',
   )
   print(f"Barrier call: {barrier:.4f}")


Variance Reduction
------------------

Compare Monte Carlo variance reduction techniques:

.. code-block:: python

   from tensorquantlib.finance.variance_reduction import compare_variance_reduction

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
   from tensorquantlib.finance.risk import var_parametric, cvar

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
