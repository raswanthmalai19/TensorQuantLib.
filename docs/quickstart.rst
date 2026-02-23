Quickstart
==========

Installation
------------

.. code-block:: bash

   pip install tensorquantlib

For development:

.. code-block:: bash

   git clone https://github.com/your-org/tensorquantlib.git
   cd tensorquantlib
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
