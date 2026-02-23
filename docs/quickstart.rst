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
       n_assets=3,
       n_points=30,
       strike=100.0,
       r=0.05,
       T=1.0,
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

   from tensorquantlib import Tensor, bs_price_tensor

   S = Tensor([100.0], requires_grad=True)
   K, r, T, sigma = 100.0, 0.05, 1.0, 0.2

   price = bs_price_tensor(S, K, r, T, sigma, option_type="call")
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
   T = np.random.randn(20, 20, 20, 20)

   # Compress with tolerance
   cores = tt_svd(T, eps=1e-6)

   # Check compression
   ratio = tt_compression_ratio(cores, T.shape)
   print(f"Compression ratio: {ratio:.1f}×")

   # Reconstruct
   T_approx = tt_to_full(cores)
   error = np.linalg.norm(T - T_approx) / np.linalg.norm(T)
   print(f"Relative error: {error:.2e}")
