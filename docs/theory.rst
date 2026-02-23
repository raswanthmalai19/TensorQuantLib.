Theory & Background
===================

This page gives a brief summary of the mathematical ideas behind
TensorQuantLib.  For full details see the references.


Tensor-Train Decomposition
---------------------------

A *d*-dimensional tensor :math:`\mathbf{T} \in \mathbb{R}^{n_1 \times \cdots \times n_d}`
is represented in **Tensor-Train (TT) format** as:

.. math::

   T(i_1, \dots, i_d) = G_1(i_1) \, G_2(i_2) \, \cdots \, G_d(i_d)

where each *core* :math:`G_k(i_k)` is an :math:`r_{k-1} \times r_k` matrix
(with :math:`r_0 = r_d = 1`).  The integers :math:`r_k` are the **TT-ranks**.

**Storage**: :math:`\sum_k r_{k-1} \, n_k \, r_k` — linear in *d* for bounded
ranks, compared to :math:`\prod_k n_k` for the full tensor.


TT-SVD Algorithm
~~~~~~~~~~~~~~~~

We use the **TT-SVD** algorithm (Oseledets, 2011):

1. Reshape the tensor into a matrix :math:`(n_1, n_2 \cdots n_d)`.
2. Compute a truncated SVD; keep singular values above threshold.
3. The left factor becomes core :math:`G_1`; the right factor is reshaped and
   the process repeats for the remaining dimensions.

The tolerance parameter :math:`\varepsilon` controls the trade-off between
compression and accuracy.


TT Arithmetic
~~~~~~~~~~~~~

TT format supports closed-form arithmetic:

- **Addition**: rank-additive (:math:`r_k^{A+B} = r_k^A + r_k^B`)
- **Hadamard product**: rank-multiplicative (:math:`r_k^{A \circ B} = r_k^A \cdot r_k^B`)
- **Scaling**: no rank increase
- **Inner product**: :math:`O(d \, r^2 \, n)` via transfer-matrix contraction

Use ``tt_round()`` after addition / Hadamard to compress ranks back down.


Reverse-Mode Automatic Differentiation
---------------------------------------

TensorQuantLib implements **reverse-mode autodiff** (back-propagation) via a
tape-based computational graph:

1. Each ``Tensor`` operation records its inputs and a ``_backward`` closure.
2. ``backward()`` performs a topological sort (DFS) of the graph, then
   traverses in reverse, accumulating gradients via the chain rule.
3. Gradients are un-broadcast where necessary.

This gives exact first-order derivatives (Delta, Vega) of any pricing function
built from the supported operations.


Black-Scholes Model
-------------------

For a European option on a single asset with spot :math:`S`, strike :math:`K`,
risk-free rate :math:`r`, time to maturity :math:`T`, and volatility
:math:`\sigma`:

.. math::

   C = S \, \Phi(d_1) - K e^{-rT} \Phi(d_2)

.. math::

   d_1 = \frac{\ln(S/K) + (r + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad
   d_2 = d_1 - \sigma\sqrt{T}


Basket Options
--------------

A basket call on :math:`d` assets with equal weights pays
:math:`\max\!\bigl(\frac{1}{d}\sum_i S_i^T - K, \; 0\bigr)`.

We price via **Monte-Carlo** simulation of correlated geometric Brownian
motions:

.. math::

   S_i^T = S_i^0 \exp\!\Bigl[\bigl(r - \tfrac{1}{2}\sigma_i^2\bigr)T
   + \sigma_i \sqrt{T}\, Z_i\Bigr]

where :math:`Z \sim \mathcal{N}(0, \Sigma)` and :math:`\Sigma` is the
correlation matrix (Cholesky-factored for sampling).


Surrogate Pricing
-----------------

The **TTSurrogate** workflow:

1. Build a pricing grid over spot prices (one axis per asset).
2. Compress the grid using TT-SVD — achieving 10–1000× compression.
3. Evaluate prices at new points via **multi-linear interpolation** on the
   compressed TT cores, avoiding full grid reconstruction.
4. Compute Greeks via **finite differences** on the surrogate.

This enables real-time pricing of 3–5 asset options on a laptop.


References
----------

- I. V. Oseledets, *Tensor-Train Decomposition*, SIAM J. Sci. Comput., 2011.
- F. Black and M. Scholes, *The Pricing of Options and Corporate Liabilities*,
  J. Political Economy, 1973.
- P. Glasserman, *Monte Carlo Methods in Financial Engineering*, Springer, 2003.
