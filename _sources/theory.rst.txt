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


Second-Order Automatic Differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Second-order Greeks are computed via a **4-point finite-difference stencil**
on the forward pricing function:

.. math::

   \Gamma = \frac{\partial^2 V}{\partial S^2}
   \approx \frac{V(S+h, \sigma) + V(S-h, \sigma) - 2V_0}{h^2}

.. math::

   \text{Volga} = \frac{\partial^2 V}{\partial \sigma^2}
   \approx \frac{V(S, \sigma+h) + V(S, \sigma-h) - 2V_0}{h^2}

.. math::

   \text{Vanna} = \frac{\partial^2 V}{\partial S \, \partial \sigma}
   \approx \frac{V_{++} - V_{+-} - V_{-+} + V_{--}}{4 h_S h_\sigma}

All three Greeks share the same 4 corner evaluations, making the combined
computation more efficient than separate calls.


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

where :math:`\Phi` is the standard normal CDF.

**Greeks** are the partial derivatives:

- **Delta** = :math:`\Phi(d_1)`
- **Gamma** = :math:`\phi(d_1) / (S \sigma \sqrt{T})`
- **Vega** = :math:`S \phi(d_1) \sqrt{T}`
- **Theta** = :math:`-\frac{S \phi(d_1) \sigma}{2\sqrt{T}} - r K e^{-rT} \Phi(d_2)`
- **Rho** = :math:`K T e^{-rT} \Phi(d_2)`


Heston Stochastic Volatility
-----------------------------

The Heston (1993) model augments Black-Scholes with a mean-reverting
variance process:

.. math::

   dS = (r - q) S \, dt + \sqrt{V} S \, dW_S

.. math::

   dV = \kappa (\theta - V) \, dt + \xi \sqrt{V} \, dW_V

.. math::

   \text{Corr}(dW_S, dW_V) = \rho \, dt

where :math:`\kappa` is the mean-reversion speed, :math:`\theta` the long-run
variance, :math:`\xi` the vol-of-vol, and :math:`\rho` the correlation.

The **Feller condition** :math:`2\kappa\theta > \xi^2` ensures the variance
process stays strictly positive.

**Semi-analytic pricing** uses Gil-Pelaez inversion of the Heston
characteristic function:

.. math::

   C = S e^{-qT} P_1 - K e^{-rT} P_2

where :math:`P_j` are computed via numerical integration of the characteristic
function :math:`\phi_j(u)`.

**Monte Carlo** uses the Quadratic-Exponential (QE) scheme
(Andersen, 2008) for efficient discretisation of the CIR variance path.


American Options — Longstaff-Schwartz
-------------------------------------

The Longstaff-Schwartz (2001) algorithm prices American options via
**least-squares Monte Carlo (LSM)**:

1. Simulate :math:`N` paths of the underlying asset.
2. At each exercise date (backwards from maturity), regress the
   *continuation value* on polynomial basis functions of the spot price.
3. Exercise if the immediate payoff exceeds the estimated continuation value.
4. Discount and average to obtain the option price.

The exercise boundary is implicitly determined by the regression.
The library uses Laguerre polynomials for the basis functions.


Basket Options & Moment Matching
---------------------------------

A basket call on :math:`d` assets with weights :math:`w_i` pays
:math:`\max\!\bigl(\sum_i w_i S_i^T - K, \; 0\bigr)`.

**Monte Carlo**: simulate correlated geometric Brownian motions:

.. math::

   S_i^T = S_i^0 \exp\!\Bigl[\bigl(r - \tfrac{1}{2}\sigma_i^2\bigr)T
   + \sigma_i \sqrt{T}\, Z_i\Bigr]

where :math:`Z \sim \mathcal{N}(0, \Sigma)` and :math:`\Sigma` is the
correlation matrix (Cholesky-factored for sampling).

**Analytic approximation** (Gentle, 1993): match the first two moments of the
basket to a lognormal distribution, then apply the Black-Scholes formula with
the matched parameters :math:`\mu_B, \sigma_B`.


SABR Model
-----------

The SABR model (Hagan et al., 2002) for the forward rate :math:`F` and
stochastic volatility :math:`\alpha`:

.. math::

   dF = \alpha F^\beta \, dW_1, \quad d\alpha = \nu \alpha \, dW_2, \quad
   \text{Corr}(dW_1, dW_2) = \rho

The Hagan (2002) asymptotic approximation gives an implied Black volatility
:math:`\sigma_B(K, F)` in closed form:

**ATM** (:math:`K = F`):

.. math::

   \sigma_{ATM} = \frac{\alpha}{F^{1-\beta}}
   \left[1 + \left(\frac{(1-\beta)^2}{24}\frac{\alpha^2}{F^{2-2\beta}}
   + \frac{\rho \beta \nu \alpha}{4 F^{1-\beta}}
   + \frac{2-3\rho^2}{24}\nu^2\right) T\right]


SVI Parameterization
---------------------

The SVI (Stochastic Volatility Inspired) raw parameterization
(Gatheral, 2004) for total implied variance :math:`w(k)` as a function
of log-moneyness :math:`k = \ln(K/F)`:

.. math::

   w(k) = a + b \left[\rho (k - m) + \sqrt{(k - m)^2 + \sigma^2}\right]

where :math:`a, b, \rho, m, \sigma` are the 5 SVI parameters.
Implied volatility is :math:`\sigma_B = \sqrt{w(k) / T}`.


Vasicek Short-Rate Model
--------------------------

The Vasicek (1977) model for the short rate:

.. math::

   dr = \kappa (\theta - r) \, dt + \sigma \, dW

Zero-coupon bond price :math:`P(0, T)`:

.. math::

   P(0, T) = A(T) \exp(-B(T) r_0)

.. math::

   B(T) = \frac{1 - e^{-\kappa T}}{\kappa}, \quad
   A(T) = \exp\!\left[\left(\theta - \frac{\sigma^2}{2\kappa^2}\right)(B - T)
   - \frac{\sigma^2}{4\kappa} B^2\right]


CIR Model
----------

The Cox-Ingersoll-Ross (1985) model:

.. math::

   dr = \kappa (\theta - r) \, dt + \sigma \sqrt{r} \, dW

The square-root diffusion ensures non-negative rates when the Feller condition
:math:`2\kappa\theta > \sigma^2` holds.

Bond pricing is analytic with expressions involving :math:`\gamma = \sqrt{\kappa^2 + 2\sigma^2}`.


Merton Structural Credit Model
---------------------------------

The Merton (1974) model treats firm equity as a call option on assets:

.. math::

   P(\text{default}) = \Phi(-d_2), \quad
   d_2 = \frac{\ln(V/D) + (r - \tfrac{1}{2}\sigma_V^2) T}{\sigma_V \sqrt{T}}

where :math:`V` is the firm asset value, :math:`D` the debt face value,
and :math:`\sigma_V` the asset volatility.

The **credit spread** is derived as:
:math:`s = -\frac{1}{T}\ln\!\left(\frac{B}{D e^{-rT}}\right)` where
:math:`B = D e^{-rT} - \text{Put}(V, D, T)`.


Garman-Kohlhagen FX Model
---------------------------

The Garman-Kohlhagen (1983) model extends Black-Scholes for FX options,
treating the foreign risk-free rate :math:`r_f` as a continuous dividend yield:

.. math::

   C = S e^{-r_f T} \Phi(d_1) - K e^{-r_d T} \Phi(d_2)

.. math::

   d_1 = \frac{\ln(S/K) + (r_d - r_f + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}


Merton Jump-Diffusion
-----------------------

The Merton (1976) model adds Poisson jumps to GBM:

.. math::

   \frac{dS}{S} = (r - \lambda k) \, dt + \sigma \, dW + J \, dN

where :math:`N \sim \text{Poisson}(\lambda T)` and
:math:`\ln(1 + J) \sim \mathcal{N}(\mu_j, \sigma_j^2)`.

The price is a weighted sum of Black-Scholes prices:

.. math::

   C = \sum_{n=0}^{\infty} \frac{e^{-\lambda' T} (\lambda' T)^n}{n!}
   \cdot \text{BS}(S, K, T, r_n, \sigma_n)


Black-76 for IR Derivatives
-----------------------------

Interest rate caps, floors, and swaptions are priced using the
Black (1976) formula applied to forward rates:

.. math::

   \text{Caplet} = \tau \cdot N \cdot \text{DF} \cdot
   [F \Phi(d_1) - K \Phi(d_2)]

.. math::

   d_1 = \frac{\ln(F/K) + \tfrac{1}{2}\sigma^2 T}{\sigma\sqrt{T}}

A **cap** is a portfolio of caplets; a **floor** is a portfolio of floorlets.
**Swaptions** use the same formula with the forward swap rate as the underlying.


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
- S. L. Heston, *A Closed-Form Solution for Options with Stochastic Volatility*, RFS, 1993.
- L. B. G. Andersen, *Efficient Simulation of the Heston Stochastic Volatility Model*, J. Comp. Fin., 2008.
- F. A. Longstaff and E. S. Schwartz, *Valuing American Options by Simulation*, RFS, 2001.
- P. S. Hagan et al., *Managing Smile Risk*, Wilmott Magazine, 2002.
- J. Gatheral, *A Parsimonious Arbitrage-Free Implied Volatility Parameterization* (SVI), 2004.
- O. A. Vasicek, *An Equilibrium Characterization of the Term Structure*, J. Fin. Econ., 1977.
- J. C. Cox, J. E. Ingersoll, S. A. Ross, *A Theory of the Term Structure of Interest Rates*, Econometrica, 1985.
- R. C. Merton, *On the Pricing of Corporate Debt*, J. Finance, 1974.
- R. C. Merton, *Option Pricing When Underlying Stock Returns Are Discontinuous*, J. Fin. Econ., 1976.
- M. B. Garman and S. W. Kohlhagen, *Foreign Currency Option Values*, J. Int. Money & Finance, 1983.
- F. Black, *The Pricing of Commodity Contracts*, J. Fin. Econ., 1976.
- F. Black and M. Scholes, *The Pricing of Options and Corporate Liabilities*, J. Pol. Econ., 1973.
- P. Glasserman, *Monte Carlo Methods in Financial Engineering*, Springer, 2003.
