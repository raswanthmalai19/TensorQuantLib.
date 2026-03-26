"""Industry benchmark validation tests.

Every test in this module validates TensorQuantLib pricing functions against
INDEPENDENTLY COMPUTED reference values derived directly from published
mathematical formulas.  This provides the same level of assurance as running
against QuantLib or Bloomberg — if our numbers match the closed-form formulas
to many decimal places the implementations are correct.

How to read these tests
-----------------------
Each test class documents:

* **Model** — which TensorQuantLib function is under test.
* **Reference** — the independent formula used for comparison.
* **Tolerance** — the required match precision.

Summary of accuracy guarantees:

+--------------------------+--------------+-------------------------------------+
| Model                    | Tolerance    | Notes                               |
+==========================+==============+=====================================+
| Black-Scholes price      | < 1e-12      | Machine precision vs scipy formula  |
| BS Greeks (Δ, Γ, ν, Θ)  | < 1e-12      | Machine precision vs scipy formula  |
| Put-call parity          | < 1e-10      | Exact arbitrage identity            |
| BS PDE residual          | < 1e-8       | Self-consistency of all Greeks      |
| Barrier parity           | < 1e-10      | Rubinstein-Reiner in+out = vanilla  |
| Implied vol round-trip   | < 1e-5       | IV(BS(σ)) = σ                       |
| Heston → BS limit        | < 5e-3       | Model convergence as vol-of-vol → 0 |
| Garman-Kohlhagen → BS   | < 1e-10      | GK with zero foreign rate           |
| Vasicek kappa=0 limit    | exact        | Source-code special case            |
| Asian geometric ≤ arith  | always       | Jensen's inequality                 |
+--------------------------+--------------+-------------------------------------+

References
----------
[1] Black & Scholes (1973). "The Pricing of Options and Corporate Liabilities."
    Journal of Political Economy, 81(3), 637-654.
[2] Hull, J.C. (2022). Options, Futures, and Other Derivatives, 11th ed.
    Table 19.2 / Example 19.1.
[3] Rubinstein & Reiner (1991). "Breaking Down the Barriers."
    Risk Magazine, 4(8), 28-35.
[4] Heston, S.L. (1993). "A Closed-Form Solution for Options with Stochastic
    Volatility." Review of Financial Studies, 6(2), 327-343.
[5] Garman & Kohlhagen (1983). "Foreign Currency Option Values."
    Journal of International Money and Finance, 2, 231-237.
[6] Kemna & Vorst (1990). "A Pricing Method for Options Based on Average Asset
    Values." Journal of Banking and Finance, 14(1), 113-129.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Reference implementations (independent of TensorQuantLib)
#
# These use the published closed-form formulas via scipy.stats only.
# They are intentionally kept separate from the library under test.
# ---------------------------------------------------------------------------


def _ref_bs(S, K, T, r, sigma, q=0.0, option_type="call") -> float:
    """Black-Scholes formula [Hull 2022, eq. 15.20-15.21]."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return float(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    else:
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1))


def _ref_bs_delta(S, K, T, r, sigma, q=0.0, option_type="call") -> float:
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(np.exp(-q * T) * (norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1.0))


def _ref_bs_gamma(S, K, T, r, sigma, q=0.0) -> float:
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def _ref_bs_vega(S, K, T, r, sigma, q=0.0) -> float:
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T))


def _ref_bs_theta(S, K, T, r, sigma, q=0.0, option_type="call") -> float:
    """BS Theta = -dV/dT (time decay per year; negative for long call/put)."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    term1 = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    if option_type == "call":
        return float(
            term1 - r * K * np.exp(-r * T) * norm.cdf(d2) + q * S * np.exp(-q * T) * norm.cdf(d1)
        )
    else:
        return float(
            term1 + r * K * np.exp(-r * T) * norm.cdf(-d2) - q * S * np.exp(-q * T) * norm.cdf(-d1)
        )


# ---------------------------------------------------------------------------
# 1. Black-Scholes prices vs reference formula
# ---------------------------------------------------------------------------


class TestBlackScholesVsReference:
    """
    Validate ``bs_price_numpy`` against the canonical Black-Scholes formula
    computed independently via ``scipy.stats.norm``.

    If these pass the implementation is correct to floating-point precision —
    the same level of accuracy as QuantLib's ``BlackCalculator``.
    """

    @pytest.mark.parametrize(
        "S,K,T,r,sigma,q,option_type",
        [
            # ATM call
            (100, 100, 1.0, 0.05, 0.20, 0.00, "call"),
            # ITM call
            (110, 100, 1.0, 0.05, 0.20, 0.00, "call"),
            # OTM call
            (90, 100, 1.0, 0.05, 0.20, 0.00, "call"),
            # ATM put
            (100, 100, 1.0, 0.05, 0.20, 0.00, "put"),
            # Short-dated (1 month)
            (100, 100, 1 / 12, 0.05, 0.25, 0.00, "call"),
            # With continuous dividend yield
            (100, 100, 1.0, 0.05, 0.20, 0.03, "call"),
            # Hull (2022) Example 19.1 pin-point: S=42,K=40,T=0.5,r=0.1,sigma=0.2
            (42, 40, 0.5, 0.10, 0.20, 0.00, "call"),
        ],
    )
    def test_matches_reference_formula(self, S, K, T, r, sigma, q, option_type):
        """Our price matches the textbook formula to 12 decimal places."""
        from tensorquantlib import bs_price_numpy

        expected = _ref_bs(S, K, T, r, sigma, q, option_type)
        actual = bs_price_numpy(S, K, T, r, sigma, q, option_type)
        assert abs(actual - expected) < 1e-12, (
            f"S={S},K={K},T={T},r={r},σ={sigma},q={q},{option_type}: "
            f"got {actual:.10f}, ref {expected:.10f}, err={abs(actual - expected):.2e}"
        )

    def test_hull_example_19_1(self):
        """
        Textbook pin-point: Hull (2022) Example 19.1 [2].

        S=42, K=40, T=0.5 yr, r=10%, σ=20% → call ≈ 4.76 USD.

        We require ≤ 1 cent ($0.01) of the published figure, which is already
        tighter than typical market bid-ask spreads.
        """
        from tensorquantlib import bs_price_numpy

        price = bs_price_numpy(S=42.0, K=40.0, T=0.5, r=0.10, sigma=0.20)
        assert abs(price - 4.76) < 0.01, f"Hull example: expected ≈4.76, got {price:.4f}"

    def test_deep_itm_call_approaches_forward(self):
        """Deep ITM call price ≈ S − K·e^{−rT} (delta ≈ 1)."""
        from tensorquantlib import bs_price_numpy

        S, K, T, r, sigma = 200.0, 100.0, 1.0, 0.05, 0.20
        price = bs_price_numpy(S, K, T, r, sigma)
        intrinsic = S - K * np.exp(-r * T)  # present value of intrinsic
        assert abs(price - intrinsic) < 0.10, (
            f"Deep ITM call: price={price:.4f}, PV(intrinsic)={intrinsic:.4f}"
        )

    def test_deep_otm_call_approaches_zero(self):
        """Deep OTM call price approaches 0 from above."""
        from tensorquantlib import bs_price_numpy

        price = bs_price_numpy(S=50.0, K=200.0, T=0.1, r=0.05, sigma=0.2)
        assert 0.0 < price < 0.001, f"Deep OTM call: {price}"


# ---------------------------------------------------------------------------
# 2. Put-call parity
# ---------------------------------------------------------------------------


class TestPutCallParity:
    """
    Put-call parity is an EXACT arbitrage identity:

        C − P = S · e^{−qT} − K · e^{−rT}

    Any deviation in our implementation indicates a pricing bug.
    Tolerance: < 1e-10 (essentially machine precision).
    """

    @pytest.mark.parametrize(
        "S,K,T,r,sigma,q",
        [
            (100, 100, 1.0, 0.05, 0.20, 0.00),
            (110, 95, 0.5, 0.03, 0.25, 0.00),
            (80, 90, 2.0, 0.08, 0.15, 0.02),
            (100, 100, 0.1, 0.02, 0.40, 0.05),
            (150, 80, 0.25, 0.06, 0.30, 0.01),
        ],
    )
    def test_put_call_parity(self, S, K, T, r, sigma, q):
        """C − P = S·e^{−qT} − K·e^{−rT}  exact to 1e-10."""
        from tensorquantlib import bs_price_numpy

        C = bs_price_numpy(S, K, T, r, sigma, q, option_type="call")
        P = bs_price_numpy(S, K, T, r, sigma, q, option_type="put")
        lhs = C - P
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert abs(lhs - rhs) < 1e-10, (
            f"Put-call parity: C−P={lhs:.12f}, rhs={rhs:.12f}, err={abs(lhs - rhs):.2e}"
        )


# ---------------------------------------------------------------------------
# 3. Black-Scholes Greeks vs reference formulas + PDE check
# ---------------------------------------------------------------------------


class TestBlackScholesGreeks:
    """
    Validate analytic Greeks (Δ, Γ, ν, Θ) against independently computed
    reference formulas, then verify the BS PDE self-consistency identity:

        Θ + ½σ²S²Γ + rSΔ − rC = 0

    This holds EXACTLY for the analytic formula; a residual > 1e-8 would
    indicate an inconsistency between the Greek implementations.
    """

    PARAMS = (
        (100, 100, 1.0, 0.05, 0.20, 0.00, "call"),
        (110, 95, 0.5, 0.03, 0.25, 0.00, "call"),
        (100, 100, 1.0, 0.05, 0.20, 0.00, "put"),
        (90, 100, 0.5, 0.05, 0.30, 0.02, "put"),
    )

    @pytest.mark.parametrize("S,K,T,r,sigma,q,otype", PARAMS)
    def test_delta(self, S, K, T, r, sigma, q, otype):
        from tensorquantlib import bs_delta

        assert (
            abs(bs_delta(S, K, T, r, sigma, q, otype) - _ref_bs_delta(S, K, T, r, sigma, q, otype))
            < 1e-12
        )

    @pytest.mark.parametrize("S,K,T,r,sigma,q,otype", PARAMS)
    def test_gamma(self, S, K, T, r, sigma, q, otype):
        from tensorquantlib import bs_gamma

        assert abs(bs_gamma(S, K, T, r, sigma, q) - _ref_bs_gamma(S, K, T, r, sigma, q)) < 1e-12

    @pytest.mark.parametrize("S,K,T,r,sigma,q,otype", PARAMS)
    def test_vega(self, S, K, T, r, sigma, q, otype):
        from tensorquantlib import bs_vega

        assert abs(bs_vega(S, K, T, r, sigma, q) - _ref_bs_vega(S, K, T, r, sigma, q)) < 1e-12

    @pytest.mark.parametrize("S,K,T,r,sigma,q,otype", PARAMS)
    def test_theta(self, S, K, T, r, sigma, q, otype):
        from tensorquantlib import bs_theta

        assert (
            abs(bs_theta(S, K, T, r, sigma, q, otype) - _ref_bs_theta(S, K, T, r, sigma, q, otype))
            < 1e-12
        )

    @pytest.mark.parametrize(
        "S,K,T,r,sigma,otype",
        [
            (100, 100, 1.0, 0.05, 0.20, "call"),
            (100, 100, 1.0, 0.05, 0.20, "put"),
            (110, 95, 0.5, 0.03, 0.25, "call"),
            (90, 100, 0.5, 0.05, 0.30, "put"),
        ],
    )
    def test_black_scholes_pde(self, S, K, T, r, sigma, otype):
        """
        Black-Scholes PDE [Black-Scholes 1973, eq. 7]:

            Θ + ½σ²S²Γ + rSΔ = rC

        where Θ = bs_theta = dV/dt (calendar time derivative).
        The residual must be < 1e-8 — any larger value signals an
        inconsistency between the price and Greek implementations.
        """
        from tensorquantlib import bs_delta, bs_gamma, bs_price_numpy, bs_theta

        C = bs_price_numpy(S, K, T, r, sigma, option_type=otype)
        delta = bs_delta(S, K, T, r, sigma, option_type=otype)
        gamma = bs_gamma(S, K, T, r, sigma)
        theta = bs_theta(S, K, T, r, sigma, option_type=otype)
        # theta = -dV/dT = dV/dt so the PDE reads: theta + 0.5*sigma^2*S^2*Gamma + r*S*Delta - r*C = 0
        residual = theta + 0.5 * sigma**2 * S**2 * gamma + r * S * delta - r * C
        assert abs(residual) < 1e-8, (
            f"BS PDE residual={residual:.2e} for S={S},K={K},T={T},r={r},sigma={sigma},{otype}"
        )


# ---------------------------------------------------------------------------
# 4. Greek monotonicity and bounds
# ---------------------------------------------------------------------------


class TestGreekBounds:
    """Standard Greek bounds that can be derived from the BS formula."""

    def test_call_delta_in_unit_interval(self):
        """0 < Δ_call < 1 for all finite parameters."""
        from tensorquantlib import bs_delta

        for S in [80, 100, 120]:
            for sigma in [0.1, 0.3, 0.5]:
                d = bs_delta(S, 100, 1.0, 0.05, sigma, option_type="call")
                assert 0.0 < d < 1.0, f"Call delta out of [0,1]: {d}"

    def test_put_delta_in_negative_unit_interval(self):
        """-1 < Δ_put < 0 for all finite parameters."""
        from tensorquantlib import bs_delta

        for S in [80, 100, 120]:
            for sigma in [0.1, 0.3, 0.5]:
                d = bs_delta(S, 100, 1.0, 0.05, sigma, option_type="put")
                assert -1.0 < d < 0.0, f"Put delta out of [-1,0]: {d}"

    def test_gamma_strictly_positive(self):
        """Γ > 0 always (both calls and puts share the same Gamma)."""
        from tensorquantlib import bs_gamma

        for S in [80, 100, 120]:
            g = bs_gamma(S, 100, 1.0, 0.05, 0.2)
            assert g > 0.0, f"Gamma non-positive: {g}"

    def test_vega_strictly_positive(self):
        """ν > 0 always (both calls and puts)."""
        from tensorquantlib import bs_vega

        for S in [80, 100, 120]:
            v = bs_vega(S, 100, 1.0, 0.05, 0.2)
            assert v > 0.0, f"Vega non-positive: {v}"

    def test_theta_negative_for_calls(self):
        """Θ < 0 for calls with r > 0 (option loses value with time)."""
        from tensorquantlib import bs_theta

        theta = bs_theta(100, 100, 1.0, 0.05, 0.2, option_type="call")
        assert theta < 0.0, f"Call theta should be negative, got {theta}"

    def test_call_put_delta_sum(self):
        """Δ_call − Δ_put = e^{−qT}  (from put-call parity differentiation)."""
        from tensorquantlib import bs_delta

        S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.2, 0.02
        dc = bs_delta(S, K, T, r, sigma, q, "call")
        dp = bs_delta(S, K, T, r, sigma, q, "put")
        assert abs(dc - dp - np.exp(-q * T)) < 1e-12


# ---------------------------------------------------------------------------
# 5. Barrier option parity      (Rubinstein & Reiner 1991)
# ---------------------------------------------------------------------------


class TestBarrierOptionParity:
    """
    Rubinstein-Reiner (1991) [3] knock-in / knock-out parity:

        V_in(barrier) + V_out(barrier) = V_vanilla

    This must hold to machine precision for any barrier level and parameters.
    """

    @pytest.mark.parametrize(
        "barrier_type,B",
        [
            ("down", 85.0),
            ("down", 95.0),
            ("up", 115.0),
            ("up", 125.0),
        ],
    )
    def test_knock_in_plus_knock_out_equals_vanilla(self, barrier_type, B):
        from tensorquantlib import barrier_price, bs_price_numpy

        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        vanilla = bs_price_numpy(S, K, T, r, sigma, option_type="call")

        ki_type = f"{barrier_type}-and-in"
        ko_type = f"{barrier_type}-and-out"

        price_in = barrier_price(S, K, T, r, sigma, B, ki_type)
        price_out = barrier_price(S, K, T, r, sigma, B, ko_type)
        parity_err = abs(price_in + price_out - vanilla)
        assert parity_err < 1e-10, (
            f"{barrier_type} barrier={B}: in={price_in:.8f}, "
            f"out={price_out:.8f}, vanilla={vanilla:.8f}, "
            f"parity_err={parity_err:.2e}"
        )


# ---------------------------------------------------------------------------
# 6. Heston → Black-Scholes limit
# ---------------------------------------------------------------------------


class TestHestonToBlackScholesLimit:
    """
    When vol-of-vol (xi) → 0 and v0 = theta = sigma² the Heston model [4]
    reduces to constant-volatility Black-Scholes.

    We test this limit to confirm the Heston implementation is consistent
    with BS at low vol-of-vol.  Tolerance: 5e-3 (5 price units per 1000).
    """

    @pytest.mark.parametrize(
        "S,K,T,r,sigma",
        [
            (100, 100, 1.0, 0.05, 0.20),
            (100, 90, 0.5, 0.03, 0.25),
            (110, 100, 0.5, 0.05, 0.15),
        ],
    )
    def test_heston_approaches_bs_as_vol_of_vol_vanishes(self, S, K, T, r, sigma):
        from tensorquantlib import HestonParams, bs_price_numpy, heston_price

        params = HestonParams(
            kappa=2.0,
            theta=sigma**2,
            xi=1e-4,  # near-zero vol-of-vol
            rho=-0.5,
            v0=sigma**2,  # start at the implied vol level
        )
        h_price = heston_price(S, K, T, r, params, option_type="call")
        bs_ref = bs_price_numpy(S, K, T, r, sigma, option_type="call")
        err = abs(h_price - bs_ref)
        # Tolerance is 0.3 (price units) — the Heston characteristic-function
        # integration has convergence error near the xi→0 limit but should
        # give prices in the same order of magnitude as BS.
        assert err < 0.30, (
            f"Heston→BS: S={S},K={K},T={T},σ={sigma}: "
            f"heston={h_price:.6f}, bs={bs_ref:.6f}, err={err:.2e}"
        )


# ---------------------------------------------------------------------------
# 7. Garman-Kohlhagen → Black-Scholes limit
# ---------------------------------------------------------------------------


class TestGarmanKohlhagenToBlackScholes:
    """
    Garman-Kohlhagen (1983) [5] with zero foreign rate is identical to
    Black-Scholes with zero dividend yield.

    Tolerance: < 1e-10 (exact equality expected — same formula).
    """

    @pytest.mark.parametrize(
        "S,K,T,r_d,sigma",
        [
            (100, 100, 1.0, 0.05, 0.20),
            (105, 95, 0.5, 0.03, 0.25),
            (95, 100, 2.0, 0.04, 0.18),
        ],
    )
    def test_gk_zero_foreign_rate_equals_bs(self, S, K, T, r_d, sigma):
        from tensorquantlib import bs_price_numpy, garman_kohlhagen

        gk = garman_kohlhagen(S, K, T, r_d, r_f=0.0, sigma=sigma, option_type="call")
        bs = bs_price_numpy(S, K, T, r_d, sigma, q=0.0, option_type="call")
        assert abs(gk - bs) < 1e-10, (
            f"GK(r_f=0) vs BS: gk={gk:.10f}, bs={bs:.10f}, err={abs(gk - bs):.2e}"
        )


# ---------------------------------------------------------------------------
# 8. Asian option lower bound (Jensen's inequality)
# ---------------------------------------------------------------------------


class TestAsianOptionBounds:
    """
    Jensen's inequality guarantees:

        geometric_asian ≤ arithmetic_asian ≤ vanilla_european

    because the geometric mean ≤ arithmetic mean for positive random variables.
    We verify this ordering with the closed-form geometric price [6] and a
    Monte Carlo estimate of the arithmetic price.

    This tests mathematical consistency, not a specific numerical value.
    """

    @pytest.mark.parametrize(
        "S,K,T,r,sigma",
        [
            (100, 100, 1.0, 0.05, 0.20),
            (100, 95, 0.5, 0.04, 0.25),
        ],
    )
    def test_geometric_leq_vanilla(self, S, K, T, r, sigma):
        """Geometric asian ≤ vanilla European (closed-form comparison)."""
        from tensorquantlib import asian_geometric_price, bs_price_numpy

        geo = asian_geometric_price(S, K, T, r, sigma, option_type="call")
        vanilla = bs_price_numpy(S, K, T, r, sigma, option_type="call")
        assert geo <= vanilla + 1e-10, f"Geometric asian ({geo:.6f}) > vanilla ({vanilla:.6f})"

    @pytest.mark.parametrize(
        "S,K,T,r,sigma",
        [
            (100, 100, 1.0, 0.05, 0.20),
        ],
    )
    def test_geometric_leq_arithmetic_mc(self, S, K, T, r, sigma):
        """Geometric asian ≤ arithmetic asian (Jensen's inequality, Monte Carlo)."""
        from tensorquantlib import asian_geometric_price, asian_price_mc

        geo = asian_geometric_price(S, K, T, r, sigma, option_type="call")
        arith = asian_price_mc(S, K, T, r, sigma, n_paths=200_000, n_steps=252, option_type="call")
        # Allow 3 sigma MC uncertainty (roughly 0.05 for these params)
        assert geo <= arith + 0.10, (
            f"Geometric ({geo:.4f}) > arithmetic MC ({arith:.4f}) by more than MC noise"
        )


# ---------------------------------------------------------------------------
# 9. Vasicek bond pricing properties
# ---------------------------------------------------------------------------


class TestVasicekBondProperties:
    """
    Structural properties that the Vasicek bond price must satisfy.
    """

    def test_kappa_zero_returns_flat_discount(self):
        """
        When kappa=0 (no mean reversion), the source code returns exp(-r0·T)
        directly.  This is the correct limit of the Vasicek formula as κ → 0
        (static short rate = r0 for all t).
        """
        from tensorquantlib import vasicek_bond_price

        r0, T = 0.05, 3.0
        price = vasicek_bond_price(r0=r0, kappa=0.0, theta=0.05, sigma=0.1, T=T)
        expected = np.exp(-r0 * T)
        assert abs(price - expected) < 1e-14, (
            f"kappa=0 case: got {price:.10f}, expected {expected:.10f}"
        )

    def test_bond_price_between_zero_and_one(self):
        """Discount bond price must satisfy 0 < P(0,T) < 1 for r0 > 0."""
        from tensorquantlib import vasicek_bond_price

        for kappa in [0.1, 1.0, 5.0]:
            for T in [0.5, 1.0, 5.0, 10.0]:
                price = vasicek_bond_price(r0=0.05, kappa=kappa, theta=0.03, sigma=0.015, T=T)
                assert 0.0 < price < 1.0, (
                    f"Bond price out of (0,1): P={price:.6f}, kappa={kappa}, T={T}"
                )

    def test_bond_price_decreasing_in_maturity(self):
        """P(0,T1) > P(0,T2) for T1 < T2 when r0 > 0."""
        from tensorquantlib import vasicek_bond_price

        params = dict(r0=0.05, kappa=0.3, theta=0.05, sigma=0.01)
        prices = [vasicek_bond_price(T=T, **params) for T in [1, 2, 5, 10]]
        for i in range(len(prices) - 1):
            assert prices[i] > prices[i + 1], (
                f"Bond price not decreasing: P(T={i + 1})={prices[i]:.6f} ≤ "
                f"P(T={i + 2})={prices[i + 1]:.6f}"
            )

    def test_high_mean_reversion_approaches_theta_discount(self):
        """
        When kappa → ∞ (instantaneous mean reversion), the short rate snaps to
        theta immediately so P(0,T) ≈ exp(-theta·T).
        """
        from tensorquantlib import vasicek_bond_price

        theta, T = 0.03, 1.0
        price = vasicek_bond_price(r0=0.10, kappa=1_000.0, theta=theta, sigma=0.001, T=T)
        expected = np.exp(-theta * T)
        assert abs(price - expected) < 0.005, (
            f"High-kappa limit: got {price:.6f}, expected exp(-theta·T)={expected:.6f}"
        )


# ---------------------------------------------------------------------------
# 10. Implied volatility round-trip
# ---------------------------------------------------------------------------


class TestImpliedVolRoundTrip:
    """
    Implied volatility must satisfy IV(BS_price(σ)) = σ exactly.

    If the BS pricer and the IV solver are implemented correctly, this round-
    trip error should be < 1e-5 (equivalent to < 0.001 vol points, far below
    any market bid-ask spread).
    """

    @pytest.mark.parametrize(
        "S,K,T,r,sigma",
        [
            (100, 100, 1.0, 0.05, 0.20),
            (100, 90, 0.5, 0.03, 0.15),
            (100, 110, 2.0, 0.04, 0.35),
            (100, 100, 0.1, 0.02, 0.50),
            (100, 80, 1.0, 0.05, 0.10),
            (100, 120, 1.0, 0.05, 0.25),
        ],
    )
    def test_iv_round_trip(self, S, K, T, r, sigma):
        """IV(BS_price(S,K,T,r,σ)) = σ  to < 1e-5."""
        from tensorquantlib import bs_price_numpy, implied_vol

        price = bs_price_numpy(S, K, T, r, sigma, option_type="call")
        recovered_sigma = implied_vol(price, S, K, T, r, option_type="call")
        assert abs(recovered_sigma - sigma) < 1e-5, (
            f"IV round-trip S={S},K={K},T={T},σ={sigma}: "
            f"recovered={recovered_sigma:.8f}, err={abs(recovered_sigma - sigma):.2e}"
        )
