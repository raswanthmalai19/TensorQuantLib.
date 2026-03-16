"""Tests for second-order automatic differentiation — tensorquantlib.core.second_order."""

import numpy as np
import pytest

from tensorquantlib.core.tensor import Tensor
from tensorquantlib.core.second_order import (
    hvp,
    hessian,
    hessian_diag,
    vhp,
    mixed_partial,
    gamma_autograd,
    vanna_autograd,
    volga_autograd,
)
from tensorquantlib.finance.black_scholes import (
    bs_gamma,
    bs_price_tensor,
    bs_vega,
)

# Standard Black-Scholes parameters for all tests
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

# Weight matrix for quadratic: f(x) = x^T W x
# W = [[1, 1.5], [1.5, 2]]  →  Hessian H = W + W^T = [[2, 3], [3, 4]]
_W_QUAD = np.array([[1.0, 1.5], [1.5, 2.0]])


def _quadratic(x: Tensor) -> Tensor:
    """f(x) = x^T W x  where W = [[1,1.5],[1.5,2]].  Hessian = [[2,3],[3,4]]."""
    Wx = x @ _W_QUAD   # Tensor @ ndarray → Tensor via _ensure_tensor
    return (x * Wx).sum()


def _scalar_sq(x: Tensor) -> Tensor:
    """f(x) = x^2.  grad = 2x, Hessian = 2."""
    return x * x


# Smooth 2D function for symmetry test: f(x) = sum(x^3) + ||x||^2
def _smooth_2d(x: Tensor) -> Tensor:
    """f(x) = sum(x^3) + ||x||^2.  Hessian is diagonal (6*x_i on diag)."""
    return (x ** 3).sum() + (x * x).sum()


# ─────────────────────────────────────────────────────────────────────
# hvp tests
# ─────────────────────────────────────────────────────────────────────

class TestHVP:
    def test_quadratic_exact(self):
        """Hessian of quadratic is constant — HVP should be exact."""
        x = Tensor(np.array([1.0, 2.0]))
        v = np.array([1.0, 0.0])
        # H = [[2, 3], [3, 4]], H @ [1, 0] = [2, 3]
        result = hvp(_quadratic, x, v)
        np.testing.assert_allclose(result, [2.0, 3.0], atol=1e-8)

    def test_quadratic_second_direction(self):
        x = Tensor(np.array([1.0, 2.0]))
        v = np.array([0.0, 1.0])
        # H @ [0, 1] = [3, 4]
        result = hvp(_quadratic, x, v)
        np.testing.assert_allclose(result, [3.0, 4.0], atol=1e-8)

    def test_scalar_function(self):
        """For f(x)=x^2, Hessian is 2. HVP with v=1 gives 2."""
        x = Tensor(np.array(3.0))
        result = hvp(_scalar_sq, x, np.array(1.0))
        np.testing.assert_allclose(result, 2.0, atol=1e-7)

    def test_hvp_vs_numerical(self):
        """HVP should agree with finite-diff Hessian-vector product."""
        x_data = np.array([0.5, -0.3])
        v = np.array([0.7, 0.4])
        x = Tensor(x_data.copy())

        h_ref = hvp(_quadratic, x, v)  # already uses FD on gradients

        # Cross-check: compute full Hessian numerically and multiply by v
        H = hessian(_quadratic, x)
        expected = H @ v
        np.testing.assert_allclose(h_ref, expected, atol=1e-7)


# ─────────────────────────────────────────────────────────────────────
# hessian_diag tests
# ─────────────────────────────────────────────────────────────────────

class TestHessianDiag:
    def test_quadratic_diagonal(self):
        """Diagonal of Hessian [[2,3],[3,4]] is [2, 4]."""
        x = Tensor(np.array([1.0, 2.0]))
        diag = hessian_diag(_quadratic, x)
        np.testing.assert_allclose(diag, [2.0, 4.0], atol=1e-8)

    def test_matches_full_hessian_diagonal(self):
        """hessian_diag should equal np.diag(hessian(...))."""
        x = Tensor(np.array([1.0, -0.5]))
        diag = hessian_diag(_quadratic, x)
        H = hessian(_quadratic, x)
        np.testing.assert_allclose(diag, np.diag(H), atol=1e-8)

    def test_scalar_squared(self):
        """For f(x) = x^2, diagonal Hessian = [2]."""
        x = Tensor(np.array(5.0))
        diag = hessian_diag(_scalar_sq, x)
        np.testing.assert_allclose(diag, 2.0, atol=1e-7)


# ─────────────────────────────────────────────────────────────────────
# hessian (full matrix) tests
# ─────────────────────────────────────────────────────────────────────

class TestHessian:
    def test_quadratic_full_hessian(self):
        """Hessian of quadratic is constant [[2,3],[3,4]]."""
        x = Tensor(np.array([0.0, 0.0]))
        H = hessian(_quadratic, x)
        expected = np.array([[2.0, 3.0], [3.0, 4.0]])
        np.testing.assert_allclose(H, expected, atol=1e-7)

    def test_hessian_symmetry(self):
        """Hessian of a smooth function should be symmetric."""
        x = Tensor(np.array([0.5, 0.8]))
        H = hessian(_smooth_2d, x)
        np.testing.assert_allclose(H, H.T, atol=1e-7)

    def test_hessian_shape(self):
        x = Tensor(np.array([1.0, 2.0]))
        H = hessian(_quadratic, x)
        assert H.shape == (2, 2)


# ─────────────────────────────────────────────────────────────────────
# vhp tests
# ─────────────────────────────────────────────────────────────────────

class TestVHP:
    def test_vhp_equals_hvp_for_symmetric(self):
        """For smooth functions, vhp(v) == hvp(v)."""
        x = Tensor(np.array([1.0, 2.0]))
        v = np.array([0.4, 0.6])
        assert np.allclose(hvp(_quadratic, x, v), vhp(_quadratic, x, v), atol=1e-10)


# ─────────────────────────────────────────────────────────────────────
# mixed_partial tests
# ─────────────────────────────────────────────────────────────────────

class TestMixedPartial:
    def test_cross_term_of_xy(self):
        """f(x,y) = x*y  → d²f/dxdy = 1."""
        def _xy(t1: Tensor, t2: Tensor) -> Tensor:
            return t1 * t2

        x1 = Tensor(np.array(2.0))
        x2 = Tensor(np.array(3.0))
        result = mixed_partial(_xy, x1, x2)
        np.testing.assert_allclose(result, 1.0, atol=1e-5)

    def test_cross_term_of_x2y(self):
        """f(x,y) = x^2 * y  → d²f/dxdy = 2x."""
        def _x2y(t1: Tensor, t2: Tensor) -> Tensor:
            return t1 ** 2 * t2

        x_val = 3.0
        x1 = Tensor(np.array(x_val))
        x2 = Tensor(np.array(1.0))
        result = mixed_partial(_x2y, x1, x2)
        expected = 2.0 * x_val  # 2x
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_no_cross_term(self):
        """f(x,y) = x^2 + y^2  → d²f/dxdy = 0."""
        def _sum_sq(t1: Tensor, t2: Tensor) -> Tensor:
            return t1 ** 2 + t2 ** 2

        x1 = Tensor(np.array(1.5))
        x2 = Tensor(np.array(2.5))
        result = mixed_partial(_sum_sq, x1, x2)
        np.testing.assert_allclose(result, 0.0, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────
# gamma_autograd tests
# ─────────────────────────────────────────────────────────────────────

class TestGammaAutograd:
    def test_gamma_matches_analytic_atm(self):
        """Gamma from autograd should match Black-Scholes analytic Gamma (ATM)."""
        g_ana = bs_gamma(S, K, T, r, sigma)
        g_auto = gamma_autograd(bs_price_tensor, S, K, T, r, sigma)
        np.testing.assert_allclose(g_auto, g_ana, rtol=1e-3)

    def test_gamma_matches_analytic_itm(self):
        g_ana = bs_gamma(110.0, K, T, r, sigma)
        g_auto = gamma_autograd(bs_price_tensor, 110.0, K, T, r, sigma)
        np.testing.assert_allclose(g_auto, g_ana, rtol=1e-3)

    def test_gamma_matches_analytic_otm(self):
        g_ana = bs_gamma(90.0, K, T, r, sigma)
        g_auto = gamma_autograd(bs_price_tensor, 90.0, K, T, r, sigma)
        np.testing.assert_allclose(g_auto, g_ana, rtol=1e-3)

    def test_gamma_positive(self):
        """Gamma is always non-negative for vanilla options."""
        g = gamma_autograd(bs_price_tensor, S, K, T, r, sigma)
        assert g >= 0.0

    def test_gamma_put_call_equal(self):
        """Put and call Gammas are equal at the same parameters."""
        g_call = gamma_autograd(bs_price_tensor, S, K, T, r, sigma, option_type="call")
        g_put = gamma_autograd(bs_price_tensor, S, K, T, r, sigma, option_type="put")
        np.testing.assert_allclose(g_call, g_put, rtol=1e-4)


# ─────────────────────────────────────────────────────────────────────
# vanna_autograd tests
# ─────────────────────────────────────────────────────────────────────

class TestVannaAutograd:
    def _analytic_vanna(self, S, K, T, r, sigma, q=0.0):
        """Black-Scholes analytic Vanna: -exp(-qT) * d2 * N'(d1) / sigma."""
        from scipy.stats import norm
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return -np.exp(-q * T) * norm.pdf(d1) * d2 / sigma

    def test_vanna_sign_itm(self):
        """Vanna for ITM call is typically negative."""
        v = vanna_autograd(bs_price_tensor, 110.0, K, T, r, sigma)
        # ITM call: d2 > 0, so vanna < 0
        assert v < 0

    def test_vanna_matches_analytic_atm(self):
        """Vanna from autograd should be close to analytic formula."""
        v_auto = vanna_autograd(bs_price_tensor, S, K, T, r, sigma)
        v_ana = self._analytic_vanna(S, K, T, r, sigma)
        np.testing.assert_allclose(v_auto, v_ana, rtol=1e-2)

    def test_vanna_call_vs_put(self):
        """Vanna is the same for calls and puts at the same strike."""
        v_call = vanna_autograd(bs_price_tensor, S, K, T, r, sigma, option_type="call")
        v_put = vanna_autograd(bs_price_tensor, S, K, T, r, sigma, option_type="put")
        np.testing.assert_allclose(v_call, v_put, rtol=1e-3)


# ─────────────────────────────────────────────────────────────────────
# volga_autograd tests
# ─────────────────────────────────────────────────────────────────────

class TestVolgaAutograd:
    def _analytic_volga(self, S, K, T, r, sigma, q=0.0):
        """Black-Scholes analytic Volga: Vega * d1 * d2 / sigma."""
        from scipy.stats import norm
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        return vega * d1 * d2 / sigma

    def test_volga_positive_atm(self):
        """Volga is non-negative for vanilla options."""
        v = volga_autograd(bs_price_tensor, S, K, T, r, sigma)
        assert v >= 0.0

    def test_volga_matches_analytic_atm(self):
        v_auto = volga_autograd(bs_price_tensor, S, K, T, r, sigma)
        v_ana = self._analytic_volga(S, K, T, r, sigma)
        np.testing.assert_allclose(v_auto, v_ana, rtol=1e-2)

    def test_volga_equals_vega_derivative(self):
        """Volga = dVega/dsigma — verify via FD on vega."""
        h = 0.001
        def _vega(sig):
            t_s = Tensor(np.array(S), requires_grad=False)
            t_sig = Tensor(np.array(sig), requires_grad=True)
            p = bs_price_tensor(t_s, K, T, r, t_sig, 0.0, "call")
            p.backward()
            return float(t_sig.grad)

        volga_fd = (_vega(sigma + h) - _vega(sigma - h)) / (2 * h)
        volga_auto = volga_autograd(bs_price_tensor, S, K, T, r, sigma)
        np.testing.assert_allclose(volga_auto, volga_fd, rtol=1e-3)
