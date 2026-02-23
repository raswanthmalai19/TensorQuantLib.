"""Tests for all differentiable operations — gradient checks via numerical differences."""

import numpy as np
import pytest
from tensorquantlib.core.tensor import Tensor
from tensorquantlib.core import ops
from tensorquantlib.utils.validation import check_grad

TOL = 1e-5


# ====================================================================== #
# Helper to run gradient check on a function
# ====================================================================== #
def _check(fn, *shapes, tol=TOL):
    """Create random inputs, run check_grad, assert pass."""
    np.random.seed(42)
    inputs = [Tensor(np.random.randn(*s) * 2 + 0.5, requires_grad=True) for s in shapes]
    result = check_grad(fn, inputs, tol=tol)
    assert result["passed"], (
        f"Gradient check failed: max_error={result['max_error']:.2e}, "
        f"errors={result['errors']}"
    )
    return result


# ====================================================================== #
# Elementwise Binary Ops
# ====================================================================== #
class TestAddGrad:
    def test_same_shape(self):
        _check(lambda a, b: a + b, (3, 4), (3, 4))

    def test_broadcast(self):
        _check(lambda a, b: a + b, (3, 1), (1, 4))

    def test_scalar_broadcast(self):
        _check(lambda a, b: a + b, (3, 4), (1,))

    def test_radd_scalar(self):
        """Test scalar + Tensor (uses __radd__)."""
        x = Tensor(np.array([2.0, 3.0]), requires_grad=True)
        z = 5.0 + x
        z.sum().backward()
        np.testing.assert_array_almost_equal(x.grad, [1.0, 1.0])


class TestSubGrad:
    def test_same_shape(self):
        _check(lambda a, b: a - b, (3, 4), (3, 4))

    def test_broadcast(self):
        _check(lambda a, b: a - b, (3, 1), (1, 4))


class TestMulGrad:
    def test_same_shape(self):
        _check(lambda a, b: a * b, (3, 4), (3, 4))

    def test_broadcast(self):
        _check(lambda a, b: a * b, (3, 1), (1, 4))

    def test_rmul_scalar(self):
        x = Tensor(np.array([2.0, 3.0]), requires_grad=True)
        z = 3.0 * x
        z.sum().backward()
        np.testing.assert_array_almost_equal(x.grad, [3.0, 3.0])


class TestDivGrad:
    def test_same_shape(self):
        def fn(a, b):
            return a / b
        # Avoid near-zero denominators
        np.random.seed(42)
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.abs(np.random.randn(3, 4)) + 0.5, requires_grad=True)
        result = check_grad(fn, [a, b], tol=TOL)
        assert result["passed"], f"max_error={result['max_error']:.2e}"

    def test_broadcast(self):
        np.random.seed(42)
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.abs(np.random.randn(1, 4)) + 0.5, requires_grad=True)
        result = check_grad(lambda a, b: a / b, [a, b], tol=TOL)
        assert result["passed"]


class TestNegGrad:
    def test_neg(self):
        _check(lambda a: -a, (3, 4))


# ====================================================================== #
# Matmul
# ====================================================================== #
class TestMatmulGrad:
    def test_2d(self):
        _check(lambda a, b: a @ b, (3, 4), (4, 5))

    def test_vector(self):
        _check(lambda a, b: a @ b, (3, 4), (4,))


# ====================================================================== #
# Unary Ops
# ====================================================================== #
class TestExpGrad:
    def test_exp(self):
        # Use moderate values to avoid huge gradients
        np.random.seed(42)
        a = Tensor(np.random.randn(3, 4) * 0.5, requires_grad=True)
        result = check_grad(lambda a: a.exp(), [a], tol=TOL)
        assert result["passed"]


class TestLogGrad:
    def test_log(self):
        np.random.seed(42)
        a = Tensor(np.abs(np.random.randn(3, 4)) + 0.5, requires_grad=True)
        result = check_grad(lambda a: a.log(), [a], tol=TOL)
        assert result["passed"]


class TestSqrtGrad:
    def test_sqrt(self):
        np.random.seed(42)
        a = Tensor(np.abs(np.random.randn(3, 4)) + 0.5, requires_grad=True)
        result = check_grad(lambda a: a.sqrt(), [a], tol=TOL)
        assert result["passed"]


class TestPowGrad:
    def test_square(self):
        _check(lambda a: a ** 2, (3, 4))

    def test_cube(self):
        _check(lambda a: a ** 3, (3, 4))

    def test_sqrt_via_pow(self):
        np.random.seed(42)
        a = Tensor(np.abs(np.random.randn(3, 4)) + 0.5, requires_grad=True)
        result = check_grad(lambda a: a ** 0.5, [a], tol=TOL)
        assert result["passed"]


# ====================================================================== #
# Reductions
# ====================================================================== #
class TestSumGrad:
    def test_sum_all(self):
        _check(lambda a: a.sum(), (3, 4))

    def test_sum_axis0(self):
        _check(lambda a: a.sum(axis=0), (3, 4))

    def test_sum_axis1(self):
        _check(lambda a: a.sum(axis=1), (3, 4))

    def test_sum_keepdims(self):
        _check(lambda a: a.sum(axis=0, keepdims=True), (3, 4))


class TestMeanGrad:
    def test_mean_all(self):
        _check(lambda a: a.mean(), (3, 4))

    def test_mean_axis0(self):
        _check(lambda a: a.mean(axis=0), (3, 4))

    def test_mean_axis1(self):
        _check(lambda a: a.mean(axis=1), (3, 4))


# ====================================================================== #
# Shape Ops
# ====================================================================== #
class TestReshapeGrad:
    def test_reshape(self):
        _check(lambda a: a.reshape(12), (3, 4))

    def test_reshape_2d(self):
        _check(lambda a: a.reshape(4, 3), (3, 4))


class TestTransposeGrad:
    def test_transpose(self):
        _check(lambda a: a.T, (3, 4))


# ====================================================================== #
# Activation / Clipping
# ====================================================================== #
class TestMaximumGrad:
    def test_relu(self):
        np.random.seed(42)
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        result = check_grad(lambda a: ops.maximum(a, 0.0), [a], tol=TOL)
        assert result["passed"]


class TestNormCdfGrad:
    def test_norm_cdf(self):
        np.random.seed(42)
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        result = check_grad(lambda a: ops.norm_cdf(a), [a], tol=TOL)
        assert result["passed"]


# ====================================================================== #
# Composite: Linear Regression
# ====================================================================== #
class TestLinearRegression:
    def test_gradient_matches_numerical(self):
        """Full integration test: linear regression MSE gradients."""
        np.random.seed(42)
        X_data = np.random.randn(20, 3)
        y_data = X_data @ np.array([1.5, -2.0, 0.5]) + 0.3 + np.random.randn(20) * 0.1

        X = Tensor(X_data)
        y = Tensor(y_data)
        W = Tensor(np.random.randn(3), requires_grad=True)
        b = Tensor(np.array([0.0]), requires_grad=True)

        def loss_fn(W, b):
            pred = X @ W + b
            diff = pred - y
            return (diff * diff).mean()

        result = check_grad(loss_fn, [W, b], tol=1e-4)
        assert result["passed"], f"Linear regression grad check failed: {result['max_error']:.2e}"


# ====================================================================== #
# Composite: Complex expression
# ====================================================================== #
class TestCompositeExpressions:
    def test_polynomial(self):
        """z = 3*x^2 + 2*x + 1, dz/dx = 6*x + 2."""
        x = Tensor(np.array([2.0]), requires_grad=True)
        z = Tensor(3.0) * x ** 2 + Tensor(2.0) * x + Tensor(1.0)
        z.sum().backward()
        expected = 6.0 * 2.0 + 2.0  # 14.0
        np.testing.assert_almost_equal(x.grad[0], expected, decimal=5)

    def test_exp_log_chain(self):
        """z = log(exp(x)) = x, so dz/dx = 1."""
        np.random.seed(42)
        x = Tensor(np.random.randn(5) * 0.5, requires_grad=True)
        z = x.exp().log()
        z.sum().backward()
        np.testing.assert_array_almost_equal(x.grad, np.ones(5), decimal=5)

    def test_complex_chain(self):
        """z = sum(sqrt(x^2 + 1)), dz/dx_i = x_i / sqrt(x_i^2 + 1)."""
        np.random.seed(42)
        x = Tensor(np.random.randn(4), requires_grad=True)
        z = (x ** 2 + Tensor(1.0)).sqrt().sum()
        z.backward()
        expected = x.data / np.sqrt(x.data ** 2 + 1)
        np.testing.assert_array_almost_equal(x.grad, expected, decimal=5)
