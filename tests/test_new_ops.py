"""Tests for the new autograd ops: sin, cos, tanh, abs, clip, where, softmax."""
from __future__ import annotations

import numpy as np
import pytest

from tensorquantlib.core.tensor import (
    Tensor,
    tensor_sin, tensor_cos, tensor_tanh, tensor_abs, tensor_clip,
    tensor_where, tensor_softmax,
)


def numerical_grad(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Compute numerical gradient via central differences."""
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        x_p, x_m = x.copy(), x.copy()
        x_p[idx] += eps
        x_m[idx] -= eps
        grad[idx] = (f(x_p) - f(x_m)) / (2 * eps)
    return grad


class TestSin:
    def test_forward(self):
        t = Tensor(np.array([0.0, np.pi / 2, np.pi]))
        out = tensor_sin(t)
        np.testing.assert_allclose(out.data, np.sin(t.data), atol=1e-10)

    def test_backward(self):
        x = np.array([0.5, 1.0, -0.5])
        t = Tensor(x.copy(), requires_grad=True)
        out = tensor_sin(t)
        out.sum().backward()
        expected = np.cos(x)
        np.testing.assert_allclose(t.grad, expected, atol=1e-10)

    def test_method_alias(self):
        t = Tensor(np.array([1.0]), requires_grad=True)
        t2 = t.sin()
        t2.backward()
        assert abs(t2.data.item() - np.sin(1.0)) < 1e-10


class TestCos:
    def test_forward(self):
        t = Tensor(np.array([0.0, np.pi / 2]))
        out = tensor_cos(t)
        np.testing.assert_allclose(out.data, np.cos(t.data), atol=1e-10)

    def test_backward(self):
        x = np.array([0.3, -0.7])
        t = Tensor(x.copy(), requires_grad=True)
        out = tensor_cos(t)
        out.sum().backward()
        np.testing.assert_allclose(t.grad, -np.sin(x), atol=1e-10)


class TestTanh:
    def test_forward(self):
        t = Tensor(np.array([0.0, 1.0, -1.0]))
        out = tensor_tanh(t)
        np.testing.assert_allclose(out.data, np.tanh(t.data), atol=1e-10)

    def test_backward(self):
        x = np.array([0.5, -0.5])
        t = Tensor(x.copy(), requires_grad=True)
        out = tensor_tanh(t)
        out.sum().backward()
        expected = 1.0 - np.tanh(x) ** 2
        np.testing.assert_allclose(t.grad, expected, atol=1e-10)

    def test_method_alias(self):
        t = Tensor(np.array([0.0]), requires_grad=True)
        t.tanh().backward()
        assert abs(t.grad.item() - 1.0) < 1e-10  # tanh'(0) = 1


class TestAbs:
    def test_forward_positive(self):
        t = Tensor(np.array([1.0, -2.0, 3.0]))
        out = tensor_abs(t)
        np.testing.assert_allclose(out.data, np.array([1.0, 2.0, 3.0]))

    def test_backward_sign(self):
        x = np.array([1.5, -0.8])
        t = Tensor(x.copy(), requires_grad=True)
        out = tensor_abs(t)
        out.sum().backward()
        np.testing.assert_allclose(t.grad, np.sign(x), atol=1e-10)

    def test_method_alias(self):
        t = Tensor(np.array([-3.0]), requires_grad=True)
        out = t.abs()
        assert out.data.item() == 3.0


class TestClip:
    def test_forward(self):
        t = Tensor(np.array([-2.0, 0.5, 3.0]))
        out = tensor_clip(t, -1.0, 2.0)
        np.testing.assert_allclose(out.data, np.array([-1.0, 0.5, 2.0]))

    def test_backward_pass_through(self):
        t = Tensor(np.array([0.5]), requires_grad=True)
        out = tensor_clip(t, 0.0, 1.0)
        out.backward()
        assert t.grad.item() == 1.0  # within bounds, gradient passes through

    def test_backward_clipped(self):
        t = Tensor(np.array([2.0]), requires_grad=True)
        out = tensor_clip(t, 0.0, 1.0)
        out.backward()
        assert t.grad.item() == 0.0  # outside bounds, gradient is 0

    def test_method_alias(self):
        t = Tensor(np.array([-5.0, 5.0]))
        out = t.clip(-2.0, 2.0)
        np.testing.assert_allclose(out.data, np.array([-2.0, 2.0]))


class TestWhere:
    def test_forward(self):
        cond = np.array([True, False, True])
        a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        b = Tensor(np.array([10.0, 20.0, 30.0]), requires_grad=True)
        out = tensor_where(cond, a, b)
        np.testing.assert_allclose(out.data, np.array([1.0, 20.0, 3.0]))

    def test_backward_a(self):
        cond = np.array([True, False])
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        b = Tensor(np.array([10.0, 20.0]), requires_grad=False)
        out = tensor_where(cond, a, b)
        out.sum().backward()
        np.testing.assert_allclose(a.grad, np.array([1.0, 0.0]))

    def test_backward_b(self):
        cond = np.array([True, False])
        a = Tensor(np.array([1.0, 2.0]), requires_grad=False)
        b = Tensor(np.array([10.0, 20.0]), requires_grad=True)
        out = tensor_where(cond, a, b)
        out.sum().backward()
        np.testing.assert_allclose(b.grad, np.array([0.0, 1.0]))


class TestSoftmax:
    def test_forward_sums_to_one(self):
        t = Tensor(np.array([1.0, 2.0, 3.0]))
        out = tensor_softmax(t, axis=0)
        assert abs(float(out.data.sum()) - 1.0) < 1e-10

    def test_backward_numerical(self):
        """Check backward against numerical gradient for scalar sum loss."""
        x_data = np.array([1.0, 2.0, 0.5])
        t = Tensor(x_data.copy(), requires_grad=True)
        out = tensor_softmax(t, axis=0)
        out.sum().backward()  # gradient of sum(softmax(x)) w.r.t. x

        # Numerical gradient
        def f(x: np.ndarray) -> float:
            ex = np.exp(x - x.max())
            return float((ex / ex.sum()).sum())

        ng = numerical_grad(f, x_data)
        np.testing.assert_allclose(t.grad, ng, atol=1e-5)

    def test_batch_softmax(self):
        """Softmax over axis 1 of a 2D tensor."""
        t = Tensor(np.ones((3, 4)))
        out = tensor_softmax(t, axis=1)
        np.testing.assert_allclose(out.data.sum(axis=1), np.ones(3), atol=1e-10)
