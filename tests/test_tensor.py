"""Tests for the Tensor class — construction, properties, backward, zero_grad."""

import numpy as np
import pytest
from tensorquantlib.core.tensor import Tensor


class TestTensorConstruction:
    def test_from_list(self):
        t = Tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)
        assert t.dtype == np.float64

    def test_from_scalar(self):
        t = Tensor(5.0)
        assert t.shape == ()
        assert t.item() == 5.0

    def test_from_numpy(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        t = Tensor(arr)
        assert t.dtype == np.float64  # always float64
        assert t.shape == (2, 2)

    def test_requires_grad_default_false(self):
        t = Tensor([1.0, 2.0])
        assert t.requires_grad is False
        assert t.grad is None

    def test_requires_grad_true(self):
        t = Tensor([1.0, 2.0], requires_grad=True)
        assert t.requires_grad is True

    def test_properties(self):
        t = Tensor(np.zeros((3, 4, 5)))
        assert t.shape == (3, 4, 5)
        assert t.ndim == 3
        assert t.size == 60

    def test_repr(self):
        t = Tensor([1.0], requires_grad=True)
        r = repr(t)
        assert "Tensor" in r
        assert "requires_grad=True" in r


class TestTensorBackward:
    def test_simple_add_backward(self):
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0], requires_grad=True)
        c = a + b
        s = c.sum()
        s.backward()
        np.testing.assert_array_equal(a.grad, [1.0, 1.0])
        np.testing.assert_array_equal(b.grad, [1.0, 1.0])

    def test_mul_backward(self):
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0], requires_grad=True)
        c = a * b
        s = c.sum()
        s.backward()
        np.testing.assert_array_almost_equal(a.grad, [4.0, 5.0])
        np.testing.assert_array_almost_equal(b.grad, [2.0, 3.0])

    def test_fan_out_gradient_accumulation(self):
        """When a tensor is used twice, gradients must accumulate."""
        x = Tensor([3.0], requires_grad=True)
        z = x + x  # dz/dx = 2
        z.sum().backward()
        np.testing.assert_array_almost_equal(x.grad, [2.0])

    def test_fan_out_mul(self):
        x = Tensor([3.0], requires_grad=True)
        z = x * x  # dz/dx = 2x = 6
        z.sum().backward()
        np.testing.assert_array_almost_equal(x.grad, [6.0])

    def test_chain_backward(self):
        """Multi-step chain: z = (x + y) * y."""
        x = Tensor([2.0], requires_grad=True)
        y = Tensor([3.0], requires_grad=True)
        s = x + y  # 5
        z = s * y  # 15; dz/dx = y = 3, dz/dy = s + y = 5 + 3 = 8
        z.sum().backward()
        np.testing.assert_array_almost_equal(x.grad, [3.0])
        np.testing.assert_array_almost_equal(y.grad, [8.0])

    def test_zero_grad(self):
        x = Tensor([2.0], requires_grad=True)
        z = x * x
        z.sum().backward()
        assert x.grad is not None
        x.zero_grad()
        assert x.grad is None


class TestTensorInplaceReject:
    def test_iadd_raises(self):
        x = Tensor([1.0], requires_grad=True)
        with pytest.raises(NotImplementedError):
            x += Tensor([1.0])

    def test_isub_raises(self):
        x = Tensor([1.0], requires_grad=True)
        with pytest.raises(NotImplementedError):
            x -= Tensor([1.0])

    def test_imul_raises(self):
        x = Tensor([1.0], requires_grad=True)
        with pytest.raises(NotImplementedError):
            x *= Tensor([2.0])
