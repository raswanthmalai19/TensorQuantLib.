"""
Tensor class with reverse-mode automatic differentiation.

This is the core data structure for TensorQuantLib. Every computation
flows through Tensor objects, which track the computational graph
and enable gradient computation via backpropagation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Union

import numpy as np


class Tensor:
    """A multi-dimensional array with automatic differentiation support.

    Stores data as a NumPy float64 array, optionally tracking gradients.
    When requires_grad=True, all operations on this tensor build a
    computational graph that enables reverse-mode autodiff via .backward().

    Attributes:
        data: The underlying NumPy array (float64).
        grad: Gradient array, populated after .backward(). None until first backward pass.
        requires_grad: Whether this tensor participates in gradient computation.
    """

    def __init__(
        self,
        data: Union[np.ndarray, list[Any], float, int],
        requires_grad: bool = False,
        _children: tuple[Tensor, ...] = (),
        _op: str = "",
    ):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None
        self._backward: Callable[[], None] = lambda: None  # closure for local backward
        self._children = set(_children)
        self._op = _op  # label for debugging

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.data.dtype

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def T(self) -> Tensor:
        """Transpose (creates a new node in the graph)."""
        return tensor_transpose(self)

    # ------------------------------------------------------------------ #
    # Backward (reverse-mode autodiff)
    # ------------------------------------------------------------------ #
    def backward(self) -> None:
        """Compute gradients via reverse-mode automatic differentiation.

        Performs a topological sort of the computational graph, then
        propagates gradients from this tensor back to all ancestors
        with requires_grad=True.

        This tensor's grad is seeded with ones (dL/dL = 1).
        """
        # Build topological order via DFS
        topo: list[Tensor] = []
        visited: set[int] = set()

        def _build_topo(v: Tensor) -> None:
            vid = id(v)
            if vid not in visited:
                visited.add(vid)
                for child in v._children:
                    _build_topo(child)
                topo.append(v)

        _build_topo(self)

        # Seed gradient
        self.grad = np.ones_like(self.data)

        # Reverse pass
        for node in reversed(topo):
            node._backward()

    def zero_grad(self) -> None:
        """Reset gradient to None."""
        self.grad = None

    # ------------------------------------------------------------------ #
    # Dunder methods — delegate to ops module functions
    # ------------------------------------------------------------------ #
    def __add__(self, other: object) -> Tensor:
        other = _ensure_tensor(other)
        return tensor_add(self, other)

    def __radd__(self, other: object) -> Tensor:
        other = _ensure_tensor(other)
        return tensor_add(other, self)

    def __sub__(self, other: object) -> Tensor:
        other = _ensure_tensor(other)
        return tensor_sub(self, other)

    def __rsub__(self, other: object) -> Tensor:
        other = _ensure_tensor(other)
        return tensor_sub(other, self)

    def __mul__(self, other: object) -> Tensor:
        other = _ensure_tensor(other)
        return tensor_mul(self, other)

    def __rmul__(self, other: object) -> Tensor:
        other = _ensure_tensor(other)
        return tensor_mul(other, self)

    def __truediv__(self, other: object) -> Tensor:
        other = _ensure_tensor(other)
        return tensor_div(self, other)

    def __rtruediv__(self, other: object) -> Tensor:
        other = _ensure_tensor(other)
        return tensor_div(other, self)

    def __neg__(self) -> Tensor:
        return tensor_neg(self)

    def __matmul__(self, other: object) -> Tensor:
        other = _ensure_tensor(other)
        return tensor_matmul(self, other)

    def __rmatmul__(self, other: object) -> Tensor:
        other = _ensure_tensor(other)
        return tensor_matmul(other, self)

    def __pow__(self, exponent: Union[int, float]) -> Tensor:
        return tensor_pow(self, exponent)

    # Forbid in-place ops to protect the computational graph
    def __iadd__(self, other: object) -> Tensor:
        raise NotImplementedError(
            "In-place operations are not supported on Tensors with autograd. "
            "Use out-of-place operations instead: z = x + y"
        )

    def __isub__(self, other: object) -> Tensor:
        raise NotImplementedError("In-place sub not supported. Use z = x - y.")

    def __imul__(self, other: object) -> Tensor:
        raise NotImplementedError("In-place mul not supported. Use z = x * y.")

    def __itruediv__(self, other: object) -> Tensor:
        raise NotImplementedError("In-place div not supported. Use z = x / y.")

    # ------------------------------------------------------------------ #
    # Convenience methods that delegate to ops
    # ------------------------------------------------------------------ #
    def sum(
        self, axis: Union[int, tuple[int, ...]] | None = None, keepdims: bool = False
    ) -> Tensor:
        return tensor_sum(self, axis=axis, keepdims=keepdims)

    def mean(
        self, axis: Union[int, tuple[int, ...]] | None = None, keepdims: bool = False
    ) -> Tensor:
        return tensor_mean(self, axis=axis, keepdims=keepdims)

    def reshape(self, *shape: Union[int, tuple[int, ...], list[int]]) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            final_shape = tuple(shape[0])
        else:
            final_shape = tuple(int(s) for s in shape)  # type: ignore[arg-type]
        return tensor_reshape(self, final_shape)

    def exp(self) -> Tensor:
        return tensor_exp(self)

    def log(self) -> Tensor:
        return tensor_log(self)

    def sqrt(self) -> Tensor:
        return tensor_sqrt(self)

    def sin(self) -> Tensor:
        return tensor_sin(self)

    def cos(self) -> Tensor:
        return tensor_cos(self)

    def tanh(self) -> Tensor:
        return tensor_tanh(self)

    def abs(self) -> Tensor:
        return tensor_abs(self)

    def clip(self, a_min: float, a_max: float) -> Tensor:
        return tensor_clip(self, a_min, a_max)

    def item(self) -> float:
        """Return scalar value (only works for size-1 tensors)."""
        return float(self.data.item())

    def detach(self) -> Tensor:
        """Return a new Tensor with the same data but detached from the graph.

        The returned tensor has ``requires_grad=False`` and no ``_children``,
        so gradients will not flow through it. Use this to treat an
        intermediate result as a constant:

            y = x * x
            y_const = y.detach()  # gradients stop here
        """
        return Tensor(self.data.copy(), requires_grad=False)

    def free_graph(self) -> None:
        """Release all references to the computational graph.

        Clears ``_children`` and the ``_backward`` closure for this node
        and all ancestors, breaking reference cycles and allowing GC to
        reclaim memory.  Call after ``backward()`` in long-running loops
        to prevent unbounded memory growth.
        """
        visited: set[int] = set()

        def _free(v: Tensor) -> None:
            vid = id(v)
            if vid in visited:
                return
            visited.add(vid)
            for child in v._children:
                _free(child)
            v._children = set()
            v._backward = lambda: None

        _free(self)

    # ------------------------------------------------------------------ #
    # Representation
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        shape = self.data.shape
        if self.data.size <= 4:
            data_str = str(self.data.tolist())
        else:
            data_str = f"shape={shape}"
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        op_str = f", op='{self._op}'" if self._op else ""
        return f"Tensor({data_str}{grad_str}{op_str})"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: Any) -> Tensor:
        """Index into the tensor, returning a Tensor that participates in autograd.

        Supports any NumPy-compatible index (int, slice, tuple, boolean mask).
        Gradients flow back to the indexed positions of the original tensor.
        """
        return tensor_getitem(self, idx)


# ====================================================================== #
# Helper: ensure value is a Tensor
# ====================================================================== #
def _ensure_tensor(val: Union[np.ndarray, list[Any], float, int, Tensor, object]) -> Tensor:
    """Wrap scalars/arrays as Tensor if needed."""
    if isinstance(val, Tensor):
        return val
    return Tensor(val, requires_grad=False)  # type: ignore[arg-type]


# ====================================================================== #
# Unbroadcast helper (critical for correct gradients)
# ====================================================================== #
def _unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Sum out dimensions that were broadcast during forward pass.

    When an operation broadcasts (e.g., shape (3,1) + (1,4) → (3,4)),
    the backward pass must sum gradients along the broadcast dimensions
    to produce the correct gradient shape for each input.
    """
    # Sum out leading dimensions that were added
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    # Sum along axes where target had size 1 but grad has size > 1
    for i, dim in enumerate(target_shape):
        if dim == 1 and grad.shape[i] > 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


# ====================================================================== #
# Core differentiable operations
# ====================================================================== #


def tensor_add(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise addition: z = a + b."""
    out = Tensor(a.data + b.data, _children=(a, b), _op="+")
    out.requires_grad = a.requires_grad or b.requires_grad

    def _backward() -> None:
        assert out.grad is not None
        if a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            a.grad += _unbroadcast(out.grad, a.shape)
        if b.requires_grad:
            if b.grad is None:
                b.grad = np.zeros_like(b.data)
            b.grad += _unbroadcast(out.grad, b.shape)

    if out.requires_grad:
        out._backward = _backward
    return out


def tensor_sub(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise subtraction: z = a - b."""
    out = Tensor(a.data - b.data, _children=(a, b), _op="-")
    out.requires_grad = a.requires_grad or b.requires_grad

    def _backward() -> None:
        assert out.grad is not None
        if a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            a.grad += _unbroadcast(out.grad, a.shape)
        if b.requires_grad:
            if b.grad is None:
                b.grad = np.zeros_like(b.data)
            b.grad += _unbroadcast(-out.grad, b.shape)

    if out.requires_grad:
        out._backward = _backward
    return out


def tensor_mul(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise multiplication: z = a * b."""
    out = Tensor(a.data * b.data, _children=(a, b), _op="*")
    out.requires_grad = a.requires_grad or b.requires_grad

    def _backward() -> None:
        assert out.grad is not None
        if a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            a.grad += _unbroadcast(out.grad * b.data, a.shape)
        if b.requires_grad:
            if b.grad is None:
                b.grad = np.zeros_like(b.data)
            b.grad += _unbroadcast(out.grad * a.data, b.shape)

    if out.requires_grad:
        out._backward = _backward
    return out


def tensor_div(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise division: z = a / b."""
    out = Tensor(a.data / b.data, _children=(a, b), _op="/")
    out.requires_grad = a.requires_grad or b.requires_grad

    def _backward() -> None:
        assert out.grad is not None
        if a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            a.grad += _unbroadcast(out.grad / b.data, a.shape)
        if b.requires_grad:
            if b.grad is None:
                b.grad = np.zeros_like(b.data)
            b.grad += _unbroadcast(-out.grad * a.data / (b.data**2), b.shape)

    if out.requires_grad:
        out._backward = _backward
    return out


def tensor_neg(a: Tensor) -> Tensor:
    """Element-wise negation: z = -a.

    Args:
        a: Input tensor.

    Returns:
        New tensor with negated values. Gradient: dz/da = -1.

    Example:
        >>> x = Tensor(np.array([3.0, -2.0]), requires_grad=True)
        >>> y = tensor_neg(x)
        >>> y.data
        array([-3.,  2.])
    """
    out = Tensor(-a.data, _children=(a,), _op="neg")
    out.requires_grad = a.requires_grad

    def _backward() -> None:
        assert out.grad is not None
        if a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            a.grad += -out.grad

    if out.requires_grad:
        out._backward = _backward
    return out


def tensor_matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication: Z = A @ B."""
    out = Tensor(a.data @ b.data, _children=(a, b), _op="@")
    out.requires_grad = a.requires_grad or b.requires_grad

    def _backward() -> None:
        assert out.grad is not None
        if a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            # dL/dA = dL/dZ @ B^T
            if b.data.ndim == 1:
                # vector case: (m,n) @ (n,) -> (m,)
                a.grad += np.outer(out.grad, b.data)
            else:
                a.grad += out.grad @ b.data.T
        if b.requires_grad:
            if b.grad is None:
                b.grad = np.zeros_like(b.data)
            # dL/dB = A^T @ dL/dZ
            if a.data.ndim == 1:
                b.grad += np.outer(a.data, out.grad)
            else:
                b.grad += a.data.T @ out.grad

    if out.requires_grad:
        out._backward = _backward
    return out


def tensor_pow(a: Tensor, exponent: Union[int, float]) -> Tensor:
    """Power: z = a^exponent (exponent is a constant, not a Tensor)."""
    out = Tensor(a.data**exponent, _children=(a,), _op=f"**{exponent}")
    out.requires_grad = a.requires_grad

    def _backward() -> None:
        assert out.grad is not None
        if a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            a.grad += out.grad * exponent * (a.data ** (exponent - 1))

    if out.requires_grad:
        out._backward = _backward
    return out


def tensor_exp(a: Tensor) -> Tensor:
    """Element-wise exponential: z = exp(a).

    Args:
        a: Input tensor.

    Returns:
        New tensor with exp(a). Gradient: dz/da = exp(a).

    Example:
        >>> x = Tensor(np.array([0.0, 1.0]), requires_grad=True)
        >>> y = tensor_exp(x)
        >>> np.allclose(y.data, [1.0, np.e])
        True
    """
    out_data = np.exp(a.data)
    out = Tensor(out_data, _children=(a,), _op="exp")
    out.requires_grad = a.requires_grad

    def _backward() -> None:
        assert out.grad is not None
        if a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            a.grad += out.grad * out_data

    if out.requires_grad:
        out._backward = _backward
    return out


def tensor_log(a: Tensor) -> Tensor:
    """Element-wise natural logarithm: z = log(a).

    Input is clamped to a minimum of 1e-12 to avoid log(0).

    Args:
        a: Input tensor (values should be positive).

    Returns:
        New tensor with log(a). Gradient: dz/da = 1/a.

    Example:
        >>> x = Tensor(np.array([1.0, np.e]), requires_grad=True)
        >>> y = tensor_log(x)
        >>> np.allclose(y.data, [0.0, 1.0])
        True
    """
    safe_data = np.maximum(a.data, 1e-12)
    out = Tensor(np.log(safe_data), _children=(a,), _op="log")
    out.requires_grad = a.requires_grad

    def _backward() -> None:
        assert out.grad is not None
        if a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            a.grad += out.grad / safe_data

    if out.requires_grad:
        out._backward = _backward
    return out


def tensor_sqrt(a: Tensor) -> Tensor:
    """Element-wise square root: z = sqrt(a).

    Args:
        a: Input tensor (values should be non-negative).

    Returns:
        New tensor with sqrt(a). Gradient: dz/da = 1 / (2 * sqrt(a)).

    Example:
        >>> x = Tensor(np.array([4.0, 9.0]), requires_grad=True)
        >>> y = tensor_sqrt(x)
        >>> np.allclose(y.data, [2.0, 3.0])
        True
    """
    out_data = np.sqrt(a.data)
    out = Tensor(out_data, _children=(a,), _op="sqrt")
    out.requires_grad = a.requires_grad

    def _backward() -> None:
        assert out.grad is not None
        if a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            # dz/da = 1 / (2 * sqrt(a)), guard against division by zero
            safe_out = np.where(out_data == 0, 1e-12, out_data)
            a.grad += out.grad / (2.0 * safe_out)

    if out.requires_grad:
        out._backward = _backward
    return out


def tensor_sum(
    a: Tensor, axis: Union[int, tuple[int, ...]] | None = None, keepdims: bool = False
) -> Tensor:
    """Sum: z = sum(a, axis)."""
    out_data = a.data.sum(axis=axis, keepdims=keepdims)
    out = Tensor(out_data, _children=(a,), _op="sum")
    out.requires_grad = a.requires_grad

    def _backward() -> None:
        assert out.grad is not None
        if a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            # Broadcast the gradient back to the original shape
            g = out.grad
            if axis is not None and not keepdims:
                # Re-expand the reduced dimension(s) for broadcasting
                if isinstance(axis, int):
                    g = np.expand_dims(g, axis=axis)
                else:
                    for ax in sorted(axis):
                        g = np.expand_dims(g, axis=ax)
            a.grad += np.broadcast_to(g, a.shape).copy()

    if out.requires_grad:
        out._backward = _backward
    return out


def tensor_mean(
    a: Tensor, axis: Union[int, tuple[int, ...]] | None = None, keepdims: bool = False
) -> Tensor:
    """Mean: z = mean(a, axis)."""
    out_data = a.data.mean(axis=axis, keepdims=keepdims)
    out = Tensor(out_data, _children=(a,), _op="mean")
    out.requires_grad = a.requires_grad

    # Compute the number of elements being averaged
    if axis is None:
        count = a.data.size
    elif isinstance(axis, int):
        count = a.data.shape[axis]
    else:
        count = 1
        for ax in axis:
            count *= a.data.shape[ax]

    def _backward() -> None:
        assert out.grad is not None
        if a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            g = out.grad
            if axis is not None and not keepdims:
                if isinstance(axis, int):
                    g = np.expand_dims(g, axis=axis)
                else:
                    for ax in sorted(axis):
                        g = np.expand_dims(g, axis=ax)
            a.grad += np.broadcast_to(g / count, a.shape).copy()

    if out.requires_grad:
        out._backward = _backward
    return out


def tensor_reshape(a: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Reshape: z = a.reshape(shape)."""
    out = Tensor(a.data.reshape(shape), _children=(a,), _op="reshape")
    out.requires_grad = a.requires_grad

    def _backward() -> None:
        assert out.grad is not None
        if a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            a.grad += out.grad.reshape(a.shape)

    if out.requires_grad:
        out._backward = _backward
    return out


def tensor_transpose(a: Tensor) -> Tensor:
    """Transpose: z = a.T."""
    out = Tensor(a.data.T, _children=(a,), _op="T")
    out.requires_grad = a.requires_grad

    def _backward() -> None:
        assert out.grad is not None
        if a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            a.grad += out.grad.T

    if out.requires_grad:
        out._backward = _backward
    return out


def tensor_maximum(a: Tensor, val: float = 0.0) -> Tensor:
    """Element-wise maximum: z = max(a, val). Used for ReLU and payoff clipping."""
    out = Tensor(np.maximum(a.data, val), _children=(a,), _op=f"max({val})")
    out.requires_grad = a.requires_grad

    def _backward() -> None:
        assert out.grad is not None
        if a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            # Subgradient: 1 where a > val, 0 where a <= val
            a.grad += out.grad * (a.data > val).astype(np.float64)

    if out.requires_grad:
        out._backward = _backward
    return out


def tensor_norm_cdf(a: Tensor) -> Tensor:
    """Standard normal CDF: z = Phi(a)."""
    from scipy.stats import norm

    out = Tensor(norm.cdf(a.data), _children=(a,), _op="Φ")
    out.requires_grad = a.requires_grad

    def _backward() -> None:
        assert out.grad is not None
        if a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            # dΦ/da = φ(a) = normal PDF
            a.grad += out.grad * norm.pdf(a.data)

    if out.requires_grad:
        out._backward = _backward
    return out


def tensor_sin(a: Tensor) -> Tensor:
    """Element-wise sine: z = sin(a).

    Args:
        a: Input tensor (in radians).

    Returns:
        New tensor with sin(a). Gradient: dz/da = cos(a).

    Example:
        >>> x = Tensor(np.array([0.0, np.pi / 2]), requires_grad=True)
        >>> y = tensor_sin(x)
        >>> np.allclose(y.data, [0.0, 1.0], atol=1e-10)
        True
    """
    out = Tensor(np.sin(a.data), _children=(a,), _op="sin")
    out.requires_grad = a.requires_grad

    def _backward() -> None:
        assert out.grad is not None
        if a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            a.grad += out.grad * np.cos(a.data)

    if out.requires_grad:
        out._backward = _backward
    return out


def tensor_cos(a: Tensor) -> Tensor:
    """Element-wise cosine: z = cos(a).

    Args:
        a: Input tensor (in radians).

    Returns:
        New tensor with cos(a). Gradient: dz/da = -sin(a).

    Example:
        >>> x = Tensor(np.array([0.0, np.pi]), requires_grad=True)
        >>> y = tensor_cos(x)
        >>> np.allclose(y.data, [1.0, -1.0], atol=1e-10)
        True
    """
    out = Tensor(np.cos(a.data), _children=(a,), _op="cos")
    out.requires_grad = a.requires_grad

    def _backward() -> None:
        assert out.grad is not None
        if a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            a.grad += out.grad * (-np.sin(a.data))

    if out.requires_grad:
        out._backward = _backward
    return out


def tensor_tanh(a: Tensor) -> Tensor:
    """Element-wise hyperbolic tangent: z = tanh(a).

    Args:
        a: Input tensor.

    Returns:
        New tensor with tanh(a). Gradient: dz/da = 1 - tanh(a)^2.

    Example:
        >>> x = Tensor(np.array([0.0, 1.0]), requires_grad=True)
        >>> y = tensor_tanh(x)
        >>> abs(y.data[0]) < 1e-10  # tanh(0) = 0
        True
    """
    out_data = np.tanh(a.data)
    out = Tensor(out_data, _children=(a,), _op="tanh")
    out.requires_grad = a.requires_grad

    def _backward() -> None:
        assert out.grad is not None
        if a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            # d tanh/da = 1 - tanh²(a)
            a.grad += out.grad * (1.0 - out_data**2)

    if out.requires_grad:
        out._backward = _backward
    return out


def tensor_abs(a: Tensor) -> Tensor:
    """Element-wise absolute value: z = |a|.

    Uses sign(a) as the subgradient, with subgradient 0 at a=0.

    Args:
        a: Input tensor.

    Returns:
        New tensor with |a|. Gradient: dz/da = sign(a).

    Example:
        >>> x = Tensor(np.array([-3.0, 0.0, 5.0]), requires_grad=True)
        >>> y = tensor_abs(x)
        >>> y.data
        array([3., 0., 5.])
    """
    out = Tensor(np.abs(a.data), _children=(a,), _op="abs")
    out.requires_grad = a.requires_grad

    def _backward() -> None:
        assert out.grad is not None
        if a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            a.grad += out.grad * np.sign(a.data)

    if out.requires_grad:
        out._backward = _backward
    return out


def tensor_clip(a: Tensor, a_min: float, a_max: float) -> Tensor:
    """Element-wise clip: z = clip(a, a_min, a_max).

    Gradient is 1 where a is within [a_min, a_max], 0 otherwise.
    """
    out = Tensor(np.clip(a.data, a_min, a_max), _children=(a,), _op="clip")
    out.requires_grad = a.requires_grad

    def _backward() -> None:
        assert out.grad is not None
        if a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            mask = ((a.data >= a_min) & (a.data <= a_max)).astype(np.float64)
            a.grad += out.grad * mask

    if out.requires_grad:
        out._backward = _backward
    return out


def tensor_where(condition: np.ndarray, a: Tensor, b: Tensor) -> Tensor:
    """Element-wise selection: z = a where condition else b.

    Args:
        condition: Boolean NumPy array.
        a: Values where condition is True.
        b: Values where condition is False.
    """
    out = Tensor(np.where(condition, a.data, b.data), _children=(a, b), _op="where")
    out.requires_grad = a.requires_grad or b.requires_grad

    def _backward() -> None:
        assert out.grad is not None
        mask = condition.astype(np.float64)
        if a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            a.grad += _unbroadcast(out.grad * mask, a.shape)
        if b.requires_grad:
            if b.grad is None:
                b.grad = np.zeros_like(b.data)
            b.grad += _unbroadcast(out.grad * (1.0 - mask), b.shape)

    if out.requires_grad:
        out._backward = _backward
    return out


def tensor_softmax(a: Tensor, axis: int = -1) -> Tensor:
    """Softmax along an axis: z_i = exp(a_i) / sum_j(exp(a_j)).

    Numerically stable via max subtraction.
    """
    shifted = a.data - a.data.max(axis=axis, keepdims=True)
    exp_a = np.exp(shifted)
    out_data = exp_a / exp_a.sum(axis=axis, keepdims=True)
    out = Tensor(out_data, _children=(a,), _op="softmax")
    out.requires_grad = a.requires_grad

    def _backward() -> None:
        assert out.grad is not None
        if a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            # Jacobian-vector product for softmax
            s = out_data
            dot = (out.grad * s).sum(axis=axis, keepdims=True)
            a.grad += s * (out.grad - dot)

    if out.requires_grad:
        out._backward = _backward
    return out


def tensor_getitem(a: Tensor, idx: Any) -> Tensor:
    """Index into a Tensor, preserving gradient flow.

    Supports any NumPy-compatible index (int, slice, array, bool mask).
    Gradient is scattered back to the original positions.

    Example::

        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x[1] ** 2   # y = 4.0; x.grad[1] = 4.0 after backward
    """
    out_data = a.data[idx]
    # Ensure out_data is an ndarray even when index yields a scalar
    out_data = np.asarray(out_data, dtype=np.float64)
    out = Tensor(out_data, _children=(a,), _op="[]")
    out.requires_grad = a.requires_grad

    def _backward() -> None:
        assert out.grad is not None
        if a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            np.add.at(a.grad, idx, out.grad)

    if out.requires_grad:
        out._backward = _backward
    return out
