"""
Public API for differentiable tensor operations.

All operations are implemented in tensor.py alongside the Tensor class
(they share the _unbroadcast helper and Tensor internals). This module
re-exports them with a clean namespace for external use.

Usage:
    from tensorquantlib.core import ops
    z = ops.exp(x)
    z = ops.norm_cdf(d1)

Note: ``tsum`` and ``tpow`` are used instead of ``sum`` / ``pow`` to avoid
shadowing Python built-ins when doing ``from ops import *``.
"""

from tensorquantlib.core.tensor import tensor_abs as abs
from tensorquantlib.core.tensor import tensor_add as add
from tensorquantlib.core.tensor import tensor_clip as clip
from tensorquantlib.core.tensor import tensor_cos as cos
from tensorquantlib.core.tensor import tensor_div as div
from tensorquantlib.core.tensor import tensor_exp as exp
from tensorquantlib.core.tensor import tensor_log as log
from tensorquantlib.core.tensor import tensor_matmul as matmul
from tensorquantlib.core.tensor import tensor_maximum as maximum
from tensorquantlib.core.tensor import tensor_mean as mean
from tensorquantlib.core.tensor import tensor_mul as mul
from tensorquantlib.core.tensor import tensor_neg as neg
from tensorquantlib.core.tensor import tensor_norm_cdf as norm_cdf
from tensorquantlib.core.tensor import tensor_pow as tpow
from tensorquantlib.core.tensor import tensor_reshape as reshape
from tensorquantlib.core.tensor import tensor_sin as sin
from tensorquantlib.core.tensor import tensor_softmax as softmax
from tensorquantlib.core.tensor import tensor_sqrt as sqrt
from tensorquantlib.core.tensor import tensor_sub as sub
from tensorquantlib.core.tensor import tensor_sum as tsum
from tensorquantlib.core.tensor import tensor_tanh as tanh
from tensorquantlib.core.tensor import tensor_transpose as transpose
from tensorquantlib.core.tensor import tensor_where as where

# Legacy aliases kept for backward compatibility (do NOT use in new code —
# they shadow Python built-ins when imported with ``from ops import *``).
pow = tpow
sum = tsum

__all__ = [
    "abs",
    "add",
    "clip",
    "cos",
    "div",
    "exp",
    "log",
    "matmul",
    "maximum",
    "mean",
    "mul",
    "neg",
    "norm_cdf",
    "reshape",
    "sin",
    "softmax",
    "sqrt",
    "sub",
    "tanh",
    "tpow",
    "transpose",
    "tsum",
    "where",
]
