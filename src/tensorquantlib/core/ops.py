"""
Public API for differentiable tensor operations.

All operations are implemented in tensor.py alongside the Tensor class
(they share the _unbroadcast helper and Tensor internals). This module
re-exports them with a clean namespace for external use.

Usage:
    from tensorquantlib.core import ops
    z = ops.exp(x)
    z = ops.norm_cdf(d1)
"""

from tensorquantlib.core.tensor import (
    tensor_add as add,
    tensor_sub as sub,
    tensor_mul as mul,
    tensor_div as div,
    tensor_neg as neg,
    tensor_matmul as matmul,
    tensor_pow as pow,
    tensor_exp as exp,
    tensor_log as log,
    tensor_sqrt as sqrt,
    tensor_sum as sum,
    tensor_mean as mean,
    tensor_reshape as reshape,
    tensor_transpose as transpose,
    tensor_maximum as maximum,
    tensor_norm_cdf as norm_cdf,
)

__all__ = [
    "add", "sub", "mul", "div", "neg",
    "matmul", "pow", "exp", "log", "sqrt",
    "sum", "mean", "reshape", "transpose",
    "maximum", "norm_cdf",
]
