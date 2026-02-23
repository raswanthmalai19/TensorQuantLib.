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
)
from tensorquantlib.core.tensor import (
    tensor_div as div,
)
from tensorquantlib.core.tensor import (
    tensor_exp as exp,
)
from tensorquantlib.core.tensor import (
    tensor_log as log,
)
from tensorquantlib.core.tensor import (
    tensor_matmul as matmul,
)
from tensorquantlib.core.tensor import (
    tensor_maximum as maximum,
)
from tensorquantlib.core.tensor import (
    tensor_mean as mean,
)
from tensorquantlib.core.tensor import (
    tensor_mul as mul,
)
from tensorquantlib.core.tensor import (
    tensor_neg as neg,
)
from tensorquantlib.core.tensor import (
    tensor_norm_cdf as norm_cdf,
)
from tensorquantlib.core.tensor import (
    tensor_pow as pow,
)
from tensorquantlib.core.tensor import (
    tensor_reshape as reshape,
)
from tensorquantlib.core.tensor import (
    tensor_sqrt as sqrt,
)
from tensorquantlib.core.tensor import (
    tensor_sub as sub,
)
from tensorquantlib.core.tensor import (
    tensor_sum as sum,
)
from tensorquantlib.core.tensor import (
    tensor_transpose as transpose,
)

__all__ = [
    "add",
    "div",
    "exp",
    "log",
    "matmul",
    "maximum",
    "mean",
    "mul",
    "neg",
    "norm_cdf",
    "pow",
    "reshape",
    "sqrt",
    "sub",
    "sum",
    "transpose",
]
