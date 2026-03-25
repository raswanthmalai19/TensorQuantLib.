"""Tensor-Train compression engine — TT-SVD, TT-Cross, evaluation, surrogate pricing."""

from .decompose import tt_cross, tt_round, tt_svd
from .ops import (
    tt_add,
    tt_compression_ratio,
    tt_dot,
    tt_error,
    tt_eval,
    tt_eval_batch,
    tt_frobenius_norm,
    tt_hadamard,
    tt_memory,
    tt_ranks,
    tt_scale,
    tt_to_full,
)
from .pricing import (
    american_surrogate,
    exotic_surrogate,
    heston_surrogate,
    jump_diffusion_surrogate,
)
from .surrogate import TTSurrogate

__all__ = [
    "TTSurrogate",
    "tt_add",
    "tt_compression_ratio",
    "tt_cross",
    "tt_dot",
    "tt_error",
    "tt_eval",
    "tt_eval_batch",
    "tt_frobenius_norm",
    "tt_hadamard",
    "tt_memory",
    "tt_ranks",
    "tt_round",
    "tt_scale",
    "tt_svd",
    "tt_to_full",
]
