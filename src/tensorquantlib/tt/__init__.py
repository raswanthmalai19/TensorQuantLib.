"""Tensor-Train compression engine — TT-SVD, evaluation, surrogate pricing."""

from .decompose import tt_svd, tt_round
from .ops import (
    tt_eval,
    tt_eval_batch,
    tt_to_full,
    tt_ranks,
    tt_memory,
    tt_error,
    tt_compression_ratio,
)
from .surrogate import TTSurrogate
