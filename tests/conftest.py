"""Shared fixtures for the TensorQuantLib test suite."""

import numpy as np
import pytest

from tensorquantlib.core.tensor import Tensor
from tensorquantlib.tt.decompose import tt_svd

# ── Random number generator (reproducible) ───────────────────────────────

@pytest.fixture
def rng():
    """Seeded NumPy random generator."""
    return np.random.default_rng(42)


# ── Tensor fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def scalar_tensor():
    """Scalar Tensor with grad tracking."""
    return Tensor([3.0], requires_grad=True)


@pytest.fixture
def vector_tensor():
    """1-D Tensor with grad tracking."""
    return Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)


@pytest.fixture
def matrix_tensor():
    """2×3 Tensor with grad tracking."""
    return Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)


# ── Market parameter fixtures ───────────────────────────────────────────

@pytest.fixture
def market_params():
    """Typical Black-Scholes market parameters."""
    return {
        "S": 100.0,
        "K": 100.0,
        "r": 0.05,
        "T": 1.0,
        "sigma": 0.2,
    }


@pytest.fixture
def basket_params():
    """Typical basket option parameters."""
    return {
        "n_assets": 3,
        "spots": [100.0, 100.0, 100.0],
        "strike": 100.0,
        "r": 0.05,
        "T": 1.0,
        "sigma": 0.2,
        "n_points": 15,
    }


# ── TT-core fixtures ────────────────────────────────────────────────────

@pytest.fixture
def random_3d_tensor(rng):
    """Random 8×6×5 tensor and its TT-cores."""
    T = rng.standard_normal((8, 6, 5))
    cores = tt_svd(T, eps=1e-12)
    return T, cores


@pytest.fixture
def random_4d_tensor(rng):
    """Random 5×4×3×4 tensor and its TT-cores."""
    T = rng.standard_normal((5, 4, 3, 4))
    cores = tt_svd(T, eps=1e-12)
    return T, cores


@pytest.fixture
def smooth_3d_tensor():
    """Smooth (low-rank) 10×10×10 tensor and its TT-cores.

    Built from an outer-product structure so TT-ranks are small.
    """
    x = np.linspace(0, 1, 10)
    T = np.exp(-x[:, None, None] - x[None, :, None] - x[None, None, :])
    cores = tt_svd(T, eps=1e-12)
    return T, cores
