"""Tests for input validation and error handling across the library."""

import numpy as np
import pytest

from tensorquantlib.finance.black_scholes import bs_price_numpy
from tensorquantlib.finance.basket import simulate_basket
from tensorquantlib.tt.decompose import tt_svd
from tensorquantlib.tt.surrogate import TTSurrogate


class TestBlackScholesValidation:
    def test_invalid_option_type(self):
        with pytest.raises(ValueError, match="option_type"):
            bs_price_numpy(100, 100, 1.0, 0.05, 0.2, option_type="straddle")

    def test_negative_spot(self):
        with pytest.raises(ValueError, match="Spot price"):
            bs_price_numpy(-100, 100, 1.0, 0.05, 0.2)

    def test_zero_strike(self):
        with pytest.raises(ValueError, match="Strike"):
            bs_price_numpy(100, 0, 1.0, 0.05, 0.2)

    def test_negative_T(self):
        with pytest.raises(ValueError, match="Time to expiry"):
            bs_price_numpy(100, 100, -1.0, 0.05, 0.2)

    def test_zero_sigma(self):
        with pytest.raises(ValueError, match="Volatility"):
            bs_price_numpy(100, 100, 1.0, 0.05, 0.0)


class TestBasketValidation:
    def test_invalid_option_type(self):
        with pytest.raises(ValueError, match="option_type"):
            simulate_basket(
                np.array([100.0]), 100, 1.0, 0.05,
                np.array([0.2]), np.eye(1), np.array([1.0]),
                option_type="butterfly",
            )

    def test_negative_spot(self):
        with pytest.raises(ValueError, match="spot prices"):
            simulate_basket(
                np.array([-100.0]), 100, 1.0, 0.05,
                np.array([0.2]), np.eye(1), np.array([1.0]),
            )

    def test_zero_strike(self):
        with pytest.raises(ValueError, match="Strike"):
            simulate_basket(
                np.array([100.0]), 0, 1.0, 0.05,
                np.array([0.2]), np.eye(1), np.array([1.0]),
            )

    def test_shape_mismatch_sigma(self):
        with pytest.raises(ValueError, match="sigma shape"):
            simulate_basket(
                np.array([100.0, 100.0]), 100, 1.0, 0.05,
                np.array([0.2]), np.eye(2), np.array([0.5, 0.5]),
            )

    def test_shape_mismatch_corr(self):
        with pytest.raises(ValueError, match="corr shape"):
            simulate_basket(
                np.array([100.0, 100.0]), 100, 1.0, 0.05,
                np.array([0.2, 0.2]), np.eye(3), np.array([0.5, 0.5]),
            )

    def test_negative_n_paths(self):
        with pytest.raises(ValueError, match="n_paths"):
            simulate_basket(
                np.array([100.0]), 100, 1.0, 0.05,
                np.array([0.2]), np.eye(1), np.array([1.0]),
                n_paths=0,
            )


class TestTTSVDValidation:
    def test_1d_tensor_raises(self):
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            tt_svd(np.array([1.0, 2.0, 3.0]))

    def test_negative_eps_raises(self):
        with pytest.raises(ValueError, match="eps must be non-negative"):
            tt_svd(np.random.randn(3, 4), eps=-0.1)

    def test_zero_max_rank_raises(self):
        with pytest.raises(ValueError, match="max_rank must be >= 1"):
            tt_svd(np.random.randn(3, 4), max_rank=0)


class TestSurrogateValidation:
    def test_grid_1d_raises(self):
        with pytest.raises(ValueError, match="at least 2D"):
            TTSurrogate.from_grid(
                np.array([1.0, 2.0, 3.0]),
                [np.linspace(0, 1, 3)],
            )

    def test_axes_count_mismatch(self):
        grid = np.random.randn(5, 6)
        with pytest.raises(ValueError, match="Number of axes"):
            TTSurrogate.from_grid(
                grid,
                [np.linspace(0, 1, 5)],  # only 1 axis for 2D grid
            )

    def test_axis_length_mismatch(self):
        grid = np.random.randn(5, 6)
        with pytest.raises(ValueError, match="Axis 1 length"):
            TTSurrogate.from_grid(
                grid,
                [np.linspace(0, 1, 5), np.linspace(0, 1, 10)],  # 10 != 6
            )

    def test_negative_eps(self):
        grid = np.random.randn(5, 6)
        with pytest.raises(ValueError, match="eps must be positive"):
            TTSurrogate.from_grid(
                grid,
                [np.linspace(0, 1, 5), np.linspace(0, 1, 6)],
                eps=-0.001,
            )
