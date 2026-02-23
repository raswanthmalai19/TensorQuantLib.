"""Smoke tests for the visualization module.

These tests create figures without displaying them (Agg backend)
to verify that plotting code runs without errors.
"""

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")  # non-interactive backend

from tensorquantlib.tt.decompose import tt_svd
from tensorquantlib.tt.ops import tt_ranks, tt_compression_ratio, tt_error
from tensorquantlib.viz import (
    plot_pricing_surface,
    plot_greeks_surface,
    plot_tt_ranks,
    plot_compression_vs_tolerance,
    plot_convergence,
    plot_rank_profile,
)


@pytest.fixture
def grid_2d():
    x = np.linspace(80, 120, 20)
    y = np.linspace(0.1, 0.5, 15)
    X, Y = np.meshgrid(x, y, indexing="ij")
    Z = np.maximum(X - 100, 0) * np.exp(-Y)
    return Z, [x, y]


@pytest.fixture
def grid_3d():
    x = np.linspace(80, 120, 10)
    y = np.linspace(0.1, 0.5, 8)
    z = np.linspace(0.5, 2.0, 6)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    V = np.maximum(X - 100, 0) * np.exp(-Y * Z)
    return V, [x, y, z]


class TestPricingSurface:
    def test_heatmap(self, grid_2d):
        Z, axes = grid_2d
        fig, ax = plot_pricing_surface(Z, axes)
        assert fig is not None

    def test_surface_3d(self, grid_2d):
        Z, axes = grid_2d
        fig, ax = plot_pricing_surface(Z, axes, mode="surface")
        assert fig is not None

    def test_3d_slice(self, grid_3d):
        V, axes = grid_3d
        fig, ax = plot_pricing_surface(V, axes, dims=(0, 1))
        assert fig is not None


class TestGreeksSurface:
    def test_multiple_greeks(self, grid_2d):
        Z, axes = grid_2d
        greeks = {"Delta": Z * 0.5, "Gamma": Z * 0.01}
        fig, axs = plot_greeks_surface(greeks, axes)
        assert len(axs) == 2

    def test_single_greek(self, grid_2d):
        Z, axes = grid_2d
        fig, axs = plot_greeks_surface({"Delta": Z}, axes)
        assert len(axs) == 1


class TestTTRanks:
    def test_rank_bar_chart(self):
        T = np.random.default_rng(42).standard_normal((8, 6, 5))
        cores = tt_svd(T, eps=1e-10)
        fig, ax = plot_tt_ranks(cores)
        assert fig is not None

    def test_rank_profile_overlay(self):
        T = np.random.default_rng(42).standard_normal((10, 8, 6))
        profiles = {}
        for eps in [1e-2, 1e-6, 1e-12]:
            cores = tt_svd(T, eps=eps)
            profiles[f"eps={eps}"] = tt_ranks(cores)
        fig, ax = plot_rank_profile(profiles)
        assert fig is not None


class TestCompressionVsTolerance:
    def test_without_errors(self):
        fig, ax = plot_compression_vs_tolerance(
            [1e-1, 1e-3, 1e-6], [2.0, 5.0, 1.0]
        )
        assert fig is not None

    def test_with_errors(self):
        fig, ax = plot_compression_vs_tolerance(
            [1e-1, 1e-3, 1e-6],
            [2.0, 5.0, 1.0],
            errors=[0.1, 0.001, 1e-6],
        )
        assert fig is not None


class TestConvergence:
    def test_convergence_plot(self):
        fig, ax = plot_convergence(
            list(range(10)), [10 / (i + 1) for i in range(10)]
        )
        assert fig is not None
