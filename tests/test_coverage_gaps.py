"""Tests for CLI (__main__), Heston coverage gaps, and exotic edge cases."""

import numpy as np
import pytest

from tensorquantlib.__main__ import build_parser, main


class TestCLIPrice:
    def test_bs_price(self, capsys):
        rc = main(
            ["price", "--S", "100", "--K", "100", "--T", "1", "--r", "0.05", "--sigma", "0.2"]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "Price" in out

    def test_bs_put(self, capsys):
        rc = main(
            [
                "price",
                "--S",
                "100",
                "--K",
                "100",
                "--T",
                "1",
                "--r",
                "0.05",
                "--sigma",
                "0.2",
                "--type",
                "put",
            ]
        )
        assert rc == 0

    def test_no_command(self, capsys):
        rc = main([])
        assert rc == 1


class TestCLIIV:
    def test_iv(self, capsys):
        rc = main(["iv", "--price", "10.45", "--S", "100", "--K", "100", "--T", "1", "--r", "0.05"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Implied Volatility" in out


class TestCLIAmerican:
    def test_american(self, capsys):
        rc = main(
            [
                "american",
                "--S",
                "100",
                "--K",
                "100",
                "--T",
                "1",
                "--r",
                "0.05",
                "--sigma",
                "0.2",
                "--paths",
                "5000",
                "--steps",
                "50",
                "--seed",
                "42",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "American" in out


class TestCLIHeston:
    def test_heston(self, capsys):
        rc = main(["heston", "--S", "100", "--K", "100", "--T", "1", "--r", "0.05"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Heston" in out

    def test_heston_put(self, capsys):
        rc = main(
            ["heston", "--S", "100", "--K", "100", "--T", "1", "--r", "0.05", "--type", "put"]
        )
        assert rc == 0


class TestCLIAsian:
    def test_asian_arithmetic(self, capsys):
        rc = main(
            [
                "asian",
                "--S",
                "100",
                "--K",
                "100",
                "--T",
                "1",
                "--r",
                "0.05",
                "--sigma",
                "0.2",
                "--paths",
                "5000",
                "--steps",
                "50",
                "--seed",
                "42",
            ]
        )
        assert rc == 0

    def test_asian_geometric(self, capsys):
        rc = main(
            [
                "asian",
                "--S",
                "100",
                "--K",
                "100",
                "--T",
                "1",
                "--r",
                "0.05",
                "--sigma",
                "0.2",
                "--avg",
                "geometric",
                "--paths",
                "5000",
                "--steps",
                "50",
                "--seed",
                "42",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "Geometric" in out


class TestCLIBarrier:
    def test_barrier(self, capsys):
        rc = main(
            [
                "barrier",
                "--S",
                "100",
                "--K",
                "100",
                "--T",
                "1",
                "--r",
                "0.05",
                "--sigma",
                "0.2",
                "--barrier",
                "90",
                "--barrier-type",
                "down-and-out",
                "--paths",
                "5000",
                "--steps",
                "50",
                "--seed",
                "42",
            ]
        )
        assert rc == 0


class TestCLIRisk:
    def test_risk(self, capsys):
        rc = main(["risk", "--sigma", "0.2"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "VaR" in out


class TestCLICompareVR:
    def test_compare_vr(self, capsys):
        rc = main(
            [
                "compare-vr",
                "--S",
                "100",
                "--K",
                "100",
                "--T",
                "1",
                "--r",
                "0.05",
                "--sigma",
                "0.2",
                "--paths",
                "5000",
            ]
        )
        assert rc == 0


class TestCLIParser:
    def test_parser_builds(self):
        parser = build_parser()
        assert parser is not None

    def test_help_text(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["price", "--help"])
        assert exc.value.code == 0


class TestHestonCoverageGaps:
    """Cover heston_greeks and HestonCalibrator."""

    def test_heston_greeks_call(self):
        from tensorquantlib.finance.heston import HestonParams, heston_greeks

        params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        g = heston_greeks(100, 100, 1.0, 0.05, params, option_type="call")
        assert "delta" in g
        assert "gamma" in g
        assert "vega" in g
        assert "theta" in g
        assert 0 < g["delta"] < 1

    def test_heston_greeks_put(self):
        from tensorquantlib.finance.heston import HestonParams, heston_greeks

        params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        g = heston_greeks(100, 100, 1.0, 0.05, params, option_type="put")
        assert g["delta"] < 0

    def test_heston_calibrator(self):
        from tensorquantlib.finance.heston import HestonCalibrator

        cal = HestonCalibrator(S=100, r=0.05)
        K_grid = np.array([90.0, 100.0, 110.0])
        T_grid = np.array([0.5, 1.0])
        iv_mkt = np.full((len(K_grid), len(T_grid)), 0.20)
        cal.fit(iv_mkt, K_grid, T_grid, n_restarts=1, maxiter=50)
        assert hasattr(cal, "params_")
        assert cal.params_.v0 > 0


class TestExoticsCoverageGaps:
    """Cover uncovered branches in exotics.py."""

    def test_digital_put(self):
        from tensorquantlib.finance.exotics import digital_price

        p = digital_price(100, 100, 1.0, 0.05, 0.2, option_type="put")
        assert p > 0

    def test_digital_mc_put(self):
        from tensorquantlib.finance.exotics import digital_price_mc

        p = digital_price_mc(100, 100, 1.0, 0.05, 0.2, option_type="put", n_paths=10_000, seed=42)
        assert p > 0

    def test_barrier_analytic_up_in(self):
        from tensorquantlib.finance.exotics import barrier_price

        p = barrier_price(100, 100, 1.0, 0.05, 0.2, 120, "up-and-in")
        assert p >= 0

    def test_barrier_mc_down_in(self):
        from tensorquantlib.finance.exotics import barrier_price_mc

        p = barrier_price_mc(100, 100, 1.0, 0.05, 0.2, 90, "down-and-in", n_paths=10_000, seed=42)
        assert p >= 0

    def test_barrier_mc_up_in(self):
        from tensorquantlib.finance.exotics import barrier_price_mc

        p = barrier_price_mc(100, 100, 1.0, 0.05, 0.2, 120, "up-and-in", n_paths=10_000, seed=42)
        assert p >= 0

    def test_lookback_floating(self):
        from tensorquantlib.finance.exotics import lookback_price_mc

        p, se = lookback_price_mc(
            100, None, 1.0, 0.05, 0.2, strike_type="floating", n_paths=10_000, seed=42
        )
        assert p > 0

    def test_lookback_put(self):
        from tensorquantlib.finance.exotics import lookback_price_mc

        p, se = lookback_price_mc(
            100, 100, 1.0, 0.05, 0.2, option_type="put", n_paths=10_000, seed=42
        )
        assert p > 0

    def test_cliquet_negative_floor(self):
        from tensorquantlib.finance.exotics import cliquet_price_mc

        result = cliquet_price_mc(
            100, 1.0, 0.05, 0.2, n_periods=4, floor=-0.05, cap=0.10, n_paths=10_000, seed=42
        )
        p = result[0] if isinstance(result, tuple) else result
        assert p > 0

    def test_rainbow_best_of(self):
        from tensorquantlib.finance.exotics import rainbow_price_mc

        result = rainbow_price_mc(
            spots=[100, 100],
            K=100,
            T=1.0,
            r=0.05,
            sigmas=[0.2, 0.25],
            corr=[[1, 0.5], [0.5, 1]],
            rainbow_type="best_of",
            n_paths=10_000,
            seed=42,
        )
        p = result[0] if isinstance(result, tuple) else result
        assert p > 0

    def test_rainbow_worst_of(self):
        from tensorquantlib.finance.exotics import rainbow_price_mc

        result = rainbow_price_mc(
            spots=[100, 100],
            K=100,
            T=1.0,
            r=0.05,
            sigmas=[0.2, 0.25],
            corr=[[1, 0.5], [0.5, 1]],
            rainbow_type="worst_of",
            n_paths=10_000,
            seed=42,
        )
        p = result[0] if isinstance(result, tuple) else result
        assert p > 0
