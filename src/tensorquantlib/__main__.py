"""
Command-line interface for TensorQuantLib.

Usage examples::

    # European option pricing
    python -m tensorquantlib price --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2

    # Implied volatility
    python -m tensorquantlib iv --price 10.45 --S 100 --K 100 --T 1 --r 0.05

    # American option (LSM)
    python -m tensorquantlib american --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2

    # Heston model price
    python -m tensorquantlib heston --S 100 --K 100 --T 1 --r 0.05 --kappa 2 --theta 0.04 --xi 0.3 --rho -0.7 --v0 0.04

    # Asian option
    python -m tensorquantlib asian --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2

    # Barrier option
    python -m tensorquantlib barrier --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --barrier 90 --barrier-type down-and-out

    # Portfolio risk
    python -m tensorquantlib risk --sigma 0.20 --horizon 1 --alpha 0.95

    # VaR comparison methods
    python -m tensorquantlib compare-vr --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2
"""

from __future__ import annotations

import argparse
import sys
from typing import NoReturn


def _print_table(rows: list[tuple[str, str]]) -> None:
    """Print a two-column table."""
    if not rows:
        return
    w = max(len(r[0]) for r in rows) + 2
    print("-" * (w + 20))
    for k, v in rows:
        print(f"  {k:<{w}}{v}")
    print("-" * (w + 20))


def cmd_price(args: argparse.Namespace) -> None:
    """European Black-Scholes pricing."""
    from tensorquantlib.finance.black_scholes import (
        bs_price_numpy, bs_delta, bs_gamma, bs_vega, bs_theta, bs_rho,
    )
    price = bs_price_numpy(args.S, args.K, args.T, args.r, args.sigma, q=args.q, option_type=args.type)
    delta = bs_delta(args.S, args.K, args.T, args.r, args.sigma, q=args.q, option_type=args.type)
    gamma = bs_gamma(args.S, args.K, args.T, args.r, args.sigma, q=args.q)
    vega  = bs_vega(args.S, args.K, args.T, args.r, args.sigma, q=args.q)
    theta = bs_theta(args.S, args.K, args.T, args.r, args.sigma, q=args.q, option_type=args.type)
    rho   = bs_rho(args.S, args.K, args.T, args.r, args.sigma, q=args.q, option_type=args.type)

    print(f"\nBlack-Scholes European {args.type.upper()}")
    _print_table([
        ("S",     f"{args.S:.4f}"),
        ("K",     f"{args.K:.4f}"),
        ("T",     f"{args.T:.4f} yr"),
        ("r",     f"{args.r:.4%}"),
        ("sigma", f"{args.sigma:.4%}"),
        ("q",     f"{args.q:.4%}"),
        ("Price", f"{float(price):.6f}"),
        ("Delta", f"{float(delta):.6f}"),
        ("Gamma", f"{float(gamma):.6f}"),
        ("Vega",  f"{float(vega):.6f}"),
        ("Theta", f"{float(theta):.6f}"),
        ("Rho",   f"{float(rho):.6f}"),
    ])


def cmd_iv(args: argparse.Namespace) -> None:
    """Implied volatility inversion."""
    from tensorquantlib.finance.implied_vol import implied_vol_nr
    try:
        iv = implied_vol_nr(args.price, args.S, args.K, args.T, args.r, q=args.q, option_type=args.type)
        print(f"\nImplied Volatility: {iv:.6f}  ({iv*100:.2f}%)")
    except ValueError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_american(args: argparse.Namespace) -> None:
    """American option pricing via Longstaff-Schwartz LSM."""
    from tensorquantlib.finance.american import american_option_lsm
    price, stderr = american_option_lsm(
        args.S, args.K, args.T, args.r, args.sigma, q=args.q,
        option_type=args.type, n_paths=args.paths, n_steps=args.steps,
        seed=args.seed, return_stderr=True,
    )
    print(f"\nAmerican LSM {args.type.upper()}")
    _print_table([
        ("Price",   f"{price:.6f}"),
        ("StdErr",  f"{stderr:.6f}"),
        ("n_paths", f"{args.paths:,}"),
        ("n_steps", f"{args.steps}"),
    ])


def cmd_heston(args: argparse.Namespace) -> None:
    """Heston stochastic volatility pricing."""
    from tensorquantlib.finance.heston import HestonParams, heston_price, heston_greeks
    params = HestonParams(kappa=args.kappa, theta=args.theta, xi=args.xi, rho=args.rho, v0=args.v0)
    price = heston_price(args.S, args.K, args.T, args.r, params, q=args.q, option_type=args.type)
    greeks = heston_greeks(args.S, args.K, args.T, args.r, params, q=args.q, option_type=args.type)
    print(f"\nHeston Model {args.type.upper()}")
    print(f"Feller condition: {'satisfied' if params.feller_satisfied() else 'VIOLATED (possible instability)'}")
    _print_table([
        ("Price",  f"{price:.6f}"),
        ("Delta",  f"{greeks['delta']:.6f}"),
        ("Gamma",  f"{greeks['gamma']:.6f}"),
        ("Theta",  f"{greeks['theta']:.6f}"),
        ("Vega",   f"{greeks['vega']:.6f}  (per unit v0)"),
    ])


def cmd_asian(args: argparse.Namespace) -> None:
    """Asian option pricing."""
    from tensorquantlib.finance.exotics import asian_price_mc, asian_geometric_price
    price_mc, stderr = asian_price_mc(
        args.S, args.K, args.T, args.r, args.sigma, q=args.q,
        option_type=args.type, average_type=args.avg,
        n_paths=args.paths, n_steps=args.steps, seed=args.seed, return_stderr=True,
    )
    print(f"\nAsian {args.avg.capitalize()} Average {args.type.upper()}")
    _print_table([
        ("MC Price",   f"{price_mc:.6f}"),
        ("MC StdErr",  f"{stderr:.6f}"),
    ])
    if args.avg == "geometric":
        analytic = asian_geometric_price(args.S, args.K, args.T, args.r, args.sigma, q=args.q, option_type=args.type)
        print(f"  {'Analytic Geo:':<20}{analytic:.6f}")


def cmd_barrier(args: argparse.Namespace) -> None:
    """Barrier option pricing."""
    from tensorquantlib.finance.exotics import barrier_price, barrier_price_mc
    try:
        analytic = barrier_price(args.S, args.K, args.T, args.r, args.sigma,
                                 args.barrier, args.barrier_type, q=args.q,
                                 option_type=args.type)
    except Exception:
        analytic = float("nan")
    mc, stderr = barrier_price_mc(
        args.S, args.K, args.T, args.r, args.sigma,
        args.barrier, args.barrier_type, q=args.q,
        option_type=args.type, n_paths=args.paths, n_steps=args.steps,
        seed=args.seed, return_stderr=True,
    )
    print(f"\nBarrier Option [{args.barrier_type}] {args.type.upper()}")
    _print_table([
        ("Barrier",  f"{args.barrier:.2f}"),
        ("Analytic", f"{analytic:.6f}"),
        ("MC Price", f"{mc:.6f}"),
        ("MC StdErr",f"{stderr:.6f}"),
    ])


def cmd_risk(args: argparse.Namespace) -> None:
    """Portfolio risk metrics — VaR, CVaR, Sharpe."""
    from tensorquantlib.finance.risk import var_parametric, var_mc, PortfolioRisk
    import numpy as np

    print(f"\nRisk Metrics  (S={args.S}, sigma={args.sigma:.1%}, alpha={args.alpha:.0%}, horizon={int(args.horizon)}d)")
    var_p = var_parametric(0.0, args.sigma, alpha=args.alpha, horizon=args.horizon / 252.0)
    var_v, cvar_v = var_mc(args.S, args.sigma, horizon=args.horizon / 252.0, alpha=args.alpha,
                            n_paths=100_000, seed=42)
    _print_table([
        ("Param VaR (1d)",   f"{var_p * args.S:.4f}  ({var_p:.4%} of S)"),
        ("MC VaR",           f"{var_v * args.S:.4f}  ({var_v:.4%} of S)"),
        ("MC CVaR (ES)",     f"{cvar_v * args.S:.4f}  ({cvar_v:.4%} of S)"),
    ])

    # Simulate a return history and report portfolio-level stats
    rng = np.random.default_rng(42)
    ret_hist = rng.normal(0.0, args.sigma / np.sqrt(252), 252)
    pr = PortfolioRisk(ret_hist, alpha=args.alpha)
    stats = pr.summary()
    print("\n  Simulated 1-year daily return history:")
    _print_table([(k, f"{v:.4f}") for k, v in stats.items()])


def cmd_compare_vr(args: argparse.Namespace) -> None:
    """Compare variance reduction methods."""
    from tensorquantlib.finance.variance_reduction import compare_variance_reduction
    results = compare_variance_reduction(
        args.S, args.K, args.T, args.r, args.sigma, n_paths=args.paths, seed=42,
    )
    print(f"\nVariance Reduction Comparison  ({args.type.upper()} S={args.S} K={args.K} T={args.T})")
    print(f"  {'Method':<25} {'Price':>10} {'StdErr':>12} {'VR Ratio':>10}")
    print("  " + "-" * 60)
    for name, res in results.items():
        print(f"  {name:<25} {res['price']:>10.5f} {res['stderr']:>12.6f} {res['vr_ratio']:>10.2f}x")


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m tensorquantlib",
        description="TensorQuantLib — quantitative finance toolkit CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ---- shared arguments ----
    def add_bs_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--S", type=float, required=True, help="Spot price")
        p.add_argument("--K", type=float, required=True, help="Strike")
        p.add_argument("--T", type=float, required=True, help="Time to expiry (years)")
        p.add_argument("--r", type=float, required=True, help="Risk-free rate")
        p.add_argument("--sigma", type=float, required=True, help="Volatility")
        p.add_argument("--q", type=float, default=0.0, help="Dividend yield (default 0)")
        p.add_argument("--type", choices=["call", "put"], default="call")

    def add_mc_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--paths", type=int, default=100_000, help="MC paths")
        p.add_argument("--steps", type=int, default=252, help="Time steps")
        p.add_argument("--seed", type=int, default=None, help="Random seed")

    # ---- price ----
    p_price = sub.add_parser("price", help="Black-Scholes European option price and Greeks")
    add_bs_args(p_price)
    p_price.set_defaults(func=cmd_price)

    # ---- iv ----
    p_iv = sub.add_parser("iv", help="Implied volatility solver")
    p_iv.add_argument("--price", type=float, required=True, help="Market price")
    p_iv.add_argument("--S", type=float, required=True)
    p_iv.add_argument("--K", type=float, required=True)
    p_iv.add_argument("--T", type=float, required=True)
    p_iv.add_argument("--r", type=float, required=True)
    p_iv.add_argument("--q", type=float, default=0.0)
    p_iv.add_argument("--type", choices=["call", "put"], default="call")
    p_iv.set_defaults(func=cmd_iv)

    # ---- american ----
    p_am = sub.add_parser("american", help="American option via LSM Monte Carlo")
    add_bs_args(p_am)
    add_mc_args(p_am)
    p_am.set_defaults(func=cmd_american)

    # ---- heston ----
    p_heston = sub.add_parser("heston", help="Heston stochastic volatility model")
    p_heston.add_argument("--S", type=float, required=True)
    p_heston.add_argument("--K", type=float, required=True)
    p_heston.add_argument("--T", type=float, required=True)
    p_heston.add_argument("--r", type=float, required=True)
    p_heston.add_argument("--q", type=float, default=0.0)
    p_heston.add_argument("--type", choices=["call", "put"], default="call")
    p_heston.add_argument("--kappa", type=float, default=2.0)
    p_heston.add_argument("--theta", type=float, default=0.04)
    p_heston.add_argument("--xi",    type=float, default=0.3)
    p_heston.add_argument("--rho",   type=float, default=-0.7)
    p_heston.add_argument("--v0",    type=float, default=0.04)
    p_heston.set_defaults(func=cmd_heston)

    # ---- asian ----
    p_asian = sub.add_parser("asian", help="Asian average-rate option")
    add_bs_args(p_asian)
    add_mc_args(p_asian)
    p_asian.add_argument("--avg", choices=["arithmetic", "geometric"], default="arithmetic")
    p_asian.set_defaults(func=cmd_asian)

    # ---- barrier ----
    p_barrier = sub.add_parser("barrier", help="Single-barrier European option")
    add_bs_args(p_barrier)
    add_mc_args(p_barrier)
    p_barrier.add_argument("--barrier", type=float, required=True, help="Barrier level")
    p_barrier.add_argument(
        "--barrier-type",
        dest="barrier_type",
        choices=["down-and-in", "down-and-out", "up-and-in", "up-and-out"],
        default="down-and-out",
    )
    p_barrier.set_defaults(func=cmd_barrier)

    # ---- risk ----
    p_risk = sub.add_parser("risk", help="Risk metrics: VaR, CVaR, Sharpe, drawdown")
    p_risk.add_argument("--S",       type=float, default=100.0, help="Spot / position size")
    p_risk.add_argument("--sigma",   type=float, required=True, help="Annualised volatility")
    p_risk.add_argument("--horizon", type=float, default=1.0,   help="Horizon in trading days (default 1)")
    p_risk.add_argument("--alpha",   type=float, default=0.95,  help="Confidence level (default 0.95)")
    p_risk.set_defaults(func=cmd_risk)

    # ---- compare-vr ----
    p_vr = sub.add_parser("compare-vr", help="Compare variance reduction methods")
    add_bs_args(p_vr)
    p_vr.add_argument("--paths", type=int, default=50_000)
    p_vr.set_defaults(func=cmd_compare_vr)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
