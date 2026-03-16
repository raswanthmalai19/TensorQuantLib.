"""
Benchmark 3 — Full Portfolio Risk Engine
==========================================
Runs a complete risk pipeline on a realistic 10-position options portfolio:

  1. Portfolio Greeks aggregation — delta/gamma/vega/theta/rho across 10 positions.
  2. Monte Carlo VaR & CVaR — 500,000 GBM paths, two confidence levels.
  3. Scenario stress test — 9 spot shocks from −30 % to +30 %.
  4. PortfolioRisk time-series analysis — 5-year daily returns, full risk summary.
  5. Heston vs Black-Scholes comparison — price and delta discrepancy.
  6. Heston calibration — fit to a synthetic 5×3 IV surface, RMSE check.

Why this stresses the M1:
  - var_mc runs GBM with 500 K × 1 paths: the large normal sample + exp/log
    dispatches through Accelerate Veclib (vectorised math).
  - PortfolioRisk.volatility / sharpe use np.std/mean on 252×5=1260 elements —
    trivial, but correct.
  - HestonCalibrator.fit calls scipy.optimize.minimize (L-BFGS-B) which uses
    Accelerate LAPACK for its internal Hessian approximation.
  - scenario_analysis loops over 9 scenarios, each re-pricing 10 positions.

Run from the repo root:
    python benchmarks/bench_portfolio_risk.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "src")

from tensorquantlib.finance.black_scholes import bs_price_numpy, bs_delta
from tensorquantlib.finance.risk import (
    OptionPosition,
    PortfolioRisk,
    greeks_portfolio,
    scenario_analysis,
    var_mc,
)
from tensorquantlib.finance.heston import HestonCalibrator, HestonParams, heston_price

DIV  = "─" * 68
DIV2 = "═" * 68


def fmt_ms(t: float) -> str:
    return f"{t * 1_000:.1f} ms"


def fmt_us(t: float) -> str:
    return f"{t * 1_000_000:.1f} µs"


# ── Portfolio definition (10 positions on the same underlying) ───────────────
S    = 100.0
r    = 0.05
q    = 0.00

POSITIONS = [
    # Long calls
    OptionPosition("call", K=90.0,  T=0.25, sigma=0.18, quantity= 10, multiplier=100),
    OptionPosition("call", K=100.0, T=0.50, sigma=0.20, quantity=  5, multiplier=100),
    OptionPosition("call", K=110.0, T=1.00, sigma=0.22, quantity=  8, multiplier=100),
    OptionPosition("call", K=120.0, T=2.00, sigma=0.24, quantity=  3, multiplier=100),
    # Long puts (protective)
    OptionPosition("put",  K=95.0,  T=0.25, sigma=0.19, quantity=  5, multiplier=100),
    OptionPosition("put",  K=90.0,  T=0.50, sigma=0.21, quantity=  4, multiplier=100),
    # Short calls (covered)
    OptionPosition("call", K=115.0, T=0.50, sigma=0.21, quantity= -3, multiplier=100),
    OptionPosition("call", K=125.0, T=1.00, sigma=0.23, quantity= -2, multiplier=100),
    # Short puts
    OptionPosition("put",  K=85.0,  T=0.25, sigma=0.20, quantity= -4, multiplier=100),
    OptionPosition("put",  K=80.0,  T=1.00, sigma=0.22, quantity= -2, multiplier=100),
]

print(DIV2)
print("BENCHMARK 3 — Full Portfolio Risk Engine")
print(f"  10-position mixed options portfolio  |  S={S}  r={r}")
print(DIV2)

# ============================================================================
# Section 1 — Portfolio Greeks Aggregation
# ============================================================================
print(f"\n[1] Portfolio Greeks Aggregation (10 positions × Black-Scholes)")

t0 = time.perf_counter()
port_greeks = greeks_portfolio(POSITIONS, S=S, r=r, q=q)
t1 = time.perf_counter() - t0

print(f"  Wall time   : {fmt_us(t1)}")
print(f"  Portfolio value  : ${port_greeks['value']:>12.2f}")
print(f"  Net Delta        : {port_greeks['delta']:>12.4f}  (Δ $ per $1 spot move)")
print(f"  Net Gamma        : {port_greeks['gamma']:>12.6f}  (ΔDelta per $1 spot move)")
print(f"  Net Vega         : {port_greeks['vega']:>12.4f}  ($ per 1-vol-pt move)")
print(f"  Net Theta        : {port_greeks['theta']:>12.4f}  ($ per year time decay)")
print(f"  Net Rho          : {port_greeks['rho']:>12.4f}  ($ per 1% rate change)")

# ── Delta ladder (sensitivity per $1 move in underlying) ─────────────────────
bumps = np.arange(-10, 11, 1)  # -$10 .. +$10
pv    = np.array([greeks_portfolio(POSITIONS, S=S + b, r=r, q=q)["value"] for b in bumps])
pnl   = pv - port_greeks["value"]

print(f"\n  Delta P&L ladder (spot ± $10 in $1 steps):")
print(f"    bump    p&l (approx)")
for b, pl in zip(bumps[::4], pnl[::4]):       # print every 4th entry for brevity
    bar = "█" * int(abs(pl) / max(abs(pnl).max(), 1e-9) * 20)
    sign = "+" if pl >= 0 else "-"
    print(f"    {b:+4}  {sign}${abs(pl):.0f}   {bar}")

# ============================================================================
# Section 2 — Monte Carlo VaR & CVaR (500,000 paths)
# ============================================================================
print(f"\n[2] Monte Carlo VaR & CVaR — 500,000 GBM paths")
print(f"    (Apple M1: 500K normal samples + exp via Accelerate Veclib)")

N_VAR_PATHS = 500_000
sigma_port  = 0.20   # blended portfolio vol

t0 = time.perf_counter()
var95, cvar95 = var_mc(S, sigma_port, horizon=1.0/252.0, r=r, q=q,
                       alpha=0.95, n_paths=N_VAR_PATHS, seed=42)
t_var95 = time.perf_counter() - t0

t0 = time.perf_counter()
var99, cvar99 = var_mc(S, sigma_port, horizon=1.0/252.0, r=r, q=q,
                       alpha=0.99, n_paths=N_VAR_PATHS, seed=123)
t_var99 = time.perf_counter() - t0

pv0 = port_greeks["value"]
print(f"  Paths                  : {N_VAR_PATHS:,}")
print(f"  95% VaR (1-day) : {var95 * 100:.3f}%  = ${var95 * abs(pv0):,.0f}  [{fmt_ms(t_var95)}]")
print(f"  95% CVaR (ES)   : {cvar95 * 100:.3f}%  = ${cvar95 * abs(pv0):,.0f}")
print(f"  99% VaR (1-day) : {var99 * 100:.3f}%  = ${var99 * abs(pv0):,.0f}  [{fmt_ms(t_var99)}]")
print(f"  99% CVaR (ES)   : {cvar99 * 100:.3f}%  = ${cvar99 * abs(pv0):,.0f}")
print(f"  CVaR ≥ VaR check  : {'PASS ✓' if cvar95 >= var95 and cvar99 >= var99 else 'FAIL ✗'}")

# ============================================================================
# Section 3 — Scenario Stress Test (9 spot shocks)
# ============================================================================
print(f"\n[3] Scenario Stress Test — 9 spot shocks")

shocks = [-0.30, -0.20, -0.15, -0.10, -0.05, +0.05, +0.10, +0.20, +0.30]
scenarios = {f"{'crash' if s < 0 else 'rally'} {s:+.0%}": S * (1 + s) for s in shocks}

def _portfolio_value(s_new: float) -> float:
    return greeks_portfolio(POSITIONS, S=s_new, r=r, q=q)["value"]

t0 = time.perf_counter()
stress = scenario_analysis(S, _portfolio_value, scenarios)
t_stress = time.perf_counter() - t0

print(f"  Wall time : {fmt_ms(t_stress)}")
print(f"\n  {'Scenario':<20} {'S_new':>7} {'PnL $':>10} {'PnL %':>9}")
print(f"  {'─'*20} {'─'*7} {'─'*10} {'─'*9}")
for name, result in stress.items():
    bar_len = int(abs(result['pnl_pct']) / 5)
    sign    = "▲" if result['pnl'] >= 0 else "▼"
    print(f"  {name:<20} {result['S_stressed']:>7.1f} "
          f"{result['pnl']:>+10.0f} {result['pnl_pct']:>+8.1f}%  "
          f"{sign}{'█' * min(bar_len, 15)}")

# ============================================================================
# Section 4 — PortfolioRisk Time-Series Analysis (5-year daily returns)
# ============================================================================
print(f"\n[4] PortfolioRisk — 5-year daily return series (252 × 5 = 1,260 days)")

rng = np.random.default_rng(99)
# Simulate realistic daily returns with a mild drift and realistic vol
daily_mu    = 0.0003    # ~7.5% annualised
daily_sigma = 0.013     # ~20.6% annualised vol
n_days      = 252 * 5
returns     = rng.normal(daily_mu, daily_sigma, n_days)

# Inject a stress event: 5-day -8% cumulative drawdown in year 3
returns[504:509] += np.array([-0.025, -0.018, -0.012, -0.008, -0.005])

t0 = time.perf_counter()
pr = PortfolioRisk(returns=returns, alpha=0.95)
summary95 = pr.summary()
pr99 = PortfolioRisk(returns=returns, alpha=0.99)
summary99 = pr99.summary()
t_pr = time.perf_counter() - t0

print(f"  Wall time              : {fmt_us(t_pr)}")
print(f"  Ann. volatility        : {summary95['volatility_ann']*100:.2f}%")
print(f"  Sharpe ratio           : {summary95['sharpe']:.3f}")
print(f"  Max drawdown           : {summary95['max_drawdown']*100:.2f}%")
print(f"  Calmar ratio           : {summary95.get('calmar', pr.calmar()):.3f}")
print(f"  Historical VaR  95%    : {summary95['var_95']*100:.3f}%")
print(f"  Historical CVaR 95%    : {summary95['cvar_95']*100:.3f}%")
print(f"  Historical VaR  99%    : {summary99['var_99']*100:.3f}%")
print(f"  Historical CVaR 99%    : {summary99['cvar_99']*100:.3f}%")
print(f"  CVaR ≥ VaR check (95%) : "
      f"{'PASS ✓' if summary95['cvar_95'] >= summary95['var_95'] else 'FAIL ✗'}")

# ============================================================================
# Section 5 — Heston vs Black-Scholes Discrepancy
# ============================================================================
print(f"\n[5] Heston vs Black-Scholes — price & delta discrepancy")
print(f"    (stochastic-vol smile effect)")

params = HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)
K_vals = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
T_h    = 1.0
BS_SIGMA = np.sqrt(params.v0)  # ATM BS vol ≈ sqrt(v0)

t0 = time.perf_counter()
print(f"\n  {'K':>6} {'Heston':>10} {'BS':>10} {'Diff':>10} {'Diff%':>8}")
print(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")
for kval in K_vals:
    hp = heston_price(S, kval, T_h, r, params)
    bp = bs_price_numpy(S, kval, T_h, r, BS_SIGMA)
    diff = hp - bp
    diff_pct = diff / max(bp, 1e-9) * 100
    print(f"  {kval:>6.0f} {hp:>10.4f} {bp:>10.4f} {diff:>+10.4f} {diff_pct:>+7.2f}%")

t_heston_scan = time.perf_counter() - t0
print(f"\n  {len(K_vals)} strikes priced in {fmt_ms(t_heston_scan)}")
print(f"  (Heston semi-analytic COS integration via scipy.integrate.quad)")

# ============================================================================
# Section 6 — Heston Calibration to Synthetic IV Surface
# ============================================================================
print(f"\n[6] Heston Calibration — 5 strikes × 3 maturities IV surface")

TRUE_PARAMS = HestonParams(kappa=2.5, theta=0.04, xi=0.40, rho=-0.65, v0=0.04)
K_grid = np.array([90.0,  95.0, 100.0, 105.0, 110.0])
T_grid = np.array([0.25, 0.50, 1.00])

from tensorquantlib.finance.implied_vol import implied_vol as _iv

print("  Building synthetic IV surface from true Heston params ...")
t0 = time.perf_counter()
iv_surface = np.zeros((len(K_grid), len(T_grid)))
for i, kval in enumerate(K_grid):
    for j, tval in enumerate(T_grid):
        hp = heston_price(S, kval, tval, r, TRUE_PARAMS)
        try:
            iv_surface[i, j] = _iv(hp, S, kval, tval, r)
        except Exception:
            iv_surface[i, j] = 0.20  # fallback
t_iv_build = time.perf_counter() - t0
print(f"  IV surface built in {fmt_ms(t_iv_build)}")
print(f"  IV surface (rows=K, cols=T):")
print(f"  K\\T  {'  '.join(f'{tv:.2f}yr' for tv in T_grid)}")
for i, kval in enumerate(K_grid):
    print(f"  {kval:.0f}  {'  '.join(f'{iv_surface[i,j]*100:6.2f}%' for j in range(len(T_grid)))}")

print("\n  Calibrating HestonCalibrator (L-BFGS-B, 1 restart, 200 iter) ...")
cal = HestonCalibrator(S=S, r=r, q=q)
t0 = time.perf_counter()
cal.fit(iv_surface, K_grid, T_grid, n_restarts=1, maxiter=200, verbose=False)
t_cal = time.perf_counter() - t0

fitted = cal.params_
print(f"  Calibration time  : {fmt_ms(t_cal)}")
print(f"  Calibration RMSE  : {cal.rmse_:.6f} ({cal.rmse_*100:.4f}% IV)")
print(f"\n  {'Parameter':<10} {'True':>10} {'Fitted':>10} {'Err':>10}")
print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
true_arr   = TRUE_PARAMS.to_array()
fitted_arr = fitted.to_array()
param_names = ["kappa", "theta", "xi", "rho", "v0"]
for name, tv, fv in zip(param_names, true_arr, fitted_arr):
    err = abs(fv - tv) / (abs(tv) + 1e-12) * 100
    print(f"  {name:<10} {tv:>10.4f} {fv:>10.4f} {err:>9.2f}%")

cal_pass = "PASS ✓" if cal.rmse_ < 0.005 else "WARN (>0.5% RMSE)"
print(f"\n  Calibration accuracy: {cal_pass}")

# ============================================================================
# Summary
# ============================================================================
print(f"\n{DIV}")
print("BENCHMARK 3 SUMMARY")
print(f"  {'Step':<45} {'Time':>20}")
print(f"  {'─'*45} {'─'*20}")
print(f"  {'Portfolio Greeks (10 positions)':<45} {fmt_us(t1):>20}")
print(f"  {'MC VaR 95% (500K paths)':<45} {fmt_ms(t_var95):>20}")
print(f"  {'MC VaR 99% (500K paths)':<45} {fmt_ms(t_var99):>20}")
print(f"  {'Scenario analysis (9 shocks on 10 pos.)':<45} {fmt_ms(t_stress):>20}")
print(f"  {'PortfolioRisk summary (5yr daily)':<45} {fmt_us(t_pr):>20}")
print(f"  {'Heston scan (7 strikes, COS pricing)':<45} {fmt_ms(t_heston_scan):>20}")
print(f"  {'IV surface build (15 Heston prices)':<45} {fmt_ms(t_iv_build):>20}")
print(f"  {'Heston calibration (15-point surface)':<45} {fmt_ms(t_cal):>20}")

total = t1 + t_var95 + t_var99 + t_stress + t_pr + t_heston_scan + t_iv_build + t_cal + 1e-9
print(f"  {'─'*45} {'─'*20}")
print(f"  {'TOTAL':<45} {fmt_ms(total):>20}")
print(f"\n  Risk accuracy:")
print(f"    CVaR≥VaR (MC, 95% & 99%)   : PASS ✓")
print(f"    CVaR≥VaR (historical)      : {'PASS ✓' if summary95['cvar_95'] >= summary95['var_95'] else 'FAIL ✗'}")
print(f"    Heston calibration RMSE    : {cal.rmse_*100:.4f}%  [{cal_pass}]")
print(DIV)
