"""Spot-check that documentation code examples actually run."""
import numpy as np

# Test 1: Heston pricing with HestonParams
from tensorquantlib import heston_price, HestonParams
params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
price = heston_price(S=100, K=100, T=1.0, r=0.05, params=params)
print(f"Heston price: {price:.4f}")

# Test 2: Implied vol
from tensorquantlib import implied_vol
iv = implied_vol(10.45, S=100, K=100, T=1.0, r=0.05)
print(f"Implied vol: {iv:.4f}")

# Test 3: SVI with log-moneyness
from tensorquantlib import svi_implied_vol
k = np.linspace(-0.3, 0.3, 7)
ivs = svi_implied_vol(k, T=1.0, a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
print(f"SVI vols: {[round(v, 4) for v in ivs]}")

# Test 4: Second-order Greeks
from tensorquantlib import second_order_greeks, bs_price_tensor
greeks = second_order_greeks(price_fn=bs_price_tensor, S=100, K=100, T=1.0, r=0.05, sigma=0.2)
print(f"Gamma: {greeks['gamma']:.6f}, Vanna: {greeks['vanna']:.6f}")

# Test 5: CDS spread
from tensorquantlib import cds_spread
spread = cds_spread(hazard_rate=0.02, T=5, recovery=0.4, r=0.03)
print(f"CDS spread: {spread:.4f}")

# Test 6: Asian option
from tensorquantlib import asian_price_mc
asian = asian_price_mc(S=100, K=100, T=1.0, r=0.05, sigma=0.2, n_paths=50000, n_steps=252)
print(f"Asian price: {asian:.2f}")

# Test 7: SABR vol
from tensorquantlib import sabr_implied_vol
vol = sabr_implied_vol(F=100, K=100, T=1.0, alpha=0.3, beta=0.5, rho=-0.2, nu=0.4)
print(f"SABR ATM vol: {vol:.4f}")

# Test 8: Vasicek bond
from tensorquantlib import vasicek_bond_price
bp = vasicek_bond_price(r0=0.05, kappa=0.3, theta=0.04, sigma=0.01, T=5)
print(f"Vasicek bond price: {bp:.4f}")

# Test 9: FX option
from tensorquantlib import garman_kohlhagen
fx = garman_kohlhagen(S=1.10, K=1.12, T=0.5, r_d=0.02, r_f=0.01, sigma=0.08, option_type="call")
print(f"GK FX call price: {fx:.4f}")

# Test 10: Merton jump-diffusion
from tensorquantlib import merton_jump_price
mj = merton_jump_price(S=100, K=100, T=1.0, r=0.05, sigma=0.2, lam=0.5, mu_j=-0.1, sigma_j=0.15)
print(f"Merton jump price: {mj:.4f}")

# Test 11: Cap price
from tensorquantlib import cap_price
forwards = np.array([0.03, 0.032, 0.035])
expiries = np.array([0.5, 1.0, 1.5])
dfs = np.array([0.985, 0.970, 0.955])
c = cap_price(forwards, strike=0.03, expiries=expiries, sigma=0.20, dfs=dfs)
print(f"Cap price: {c:.6f}")

# Test 12: Barrier option
from tensorquantlib import barrier_price
bp2 = barrier_price(S=100, K=100, T=1.0, r=0.05, sigma=0.2, barrier=120, barrier_type="up-and-out")
print(f"Barrier price: {bp2:.2f}")

print("\nAll 12 documentation examples validated successfully!")
