"""Quick verification of all audit fixes."""
import numpy as np
import tensorquantlib as tql
from tensorquantlib.core.tensor import Tensor
from tensorquantlib.core import ops

# 1. ops has 7 new exports + tsum/tpow
for name in ['sin', 'cos', 'tanh', 'abs', 'clip', 'where', 'softmax', 'tsum', 'tpow']:
    assert hasattr(ops, name), f"ops.{name} missing"
print("PASS: ops has all 9 new exports")

# 2. __getitem__ autograd
x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
y = x[0] ** 2 + x[2] ** 2
y.backward()
np.testing.assert_allclose(x.grad, [2.0, 0.0, 6.0])
print(f"PASS: getitem autograd correct, grad = {x.grad}")

# 3. detach
z = x.detach()
assert not z.requires_grad
print("PASS: detach() works")

# 4. repr compact for large tensor
big = Tensor(np.ones((100, 200)), requires_grad=True)
r = repr(big)
assert "shape=" in r and "100" in r, f"repr not compact: {r[:80]}"
print(f"PASS: repr compact: {r}")

# 5. free_graph
x2 = Tensor(np.array([1.0, 2.0]), requires_grad=True)
y2 = x2 ** 2
y2.backward()
y2.free_graph()
assert len(y2._children) == 0
print("PASS: free_graph() clears children")

# 6. vega shape fix
S_arr = np.linspace(80, 120, 50)
result = tql.compute_greeks_vectorized(tql.bs_price_tensor, S_arr, 100, 1.0, 0.05, 0.2)
assert result['vega'].shape == (50,), f"vega shape {result['vega'].shape} != (50,)"
assert result['delta'].shape == (50,)
print(f"PASS: vega shape correct: {result['vega'].shape}")

# 7. second_order_greeks (fast combined path)
so = tql.second_order_greeks(tql.bs_price_tensor, 100.0, 100.0, 1.0, 0.05, 0.2)
assert 'gamma' in so and 'vanna' in so and 'volga' in so
# Analytical gamma for ATM BS: gamma = N'(d1) / (S * sigma * sqrt(T))
import scipy.stats
d1 = (np.log(1.0) + (0.05 + 0.5 * 0.04) * 1.0) / (0.2 * 1.0)
gamma_analytic = scipy.stats.norm.pdf(d1) / (100.0 * 0.2 * 1.0)
assert abs(so['gamma'] - gamma_analytic) < 0.01, f"gamma err: {abs(so['gamma'] - gamma_analytic)}"
print(f"PASS: second_order_greeks: {so}")

# 8. basket analytic uses moment matching (not intrinsic forward value)
from tensorquantlib.finance.basket import build_pricing_grid_analytic
grid, axes = build_pricing_grid_analytic(
    [(80, 120), (80, 120)], 100, 1.0, 0.05,
    np.array([0.2, 0.25]), np.array([0.5, 0.5]), n_points=10,
)
centre = float(grid[5, 5])
assert centre > 1.0, f"analytic value too low: {centre}"
# Deep ITM (both assets at 120) should be worth more than OTM (both at 80)
assert grid[9, 9] > grid[0, 0], "ITM > OTM failed"
print(f"PASS: analytic basket grid centre = {centre:.4f}")

# 9. TTSurrogate has plot methods
surr = tql.TTSurrogate.from_basket_analytic(
    [(80, 120), (80, 120)], K=100, T=1.0, r=0.05,
    sigma=[0.2, 0.25], weights=[0.5, 0.5], n_points=15, eps=1e-3,
)
assert hasattr(surr, 'plot_surface'), "TTSurrogate.plot_surface missing"
assert hasattr(surr, 'plot_greeks'), "TTSurrogate.plot_greeks missing"
assert hasattr(surr, 'plot_ranks'), "TTSurrogate.plot_ranks missing"
print("PASS: TTSurrogate has plot_surface, plot_greeks, plot_ranks")

print()
print("ALL CHECKS PASSED")
