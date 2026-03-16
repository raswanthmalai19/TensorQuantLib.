"""Debug script for TT-Cross trig function issue."""
import numpy as np
from tensorquantlib.tt.decompose import tt_cross
from tensorquantlib.tt.ops import tt_to_full

n = 8
# Axes starting at 0 — sin(0) = 0, so rows starting with axis[0]=0 are all zero
axes0 = [np.linspace(0.0, np.pi/2, n)] * 3
# Axes starting at offset — avoid zero
axes1 = [np.linspace(0.1, np.pi/2, n)] * 3


def make_fn(axes):
    def fn(*indices):
        vals = [axes[k][i] for k, i in enumerate(indices)]
        result = 1.0
        for j, v in enumerate(vals):
            result *= (np.sin(v) if j % 2 == 0 else np.cos(v))
        return float(result)
    return fn

fn0 = make_fn(axes0)
fn1 = make_fn(axes1)

print("f0(0,1,1)=", fn0(0, 1, 1), " (should be 0 since sin(0)=0)")
print("f0(1,1,1)=", fn0(1, 1, 1))

print("\nWith axes starting at 0.0:")
cores0 = tt_cross(fn0, shape=(n, n, n), eps=0.05, max_rank=10, n_sweeps=6)
A_ref0 = np.array([[[fn0(i, j, k) for k in range(n)] for j in range(n)] for i in range(n)])
A_tt0 = tt_to_full(cores0)
rel_err0 = np.linalg.norm(A_tt0 - A_ref0) / (np.linalg.norm(A_ref0) + 1e-15)
print(f"rel_err={rel_err0:.4e}  A_tt norm={np.linalg.norm(A_tt0):.4f}  ref norm={np.linalg.norm(A_ref0):.4f}")

print("\nWith axes starting at 0.1 (offset):")
cores1 = tt_cross(fn1, shape=(n, n, n), eps=0.05, max_rank=10, n_sweeps=6)
A_ref1 = np.array([[[fn1(i, j, k) for k in range(n)] for j in range(n)] for i in range(n)])
A_tt1 = tt_to_full(cores1)
rel_err1 = np.linalg.norm(A_tt1 - A_ref1) / (np.linalg.norm(A_ref1) + 1e-15)
print(f"rel_err={rel_err1:.4e}  A_tt norm={np.linalg.norm(A_tt1):.4f}  ref norm={np.linalg.norm(A_ref1):.4f}")
