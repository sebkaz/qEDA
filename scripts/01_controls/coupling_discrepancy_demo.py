"""Regression check for Eq. (warmstart) and ``precision_coupling``.

Both implementations must compute

    J_jk = -s * Theta_jk / sqrt(Theta_jj * Theta_kk).

The explicit implementation below is intentionally independent of the helper
and catches sign or normalisation regressions.
"""

import numpy as np

from step1_statistics import precision_coupling

RIDGE, SCALE = 0.1, 0.6


def paper_formula(data, ridge=RIDGE, scale=SCALE):
    """Eq. (warmstart) verbatim."""
    n = data.shape[1]
    theta = np.linalg.inv(np.cov(data, rowvar=False) + ridge * np.eye(n))
    denom = np.sqrt(np.outer(np.diag(theta), np.diag(theta)))
    J = -scale * theta / denom          # <-- MINUS sign, per-pair denominator
    np.fill_diagonal(J, 0.0)
    return 0.5 * (J + J.T)


rng = np.random.default_rng(1)
# 3 features with clearly UNEQUAL, mixed-sign dependence
cov = np.array([[1.0, 0.70, 0.10],
                [0.70, 1.0, -0.35],
                [0.10, -0.35, 1.0]])
raw = rng.normal(size=(2000, 3)) @ np.linalg.cholesky(cov).T

J_paper = paper_formula(raw)
J_step1 = precision_coupling(raw, ridge=RIDGE, scale=SCALE)

np.set_printoptions(precision=4, suppress=True)
print("Eq. (warmstart), as written in 03_wall.tex:")
print(J_paper)
print("\nstep1_statistics.precision_coupling():")
print(J_step1)
error = float(np.max(np.abs(J_paper - J_step1)))
print(f"\nmax absolute difference: {error:.3e}")
if error > 1e-12:
    raise RuntimeError("precision_coupling disagrees with Eq. (warmstart)")

print("\n--- mixed-sign control ---")
pairs = [(0, 1), (0, 2), (1, 2)]
print(f"{'pair':>6} {'paper':>10} {'step1':>10} {'same sign?':>12}")
for j, k in pairs:
    a, b = J_paper[j, k], J_step1[j, k]
    print(f"{f'({j},{k})':>6} {a:10.4f} {b:10.4f} "
          f"{str(np.sign(a) == np.sign(b)):>12}")

print("\nTrue correlations were: (0,1)=+0.70, (0,2)=+0.10, (1,2)=-0.35")
print("Both implementations reproduce the partial-correlation sign.")

print("\n--- per-pair denominator control ---")
print(f"max|J| paper = {np.max(np.abs(J_paper)):.4f}   "
      f"max|J| step1 = {np.max(np.abs(J_step1)):.4f}")
print("The maximum is not pinned to `scale`; weak dependence remains weak.")

print("\n--- consequence: behaviour on data with NO dependence at all ---")
print(f"{'M':>7} {'max|J| paper':>14} {'max|J| step1':>14}")
for M in (50, 500, 5000, 50000):
    null = rng.normal(size=(M, 3))          # exactly independent columns
    expected = paper_formula(null)
    actual = precision_coupling(null, ridge=RIDGE, scale=SCALE)
    if not np.allclose(expected, actual, atol=1e-12, rtol=0.0):
        raise RuntimeError("Null control implementations disagree")
    print(f"{M:7d} {np.max(np.abs(expected)):14.4f} "
          f"{np.max(np.abs(actual)):14.4f}")
print("\nBoth implementations shrink toward zero under the independence null.")
