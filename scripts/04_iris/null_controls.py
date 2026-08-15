"""null_controls.py -- calibration nulls for the operator statistics.

WHY
---
The manuscript reports absolute values of the subsystem statistics, e.g.
Q = 0.420 on Iris.  Without a reference distribution such a number is not
interpretable: the reader cannot tell whether it reflects the data, the
size of the coupling, or merely the fact that an entangling circuit was
applied at all.  This script supplies two nulls.

  NULL 1 (structure)  Replace the partial-correlation J by a RANDOM
                      symmetric J with the SAME Frobenius norm.  If the
                      reported statistics are driven by the conditional
                      dependence structure of the data, the true J should
                      sit far in the tail of the random-J distribution.

  NULL 2 (factorisation)  Apply one FIXED Haar-random unitary after the
                      real product encoding.  This is entirely
                      data-independent.  Because it is a global unitary
                      it cannot change the spectrum, so purity, mass gap
                      and the frame potentials must be unchanged; but it
                      does change the tensor factorisation, so it shows
                      how much I and Q respond to factorisation alone.

Datasets: Iris (used in E4 of the manuscript) and a matched synthetic
Gaussian with deliberately ASYMMETRIC covariance.  Asymmetry matters: an
equicorrelated sample produces a near-uniform J, which suppresses the
encoded current by symmetry and would make the current null vacuous.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris

from qeda.engine import (
    log_negativity,
    mass_gap,
    maximum_current,
    mutual_information,
    purity,
)
from encoding_zoo import (
    _haar_unitary,
    _partial_correlation_coupling,
    _random_coupling_like,
    enc_haar_fixed,
    enc_product,
    enc_sandwich,
    enc_zzmap,
    frame_potential,
)

SEED = 20260812
COUPLING_SCALE = 0.8      # the manuscript's value for the Iris audit
RIDGE = 1e-3
N_DRAWS = 200


def profile(states):
    A = states.T
    rho = A @ A.conj().T / A.shape[1]
    rho = 0.5 * (rho + rho.conj().T)
    g = mass_gap(rho)
    return {
        "purity": purity(rho),
        "gap": g if np.isfinite(g) else np.nan,
        "I": mutual_information(rho),
        "Q": log_negativity(rho),
        "current": maximum_current(rho),
        "F2": frame_potential(states, 2),
    }


def scale_to_angles(raw):
    lo, hi = raw.min(0), raw.max(0)
    return (raw - lo) / (hi - lo) * np.pi


def class_null(raw, X, rng, label=""):
    """Return the true-J profile and the random-J null distribution."""
    J = _partial_correlation_coupling(raw, ridge=RIDGE, scale=COUPLING_SCALE)
    true = profile(enc_sandwich(X, J=J))
    draws = {k: [] for k in ("purity", "gap", "I", "Q", "current")}
    for _ in range(N_DRAWS):
        p = profile(enc_sandwich(X, J=_random_coupling_like(J, rng)))
        for k in draws:
            draws[k].append(p[k])
    return J, true, {k: np.asarray(v) for k, v in draws.items()}


def report(name, J, true, draws):
    off = J[np.triu_indices(len(J), 1)]
    print(f"\n--- {name} ---")
    print(f"  |J|_F = {np.linalg.norm(J):.4f}   off-diagonals "
          f"{np.round(off, 3)}   sd {np.std(off):.3f}")
    print(f"  {'statistic':10}{'true J':>10}{'null mean':>11}{'null sd':>10}"
          f"{'z':>8}{'p(2-sided)':>12}")
    from scipy.stats import norm
    for k, v in draws.items():
        z = (true[k] - v.mean()) / v.std() if v.std() > 1e-12 else np.nan
        p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
        print(f"  {k:10}{true[k]:10.4f}{v.mean():11.4f}{v.std():10.4f}"
              f"{z:8.2f}{p:12.3f}")


def main():
    rng = np.random.default_rng(SEED)

    print("=" * 78)
    print("NULL 1 -- is the partial-correlation J distinguishable from a")
    print("          random J of the same Frobenius norm?  (%d draws)" % N_DRAWS)
    print("=" * 78)

    iris = load_iris()
    for c in range(3):
        raw = iris.data[iris.target == c]
        J, true, draws = class_null(raw, scale_to_angles(raw), rng,
                                    label=iris.target_names[c])
        report(f"Iris / {iris.target_names[c]}  (n={len(raw)}, 4 qubits)",
               J, true, draws)

    cov = np.array([[1.0, 0.75, 0.05], [0.75, 1.0, -0.40], [0.05, -0.40, 1.0]])
    raw = rng.normal(size=(300, 3)) @ np.linalg.cholesky(cov).T
    J, true, draws = class_null(raw, scale_to_angles(raw), rng)
    report("Synthetic, asymmetric covariance (n=300, 3 qubits)", J, true, draws)

    print("\n" + "=" * 78)
    print("NULL 2 -- a fixed, DATA-INDEPENDENT Haar unitary after the real")
    print("          product encoding")
    print("=" * 78)
    raw = iris.data[iris.target == 0]
    X = scale_to_angles(raw)
    U = _haar_unitary(2 ** X.shape[1], rng)
    Jp = _partial_correlation_coupling(raw, ridge=RIDGE, scale=COUPLING_SCALE)
    rows = {
        "product (J=0 control)": enc_product(X),
        "sandwich (partial-corr J)": enc_sandwich(X, J=Jp),
        "fixed Haar unitary": enc_haar_fixed(X, U=U),
        "ZZFeatureMap reps=1": enc_zzmap(X, reps=1),
        "ZZFeatureMap reps=2": enc_zzmap(X, reps=2),
    }
    hdr = (f"{'encoding':28}{'purity':>9}{'gap':>8}{'I':>8}{'Q':>8}"
           f"{'current':>9}{'F2':>8}")
    print("\nIris / setosa, 4 qubits")
    print(hdr)
    print("-" * len(hdr))
    for name, st in rows.items():
        p = profile(st)
        print(f"{name:28}{p['purity']:9.4f}{p['gap']:8.3f}{p['I']:8.4f}"
              f"{p['Q']:8.4f}{p['current']:9.4f}{p['F2']:8.4f}")
    print("\n  Note: the Haar row must reproduce the product row exactly in")
    print("  purity, gap and F2 -- a global unitary cannot move the spectrum.")
    print("  Any difference there would indicate an implementation error.")


if __name__ == "__main__":
    main()
