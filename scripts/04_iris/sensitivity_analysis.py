"""sensitivity_analysis.py -- declared-parameter sensitivity for Sect. 8.6.

Section 5.4 of the manuscript states that the ridge ``epsilon`` and the
coupling scale ``s`` are declared analysis parameters requiring stability
checks rather than hidden tuning knobs.  This script produces the grid
reported in Sect. 8.6: the matched contrasts (coupled minus J=0) over

    epsilon in {1e-4, 1e-3, 1e-2, 1e-1}
    s       in {0.4, 0.6, 0.8, 1.0, 1.2}

on Iris setosa, four qubits.  The reported conclusion is that every
contrast keeps its sign at all twenty settings while the magnitudes vary
by up to a factor of twenty, so qualitative readings are stable and
quantitative ones must be quoted with their (epsilon, s).

Run from the repository root::

    python3 sensitivity_analysis.py
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris

from encoding_zoo import _partial_correlation_coupling, enc_product, enc_sandwich
from null_controls import profile, scale_to_angles

RIDGES = (1e-4, 1e-3, 1e-2, 1e-1)
SCALES = (0.4, 0.6, 0.8, 1.0, 1.2)
KEYS = ("d_purity", "d_massgap", "d_I", "Q", "current")


def main() -> None:
    iris = load_iris()
    raw = iris.data[iris.target == 0]
    X = scale_to_angles(raw)
    control = profile(enc_product(X))

    print("Sensitivity of the matched contrasts, Iris setosa, four qubits")
    print(f"{'ridge':>8}{'scale':>7}{'d purity':>11}{'d massgap':>11}"
          f"{'d I':>9}{'Q':>9}{'current':>9}")
    rows = []
    for ridge in RIDGES:
        for scale in SCALES:
            J = _partial_correlation_coupling(raw, ridge=ridge, scale=scale)
            p = profile(enc_sandwich(X, J=J))
            row = (
                p["purity"] - control["purity"],
                p["gap"] - control["gap"],
                p["I"] - control["I"],
                p["Q"],
                p["current"],
            )
            rows.append(row)
            print(f"{ridge:8.0e}{scale:7.1f}" + "".join(f"{v:11.4f}" if i < 3
                                                        else f"{v:9.4f}"
                                                        for i, v in enumerate(row)))

    grid = np.asarray(rows)
    print("\nSign stability across the 4x5 grid:")
    for i, name in enumerate(KEYS):
        v = grid[:, i]
        stable = bool(np.all(v >= -1e-12) or np.all(v <= 1e-12))
        print(f"  {name:10} min={v.min():+.4f} max={v.max():+.4f} "
              f"sign_stable={stable}")
    print(f"\nRange of variation for d_purity: "
          f"{grid[:, 0].max() / grid[:, 0].min():.1f}x")


if __name__ == "__main__":
    main()
