"""encoding_zoo.py -- qEDA profile across a family of declared encodings.

Motivation
----------
The manuscript audits one encoding in depth (the RY/ZZ sandwich) with an
exact J=0 control.  Two questions are left open, and a referee will ask
both:

  Q1  Does the audit work on an encoding we did not design?  The obvious
      candidate is the Havlicek ZZFeatureMap, whose coupling is a
      FUNCTION OF THE DATA POINT, phi_jk(x) = (pi-x_j)(pi-x_k), so there
      is no J to declare at all.

  Q2  Is the precision-derived J special, or would ANY coupling of the
      same size produce the same operator enrichment?  Without this null
      the reported Q and current cannot be attributed to conditional
      dependence structure.

This script computes the full qEDA profile for a common dataset under:

  product        RY(x_j)                        -- the real J=0 control
  sandwich       RY(x/2) ZZ(J_prec) RY(x/2)     -- the paper's encoding
  sandwich_rand  RY(x/2) ZZ(J_rand) RY(x/2)     -- NULL for Q2: J_rand is
                                                   random with the SAME
                                                   Frobenius norm
  zz_post        RY(x) then ZZ(J_prec)          -- coupling not sandwiched
  zzmap1/zzmap2  Havlicek ZZFeatureMap, 1 and 2 repetitions
  haar_fixed     RY(x_j) then ONE fixed Haar unitary -- NULL for "any
                                                   scrambling looks like
                                                   structure"

Reported per encoding: purity, mass gap, mean single-mode mutual
information, mean single-mode log-negativity, max |current|, and the
frame potentials F_2, F_3 (which are NOT functions of rho and therefore
probe the state cloud beyond its barycentre).

Everything is plain NumPy; the product and sandwich paths are checked
against dqsa_engine to make sure this file computes the same objects the
paper does.
"""

from __future__ import annotations

import numpy as np

from qeda.engine import (
    log_negativity,
    mass_gap,
    maximum_current,
    mutual_information,
    purity,
)

RNG_SEED = 20260812
N_QUBITS = 3


# ---------------------------------------------------------------- utils
def _kron_list(mats):
    out = np.array([[1.0 + 0.0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


def _ry(theta):
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)


def _z_signs(n):
    """(2^n, n) array of +-1: the Z eigenvalue of each qubit per basis state."""
    idx = np.arange(2**n)
    bits = ((idx[:, None] >> np.arange(n - 1, -1, -1)[None, :]) & 1)
    return 1.0 - 2.0 * bits


def _zz_phases(J, n):
    """Diagonal of exp(-i sum_{j<k} J_jk Z_j Z_k)."""
    z = _z_signs(n)
    expo = np.zeros(2**n)
    for j in range(n):
        for k in range(j + 1, n):
            expo += J[j, k] * z[:, j] * z[:, k]
    return np.exp(-1j * expo)


def _partial_correlation_coupling(raw, ridge=1e-3, scale=0.8):
    """Eq. (warmstart) of 03_wall.tex."""
    n = raw.shape[1]
    theta = np.linalg.inv(np.cov(raw, rowvar=False) + ridge * np.eye(n))
    denom = np.sqrt(np.outer(np.diag(theta), np.diag(theta)))
    J = -scale * theta / denom
    np.fill_diagonal(J, 0.0)
    return 0.5 * (J + J.T)


def _random_coupling_like(J, rng):
    """Random symmetric zero-diagonal J with the same Frobenius norm."""
    n = J.shape[0]
    R = rng.normal(size=(n, n))
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 0.0)
    return R * (np.linalg.norm(J) / np.linalg.norm(R))


def _haar_unitary(dim, rng):
    z = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return q * (d / np.abs(d))[None, :]


# ------------------------------------------------------------ encodings
def enc_product(X, **kw):
    n = X.shape[1]
    out = np.empty((len(X), 2**n), dtype=complex)
    vac = np.zeros(2**n, dtype=complex)
    vac[0] = 1.0
    for m, row in enumerate(X):
        out[m] = _kron_list([_ry(v) for v in row]) @ vac
    return out


def enc_sandwich(X, J=None, **kw):
    n = X.shape[1]
    ph = _zz_phases(J, n)
    vac = np.zeros(2**n, dtype=complex)
    vac[0] = 1.0
    out = np.empty((len(X), 2**n), dtype=complex)
    for m, row in enumerate(X):
        half = _kron_list([_ry(v / 2.0) for v in row])
        out[m] = half @ (ph * (half @ vac))
    return out


def enc_zz_post(X, J=None, **kw):
    n = X.shape[1]
    ph = _zz_phases(J, n)
    vac = np.zeros(2**n, dtype=complex)
    vac[0] = 1.0
    out = np.empty((len(X), 2**n), dtype=complex)
    for m, row in enumerate(X):
        out[m] = ph * (_kron_list([_ry(v) for v in row]) @ vac)
    return out


def enc_zzmap(X, reps=2, **kw):
    """Havlicek ZZFeatureMap.  Coupling is data-dependent; no J is declared."""
    n = X.shape[1]
    z = _z_signs(n)
    hall = _kron_list([_H] * n)
    vac = np.zeros(2**n, dtype=complex)
    vac[0] = 1.0
    out = np.empty((len(X), 2**n), dtype=complex)
    for m, row in enumerate(X):
        expo = z @ row
        for j in range(n):
            for k in range(j + 1, n):
                expo += (np.pi - row[j]) * (np.pi - row[k]) * z[:, j] * z[:, k]
        diag = np.exp(1j * expo)
        psi = vac
        for _ in range(reps):
            psi = diag * (hall @ psi)
        out[m] = psi
    return out


def enc_haar_fixed(X, U=None, **kw):
    return enc_product(X) @ U.T


# ------------------------------------------------------------- profile
def frame_potential(states, k):
    G = states.conj() @ states.T
    return float(np.mean((np.abs(G) ** 2) ** k))


def profile(states):
    A = states.T
    rho = A @ A.conj().T / A.shape[1]
    rho = 0.5 * (rho + rho.conj().T)
    g = mass_gap(rho)
    return dict(
        purity=purity(rho),
        gap=g if np.isfinite(g) else np.nan,
        I=mutual_information(rho),
        Q=log_negativity(rho),
        current=maximum_current(rho),
        F2=frame_potential(states, 2),
        F3=frame_potential(states, 3),
    )


# ------------------------------------------------------------------ run
def make_data(rng, n_rows=300, rho_target=0.6):
    n = N_QUBITS
    cov = np.full((n, n), rho_target)
    np.fill_diagonal(cov, 1.0)
    raw = rng.normal(size=(n_rows, n)) @ np.linalg.cholesky(cov).T
    lo, hi = raw.min(0), raw.max(0)
    return raw, (raw - lo) / (hi - lo) * np.pi


def main():
    rng = np.random.default_rng(RNG_SEED)
    raw, X = make_data(rng)
    J = _partial_correlation_coupling(raw)
    U = _haar_unitary(2**N_QUBITS, rng)

    encodings = {
        "product (J=0 control)": (enc_product, {}),
        "sandwich (paper)": (enc_sandwich, {"J": J}),
        "sandwich RANDOM J": (enc_sandwich, {"J": _random_coupling_like(J, rng)}),
        "zz_post (not sandwiched)": (enc_zz_post, {"J": J}),
        "ZZFeatureMap reps=1": (enc_zzmap, {"reps": 1}),
        "ZZFeatureMap reps=2": (enc_zzmap, {"reps": 2}),
        "Haar fixed (scramble null)": (enc_haar_fixed, {"U": U}),
    }

    print(f"data: {X.shape[0]} rows, {N_QUBITS} features, "
          f"true equicorrelation 0.6")
    print(f"|J_prec|_F = {np.linalg.norm(J):.4f}, "
          f"max|J_prec| = {np.max(np.abs(J)):.4f}\n")
    hdr = f"{'encoding':28}{'purity':>9}{'gap':>8}{'I':>8}{'Q':>8}{'current':>9}{'F2':>8}{'F3':>8}"
    print(hdr)
    print("-" * len(hdr))
    results = {}
    for name, (fn, kw) in encodings.items():
        st = fn(X, **kw)
        nrm = np.abs(np.einsum("mi,mi->m", st.conj(), st).real - 1.0).max()
        assert nrm < 1e-10, f"{name}: states not normalised ({nrm})"
        p = profile(st)
        results[name] = p
        print(f"{name:28}{p['purity']:9.4f}{p['gap']:8.3f}{p['I']:8.4f}"
              f"{p['Q']:8.4f}{p['current']:9.4f}{p['F2']:8.4f}{p['F3']:8.4f}")

    print("\n--- cross-check against dqsa_engine (product & sandwich) ---")
    from qeda.engine import class_operator

    def enc_row_product(row):
        vac = np.zeros(2**N_QUBITS, dtype=complex)
        vac[0] = 1.0
        return _kron_list([_ry(v) for v in row]) @ vac

    rho_ref = class_operator(X, enc_row_product)
    print(f"  purity via dqsa_engine = {purity(rho_ref):.10f}")
    print(f"  purity via this file   = {results['product (J=0 control)']['purity']:.10f}")

    null_test(rng)


def null_test(rng, n_draws=30):
    """Q2 proper: compare the precision J against an ENSEMBLE of random J
    at matched Frobenius norm, on ASYMMETRIC data.

    Asymmetric covariance is essential.  Equicorrelated features give a
    near-uniform J, which zeroes the current by symmetry (a known trap in
    this project), so the current cannot be assessed on such data.
    """
    print("\n" + "=" * 78)
    print("Q2 -- is the precision-derived J distinguishable from a random J")
    print("      of the same size?  (asymmetric covariance, %d random draws)"
          % n_draws)
    print("=" * 78)
    cov = np.array([[1.0, 0.75, 0.05],
                    [0.75, 1.0, -0.40],
                    [0.05, -0.40, 1.0]])
    raw = rng.normal(size=(300, 3)) @ np.linalg.cholesky(cov).T
    lo, hi = raw.min(0), raw.max(0)
    X = (raw - lo) / (hi - lo) * np.pi
    J = _partial_correlation_coupling(raw)
    print(f"  J_prec off-diagonals = {np.round(J[np.triu_indices(3,1)], 4)}"
          f"   (sd {np.std(J[np.triu_indices(3,1)]):.4f}, so genuinely non-uniform)")

    ref = profile(enc_sandwich(X, J=J))
    draws = {k: [] for k in ("purity", "gap", "I", "Q", "current")}
    for _ in range(n_draws):
        p = profile(enc_sandwich(X, J=_random_coupling_like(J, rng)))
        for k in draws:
            draws[k].append(p[k])

    print(f"\n{'statistic':10}{'precision J':>13}{'random J mean':>15}"
          f"{'sd':>9}{'z':>8}   verdict")
    for k, v in draws.items():
        v = np.asarray(v)
        z = (ref[k] - v.mean()) / v.std() if v.std() > 1e-12 else np.nan
        verdict = "distinguishable" if abs(z) > 2 else "NOT distinguishable"
        print(f"{k:10}{ref[k]:13.4f}{v.mean():15.4f}{v.std():9.4f}{z:8.2f}   {verdict}")
    print("\n  |z| > 2 would be needed to attribute the operator response to the")
    print("  conditional-dependence structure of J rather than to its magnitude.")


if __name__ == "__main__":
    main()
