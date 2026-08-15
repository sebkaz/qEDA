"""
sandwich_check.py  --  what the L=1 sandwich does and does not do.

Two questions that must not be confused:

  Q1.  Does J != 0 change the encoded state / kernel / correlations
       relative to J = 0?
  Q2.  Is the Berry curvature (imaginary part of the quantum geometric
       tensor) zero at depth L = 1?

The claim under test is that the answer to Q1 is YES and to Q2 is also
YES -- i.e. the coupling produces genuinely complex amplitudes and real
entanglement, while the phase it writes carries no holonomy at depth 1.
Complex amplitudes are necessary but not sufficient for curvature.

Encoding (one sandwich block):
    U(x, J) = Uenc(x/2) . Uzz(J) . Uenc(x/2),      Uenc(y) = ⊗_j Ry(y_j)
    Uzz(J)  = exp(-i mu sum_{j<k} J_jk Z_j Z_k)

Depth L repeats the block L times.

Pure numpy, no quantum SDK. Run:  python3 sandwich_check.py
"""

import numpy as np
import itertools

np.set_printoptions(precision=6, suppress=True)


# ----------------------------------------------------------------------
# circuit
# ----------------------------------------------------------------------
def ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def kron_all(ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def u_enc(y):
    """Product layer ⊗_j Ry(y_j).  Real orthogonal."""
    return kron_all([ry(yj) for yj in y])


def zz_phases(J, n, mu=1.0):
    """Diagonal of exp(-i mu sum_{j<k} J_jk Z_j Z_k)."""
    d = 2 ** n
    ph = np.zeros(d)
    for b in range(d):
        z = [1 - 2 * ((b >> (n - 1 - q)) & 1) for q in range(n)]
        ph[b] = sum(J[j, k] * z[j] * z[k]
                    for j in range(n) for k in range(j + 1, n))
    return np.exp(-1j * mu * ph)


def sandwich_state(x, J, L=1, mu=1.0):
    """|psi(x,J)> for L stacked sandwich blocks."""
    n = len(x)
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    D = zz_phases(J, n, mu)
    half = u_enc(np.asarray(x) / (2 * L))
    for _ in range(L):
        psi = half @ psi
        psi = D * psi
        psi = half @ psi
    return psi


# ----------------------------------------------------------------------
# geometry: quantum geometric tensor by central differences
#   Q_uv = <d_u psi|d_v psi> - <d_u psi|psi><psi|d_v psi>
#   Re Q = Fubini-Study metric      Im Q = -(1/2) Berry curvature
# ----------------------------------------------------------------------
def qgt(x, J, L=1, mu=1.0, h=1e-5):
    x = np.asarray(x, dtype=float)
    n = len(x)
    psi = sandwich_state(x, J, L, mu)
    d = np.empty((n, len(psi)), dtype=complex)
    for m in range(n):
        xp, xm = x.copy(), x.copy()
        xp[m] += h
        xm[m] -= h
        d[m] = (sandwich_state(xp, J, L, mu) - sandwich_state(xm, J, L, mu)) / (2 * h)
    Q = np.empty((n, n), dtype=complex)
    for a in range(n):
        for b in range(n):
            Q[a, b] = np.vdot(d[a], d[b]) - np.vdot(d[a], psi) * np.vdot(psi, d[b])
    return Q


def berry_curvature(x, J, L=1, mu=1.0, h=1e-5):
    return -2.0 * np.imag(qgt(x, J, L, mu, h))


def berry_phase_loop(J, L=1, mu=1.0, n=2, radius=0.4, centre=None, steps=400):
    """
    Gauge-invariant discrete Berry phase around a closed loop in the
    (x_0, x_1) plane: -arg prod_k <psi_k|psi_{k+1}>.
    Any non-zero value is genuine holonomy.
    """
    if centre is None:
        centre = np.full(n, 1.0)
    pts = []
    for k in range(steps):
        t = 2 * np.pi * k / steps
        x = np.array(centre, dtype=float)
        x[0] += radius * np.cos(t)
        x[1] += radius * np.sin(t)
        pts.append(sandwich_state(x, J, L, mu))
    prod = 1.0 + 0j
    for k in range(steps):
        prod *= np.vdot(pts[k], pts[(k + 1) % steps])
    return -np.angle(prod)


# ----------------------------------------------------------------------
# diagnostics: does J change anything at all?
# ----------------------------------------------------------------------
def partial_trace_keep(rho, keep, n):
    rho = rho.reshape([2] * n + [2] * n)
    drop = [q for q in range(n) if q not in keep]
    for q in sorted(drop, reverse=True):
        rho = np.trace(rho, axis1=q, axis2=q + rho.ndim // 2)
        n -= 1
    d = 2 ** len(keep)
    return rho.reshape(d, d)


def von_neumann(rho, eps=1e-9):
    rho = (1 - eps) * rho + eps * np.eye(len(rho)) / len(rho)
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-15]
    return float(-np.sum(w * np.log(w)))


def mutual_information(rho, n):
    """Mean over single-qubit-vs-rest cuts."""
    S_tot = von_neumann(rho)
    vals = []
    for k in range(n):
        rk = partial_trace_keep(rho, [k], n)
        rr = partial_trace_keep(rho, [q for q in range(n) if q != k], n)
        vals.append(von_neumann(rk) + von_neumann(rr) - S_tot)
    return float(np.mean(vals))


def log_negativity(rho, n):
    """Mean log2||rho^{T_k}||_1 over single-qubit cuts."""
    vals = []
    for k in range(n):
        t = rho.reshape([2] * n + [2] * n)
        ax = list(range(2 * n))
        ax[k], ax[k + n] = ax[k + n], ax[k]
        t = np.transpose(t, ax).reshape(2 ** n, 2 ** n)
        s = np.linalg.svd(t, compute_uv=False)
        vals.append(np.log2(np.sum(s)))
    return float(np.mean(vals))


def class_rho(X, J, L=1, mu=1.0):
    n = X.shape[1]
    rho = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for x in X:
        p = sandwich_state(x, J, L, mu)
        rho += np.outer(p, p.conj())
    return rho / len(X)


# ----------------------------------------------------------------------
def main():
    rng = np.random.default_rng(7)

    print("=" * 70)
    print("Q1.  DOES J CHANGE THE STATE?   (expected answer: YES)")
    print("=" * 70)
    for n in (2, 3, 4):
        J = np.triu(rng.normal(size=(n, n)), 1)
        J = J + J.T
        x = rng.uniform(0, np.pi, n)
        p0 = sandwich_state(x, np.zeros((n, n)))
        pJ = sandwich_state(x, J)
        print(f"\nn={n}")
        print(f"  |<psi(J=0)|psi(J)>|^2            = {abs(np.vdot(p0,pJ))**2:.6f}")
        print(f"  max |Im amplitude|, J=0          = {np.abs(p0.imag).max():.3e}"
              "   <- real: this is the RAE regime")
        print(f"  max |Im amplitude|, J!=0         = {np.abs(pJ.imag).max():.3e}"
              "   <- complex: RAE theorem no longer applies")

        X = rng.uniform(0, np.pi, size=(60, n))
        r0, rJ = class_rho(X, np.zeros((n, n))), class_rho(X, J)
        print(f"  purity   J=0 / J*                = "
              f"{np.trace(r0@r0).real:.4f} / {np.trace(rJ@rJ).real:.4f}")
        print(f"  I        J=0 / J*                = "
              f"{mutual_information(r0,n):.4f} / {mutual_information(rJ,n):.4f}")
        print(f"  Q (logneg) J=0 / J*              = "
              f"{log_negativity(r0,n):.2e} / {log_negativity(rJ,n):.4f}"
              "   <- Q=0 at J=0 exactly")

    print()
    print("=" * 70)
    print("Q2.  IS THE BERRY CURVATURE ZERO AT L=1?")
    print("=" * 70)
    for n in (2, 3, 4):
        J = np.triu(rng.normal(size=(n, n)), 1)
        J = J + J.T
        x = rng.uniform(0.3, np.pi - 0.3, n)
        for L in (1, 2):
            F = berry_curvature(x, J, L=L)
            g = np.real(qgt(x, J, L=L))
            print(f"  n={n}  L={L}   ||F||_max = {np.abs(F).max():.3e}"
                  f"     ||Re Q||_max = {np.abs(g).max():.3e}")
        print()

    print("=" * 70)
    print("Q2b.  HOLONOMY: gauge-invariant Berry phase around a loop")
    print("=" * 70)
    for n in (2, 3, 4):
        J = np.triu(rng.normal(size=(n, n)), 1)
        J = J + J.T
        for L in (1, 2):
            for r in (0.2, 0.5):
                ph = berry_phase_loop(J, L=L, n=n, radius=r)
                print(f"  n={n}  L={L}  radius={r}   Berry phase = {ph:+.3e}")
        print()


if __name__ == "__main__":
    main()
