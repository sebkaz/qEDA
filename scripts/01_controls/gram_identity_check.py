import numpy as np
from qeda.sandwich import sandwich_state, class_rho
np.set_printoptions(precision=6, suppress=True)
rng = np.random.default_rng(3)

n, M = 3, 12
X = rng.uniform(0, np.pi, size=(M, n))
J = np.triu(rng.uniform(0.2,1.0,size=(n,n)),1); J = J+J.T

for label, Ju in [("J = 0", np.zeros((n,n))), ("J != 0", J)]:
    P = np.array([sandwich_state(x, Ju) for x in X])       # M x 2^n
    rho = class_rho(X, Ju)
    G   = P.conj() @ P.T                                    # COMPLEX Gram <psi_m|psi_n>
    K   = np.abs(G)**2                                      # the quantum KERNEL

    sr = np.sort(np.linalg.eigvalsh(rho).real)[::-1][:5]
    sg = np.sort(np.linalg.eigvalsh(G/M).real)[::-1][:5]

    print(f"--- {label} ---")
    print("  spec(rho_c)      ", sr)
    print("  spec(Gram/M)     ", sg)
    print(f"  max |difference| = {np.abs(sr-sg).max():.2e}   <- identical spectra")
    print(f"  is Gram real?      max|Im G| = {np.abs(G.imag).max():.3e}")
    # Bargmann invariant: gauge-invariant, third order, INVISIBLE to K=|G|^2
    b = G[0,1]*G[1,2]*G[2,0]
    print(f"  Bargmann Delta_3 = {b:.5f}   arg = {np.angle(b):+.5f}")
    print()

print("KEY POINT: spec(rho_c) == spec(Gram/M).  The spectrum is NOT lost by")
print("going pairwise -- it is lost by taking the MODULUS.  The kernel keeps")
print("|G_mn| and discards arg G_mn, and the phase is exactly where the")
print("Bargmann invariant and closed-loop geometric phase live.")
