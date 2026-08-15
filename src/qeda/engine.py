"""Encoding-conditional operator qEDA.

The module implements the instrument defined in Sections 4--6 of the
paper.  A dataset and a declared encoding produce one empirical density
operator.  Spectral, subsystem, and mode-coherence statistics are read
from that same operator.  No predictive model is fitted here.

The numerical core depends only on NumPy.  PennyLane is imported lazily
by the illustrative encodings at the bottom of the file.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np


ComplexArray = np.ndarray
RealArray = np.ndarray
Encoding = Callable[[RealArray], ComplexArray]

_PAULI: dict[str, ComplexArray] = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def n_qubits(rho: ComplexArray) -> int:
    """Return the qubit count associated with a square density matrix."""
    matrix = np.asarray(rho)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("rho must be a square matrix")
    n = int(round(np.log2(matrix.shape[0])))
    if 2**n != matrix.shape[0]:
        raise ValueError("rho dimension must be a power of two")
    return n


def encode_rows(
    data: RealArray,
    encoding: Encoding,
    *,
    tolerance: float = 1e-9,
) -> ComplexArray:
    """Encode rows and return the state matrix ``A``.

    Parameters
    ----------
    data
        Nonempty array with shape ``(n_samples, n_features)``.
    encoding
        Callable returning a normalized statevector for one row.
    tolerance
        Absolute tolerance used to validate state normalization.

    Returns
    -------
    numpy.ndarray
        Matrix whose columns are the encoded statevectors.
    """
    rows = np.asarray(data, dtype=float)
    if rows.ndim != 2 or rows.shape[0] == 0:
        raise ValueError("data must be a nonempty two-dimensional array")

    states: list[ComplexArray] = []
    dimension: int | None = None
    for row in rows:
        state = np.asarray(encoding(row), dtype=complex).reshape(-1)
        if dimension is None:
            dimension = state.size
            if dimension == 0:
                raise ValueError("encoding returned an empty statevector")
            qubits = int(round(np.log2(dimension)))
            if 2**qubits != dimension:
                raise ValueError("statevector dimension must be a power of two")
        elif state.size != dimension:
            raise ValueError("encoding returned inconsistent state dimensions")

        norm = float(np.vdot(state, state).real)
        if not np.isclose(norm, 1.0, atol=tolerance, rtol=0.0):
            raise ValueError(f"encoding returned a state with norm squared {norm}")
        states.append(state)

    return np.column_stack(states)


def class_operator_from_states(states: ComplexArray) -> ComplexArray:
    """Compute the empirical operator ``rho = A A^dagger / M``."""
    matrix = np.asarray(states, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("states must be a nonempty state matrix")
    rho = matrix @ matrix.conj().T / matrix.shape[1]
    return (rho + rho.conj().T) / 2.0


def class_operator(data: RealArray, encoding: Encoding) -> ComplexArray:
    """Construct ``rho_c = M^-1 sum_m |psi_m><psi_m|``."""
    return class_operator_from_states(encode_rows(data, encoding))


def gram_matrix(states: ComplexArray) -> ComplexArray:
    """Compute the complex Gram matrix ``G = A^dagger A``."""
    matrix = np.asarray(states, dtype=complex)
    if matrix.ndim != 2:
        raise ValueError("states must be a two-dimensional matrix")
    return matrix.conj().T @ matrix


def fidelity_kernel(states: ComplexArray) -> RealArray:
    """Compute the fidelity kernel ``K_mn = |G_mn|^2``."""
    return np.abs(gram_matrix(states)) ** 2


def density_eigenvalues(
    rho: ComplexArray,
    *,
    tolerance: float = 1e-10,
) -> RealArray:
    """Return descending eigenvalues after round-off validation.

    Negative eigenvalues inside ``tolerance`` are clipped and the result
    is normalized.  A more negative eigenvalue raises ``ValueError``.
    """
    matrix = np.asarray(rho, dtype=complex)
    n_qubits(matrix)
    hermitian = (matrix + matrix.conj().T) / 2.0
    eigenvalues = np.linalg.eigvalsh(hermitian).real
    if eigenvalues[0] < -tolerance:
        raise ValueError("rho is not positive semidefinite")
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    trace = float(eigenvalues.sum())
    if trace <= tolerance:
        raise ValueError("rho has zero trace")
    eigenvalues /= trace
    return eigenvalues[::-1]


def purity(rho: ComplexArray) -> float:
    """Compute field purity ``gamma = Tr(rho^2)``."""
    matrix = np.asarray(rho, dtype=complex)
    return float(np.trace(matrix @ matrix).real)


def dispersion(rho: ComplexArray) -> float:
    """Compute linear entropy ``S_L = 1 - Tr(rho^2)``."""
    return 1.0 - purity(rho)


def participation_rank(rho: ComplexArray) -> float:
    """Compute the spectral participation rank ``1 / Tr(rho^2)``."""
    return 1.0 / purity(rho)


def mass_gap(rho: ComplexArray, *, tolerance: float = 1e-12) -> float:
    """Compute ``log(lambda_0 / lambda_1)`` for the nonzero spectrum."""
    eigenvalues = density_eigenvalues(rho)
    nonzero = eigenvalues[eigenvalues > tolerance]
    if nonzero.size < 2:
        return float("inf")
    return float(np.log(nonzero[0] / nonzero[1]))


def third_moment(rho: ComplexArray) -> float:
    """Compute the phase-sensitive moment ``Tr(rho^3)``."""
    matrix = np.asarray(rho, dtype=complex)
    return float(np.trace(matrix @ matrix @ matrix).real)


def spectrum(rho: ComplexArray, k: int | None = None) -> RealArray:
    """Return the full or leading density-operator spectrum."""
    eigenvalues = density_eigenvalues(rho)
    if k is None:
        return eigenvalues
    if k < 1:
        raise ValueError("k must be positive")
    return eigenvalues[:k]


def von_neumann_entropy(
    rho: ComplexArray,
    *,
    base: float = 2.0,
    tolerance: float = 1e-12,
) -> float:
    """Compute ``-Tr(rho log_base rho)`` with ``0 log 0 = 0``.

    No depolarizing regularizer is applied.  Eigenvalues below the
    numerical tolerance do not contribute to the entropy.
    """
    eigenvalues = density_eigenvalues(rho)
    support = eigenvalues[eigenvalues > tolerance]
    logarithm = np.log(support) / np.log(base)
    return float(-np.sum(support * logarithm))


def reduced_density_matrix(rho: ComplexArray, keep: list[int]) -> ComplexArray:
    """Trace out all qubits not listed in ``keep``."""
    matrix = np.asarray(rho, dtype=complex)
    n = n_qubits(matrix)
    kept = list(dict.fromkeys(keep))
    if any(index < 0 or index >= n for index in kept):
        raise ValueError("keep contains an invalid qubit index")
    traced = [index for index in range(n) if index not in kept]
    permutation = (
        kept
        + traced
        + [index + n for index in kept]
        + [index + n for index in traced]
    )
    kept_dim = 2 ** len(kept)
    traced_dim = 2 ** len(traced)
    tensor = matrix.reshape([2] * (2 * n)).transpose(permutation)
    tensor = tensor.reshape(kept_dim, traced_dim, kept_dim, traced_dim)
    reduced = np.einsum("aibi->ab", tensor)
    return (reduced + reduced.conj().T) / 2.0


def mode_mutual_information(rho: ComplexArray) -> RealArray:
    """Return mutual information for every single-mode/rest cut."""
    n = n_qubits(rho)
    if n < 2:
        return np.zeros(n, dtype=float)
    total_entropy = von_neumann_entropy(rho)
    values = np.empty(n, dtype=float)
    for mode in range(n):
        complement = [index for index in range(n) if index != mode]
        value = (
            von_neumann_entropy(reduced_density_matrix(rho, [mode]))
            + von_neumann_entropy(reduced_density_matrix(rho, complement))
            - total_entropy
        )
        values[mode] = max(0.0, value)
    return values


def mutual_information(rho: ComplexArray) -> float:
    """Return mean single-mode/rest mutual information in bits."""
    values = mode_mutual_information(rho)
    return float(values.mean()) if values.size else 0.0


def partial_transpose(rho: ComplexArray, mode: int) -> ComplexArray:
    """Partially transpose ``rho`` on one qubit."""
    matrix = np.asarray(rho, dtype=complex)
    n = n_qubits(matrix)
    if mode < 0 or mode >= n:
        raise ValueError("mode is outside the register")
    tensor = matrix.reshape([2] * (2 * n))
    axes = list(range(2 * n))
    axes[mode], axes[mode + n] = axes[mode + n], axes[mode]
    return tensor.transpose(axes).reshape(matrix.shape)


def mode_log_negativity(rho: ComplexArray) -> RealArray:
    """Return logarithmic negativity for each single-mode/rest cut."""
    n = n_qubits(rho)
    if n < 2:
        return np.zeros(n, dtype=float)
    values = np.empty(n, dtype=float)
    for mode in range(n):
        singular_values = np.linalg.svd(
            partial_transpose(rho, mode),
            compute_uv=False,
        )
        values[mode] = max(0.0, float(np.log2(singular_values.sum())))
    return values


def log_negativity(rho: ComplexArray) -> float:
    """Return mean logarithmic negativity over single-mode cuts."""
    values = mode_log_negativity(rho)
    return float(values.mean()) if values.size else 0.0


def _pauli_expectation(
    rho: ComplexArray,
    operators: Mapping[int, str],
) -> complex:
    """Evaluate a tensor Pauli observable on ``rho``."""
    n = n_qubits(rho)
    operator = np.array([[1.0 + 0.0j]])
    for mode in range(n):
        label = operators.get(mode, "I")
        if label not in _PAULI:
            raise ValueError(f"unknown Pauli label: {label}")
        operator = np.kron(operator, _PAULI[label])
    return complex(np.trace(np.asarray(rho) @ operator))


def mode_coherence_matrix(rho: ComplexArray) -> ComplexArray:
    """Compute ``Gamma_jk = <sigma_j^+ sigma_k^->``.

    The local hard-core convention is
    ``sigma^- = (X + iY) / 2``.  No Jordan--Wigner strings are used.
    """
    n = n_qubits(rho)
    gamma = np.zeros((n, n), dtype=complex)
    for j in range(n):
        z = float(_pauli_expectation(rho, {j: "Z"}).real)
        gamma[j, j] = 0.5 * (1.0 - z)

    for j in range(n):
        for k in range(j + 1, n):
            xx = float(_pauli_expectation(rho, {j: "X", k: "X"}).real)
            yy = float(_pauli_expectation(rho, {j: "Y", k: "Y"}).real)
            xy = float(_pauli_expectation(rho, {j: "X", k: "Y"}).real)
            yx = float(_pauli_expectation(rho, {j: "Y", k: "X"}).real)
            value = 0.25 * ((xx + yy) + 1j * (xy - yx))
            gamma[j, k] = value
            gamma[k, j] = value.conjugate()
    return gamma


def one_particle_matrix(rho: ComplexArray) -> ComplexArray:
    """Compatibility alias for :func:`mode_coherence_matrix`."""
    return mode_coherence_matrix(rho)


def encoded_current(rho: ComplexArray) -> RealArray:
    """Return ``J = Im(Gamma)``, a real antisymmetric current matrix."""
    return np.asarray(mode_coherence_matrix(rho).imag, dtype=float)


def maximum_current(rho: ComplexArray) -> float:
    """Return the largest absolute encoded-current component."""
    current = encoded_current(rho)
    return float(np.max(np.abs(current))) if current.size else 0.0


def occupations(rho: ComplexArray, *, tolerance: float = 1e-10) -> RealArray:
    """Return descending eigenvalues of the mode-coherence matrix."""
    values = np.linalg.eigvalsh(mode_coherence_matrix(rho)).real
    if values[0] < -tolerance:
        raise ValueError("mode-coherence matrix is not positive semidefinite")
    return np.clip(values, 0.0, None)[::-1]


def mode_participation(rho: ComplexArray, *, tolerance: float = 1e-12) -> float:
    """Compute ``Tr(Gamma)^2 / Tr(Gamma^2)``."""
    gamma = mode_coherence_matrix(rho)
    trace = float(np.trace(gamma).real)
    denominator = float(np.trace(gamma @ gamma).real)
    if trace <= tolerance or denominator <= tolerance:
        return float("nan")
    return trace**2 / denominator


def leading_subspace_projector(rho: ComplexArray, k: int) -> ComplexArray:
    """Return the projector onto the leading ``k`` eigenvectors."""
    matrix = np.asarray(rho, dtype=complex)
    if k < 1 or k > matrix.shape[0]:
        raise ValueError("k must lie between 1 and the Hilbert dimension")
    _, eigenvectors = np.linalg.eigh((matrix + matrix.conj().T) / 2.0)
    leading = eigenvectors[:, -k:]
    return leading @ leading.conj().T


def leading_subspace_score(
    state: ComplexArray,
    projector: ComplexArray,
) -> float:
    """Compute the Born weight of ``state`` in ``projector``."""
    vector = np.asarray(state, dtype=complex).reshape(-1)
    value = np.vdot(vector, np.asarray(projector) @ vector).real
    return float(np.clip(value, 0.0, 1.0))


def profile_from_density(rho: ComplexArray) -> dict[str, Any]:
    """Compute the complete qEDA profile of one empirical operator."""
    gamma = mode_coherence_matrix(rho)
    current = gamma.imag
    return {
        "purity": purity(rho),
        "dispersion": dispersion(rho),
        "effective_rank": participation_rank(rho),
        "mass_gap": mass_gap(rho),
        "third_moment": third_moment(rho),
        "entropy_bits": von_neumann_entropy(rho),
        "I": mutual_information(rho),
        "I_by_mode": mode_mutual_information(rho),
        "Q": log_negativity(rho),
        "Q_by_mode": mode_log_negativity(rho),
        "Gamma": gamma,
        "current_matrix": current,
        "current": float(np.max(np.abs(current))) if current.size else 0.0,
        "occupations": occupations(rho),
        "mode_participation": mode_participation(rho),
        "spectrum": spectrum(rho),
    }


def profile(data: RealArray, encoding: Encoding) -> dict[str, Any]:
    """Encode ``data`` and compute its operator-qEDA profile."""
    return profile_from_density(class_operator(data, encoding))


def matched_profile(
    data: RealArray,
    control_encoding: Encoding,
    coupled_encoding: Encoding,
) -> dict[str, Any]:
    """Compute matched profiles and signed coupled-minus-control changes."""
    control = profile(data, control_encoding)
    coupled = profile(data, coupled_encoding)
    comparable = (
        "purity",
        "dispersion",
        "effective_rank",
        "mass_gap",
        "third_moment",
        "entropy_bits",
        "I",
        "I_by_mode",
        "Q",
        "Q_by_mode",
        "Gamma",
        "current_matrix",
        "current",
        "occupations",
        "mode_participation",
        "spectrum",
    )
    delta = {
        key: np.asarray(coupled[key]) - np.asarray(control[key])
        for key in comparable
    }
    return {"control": control, "coupled": coupled, "delta": delta}


def _anscombe() -> dict[str, tuple[RealArray, RealArray]]:
    """Return Anscombe's four datasets."""
    x = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], dtype=float)
    y1 = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])
    y2 = np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74])
    y3 = np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73])
    x4 = np.array([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], dtype=float)
    y4 = np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89])
    return {"I": (x, y1), "II": (x, y2), "III": (x, y3), "IV": (x4, y4)}


def _encodings(coupling: float) -> dict[str, Encoding]:
    """Create the five illustrative PennyLane encodings used in the figure."""
    import pennylane as qml

    one_qubit = qml.device("default.qubit", wires=1)
    two_qubit = qml.device("default.qubit", wires=2)

    @qml.qnode(one_qubit)
    def amplitude(x: RealArray) -> ComplexArray:
        vector = np.asarray([x[0], x[1]], dtype=float)
        norm = np.linalg.norm(vector)
        vector = vector / norm if norm > 1e-12 else np.array([1.0, 0.0])
        qml.MottonenStatePreparation(vector, wires=0)
        return qml.state()

    @qml.qnode(two_qubit)
    def angle(x: RealArray) -> ComplexArray:
        qml.RY(x[0], wires=0)
        qml.RY(x[1], wires=1)
        return qml.state()

    @qml.qnode(two_qubit)
    def angle_cnot(x: RealArray) -> ComplexArray:
        qml.RY(x[0], wires=0)
        qml.RY(x[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.state()

    @qml.qnode(two_qubit)
    def sandwich(x: RealArray) -> ComplexArray:
        qml.RY(x[0] / 2.0, wires=0)
        qml.RY(x[1] / 2.0, wires=1)
        qml.IsingZZ(2.0 * coupling, wires=[0, 1])
        qml.RY(x[0] / 2.0, wires=0)
        qml.RY(x[1] / 2.0, wires=1)
        return qml.state()

    @qml.qnode(two_qubit)
    def reupload_fixed(x: RealArray) -> ComplexArray:
        for _ in range(3):
            qml.RY(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.IsingZZ(2.0 * coupling, wires=[0, 1])
        return qml.state()

    return {
        "amplitude": amplitude,
        "angle": angle,
        "angle+CNOT": angle_cnot,
        "sandwich": sandwich,
        "reupload-fixed": reupload_fixed,
    }


def main() -> None:
    """Print an encoding-conditional profile of Anscombe's quartet."""
    datasets = _anscombe()
    all_x = np.concatenate([values[0] for values in datasets.values()])
    all_y = np.concatenate([values[1] for values in datasets.values()])

    def scale(x: RealArray, y: RealArray) -> RealArray:
        x_scaled = (x - all_x.min()) / (all_x.max() - all_x.min()) * np.pi
        y_scaled = (y - all_y.min()) / (all_y.max() - all_y.min()) * np.pi
        return np.stack([x_scaled, y_scaled], axis=1)

    for name, encoding in _encodings(coupling=0.6).items():
        print(f"\n=== encoding: {name} ===")
        print(f"{'set':>4} {'purity':>8} {'gap':>8} {'Tr3':>8} {'I':>8} {'Q':>8} {'Jmax':>8}")
        for label, (x, y) in datasets.items():
            result = profile(scale(x, y), encoding)
            print(
                f"{label:>4} {result['purity']:8.4f} "
                f"{result['mass_gap']:8.3f} {result['third_moment']:8.4f} "
                f"{result['I']:8.4f} {result['Q']:8.4f} "
                f"{result['current']:8.4f}"
            )


if __name__ == "__main__":
    main()
