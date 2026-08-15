"""Independent algebraic checks for the operator-qEDA implementation.

The checks use explicit NumPy matrices and therefore run without a
quantum framework.  When PennyLane is available, the final check also
compares the matrix sandwich with a PennyLane QNode.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from qeda.engine import (
    class_operator,
    encoded_current,
    encode_rows,
    fidelity_kernel,
    gram_matrix,
    log_negativity,
    mode_coherence_matrix,
    mode_mutual_information,
    mutual_information,
    occupations,
    purity,
    third_moment,
)


ComplexArray = np.ndarray
RealArray = np.ndarray
Encoding = Callable[[RealArray], ComplexArray]

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
LOWER = (X + 1j * Y) / 2.0  # sigma^- = |0><1|
RAISE = LOWER.conj().T


def ry(angle: float) -> ComplexArray:
    """Return the single-qubit ``R_Y(angle)`` matrix."""
    cosine = np.cos(angle / 2.0)
    sine = np.sin(angle / 2.0)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=complex)


def tensor_product(operators: list[ComplexArray]) -> ComplexArray:
    """Return an ordered Kronecker product."""
    result = np.array([[1.0 + 0.0j]])
    for operator in operators:
        result = np.kron(result, operator)
    return result


def product_ry_encoding(n_qubits: int) -> Encoding:
    """Create the real product ``R_Y`` control encoding."""
    vacuum = np.zeros(2**n_qubits, dtype=complex)
    vacuum[0] = 1.0

    def encoding(row: RealArray) -> ComplexArray:
        unitary = tensor_product([ry(float(value)) for value in row])
        return unitary @ vacuum

    return encoding


def sandwich_encoding(coupling: RealArray) -> Encoding:
    """Create ``U_RY(x/2) U_ZZ(J) U_RY(x/2) |0>`` explicitly."""
    matrix = np.asarray(coupling, dtype=float)
    n = matrix.shape[0]
    if matrix.shape != (n, n) or not np.allclose(matrix, matrix.T):
        raise ValueError("coupling must be square and symmetric")
    vacuum = np.zeros(2**n, dtype=complex)
    vacuum[0] = 1.0

    phases = np.empty(2**n, dtype=complex)
    for basis_index in range(2**n):
        bits = [(basis_index >> (n - 1 - mode)) & 1 for mode in range(n)]
        z_values = np.array([1.0 if bit == 0 else -1.0 for bit in bits])
        exponent = sum(
            matrix[j, k] * z_values[j] * z_values[k]
            for j in range(n)
            for k in range(j + 1, n)
        )
        phases[basis_index] = np.exp(-1j * exponent)

    def encoding(row: RealArray) -> ComplexArray:
        half_layer = tensor_product([ry(float(value) / 2.0) for value in row])
        return half_layer @ (phases * (half_layer @ vacuum))

    return encoding


def operator_on_mode(
    operator: ComplexArray,
    mode: int,
    n_qubits: int,
) -> ComplexArray:
    """Embed a single-qubit operator into an ``n_qubits`` register."""
    factors = [I2] * n_qubits
    factors[mode] = operator
    return tensor_product(factors)


def direct_mode_coherence(rho: ComplexArray) -> ComplexArray:
    """Compute ``Gamma_jk`` directly from raising/lowering matrices."""
    n = int(round(np.log2(rho.shape[0])))
    lowering = [operator_on_mode(LOWER, mode, n) for mode in range(n)]
    raising = [operator_on_mode(RAISE, mode, n) for mode in range(n)]
    gamma = np.empty((n, n), dtype=complex)
    for j in range(n):
        for k in range(n):
            gamma[j, k] = np.trace(rho @ raising[j] @ lowering[k])
    return gamma


def precision_coupling(
    data: RealArray,
    *,
    ridge: float = 0.1,
    scale: float = 0.6,
) -> RealArray:
    """Construct the manuscript's partial-correlation ridge coupling."""
    covariance = np.cov(np.asarray(data), rowvar=False)
    precision = np.linalg.inv(covariance + ridge * np.eye(covariance.shape[0]))
    diagonal = np.diag(precision)
    denominator = np.sqrt(np.outer(diagonal, diagonal))
    coupling = -scale * precision / denominator
    np.fill_diagonal(coupling, 0.0)
    return 0.5 * (coupling + coupling.T)


def random_local_unitary(rng: np.random.Generator, n_qubits: int) -> ComplexArray:
    """Construct a fixed tensor product of Haar-like two-dimensional unitaries."""
    factors: list[ComplexArray] = []
    for _ in range(n_qubits):
        matrix = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        q, r = np.linalg.qr(matrix)
        phases = np.diag(r)
        phases = np.where(np.abs(phases) > 0.0, phases / np.abs(phases), 1.0)
        factors.append(q @ np.diag(phases.conj()))
    return tensor_product(factors)


def check_pennylane_sandwich(
    row: RealArray,
    coupling: RealArray,
    matrix_encoding: Encoding,
) -> None:
    """Compare the explicit sandwich with PennyLane when available."""
    try:
        import pennylane as qml
    except ImportError:
        print("PennyLane check: skipped (PennyLane is not installed)")
        return

    n = len(row)
    device = qml.device("default.qubit", wires=n)

    @qml.qnode(device)
    def circuit(values: RealArray) -> ComplexArray:
        for mode in range(n):
            qml.RY(values[mode] / 2.0, wires=mode)
        for j in range(n):
            for k in range(j + 1, n):
                qml.IsingZZ(2.0 * coupling[j, k], wires=[j, k])
        for mode in range(n):
            qml.RY(values[mode] / 2.0, wires=mode)
        return qml.state()

    expected = matrix_encoding(row)
    observed = np.asarray(circuit(row))
    fidelity = abs(np.vdot(expected, observed)) ** 2
    np.testing.assert_allclose(fidelity, 1.0, atol=1e-12)
    print(f"PennyLane sandwich fidelity: {fidelity:.15f}")


def main() -> None:
    """Run the identities required by the paper and implementation."""
    rng = np.random.default_rng(7)
    n = 3
    covariance = np.array(
        [[1.0, 0.55, -0.20], [0.55, 1.0, 0.35], [-0.20, 0.35, 1.0]]
    )
    raw = rng.normal(size=(80, n)) @ np.linalg.cholesky(covariance).T
    minimum = raw.min(axis=0)
    maximum = raw.max(axis=0)
    data = (raw - minimum) / (maximum - minimum) * np.pi

    control_encoding = product_ry_encoding(n)
    coupling = precision_coupling(raw)
    coupled_encoding = sandwich_encoding(coupling)

    states = encode_rows(data, coupled_encoding)
    rho = class_operator(data, coupled_encoding)
    gram = gram_matrix(states)
    kernel = fidelity_kernel(states)
    sample_count = data.shape[0]

    # rho = A A^dagger / M and G/M have the same nonzero spectrum.
    rho_spectrum = np.linalg.eigvalsh(rho)
    rho_spectrum = rho_spectrum[rho_spectrum > 1e-12]
    gram_spectrum = np.linalg.eigvalsh(gram / sample_count)
    gram_spectrum = gram_spectrum[gram_spectrum > 1e-12]
    np.testing.assert_allclose(rho_spectrum, gram_spectrum, atol=1e-11)

    # Purity is kernel-visible; the third moment requires the complex G.
    np.testing.assert_allclose(purity(rho), kernel.sum() / sample_count**2, atol=1e-12)
    np.testing.assert_allclose(
        third_moment(rho),
        np.trace(gram @ gram @ gram).real / sample_count**3,
        atol=1e-12,
    )

    # The Pauli formula for Gamma must equal the direct operator formula.
    gamma_pauli = mode_coherence_matrix(rho)
    gamma_direct = direct_mode_coherence(rho)
    np.testing.assert_allclose(gamma_pauli, gamma_direct, atol=1e-12)
    np.testing.assert_allclose(gamma_pauli, gamma_pauli.conj().T, atol=1e-12)
    if np.min(np.linalg.eigvalsh(gamma_pauli)) < -1e-10:
        raise AssertionError("Gamma is not positive semidefinite")
    np.testing.assert_allclose(encoded_current(rho), gamma_pauli.imag, atol=1e-12)

    # The real RY control is separable and carries no encoded current.
    rho_control = class_operator(data, control_encoding)
    np.testing.assert_allclose(log_negativity(rho_control), 0.0, atol=1e-12)
    np.testing.assert_allclose(encoded_current(rho_control), 0.0, atol=1e-12)

    # Mutual information and negativity are invariant under fixed local unitaries.
    local_unitary = random_local_unitary(rng, n)
    transformed = local_unitary @ rho @ local_unitary.conj().T
    np.testing.assert_allclose(
        mode_mutual_information(transformed),
        mode_mutual_information(rho),
        atol=1e-11,
    )
    np.testing.assert_allclose(
        log_negativity(transformed),
        log_negativity(rho),
        atol=1e-11,
    )

    check_pennylane_sandwich(data[0], coupling, coupled_encoding)

    print("All operator-qEDA checks passed.")
    print(f"purity J=0 / Jc: {purity(rho_control):.6f} / {purity(rho):.6f}")
    print(
        "mutual information J=0 / Jc: "
        f"{mutual_information(rho_control):.6f} / {mutual_information(rho):.6f}"
    )
    print(
        "log-negativity J=0 / Jc: "
        f"{log_negativity(rho_control):.6f} / {log_negativity(rho):.6f}"
    )
    print(
        "max current J=0 / Jc: "
        f"{np.max(np.abs(encoded_current(rho_control))):.6f} / "
        f"{np.max(np.abs(encoded_current(rho))):.6f}"
    )
    print(f"Gamma occupations: {np.array2string(occupations(rho), precision=6)}")


if __name__ == "__main__":
    main()
