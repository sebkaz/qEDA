"""Separate commuting gate generators from operational classical collapse.

The four controls use the same 50 Iris-setosa rows and the same fixed
partial-correlation coupling.  They distinguish: (i) a circuit entirely
diagonal in the preparation/readout basis, (ii) a real product encoding,
(iii) mutually commuting Y and YY generators with incompatible preparation
and readout, and (iv) the RY-ZZ-RY sandwich used in the manuscript.

Run from the repository root:
    PYTHONPATH=src python scripts/01_controls/operational_abelian_controls.py
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

from qeda.engine import (
    class_operator_from_states,
    gram_matrix,
    log_negativity,
    mass_gap,
    maximum_current,
    mutual_information,
    purity,
)

RESULTS = Path("results/data")
N_QUBITS = 4
RIDGE = 1e-3
ALPHA = 0.8
TOLERANCE = 1e-12

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_all(operators: list[np.ndarray]) -> np.ndarray:
    """Return an ordered tensor product."""
    result = np.array([[1.0 + 0.0j]])
    for operator in operators:
        result = np.kron(result, operator)
    return result


def local(operator: np.ndarray, mode: int) -> np.ndarray:
    """Embed a one-qubit operator in the four-qubit register."""
    return kron_all([operator if index == mode else I2 for index in range(N_QUBITS)])


def pair(operator: np.ndarray, left: int, right: int) -> np.ndarray:
    """Embed the same Pauli operator on one unordered pair."""
    return kron_all(
        [operator if index in (left, right) else I2 for index in range(N_QUBITS)]
    )


Y_LOCAL = [local(Y, mode) for mode in range(N_QUBITS)]
Z_LOCAL = [local(Z, mode) for mode in range(N_QUBITS)]
YY = {(left, right): pair(Y, left, right) for left, right in combinations(range(N_QUBITS), 2)}
ZZ = {(left, right): pair(Z, left, right) for left, right in combinations(range(N_QUBITS), 2)}
VACUUM = np.zeros(2**N_QUBITS, dtype=complex)
VACUUM[0] = 1.0
RHO_VACUUM = np.outer(VACUUM, VACUUM.conj())


def rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return ``exp(-i angle axis / 2)`` for a Pauli axis."""
    return np.cos(angle / 2.0) * I2 - 1j * np.sin(angle / 2.0) * axis


def local_rotation(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Return the product of one-qubit rotations about one common axis."""
    return kron_all([rotation(axis, float(angle)) for angle in angles])


def unitary_from_hamiltonian(hamiltonian: np.ndarray) -> np.ndarray:
    """Exponentiate a small Hermitian Hamiltonian without a simulator SDK."""
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    return (eigenvectors * np.exp(-1j * eigenvalues)) @ eigenvectors.conj().T


def coupling_from_data(values: np.ndarray) -> np.ndarray:
    """Return the fixed-scale partial-correlation coupling used in the audit."""
    precision = np.linalg.inv(np.cov(values, rowvar=False) + RIDGE * np.eye(N_QUBITS))
    denominator = np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    coupling = -ALPHA * precision / denominator
    np.fill_diagonal(coupling, 0.0)
    return 0.5 * (coupling + coupling.T)


def interaction(coupling: np.ndarray, terms: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    """Build ``sum_{j<k} J_jk P_j P_k``."""
    return sum(
        coupling[left, right] * terms[left, right]
        for left, right in combinations(range(N_QUBITS), 2)
    )


def normalized_commutator(left: np.ndarray, right: np.ndarray) -> float:
    """Return a scale-free Frobenius commutator norm."""
    denominator = np.linalg.norm(left, "fro") * np.linalg.norm(right, "fro")
    return float(np.linalg.norm(left @ right - right @ left, "fro") / denominator)


def maximum_pair_commutator(operators: list[np.ndarray]) -> float:
    """Return the maximum pairwise normalized commutator."""
    return max(
        (normalized_commutator(left, right) for left, right in combinations(operators, 2)),
        default=0.0,
    )


def state_family_commutator(states: np.ndarray) -> float:
    """Mean commutator norm of distinct encoded pure-state projectors."""
    projectors = [np.outer(state, state.conj()) for state in states.T]
    values = [
        normalized_commutator(left, right)
        for left, right in combinations(projectors, 2)
    ]
    return float(np.mean(values))


def maximum_bargmann_phase(gram: np.ndarray) -> float:
    """Return the largest gauge-invariant triangular overlap phase."""
    phases = [
        abs(np.angle(gram[i, j] * gram[j, k] * gram[k, i]))
        for i, j, k in combinations(range(gram.shape[0]), 3)
    ]
    return float(max(phases, default=0.0))


def markdown_table(frame: pd.DataFrame) -> str:
    """Format a small DataFrame as Markdown without an optional dependency."""
    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    separator = ["---"] * len(headers)
    return "\n".join(
        "| " + " | ".join(row) + " |"
        for row in [headers, separator, *rows]
    )


def profile(name: str, states: np.ndarray, generators: list[np.ndarray]) -> dict[str, float | str | bool]:
    """Compute algebraic compatibility and qEDA statistics for one control."""
    rho = class_operator_from_states(states)
    gram = gram_matrix(states)
    gate_commutator = maximum_pair_commutator(generators)
    input_commutator = max(
        (normalized_commutator(RHO_VACUUM, generator) for generator in generators),
        default=0.0,
    )
    readout_commutator = max(
        (normalized_commutator(readout, generator) for readout in Z_LOCAL for generator in generators),
        default=0.0,
    )
    return {
        "encoding": name,
        "gate_algebra_abelian": gate_commutator < TOLERANCE,
        "input_compatible": input_commutator < TOLERANCE,
        "Z_readout_compatible": readout_commutator < TOLERANCE,
        "operationally_abelian": (
            gate_commutator < TOLERANCE
            and input_commutator < TOLERANCE
            and readout_commutator < TOLERANCE
        ),
        "max_gate_commutator": gate_commutator,
        "max_input_commutator": input_commutator,
        "max_readout_commutator": readout_commutator,
        "mean_state_projector_commutator": state_family_commutator(states),
        "purity": purity(rho),
        "mass_gap": mass_gap(rho),
        "mutual_information": mutual_information(rho),
        "log_negativity": log_negativity(rho),
        "max_current": maximum_current(rho),
        "rho_imag_frobenius": float(np.linalg.norm(rho.imag, "fro")),
        "max_gram_imag": float(np.max(np.abs(gram.imag))),
        "max_bargmann_phase_rad": maximum_bargmann_phase(gram),
    }


def main() -> None:
    """Run matched controls and write a compact comparison table."""
    iris = load_iris()
    raw = iris.data[iris.target == 0]
    angles = MinMaxScaler(feature_range=(0.0, np.pi), clip=True).fit_transform(raw)
    coupling = coupling_from_data(raw)
    h_yy = interaction(coupling, YY)
    h_zz = interaction(coupling, ZZ)
    u_yy = unitary_from_hamiltonian(h_yy)
    u_zz = unitary_from_hamiltonian(h_zz)

    rz_zz_states = np.column_stack(
        [local_rotation(Z, row / 2.0) @ u_zz @ local_rotation(Z, row / 2.0) @ VACUUM for row in angles]
    )
    ry_product_states = np.column_stack(
        [local_rotation(Y, row) @ VACUUM for row in angles]
    )
    ry_yy_states = np.column_stack(
        [local_rotation(Y, row / 2.0) @ u_yy @ local_rotation(Y, row / 2.0) @ VACUUM for row in angles]
    )
    ry_zz_states = np.column_stack(
        [local_rotation(Y, row / 2.0) @ u_zz @ local_rotation(Y, row / 2.0) @ VACUUM for row in angles]
    )

    z_generators = Z_LOCAL + list(ZZ.values())
    y_generators = Y_LOCAL + list(YY.values())
    sandwich_generators = Y_LOCAL + list(ZZ.values())
    table = pd.DataFrame(
        [
            profile("RZ-ZZ-RZ (computationally diagonal)", rz_zz_states, z_generators),
            profile("RY product control", ry_product_states, Y_LOCAL),
            profile("RY-YY-RY (commuting generators)", ry_yy_states, y_generators),
            profile("RY-ZZ-RY (sandwich)", ry_zz_states, sandwich_generators),
        ]
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / "operational_abelian_controls.csv"
    markdown_path = RESULTS / "operational_abelian_controls.md"
    table.to_csv(csv_path, index=False)

    display = table[
        [
            "encoding", "gate_algebra_abelian", "input_compatible",
            "Z_readout_compatible", "operationally_abelian", "log_negativity",
            "max_current", "rho_imag_frobenius", "max_bargmann_phase_rad",
        ]
    ].copy()
    for column in display.select_dtypes(include="number"):
        display[column] = display[column].map(lambda value: f"{value:.6g}")
    markdown_path.write_text(
        "# Operational abelian controls\n\n"
        "All rows use Iris setosa (50 rows, four features), the same angle scaling, "
        "and the same fixed partial-correlation coupling where applicable.\n\n"
        "Only the first row is operationally abelian in the computational basis: "
        "the gate generators, input state, and Z readout are jointly compatible. "
        "The RY-YY-RY row proves that mutually commuting gate generators alone do "
        "not imply a classical collapse. The diagonal RZ-ZZ-RZ row can have "
        "gauge-dependent complex Gram entries from sample-wise global phases, but "
        "its class operator and Bargmann phases remain real/zero.\n\n"
        + markdown_table(display)
        + "\n",
        encoding="utf-8",
    )
    print(table.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"\nWrote {csv_path} and {markdown_path}")


if __name__ == "__main__":
    main()
