"""PennyLane adapters and reference circuits for qEDA.

qEDA consumes a callable ``row -> statevector``.  This module turns a
declared PennyLane circuit into that callable, so the same operator audit can
be applied to the sandwich or to an independently designed feature circuit.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .engine import ComplexArray, Encoding, RealArray


PennyLaneCircuit = Callable[[RealArray, Sequence[int]], None]


def pennylane_encoding(
    circuit: PennyLaneCircuit,
    n_qubits: int,
    *,
    device_name: str = "default.qubit",
) -> Encoding:
    """Adapt a PennyLane state-preparation circuit to a qEDA encoding.

    Parameters
    ----------
    circuit
        Callable with signature ``circuit(features, wires)``.  It must apply
        state-preparation operations only; qEDA appends ``qml.state()``.
    n_qubits
        Number of wires and hence the dimension of the returned statevector.
    device_name
        PennyLane simulator device.  The default is the exact statevector
        simulator required by the empirical-operator construction.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")

    import pennylane as qml

    wires = tuple(range(n_qubits))
    device = qml.device(device_name, wires=wires, shots=None)

    @qml.qnode(device)
    def statevector(features: RealArray) -> ComplexArray:
        circuit(features, wires)
        return qml.state()

    def encoding(row: RealArray) -> ComplexArray:
        features = np.asarray(row, dtype=float)
        if features.ndim != 1:
            raise ValueError("each encoded row must be one-dimensional")
        return np.asarray(statevector(features), dtype=complex)

    return encoding


def product_ry_circuit(features: RealArray, wires: Sequence[int]) -> None:
    """Apply the real product control ``prod_j RY(x_j)``."""
    import pennylane as qml

    values = np.asarray(features, dtype=float)
    if values.shape != (len(wires),):
        raise ValueError("features and wires must have the same length")
    for value, wire in zip(values, wires, strict=True):
        qml.RY(value, wires=wire)


def sandwich_circuit(
    coupling: RealArray,
    *,
    layers: int = 1,
    rescale_features: bool = True,
) -> PennyLaneCircuit:
    """Return a layered ``RY(x/2) -- ZZ(J) -- RY(x/2)`` circuit.

    ``layers=1`` is the manuscript sandwich.  With ``rescale_features=True``
    each block receives ``x/layers`` so that the uncoupled circuit retains the
    same total product-angle map for every depth.  Since PennyLane's
    ``IsingZZ(phi)`` equals ``exp(-i phi Z⊗Z / 2)``, the gate receives
    ``2 * J_jk`` to implement the coupling convention of the manuscript.
    """
    matrix = np.asarray(coupling, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("coupling must be a square matrix")
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("coupling must be symmetric")
    if layers < 1:
        raise ValueError("layers must be positive")

    n_qubits = matrix.shape[0]

    def circuit(features: RealArray, wires: Sequence[int]) -> None:
        import pennylane as qml

        values = np.asarray(features, dtype=float)
        if len(wires) != n_qubits or values.shape != (n_qubits,):
            raise ValueError("features, wires, and coupling must have one common size")
        block_features = values / layers if rescale_features else values
        for _ in range(layers):
            for value, wire in zip(block_features, wires, strict=True):
                qml.RY(value / 2.0, wires=wire)
            for left in range(n_qubits):
                for right in range(left + 1, n_qubits):
                    strength = matrix[left, right]
                    if strength != 0.0:
                        qml.IsingZZ(2.0 * strength, wires=[wires[left], wires[right]])
            for value, wire in zip(block_features, wires, strict=True):
                qml.RY(value / 2.0, wires=wire)

    return circuit
