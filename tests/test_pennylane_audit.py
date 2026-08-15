"""Focused checks for the public circuit-to-report qEDA API."""

from __future__ import annotations

import unittest

import numpy as np

from qeda import audit, matched_audit, pennylane_encoding, product_ry_circuit, sandwich_circuit


class PennyLaneAuditTest(unittest.TestCase):
    """Verify the adapter, matched report, and multi-layer sandwich control."""

    def setUp(self) -> None:
        self.data = np.array(
            [[0.2, 0.4], [0.6, 0.8], [1.0, 1.2], [1.4, 1.6]], dtype=float
        )
        self.control = pennylane_encoding(product_ry_circuit, n_qubits=2)

    def test_product_control_has_zero_encoded_current(self) -> None:
        report = audit(self.data, self.control, name="product")
        self.assertAlmostEqual(report.scalars()["current"], 0.0, places=12)
        self.assertIn("## Spectral reading", report.to_markdown())
        self.assertIn("## Declared-subsystem reading", report.to_markdown())
        self.assertIn("## Feature-mode reading", report.to_markdown())

    def test_layered_sandwich_produces_a_matched_report(self) -> None:
        coupling = np.array([[0.0, 0.35], [0.35, 0.0]])
        candidate = pennylane_encoding(sandwich_circuit(coupling, layers=2), 2)
        report = matched_audit(self.data, self.control, candidate)
        self.assertEqual(set(report.scalar_contrasts()), set(report.control.scalars()))
        self.assertIn("candidate - control", report.to_markdown())

    def test_zero_coupling_recovers_the_product_control_at_any_depth(self) -> None:
        zero = np.zeros((2, 2))
        candidate = pennylane_encoding(sandwich_circuit(zero, layers=3), 2)
        report = matched_audit(self.data, self.control, candidate)
        for contrast in report.scalar_contrasts().values():
            self.assertAlmostEqual(contrast, 0.0, places=12)

    def test_user_supplied_pennylane_circuit_is_accepted(self) -> None:
        import pennylane as qml

        def circuit(features: np.ndarray, wires: tuple[int, ...]) -> None:
            qml.RX(features[0], wires=wires[0])
            qml.RY(features[1], wires=wires[1])
            qml.CNOT(wires=[wires[0], wires[1]])

        report = audit(self.data, pennylane_encoding(circuit, 2), name="custom")
        self.assertEqual(report.n_qubits, 2)
        self.assertGreaterEqual(report.scalars()["Q"], 0.0)


if __name__ == "__main__":
    unittest.main()
