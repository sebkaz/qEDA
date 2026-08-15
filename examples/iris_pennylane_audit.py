"""Compare the product control with one- and two-layer sandwich encodings on Iris.

Run from the repository root after ``pip install -e .``:

    python examples/iris_pennylane_audit.py

The example uses Iris setosa only so it forms one empirical class operator.
It is a representation audit, not a classification experiment.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

from qeda import (
    matched_audit,
    pennylane_encoding,
    product_ry_circuit,
    sandwich_circuit,
)


def partial_correlation_coupling(data: np.ndarray, *, scale: float = 0.18) -> np.ndarray:
    """Construct a symmetric partial-correlation coupling from one feature block."""
    covariance = np.cov(data, rowvar=False)
    precision = np.linalg.pinv(covariance)
    diagonal = np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    coupling = -precision / diagonal
    np.fill_diagonal(coupling, 0.0)
    return scale * coupling


def main() -> None:
    iris = load_iris()
    raw = iris.data[iris.target == 0]
    features = MinMaxScaler(feature_range=(0.0, np.pi)).fit_transform(raw)
    coupling = partial_correlation_coupling(raw)
    n_qubits = features.shape[1]

    control = pennylane_encoding(product_ry_circuit, n_qubits)
    one_layer = pennylane_encoding(
        sandwich_circuit(coupling, layers=1), n_qubits
    )
    two_layer = pennylane_encoding(
        sandwich_circuit(coupling, layers=2), n_qubits
    )

    print(matched_audit(
        features,
        control,
        one_layer,
        control_name="product RY (J=0)",
        candidate_name="sandwich (1 layer)",
    ).to_markdown())
    print()
    print(matched_audit(
        features,
        control,
        two_layer,
        control_name="product RY (J=0)",
        candidate_name="sandwich (2 layers)",
    ).to_markdown())


if __name__ == "__main__":
    main()
