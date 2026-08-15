"""Cross-fitted E4 experiment on native tabular features.

The experiment deliberately avoids PCA and feature truncation.  Every fitted
object -- standardisation, angle scaling, class precision coupling, class
density operator, spectral projector, and classical classifier -- is estimated
inside the training part of a stratified fold.  Spectral fidelity is then
evaluated on held-out rows.

Iris is currently the only bundled real dataset used here because all four of
its native features fit the full-density-operator budget.  Wine and Breast
Cancer are not silently reduced: they require either a scalable native-feature
diagnostic or a separately justified blockwise protocol.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pennylane as qml
from numpy.typing import NDArray
from scipy.stats import mannwhitneyu
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

Array = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

N_SPLITS = 5
RANDOM_STATE = 17
RIDGE = 1e-3
COUPLING_SCALE = 0.8


def _make_encoder(n_qubits: int):
    """Create the canonical PennyLane sandwich encoder."""
    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device, interface=None, diff_method=None)
    def encode(x: Array, coupling: Array):
        # Apply U_enc(x/2) = tensor_j RY(x_j/2).
        for qubit in range(n_qubits):
            qml.RY(x[qubit] / 2.0, wires=qubit)

        # PennyLane IsingZZ(phi) = exp(-i phi Z_j Z_k / 2).
        for j, k in combinations(range(n_qubits), 2):
            qml.IsingZZ(2.0 * coupling[j, k], wires=[j, k])

        # Apply the closing U_enc(x/2), including at J = 0.
        for qubit in range(n_qubits):
            qml.RY(x[qubit] / 2.0, wires=qubit)
        return qml.state()

    return encode


def _class_operator(states: list[ComplexArray]) -> ComplexArray:
    """Compute the class density operator ``rho_c = mean |psi><psi|``."""
    rho = np.zeros((len(states[0]), len(states[0])), dtype=complex)
    for state in states:
        rho += np.outer(state, state.conj())
    return rho / len(states)


def _precision_coupling(X_class: Array) -> Array:
    """Construct the fixed class coupling from partial correlations."""
    n_features = X_class.shape[1]
    covariance = np.cov(X_class, rowvar=False)
    precision = np.linalg.inv(covariance + RIDGE * np.eye(n_features))
    diagonal = np.diag(precision)
    denominator = np.sqrt(np.outer(diagonal, diagonal))
    coupling = -COUPLING_SCALE * precision / denominator
    np.fill_diagonal(coupling, 0.0)
    return 0.5 * (coupling + coupling.T)


def _partial_transpose(
    rho: ComplexArray,
    qubit: int,
    n_qubits: int,
) -> ComplexArray:
    """Partially transpose ``rho`` on one qubit."""
    tensor = rho.reshape((2,) * (2 * n_qubits))
    return np.swapaxes(tensor, qubit, qubit + n_qubits).reshape(rho.shape)


def _logarithmic_negativity(rho: ComplexArray, n_qubits: int) -> float:
    """Compute mean logarithmic negativity over single-mode cuts."""
    values = []
    for qubit in range(n_qubits):
        transposed = _partial_transpose(rho, qubit, n_qubits)
        trace_norm = np.linalg.svd(transposed, compute_uv=False).sum()
        values.append(np.log2(trace_norm))
    return float(np.mean(values))


def _pauli_word(
    n_qubits: int,
    factors: dict[int, ComplexArray],
) -> ComplexArray:
    """Construct a tensor-product Pauli word."""
    identity = np.eye(2, dtype=complex)
    operator = np.array([[1.0 + 0.0j]])
    for qubit in range(n_qubits):
        operator = np.kron(operator, factors.get(qubit, identity))
    return operator


def _maximum_current(rho: ComplexArray, n_qubits: int) -> float:
    """Compute ``max |Im Gamma_jk|`` for local hard-core modes."""
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    pauli_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    values = []
    for j, k in combinations(range(n_qubits), 2):
        xy = _pauli_word(n_qubits, {j: pauli_x, k: pauli_y})
        yx = _pauli_word(n_qubits, {j: pauli_y, k: pauli_x})
        # Im Gamma_jk = <X_j Y_k - Y_j X_k> / 4.
        current = 0.25 * np.trace(rho @ (xy - yx)).real
        values.append(abs(float(current)))
    return max(values, default=0.0)


def _leading_projector(rho: ComplexArray) -> tuple[ComplexArray, int]:
    """Select the leading eigenspace before the largest logarithmic drop."""
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues > 1e-12
    eigenvalues = eigenvalues[positive]
    eigenvectors = eigenvectors[:, positive]

    if len(eigenvalues) <= 1:
        n_modes = 1
    else:
        log_drops = np.log(eigenvalues[:-1]) - np.log(eigenvalues[1:])
        n_modes = int(np.argmax(log_drops) + 1)
    projector_basis = eigenvectors[:, :n_modes]
    return projector_basis, n_modes


def _classical_models() -> list[object]:
    """Return the three fixed classical error probes."""
    return [
        make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        ),
        make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        make_pipeline(StandardScaler(), SVC(kernel="rbf")),
    ]


def evaluate_iris() -> dict[str, float]:
    """Run cross-fitted E4 on all four native Iris features."""
    dataset = load_iris()
    X = np.asarray(dataset.data, dtype=float)
    y = np.asarray(dataset.target)
    n_qubits = X.shape[1]
    encoder = _make_encoder(n_qubits)

    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    wrong = np.zeros(len(y), dtype=bool)
    fidelities = np.full(len(y), np.nan, dtype=float)
    fold_negativities = []
    fold_currents = []
    retained_modes = []

    for train_indices, test_indices in splitter.split(X, y):
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]

        # Fit every preprocessing transformation on the fold's training rows.
        standardizer = StandardScaler().fit(X_train)
        Z_train = standardizer.transform(X_train)
        Z_test = standardizer.transform(X_test)
        angle_scaler = MinMaxScaler(
            feature_range=(0.0, np.pi),
            clip=True,
        ).fit(Z_train)
        angles_train = angle_scaler.transform(Z_train)
        angles_test = angle_scaler.transform(Z_test)

        class_couplings = {
            label: _precision_coupling(Z_train[y_train == label])
            for label in np.unique(y_train)
        }

        class_projectors = {}
        class_negativities = []
        class_currents = []
        for label, coupling in class_couplings.items():
            states = [
                np.asarray(encoder(row, coupling), dtype=complex)
                for row in angles_train[y_train == label]
            ]
            rho = _class_operator(states)
            projector, n_modes = _leading_projector(rho)
            class_projectors[label] = projector
            retained_modes.append(n_modes)
            class_negativities.append(_logarithmic_negativity(rho, n_qubits))
            class_currents.append(_maximum_current(rho, n_qubits))

        fold_negativities.append(float(np.mean(class_negativities)))
        fold_currents.append(float(np.mean(class_currents)))

        # Compute held-out spectral fidelity against the row's true class.
        for local_index, global_index in enumerate(test_indices):
            label = y_test[local_index]
            state = np.asarray(
                encoder(
                    angles_test[local_index],
                    class_couplings[label],
                ),
                dtype=complex,
            )
            projector = class_projectors[label]
            fidelities[global_index] = float(
                np.sum(np.abs(projector.conj().T @ state) ** 2)
            )

        # Define model errors out of fold, using the same held-out rows.
        for model in _classical_models():
            model.fit(X_train, y_train)
            wrong[test_indices] |= model.predict(X_test) != y_test

    if np.isnan(fidelities).any():
        raise RuntimeError("Cross-fitting did not assign every fidelity")
    if not wrong.any():
        raise RuntimeError("No out-of-fold errors were detected")

    p_value = mannwhitneyu(
        fidelities[~wrong],
        fidelities[wrong],
        alternative="greater",
    ).pvalue
    result = {
        "n_samples": float(len(y)),
        "n_features": float(n_qubits),
        "n_errors": float(wrong.sum()),
        "Q": float(np.mean(fold_negativities)),
        "current": float(np.mean(fold_currents)),
        "fidelity_correct": float(fidelities[~wrong].mean()),
        "fidelity_wrong": float(fidelities[wrong].mean()),
        "p_value": float(p_value),
        "median_retained_modes": float(np.median(retained_modes)),
    }
    return result


def main() -> None:
    """Run E4 and print a manuscript-ready summary."""
    result = evaluate_iris()
    print(
        "Iris [native 4 features, cross-fitted] "
        f"Q={result['Q']:.3f}, current={result['current']:.3f}, "
        f"errors={int(result['n_errors'])}/{int(result['n_samples'])}, "
        f"Fs correct={result['fidelity_correct']:.3f}, "
        f"Fs wrong={result['fidelity_wrong']:.3f}, "
        f"p={result['p_value']:.2e}, "
        f"median k={result['median_retained_modes']:.0f}"
    )


if __name__ == "__main__":
    main()
