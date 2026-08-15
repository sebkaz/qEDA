"""One-class qEDA case study on the credit-card fraud dataset.

The released dataset contains anonymised PCA coordinates V1--V28 together
with Time and Amount.  This script analyses that released representation; it
does not claim access to the original transaction variables.

Only normal transactions from the training split are used to fit scaling,
partial-correlation couplings, empirical density operators, spectral
projectors, and false-positive-rate thresholds.  Fraud labels are used only
for held-out evaluation.  No predictive feature selector is fitted.  Thirty
features are handled by the balanced cyclic marginal protocol from Section 8
of the manuscript: four seeded base permutations, all thirty cyclic shifts,
and four features per path, giving 120 equally balanced paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/qeda-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/qeda-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from qeda.engine import (
    class_operator_from_states,
    log_negativity,
    mass_gap,
    maximum_current,
    participation_rank,
    purity,
)

RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

SEED = 1729
SPLIT_SEED = 42
TEST_SIZE = 0.20
N_REFERENCE = 5_000
N_CALIBRATION = 10_000
N_BASE_PERMUTATIONS = 4
N_QUBITS = 4
RIDGE = 1e-3
ALPHA = 0.8
BATCH_SIZE = 8_192
TARGET_FPRS = (0.001, 0.005, 0.01)

DATA_PATH = Path("data/credit-card/credit.csv")
OUTPUT_DATA = Path("results/data")
OUTPUT_FIGURES = Path("results/figures")


@dataclass(frozen=True)
class PathResult:
    """Scores and diagnostics for one balanced feature path."""

    test_j0: RealArray
    test_j: RealArray
    calibration_j0: RealArray
    calibration_j: RealArray
    diagnostics: dict[str, float | int | str]


def balanced_feature_paths(
    feature_names: list[str],
    rng: np.random.Generator,
) -> list[tuple[str, ...]]:
    """Return four-permutation cyclic paths at the fixed qubit budget."""
    canonical = np.array(sorted(feature_names), dtype=object)
    paths: list[tuple[str, ...]] = []
    for _ in range(N_BASE_PERMUTATIONS):
        base = rng.permutation(canonical)
        for shift in range(len(base)):
            shifted = np.roll(base, -shift)
            paths.append(tuple(sorted(shifted[:N_QUBITS].tolist())))
    return paths


def partial_correlation_coupling(
    standardised: RealArray,
    alpha: float = ALPHA,
) -> RealArray:
    """Estimate the fixed-scale class coupling from normal reference rows."""
    n_features = standardised.shape[1]
    covariance = np.cov(standardised, rowvar=False)
    precision = np.linalg.inv(covariance + RIDGE * np.eye(n_features))
    denominator = np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    coupling = -alpha * precision / denominator
    np.fill_diagonal(coupling, 0.0)
    return 0.5 * (coupling + coupling.T)


def _apply_ry_layer(states: ComplexArray, angles: RealArray) -> ComplexArray:
    """Apply row-dependent ``tensor_j RY(angles_j / 2)`` in batches."""
    result = states
    n_rows = len(angles)
    for wire in range(N_QUBITS):
        left_dimension = 2**wire
        right_dimension = 2 ** (N_QUBITS - wire - 1)
        cosine = np.cos(angles[:, wire] / 4.0)
        sine = np.sin(angles[:, wire] / 4.0)
        rotations = np.empty((n_rows, 2, 2), dtype=float)
        rotations[:, 0, 0] = cosine
        rotations[:, 0, 1] = -sine
        rotations[:, 1, 0] = sine
        rotations[:, 1, 1] = cosine
        tensor = result.reshape(n_rows, left_dimension, 2, right_dimension)
        result = np.einsum("nlbr,nab->nlar", tensor, rotations).reshape(
            n_rows, -1
        )
    return result


def encode_numpy(angles: RealArray, coupling: RealArray) -> ComplexArray:
    """Vectorised statevector implementation of the canonical sandwich."""
    n_rows = len(angles)
    dimension = 2**N_QUBITS
    states = np.zeros((n_rows, dimension), dtype=complex)
    states[:, 0] = 1.0
    states = _apply_ry_layer(states, angles)

    basis = np.arange(dimension, dtype=int)
    shifts = N_QUBITS - 1 - np.arange(N_QUBITS)
    bits = (basis[:, None] >> shifts[None, :]) & 1
    z_values = 1.0 - 2.0 * bits
    energies = 0.5 * np.einsum("bi,ij,bj->b", z_values, coupling, z_values)
    states *= np.exp(-1j * energies)[None, :]
    return _apply_ry_layer(states, angles)


def validate_numpy_encoder(rng: np.random.Generator) -> float:
    """Cross-check the vectorised implementation against PennyLane."""
    device = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(device, interface=None, diff_method=None)
    def encode_pennylane(angles: RealArray, coupling: RealArray):
        for qubit in range(N_QUBITS):
            qml.RY(angles[qubit] / 2.0, wires=qubit)
        for j, k in combinations(range(N_QUBITS), 2):
            qml.IsingZZ(2.0 * coupling[j, k], wires=[j, k])
        for qubit in range(N_QUBITS):
            qml.RY(angles[qubit] / 2.0, wires=qubit)
        return qml.state()

    angles = rng.uniform(0.0, np.pi, size=(3, N_QUBITS))
    raw_coupling = rng.uniform(-0.5, 0.5, size=(N_QUBITS, N_QUBITS))
    coupling = np.triu(raw_coupling, 1)
    coupling += coupling.T
    numpy_states = encode_numpy(angles, coupling)
    errors = []
    for row, numpy_state in zip(angles, numpy_states):
        reference = np.asarray(encode_pennylane(row, coupling), dtype=complex)
        errors.append(float(np.max(np.abs(reference - numpy_state))))
    return max(errors)


def encode_in_batches(angles: RealArray, coupling: RealArray) -> ComplexArray:
    """Encode a large table without a large temporary tensor."""
    batches = []
    for start in range(0, len(angles), BATCH_SIZE):
        batches.append(encode_numpy(angles[start : start + BATCH_SIZE], coupling))
    return np.concatenate(batches, axis=0)


def leading_basis(rho: ComplexArray) -> tuple[ComplexArray, int]:
    """Select modes before the largest logarithmic spectral drop."""
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]
    support = eigenvalues > 1e-12
    eigenvalues = eigenvalues[support]
    eigenvectors = eigenvectors[:, support]
    if len(eigenvalues) <= 1:
        retained = 1
    else:
        retained = int(np.argmax(np.log(eigenvalues[:-1] / eigenvalues[1:])) + 1)
    return eigenvectors[:, :retained], retained


def anomaly_scores(states: ComplexArray, basis: ComplexArray) -> RealArray:
    """Return one minus the leading-subspace Born weight."""
    overlaps = states @ basis.conj()
    fidelity = np.sum(np.abs(overlaps) ** 2, axis=1)
    return np.clip(1.0 - fidelity, 0.0, 1.0)


def transform_angles(
    values: pd.DataFrame,
    standardizer: StandardScaler,
    angle_scaler: MinMaxScaler,
) -> RealArray:
    """Apply reference-fitted standardisation and angle scaling."""
    standardised = standardizer.transform(values)
    return np.asarray(angle_scaler.transform(standardised), dtype=float)


def evaluate_path(
    path_index: int,
    features: tuple[str, ...],
    reference: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    y_test: RealArray,
    alpha: float = ALPHA,
) -> PathResult:
    """Fit one normal-reference marginal and score calibration plus test rows."""
    reference_subset = reference.loc[:, features]
    standardizer = StandardScaler().fit(reference_subset)
    reference_standardised = standardizer.transform(reference_subset)
    angle_scaler = MinMaxScaler(
        feature_range=(0.0, np.pi),
        clip=True,
    ).fit(reference_standardised)

    reference_angles = np.asarray(
        angle_scaler.transform(reference_standardised), dtype=float
    )
    calibration_angles = transform_angles(
        calibration.loc[:, features], standardizer, angle_scaler
    )
    test_angles = transform_angles(test.loc[:, features], standardizer, angle_scaler)
    coupling = partial_correlation_coupling(reference_standardised, alpha=alpha)

    diagnostics: dict[str, float | int | str] = {
        "path": path_index,
        "features": "|".join(features),
        "coupling_frobenius": float(np.linalg.norm(coupling)),
        "coupling_max_abs": float(np.max(np.abs(coupling))),
    }
    score_sets: dict[str, RealArray] = {}

    for encoding_name, active_coupling in (
        ("J0", np.zeros_like(coupling)),
        ("J_partial", coupling),
    ):
        reference_states = encode_in_batches(reference_angles, active_coupling)
        rho = class_operator_from_states(reference_states.T)
        basis, retained = leading_basis(rho)
        calibration_states = encode_in_batches(calibration_angles, active_coupling)
        test_states = encode_in_batches(test_angles, active_coupling)
        score_sets[f"calibration_{encoding_name}"] = anomaly_scores(
            calibration_states, basis
        )
        score_sets[f"test_{encoding_name}"] = anomaly_scores(test_states, basis)

        diagnostics[f"retained_{encoding_name}"] = retained
        diagnostics[f"purity_{encoding_name}"] = purity(rho)
        diagnostics[f"effective_rank_{encoding_name}"] = participation_rank(rho)
        diagnostics[f"mass_gap_{encoding_name}"] = mass_gap(rho)
        diagnostics[f"Q_{encoding_name}"] = log_negativity(rho)
        diagnostics[f"current_{encoding_name}"] = maximum_current(rho)
        diagnostics[f"roc_auc_{encoding_name}"] = roc_auc_score(
            y_test, score_sets[f"test_{encoding_name}"]
        )
        diagnostics[f"average_precision_{encoding_name}"] = average_precision_score(
            y_test, score_sets[f"test_{encoding_name}"]
        )

    return PathResult(
        test_j0=score_sets["test_J0"],
        test_j=score_sets["test_J_partial"],
        calibration_j0=score_sets["calibration_J0"],
        calibration_j=score_sets["calibration_J_partial"],
        diagnostics=diagnostics,
    )


def evaluation_rows(
    name: str,
    test_scores: RealArray,
    calibration_scores: RealArray,
    labels: RealArray,
) -> list[dict[str, float | int | str]]:
    """Evaluate ranking and calibration-fitted operating points."""
    rows = []
    for target_fpr in TARGET_FPRS:
        threshold = float(np.quantile(calibration_scores, 1.0 - target_fpr))
        predictions = test_scores >= threshold
        normal = labels == 0
        fraud = labels == 1
        false_positive_rate = float(np.mean(predictions[normal]))
        recall = float(np.mean(predictions[fraud]))
        precision = float(
            labels[predictions].mean() if np.any(predictions) else 0.0
        )
        f1 = float(
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
        rows.append(
            {
                "method": name,
                "roc_auc": roc_auc_score(labels, test_scores),
                "average_precision": average_precision_score(labels, test_scores),
                "target_fpr": target_fpr,
                "calibration_threshold": threshold,
                "observed_test_fpr": false_positive_rate,
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "n_flagged": int(predictions.sum()),
            }
        )
    return rows


def make_figure(
    labels: RealArray,
    scores: pd.DataFrame,
    path_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot score distributions, PR/ROC curves, and path variability."""
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.5))
    plot_methods = {
        "qEDA J=0": "qeda_j0_median",
        "qEDA partial J": "qeda_j_median",
        "Isolation Forest": "isolation_forest",
    }

    for label, color in ((0, "#0072B2"), (1, "#D55E00")):
        values = scores.loc[labels == label, "qeda_j_median"]
        axes[0, 0].hist(
            values,
            bins=60,
            density=True,
            histtype="step",
            linewidth=1.5,
            color=color,
            label="normal" if label == 0 else "fraud",
        )
    axes[0, 0].set_title("Held-out qEDA anomaly scores")
    axes[0, 0].set_xlabel(r"median $1-F_s$ across balanced paths")
    axes[0, 0].set_ylabel("density")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(frameon=False)

    for display_name, column in plot_methods.items():
        precision_values, recall_values, _ = precision_recall_curve(
            labels, scores[column]
        )
        average_precision = average_precision_score(labels, scores[column])
        axes[0, 1].plot(
            recall_values,
            precision_values,
            label=f"{display_name} (AP={average_precision:.3f})",
        )
    axes[0, 1].axhline(labels.mean(), color="black", linestyle="--", linewidth=0.8)
    axes[0, 1].set_title("Natural-prevalence precision--recall")
    axes[0, 1].set_xlabel("recall")
    axes[0, 1].set_ylabel("precision")
    axes[0, 1].legend(frameon=False, fontsize=8)

    for display_name, column in plot_methods.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores[column])
        auc = roc_auc_score(labels, scores[column])
        axes[1, 0].plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{display_name} (AUC={auc:.3f})",
        )
    axes[1, 0].plot([0.0, 1.0], [0.0, 1.0], "k--", linewidth=0.8)
    axes[1, 0].set_xlim(0.0, 0.05)
    axes[1, 0].set_title("ROC detail at low false-positive rates")
    axes[1, 0].set_xlabel("false-positive rate")
    axes[1, 0].set_ylabel("true-positive rate")
    axes[1, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].boxplot(
        [
            path_metrics["roc_auc_J0"],
            path_metrics["roc_auc_J_partial"],
        ],
        tick_labels=["J=0", "partial J"],
        showfliers=False,
    )
    axes[1, 1].set_title("Variability across 120 balanced marginals")
    axes[1, 1].set_ylabel("path-wise ROC AUC")

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Run the complete leakage-controlled fraud case study."""
    OUTPUT_DATA.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    validation_rng = np.random.default_rng(SEED + 2)
    reference_rng = np.random.default_rng(SEED + 1)
    path_rng = np.random.default_rng(SEED)
    encoder_error = validate_numpy_encoder(validation_rng)
    if encoder_error > 1e-10:
        raise RuntimeError(f"NumPy/PennyLane encoder mismatch: {encoder_error}")

    data = pd.read_csv(DATA_PATH)
    features = [column for column in data.columns if column != "Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        data[features],
        data["Class"].to_numpy(dtype=int),
        test_size=TEST_SIZE,
        random_state=SPLIT_SEED,
        stratify=data["Class"],
    )
    normal_train = X_train.loc[y_train == 0]
    selected = reference_rng.choice(
        len(normal_train),
        size=N_REFERENCE + N_CALIBRATION,
        replace=False,
    )
    reference = normal_train.iloc[selected[:N_REFERENCE]].copy()
    calibration = normal_train.iloc[selected[N_REFERENCE:]].copy()

    paths = balanced_feature_paths(features, path_rng)
    expected_path_count = N_BASE_PERMUTATIONS * len(features)
    expected_feature_count = N_BASE_PERMUTATIONS * N_QUBITS
    feature_counts = {
        feature: sum(feature in path for path in paths) for feature in features
    }
    if len(paths) != expected_path_count or set(feature_counts.values()) != {
        expected_feature_count
    }:
        raise RuntimeError("Balanced marginal path construction failed")

    test_j0 = np.empty((len(X_test), len(paths)), dtype=np.float32)
    test_j = np.empty_like(test_j0)
    calibration_j0 = np.empty((len(calibration), len(paths)), dtype=np.float32)
    calibration_j = np.empty_like(calibration_j0)
    diagnostics = []

    for path_index, path_features in enumerate(paths):
        result = evaluate_path(
            path_index,
            path_features,
            reference,
            calibration,
            X_test,
            y_test,
        )
        test_j0[:, path_index] = result.test_j0
        test_j[:, path_index] = result.test_j
        calibration_j0[:, path_index] = result.calibration_j0
        calibration_j[:, path_index] = result.calibration_j
        diagnostics.append(result.diagnostics)
        if (path_index + 1) % 10 == 0 or path_index + 1 == len(paths):
            print(f"Profiled {path_index + 1}/{len(paths)} paths", flush=True)

    aggregate_scores = pd.DataFrame(
        {
            "qeda_j0_median": np.median(test_j0, axis=1),
            "qeda_j_median": np.median(test_j, axis=1),
            "qeda_j0_q90": np.quantile(test_j0, 0.90, axis=1),
            "qeda_j_q90": np.quantile(test_j, 0.90, axis=1),
        },
        index=X_test.index,
    )
    calibration_scores = {
        "qeda_j0_median": np.median(calibration_j0, axis=1),
        "qeda_j_median": np.median(calibration_j, axis=1),
        "qeda_j0_q90": np.quantile(calibration_j0, 0.90, axis=1),
        "qeda_j_q90": np.quantile(calibration_j, 0.90, axis=1),
    }

    # Isolation Forest is a contextual classical anomaly detector, fitted on
    # exactly the same normal reference rows and never used to define qEDA.
    full_standardizer = StandardScaler().fit(reference[features])
    reference_standardised = full_standardizer.transform(reference[features])
    calibration_standardised = full_standardizer.transform(calibration[features])
    test_standardised = full_standardizer.transform(X_test[features])
    isolation_forest = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=SEED,
        n_jobs=-1,
    ).fit(reference_standardised)
    aggregate_scores["isolation_forest"] = -isolation_forest.decision_function(
        test_standardised
    )
    calibration_scores["isolation_forest"] = -isolation_forest.decision_function(
        calibration_standardised
    )

    summary_rows = []
    for method, test_score in aggregate_scores.items():
        summary_rows.extend(
            evaluation_rows(
                method,
                test_score.to_numpy(),
                np.asarray(calibration_scores[method]),
                y_test,
            )
        )

    path_metrics = pd.DataFrame(diagnostics)
    summary = pd.DataFrame(summary_rows)
    score_output = aggregate_scores.copy()
    score_output.insert(0, "label", y_test)
    score_output.insert(0, "source_index", X_test.index)

    path_metrics.to_csv(OUTPUT_DATA / "fraud_qeda_paths.csv", index=False)
    summary.to_csv(OUTPUT_DATA / "fraud_qeda_summary.csv", index=False)
    score_output.to_csv(OUTPUT_DATA / "fraud_qeda_scores.csv", index=False)
    make_figure(
        y_test,
        aggregate_scores.reset_index(drop=True),
        path_metrics,
        OUTPUT_FIGURES / "fraud_qeda_case_study.pdf",
    )

    print(f"PennyLane validation max error: {encoder_error:.3e}")
    print(f"Train normal reference: {len(reference)}")
    print(f"Normal calibration: {len(calibration)}")
    print(f"Held-out test: {len(y_test)} rows, {int(y_test.sum())} frauds")
    print("\nPrimary summary at target FPR=0.005")
    print(
        summary[summary["target_fpr"] == 0.005].to_string(
            index=False,
            float_format=lambda value: f"{value:.6g}",
        )
    )


if __name__ == "__main__":
    main()
