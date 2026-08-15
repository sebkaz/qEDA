"""Classical and PennyLane spectral EDA of the Bowles benchmark datasets.

The benchmark datasets are the object of study; predictive models are not.
For every dataset variant, the script writes three deliberately separate
tables:

* standard EDA of the native training features;
* class-resolved spectra of the separable and coupled density operators;
* published Bowles accuracy ranges as contextual metadata only.

No predictive model is fitted here and no PCA or feature truncation is used.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import pennylane as qml
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import datasets as benchmark_datasets
from retrodiction import collect as collect_published_accuracies

Array = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class SpectralConfiguration:
    """Fixed configuration of the descriptive benchmark."""

    coupling_scale: float = 1.0
    ridge: float = 1e-3
    angle_scale: float = 1.0
    max_qubits: int = 10


def make_encoder(n_qubits: int):
    """Create the canonical PennyLane sandwich encoder for ``n_qubits``."""
    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device, interface=None, diff_method=None)
    def encode(x: Array, coupling: Array):
        # Apply U_enc(x/2) = tensor_j RY(x_j/2).
        for qubit in range(n_qubits):
            qml.RY(x[qubit] / 2.0, wires=qubit)

        # Apply U_ZZ(J) = product_(j<k) exp(-i J_jk Z_j Z_k).
        for j in range(n_qubits):
            for k in range(j + 1, n_qubits):
                qml.IsingZZ(2.0 * coupling[j, k], wires=[j, k])

        # Apply the closing U_enc(x/2), including at J = 0.
        for qubit in range(n_qubits):
            qml.RY(x[qubit] / 2.0, wires=qubit)
        return qml.state()

    return encode


def prepare_features(X_train: Array) -> tuple[Array, Array]:
    """Return standardised features and train-fitted angles in ``[0, pi]``."""
    standardised = StandardScaler().fit_transform(X_train)
    angles = MinMaxScaler(feature_range=(0.0, np.pi)).fit_transform(standardised)
    return standardised, angles


def _normalised_spectrum(matrix: Array) -> Array:
    """Return the non-negative unit-trace spectrum of a symmetric matrix."""
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    eigenvalues = np.clip(eigenvalues, 0.0, None)[::-1]
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return np.zeros_like(eigenvalues)
    return eigenvalues / total


def classical_eda_statistics(X_train: Array, y_train: NDArray) -> dict[str, float]:
    """Describe native features without projecting or rotating the data."""
    standardised = StandardScaler().fit_transform(X_train)
    correlation = np.atleast_2d(np.corrcoef(standardised, rowvar=False))
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 1.0)

    spectrum = _normalised_spectrum(correlation)
    positive = spectrum[spectrum > 1e-12]
    entropy = float(-np.sum(positive * np.log(positive)))
    off_diagonal = correlation[~np.eye(correlation.shape[0], dtype=bool)]

    labels, counts = np.unique(y_train, return_counts=True)
    probabilities = counts / counts.sum()
    class_entropy = float(-np.sum(probabilities * np.log2(probabilities)))

    overall_mean = standardised.mean(axis=0)
    within_trace = 0.0
    between_trace = 0.0
    for label, count in zip(labels, counts):
        X_class = standardised[y_train == label]
        class_mean = X_class.mean(axis=0)
        within_trace += float(np.sum((X_class - class_mean) ** 2))
        between_trace += float(count * np.sum((class_mean - overall_mean) ** 2))

    return {
        "n_train": float(len(X_train)),
        "n_features": float(X_train.shape[1]),
        "n_classes": float(len(labels)),
        "minority_fraction": float(counts.min() / counts.sum()),
        "class_entropy_bits": class_entropy,
        "mean_abs_correlation": float(np.mean(np.abs(off_diagonal))),
        "max_abs_correlation": float(np.max(np.abs(off_diagonal))),
        "correlation_effective_rank": float(1.0 / np.sum(spectrum**2)),
        "correlation_entropy_rank": float(np.exp(entropy)),
        "correlation_lambda_0": float(spectrum[0]),
        "fisher_trace_ratio": float(
            between_trace / max(within_trace, np.finfo(float).eps)
        ),
    }


@lru_cache(maxsize=None)
def _published_table(family: str) -> pd.DataFrame:
    """Cache the local Bowles result table for one dataset family."""
    return collect_published_accuracies(family)


def accuracy_context(family: str, size: int) -> dict[str, float]:
    """Summarise published model accuracies without using them as targets."""
    published = _published_table(family)
    if published.empty or size not in published.index:
        return {
            "n_published_models": 0.0,
            "published_accuracy_min": np.nan,
            "published_accuracy_median": np.nan,
            "published_accuracy_max": np.nan,
        }
    values = published.loc[size].dropna().to_numpy(dtype=float)
    return {
        "n_published_models": float(len(values)),
        "published_accuracy_min": float(np.min(values)),
        "published_accuracy_median": float(np.median(values)),
        "published_accuracy_max": float(np.max(values)),
    }


def precision_coupling(
    X_class: Array,
    coupling_scale: float,
    ridge: float,
) -> Array:
    """Construct a fixed-scale coupling from class partial correlations."""
    n_features = X_class.shape[1]
    covariance = np.atleast_2d(np.cov(X_class, rowvar=False))
    precision = np.linalg.inv(covariance + ridge * np.eye(n_features))
    diagonal = np.diag(precision)
    denominator = np.sqrt(np.outer(diagonal, diagonal))
    coupling = -coupling_scale * precision / denominator
    np.fill_diagonal(coupling, 0.0)
    return 0.5 * (coupling + coupling.T)


def class_operator(
    X_class: Array,
    coupling: Array,
    encoder,
) -> ComplexArray:
    """Compute ``rho_c = M_c^-1 sum_m |psi_m><psi_m|``."""
    dimension = 2 ** X_class.shape[1]
    rho = np.zeros((dimension, dimension), dtype=complex)
    for row in X_class:
        state = np.asarray(encoder(row, coupling), dtype=complex)
        rho += np.outer(state, state.conj())
    rho /= len(X_class)
    return 0.5 * (rho + rho.conj().T)


def spectral_statistics(rho: ComplexArray) -> dict[str, float]:
    """Compute the spectral summaries used by the manuscript."""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = np.clip(eigenvalues, 0.0, None)[::-1]
    eigenvalues /= eigenvalues.sum()
    positive = eigenvalues[eigenvalues > 1e-12]

    purity = float(np.sum(eigenvalues**2))
    entropy = float(-np.sum(positive * np.log2(positive)))
    if len(positive) > 1:
        mass_gap = float(np.log(positive[0] / positive[1]))
    else:
        mass_gap = float("inf")

    return {
        "purity": purity,
        "dispersion": 1.0 - purity,
        "effective_rank": 1.0 / purity,
        "entropy_bits": entropy,
        "mass_gap": mass_gap,
        "rank": float(len(positive)),
        "lambda_0": float(positive[0]),
        "lambda_1": float(positive[1]) if len(positive) > 1 else 0.0,
        "third_moment": float(np.sum(eigenvalues**3)),
    }


def profile_variant(
    family: str,
    size: int,
    X_train: Array,
    y_train: NDArray,
    configuration: SpectralConfiguration,
) -> list[dict[str, float | int | str]]:
    """Profile every class of one dataset variant."""
    n_qubits = X_train.shape[1]
    standardised, angles = prepare_features(X_train)
    encoder = make_encoder(n_qubits)
    rows = []

    for label in np.unique(y_train):
        class_mask = y_train == label
        coupling = precision_coupling(
            standardised[class_mask],
            coupling_scale=configuration.coupling_scale,
            ridge=configuration.ridge,
        )
        baseline = np.zeros_like(coupling)

        rho_0 = class_operator(
            configuration.angle_scale * angles[class_mask], baseline, encoder
        )
        rho_j = class_operator(
            configuration.angle_scale * angles[class_mask], coupling, encoder
        )
        stats_0 = spectral_statistics(rho_0)
        stats_j = spectral_statistics(rho_j)

        row: dict[str, float | int | str] = {
            "family": family,
            "size": int(size),
            "class_label": str(label),
            "n_qubits": int(n_qubits),
            "n_class_samples": int(class_mask.sum()),
            "coupling_frobenius": float(np.linalg.norm(coupling)),
            "coupling_max_abs": float(np.max(np.abs(coupling))),
        }
        for name, value in stats_0.items():
            row[f"{name}_J0"] = value
        for name, value in stats_j.items():
            row[f"{name}_J"] = value
            row[f"delta_{name}"] = value - stats_0[name]
        rows.append(row)
    return rows


def load_family(family: str, max_size: int):
    """Regenerate one benchmark family with its original split."""
    if family == "bars_and_stripes":
        return benchmark_datasets.bars_and_stripes(sizes=(4,))
    if family == "hyperplanes_diff":
        return benchmark_datasets.hyperplanes_diff(max_n=max_size)
    if family in ("hidden_manifold_diff", "two_curves_diff"):
        return benchmark_datasets.FAMILIES[family](max_n=max_size)
    return benchmark_datasets.FAMILIES[family](max_d=max_size)


def run_benchmark(
    families: list[str],
    max_size: int,
    configuration: SpectralConfiguration,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return quantum spectra, classical EDA, accuracy context, and coverage."""
    spectral_rows = []
    classical_rows = []
    accuracy_rows = []
    coverage = []
    for family in families:
        print(f"[{family}]", flush=True)
        for size, (X_train, y_train, _, _) in sorted(
            load_family(family, max_size).items()
        ):
            X_train = np.asarray(X_train, dtype=float)
            y_train = np.asarray(y_train)
            n_qubits = X_train.shape[1]
            classical_rows.append(
                {
                    "family": family,
                    "size": int(size),
                    **classical_eda_statistics(X_train, y_train),
                }
            )
            accuracy_rows.append(
                {
                    "family": family,
                    "size": int(size),
                    **accuracy_context(family, size),
                }
            )
            if n_qubits > configuration.max_qubits:
                coverage.append(
                    {
                        "family": family,
                        "size": size,
                        "n_qubits": n_qubits,
                        "status": "skipped_full_operator_budget",
                    }
                )
                print(
                    f"  size={size}: skipped ({n_qubits} qubits > "
                    f"{configuration.max_qubits})",
                    flush=True,
                )
                continue

            spectral_rows.extend(
                profile_variant(
                    family,
                    size,
                    X_train,
                    y_train,
                    configuration,
                )
            )
            coverage.append(
                {
                    "family": family,
                    "size": size,
                    "n_qubits": n_qubits,
                    "status": "profiled",
                }
            )
            print(f"  size={size}: profiled", flush=True)

    return (
        pd.DataFrame(spectral_rows),
        pd.DataFrame(classical_rows),
        pd.DataFrame(accuracy_rows),
        pd.DataFrame(coverage),
    )


def main() -> None:
    """Run PennyLane spectral profiling from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-size", type=int, default=10)
    parser.add_argument("--max-qubits", type=int, default=10)
    parser.add_argument("--coupling-scale", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--angle-scale", type=float, default=1.0)
    parser.add_argument(
        "--families",
        nargs="*",
        default=list(benchmark_datasets.FAMILIES),
    )
    parser.add_argument("--out", default="spectral_benchmark.csv")
    parser.add_argument("--classical-out", default="classical_eda.csv")
    parser.add_argument("--accuracy-out", default="accuracy_context.csv")
    parser.add_argument("--coverage-out", default="spectral_coverage.csv")
    args = parser.parse_args()

    configuration = SpectralConfiguration(
        coupling_scale=args.coupling_scale,
        ridge=args.ridge,
        angle_scale=args.angle_scale,
        max_qubits=args.max_qubits,
    )
    results, classical, accuracy, coverage = run_benchmark(
        args.families,
        max_size=args.max_size,
        configuration=configuration,
    )
    results.to_csv(args.out, index=False)
    classical.to_csv(args.classical_out, index=False)
    accuracy.to_csv(args.accuracy_out, index=False)
    coverage.to_csv(args.coverage_out, index=False)
    print(f"Saved spectral results: {args.out} ({len(results)} class rows)")
    print(f"Saved classical EDA: {args.classical_out} ({len(classical)} variants)")
    print(f"Saved accuracy context: {args.accuracy_out} ({len(accuracy)} variants)")
    print(f"Saved coverage: {args.coverage_out} ({len(coverage)} variants)")


if __name__ == "__main__":
    main()
