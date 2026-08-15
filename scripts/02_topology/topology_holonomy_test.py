"""Matched-moment topology and holonomy test for operator qEDA.

The experiment compares a ring-like cloud with a filled-disk cloud.  An
invertible affine recolouring makes their empirical means and covariance
matrices identical to numerical precision.  Both datasets are then analysed
with the same angle map and with two encodings:

* the real product control ``J = 0``;
* the symmetric ``RY-ZZ-RY`` sandwich with a coupling obtained from partial
  correlations and one global scale ``alpha``.

The test deliberately separates two questions.  A permutation test asks
whether the current operator summaries distinguish the matched-moment point
clouds.  A family of closed loops then asks whether the sandwich has geometric
holonomy and whether that holonomy is stable under contraction.  Nonzero loop
phase alone is not interpreted as a certificate of a topological hole.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qeda-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/qeda-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
from numpy.typing import NDArray

from qeda.engine import (
    class_operator_from_states,
    log_negativity,
    mass_gap,
    maximum_current,
    mutual_information,
    profile_from_density,
    purity,
    third_moment,
)

RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

SEED = 1729
N_SAMPLES = 300
N_LOOP_POINTS = 300
N_PERMUTATIONS = 500
RIDGE = 1e-3
ALPHA = 0.8
ANGLE_MARGIN = 0.1

DATA_DIR = Path("results/data")
FIGURE_DIR = Path("results/figures")

TARGET_COVARIANCE = np.array(
    [
        [1.00, 0.65, 0.15],
        [0.65, 1.00, 0.45],
        [0.15, 0.45, 1.00],
    ],
    dtype=float,
)

PROFILE_METRICS = (
    "purity",
    "mass_gap",
    "third_moment",
    "I",
    "Q",
    "current",
)


@dataclass(frozen=True)
class AngleMap:
    """Common affine map from feature coordinates to rotation angles."""

    lower: RealArray
    upper: RealArray
    margin: float = ANGLE_MARGIN

    def transform(self, values: RealArray) -> RealArray:
        """Map values to the common interval ``[margin, pi - margin]``."""
        scaled = (values - self.lower) / (self.upper - self.lower)
        return self.margin + scaled * (np.pi - 2.0 * self.margin)


def _symmetric_inverse_sqrt(matrix: RealArray) -> RealArray:
    """Return the inverse square root of a positive-definite matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if np.min(eigenvalues) <= 0.0:
        raise ValueError("matrix must be positive definite")
    return eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T


def match_mean_and_covariance(
    values: RealArray,
    target_covariance: RealArray,
) -> RealArray:
    """Recolour rows to zero mean and the declared target covariance."""
    centered = values - values.mean(axis=0)
    empirical_covariance = np.cov(centered, rowvar=False)
    whitened = centered @ _symmetric_inverse_sqrt(empirical_covariance)
    target_cholesky = np.linalg.cholesky(target_covariance)
    return whitened @ target_cholesky.T


def make_matched_clouds(
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[RealArray, RealArray, RealArray, RealArray]:
    """Generate ring and filled-disk clouds with matched first two moments."""
    angles = 2.0 * np.pi * np.arange(n_samples) / n_samples
    vertical = np.linspace(-1.0, 1.0, n_samples)
    vertical = vertical[rng.permutation(n_samples)]

    ring_latent = np.column_stack(
        (np.cos(angles), np.sin(angles), 0.25 * vertical)
    )

    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    disk_angles = (golden_angle * np.arange(n_samples)) % (2.0 * np.pi)
    disk_radii = np.sqrt((np.arange(n_samples) + 0.5) / n_samples)
    disk_latent = np.column_stack(
        (
            disk_radii * np.cos(disk_angles),
            disk_radii * np.sin(disk_angles),
            0.25 * vertical[::-1],
        )
    )

    ring = match_mean_and_covariance(ring_latent, TARGET_COVARIANCE)
    disk = match_mean_and_covariance(disk_latent, TARGET_COVARIANCE)
    return ring, disk, ring_latent, disk_latent


def partial_correlation_coupling(
    covariance: RealArray,
    alpha: float,
    ridge: float,
) -> RealArray:
    """Construct ``J`` from partial correlations at one global scale."""
    precision = np.linalg.inv(covariance + ridge * np.eye(len(covariance)))
    diagonal_scale = np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    partial_correlation = -precision / diagonal_scale
    np.fill_diagonal(partial_correlation, 0.0)
    return alpha * partial_correlation


def make_encoder(n_qubits: int):
    """Create the canonical PennyLane ``RY-ZZ-RY`` sandwich."""
    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device, interface=None, diff_method=None)
    def encode(angles: RealArray, coupling: RealArray):
        # Apply the first half of the local feature map.
        for qubit in range(n_qubits):
            qml.RY(angles[qubit] / 2.0, wires=qubit)

        # PennyLane uses IsingZZ(phi) = exp(-i phi Z_j Z_k / 2).
        for j, k in combinations(range(n_qubits), 2):
            qml.IsingZZ(2.0 * coupling[j, k], wires=[j, k])

        # Apply the closing half, including for the exact J = 0 control.
        for qubit in range(n_qubits):
            qml.RY(angles[qubit] / 2.0, wires=qubit)
        return qml.state()

    return encode


def encode_states(
    angles: RealArray,
    coupling: RealArray,
    encoder,
) -> ComplexArray:
    """Encode rows and return states with observations along axis zero."""
    return np.asarray(
        [np.asarray(encoder(row, coupling), dtype=complex) for row in angles]
    )


def state_outer_products(states: ComplexArray) -> ComplexArray:
    """Return one rank-one density operator for every encoded row."""
    return np.einsum("ni,nj->nij", states, states.conj())


def scalar_profile(rho: ComplexArray) -> dict[str, float]:
    """Return the scalar summaries used in the permutation comparison."""
    profile = profile_from_density(rho)
    return {metric: float(profile[metric]) for metric in PROFILE_METRICS}


def _profile_vector(rho: ComplexArray) -> RealArray:
    """Compute the fixed metric vector without allocating profile metadata."""
    return np.array(
        [
            purity(rho),
            mass_gap(rho),
            third_moment(rho),
            mutual_information(rho),
            log_negativity(rho),
            maximum_current(rho),
        ],
        dtype=float,
    )


def permutation_comparison(
    ring_outer: ComplexArray,
    disk_outer: ComplexArray,
    encoding_name: str,
    rng: np.random.Generator,
    n_permutations: int,
) -> tuple[pd.DataFrame, RealArray]:
    """Compare profiles under random reassignment of rows to the two clouds."""
    n_samples = len(ring_outer)
    observed = _profile_vector(ring_outer.mean(axis=0)) - _profile_vector(
        disk_outer.mean(axis=0)
    )
    pooled = np.concatenate((ring_outer, disk_outer), axis=0)
    null = np.empty((n_permutations, len(PROFILE_METRICS)), dtype=float)

    for replicate in range(n_permutations):
        order = rng.permutation(2 * n_samples)
        first = pooled[order[:n_samples]].mean(axis=0)
        second = pooled[order[n_samples:]].mean(axis=0)
        null[replicate] = _profile_vector(first) - _profile_vector(second)

    rows = []
    for index, metric in enumerate(PROFILE_METRICS):
        two_sided_p = (
            1.0
            + np.count_nonzero(np.abs(null[:, index]) >= abs(observed[index]))
        ) / (n_permutations + 1.0)
        null_standard_deviation = float(np.std(null[:, index], ddof=1))
        rows.append(
            {
                "encoding": encoding_name,
                "metric": metric,
                "ring_minus_disk": observed[index],
                "null_q025": np.quantile(null[:, index], 0.025),
                "null_q975": np.quantile(null[:, index], 0.975),
                "permutation_p_two_sided": two_sided_p,
                "null_standardized_difference": (
                    observed[index] / null_standard_deviation
                    if null_standard_deviation > 0.0
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows), null


def wrapped_phase_difference(first: float, second: float) -> float:
    """Return the absolute difference between two angles modulo ``2 pi``."""
    return float(abs(np.angle(np.exp(1j * (first - second)))))


def bargmann_loop(states: ComplexArray) -> tuple[float, float]:
    """Return phase and magnitude of the closed discrete Bargmann product."""
    overlaps = np.array(
        [
            np.vdot(states[index], states[(index + 1) % len(states)])
            for index in range(len(states))
        ]
    )
    phase = float(np.angle(np.exp(1j * np.sum(np.angle(overlaps)))))
    log_magnitude = float(np.sum(np.log(np.maximum(np.abs(overlaps), 1e-300))))
    return phase, float(np.exp(log_magnitude))


def loop_scan(
    encoder,
    coupling: RealArray,
    angle_map: AngleMap,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Measure holonomy while contracting a common family of closed loops."""
    target_cholesky = np.linalg.cholesky(TARGET_COVARIANCE)
    loop_angles = 2.0 * np.pi * np.arange(N_LOOP_POINTS) / N_LOOP_POINTS
    rows = []

    for radius in np.linspace(0.1, 1.2, 12):
        latent_loop = np.column_stack(
            (
                radius * np.cos(loop_angles),
                radius * np.sin(loop_angles),
                np.zeros(N_LOOP_POINTS),
            )
        )
        feature_loop = latent_loop @ target_cholesky.T
        encoded_angles = angle_map.transform(feature_loop)

        for encoding_name, active_coupling in (
            ("J0", np.zeros_like(coupling)),
            ("J_partial", coupling),
        ):
            states = encode_states(encoded_angles, active_coupling, encoder)
            phase, magnitude = bargmann_loop(states)

            # A row-dependent phase is a gauge change and must cancel around
            # the closed Bargmann product.
            gauge_angles = rng.uniform(-np.pi, np.pi, len(states))
            gauge_states = states * np.exp(1j * gauge_angles)[:, None]
            gauge_phase, _ = bargmann_loop(gauge_states)

            reversed_phase, _ = bargmann_loop(states[::-1])
            rows.append(
                {
                    "encoding": encoding_name,
                    "radius": radius,
                    "bargmann_phase": phase,
                    "bargmann_magnitude": magnitude,
                    "gauge_phase_error": wrapped_phase_difference(
                        phase, gauge_phase
                    ),
                    "orientation_reversal_error": wrapped_phase_difference(
                        phase, -reversed_phase
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_figure(
    ring_latent: RealArray,
    disk_latent: RealArray,
    coupling: RealArray,
    permutation_results: pd.DataFrame,
    loop_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create a compact diagnostic figure for the complete experiment."""
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.5))

    axes[0, 0].scatter(
        disk_latent[:, 0],
        disk_latent[:, 1],
        s=10,
        alpha=0.55,
        label="filled disk",
    )
    axes[0, 0].scatter(
        ring_latent[:, 0],
        ring_latent[:, 1],
        s=10,
        alpha=0.75,
        label="ring",
    )
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set_title("Different support topology")
    axes[0, 0].set_xlabel("latent coordinate 1")
    axes[0, 0].set_ylabel("latent coordinate 2")
    axes[0, 0].legend(frameon=False)

    image = axes[0, 1].imshow(
        coupling,
        cmap="coolwarm",
        vmin=-ALPHA,
        vmax=ALPHA,
    )
    axes[0, 1].set_title(r"Common $J=\alpha R_{\mathrm{partial}}$")
    axes[0, 1].set_xlabel("mode")
    axes[0, 1].set_ylabel("mode")
    figure.colorbar(image, ax=axes[0, 1], shrink=0.82)

    for encoding_name, group in loop_results.groupby("encoding"):
        axes[1, 0].plot(
            group["radius"],
            group["bargmann_phase"],
            marker="o",
            label=encoding_name,
        )
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_title("Loop phase contracts continuously")
    axes[1, 0].set_xlabel("loop radius")
    axes[1, 0].set_ylabel("Bargmann phase [rad]")
    axes[1, 0].legend(frameon=False)

    metric_positions = np.arange(len(PROFILE_METRICS))
    width = 0.36
    for offset, (encoding_name, group) in zip(
        (-width / 2.0, width / 2.0),
        permutation_results.groupby("encoding", sort=False),
    ):
        ordered = group.set_index("metric").loc[list(PROFILE_METRICS)]
        axes[1, 1].bar(
            metric_positions + offset,
            ordered["null_standardized_difference"],
            width=width,
            label=encoding_name,
        )
    axes[1, 1].axhline(1.96, color="black", linestyle="--", linewidth=0.8)
    axes[1, 1].axhline(-1.96, color="black", linestyle="--", linewidth=0.8)
    axes[1, 1].set_xticks(metric_positions)
    axes[1, 1].set_xticklabels(PROFILE_METRICS, rotation=35, ha="right")
    axes[1, 1].set_ylabel("ring-disk difference / null SD")
    axes[1, 1].set_title("No profile statistic crosses the null band")
    axes[1, 1].legend(frameon=False)

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Run the experiment and write data tables plus a diagnostic figure."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    ring, disk, ring_latent, disk_latent = make_matched_clouds(N_SAMPLES, rng)
    mean_error = float(np.max(np.abs(ring.mean(axis=0) - disk.mean(axis=0))))
    covariance_error = float(
        np.max(
            np.abs(
                np.cov(ring, rowvar=False) - np.cov(disk, rowvar=False)
            )
        )
    )

    pooled = np.concatenate((ring, disk), axis=0)
    angle_map = AngleMap(lower=pooled.min(axis=0), upper=pooled.max(axis=0))
    ring_angles = angle_map.transform(ring)
    disk_angles = angle_map.transform(disk)

    coupling = partial_correlation_coupling(
        TARGET_COVARIANCE,
        alpha=ALPHA,
        ridge=RIDGE,
    )
    encoder = make_encoder(n_qubits=ring.shape[1])

    profile_rows = []
    permutation_tables = []
    for encoding_name, active_coupling in (
        ("J0", np.zeros_like(coupling)),
        ("J_partial", coupling),
    ):
        ring_states = encode_states(ring_angles, active_coupling, encoder)
        disk_states = encode_states(disk_angles, active_coupling, encoder)
        ring_outer = state_outer_products(ring_states)
        disk_outer = state_outer_products(disk_states)

        for dataset_name, outer_products in (
            ("ring", ring_outer),
            ("filled_disk", disk_outer),
        ):
            profile_rows.append(
                {
                    "dataset": dataset_name,
                    "encoding": encoding_name,
                    **scalar_profile(outer_products.mean(axis=0)),
                }
            )

        permutation_table, _ = permutation_comparison(
            ring_outer,
            disk_outer,
            encoding_name=encoding_name,
            rng=rng,
            n_permutations=N_PERMUTATIONS,
        )
        permutation_tables.append(permutation_table)

    profiles = pd.DataFrame(profile_rows)
    permutation_results = pd.concat(permutation_tables, ignore_index=True)
    loop_results = loop_scan(encoder, coupling, angle_map, rng)

    profiles["mean_match_max_abs_error"] = mean_error
    profiles["covariance_match_max_abs_error"] = covariance_error
    profiles.to_csv(DATA_DIR / "topology_holonomy_profiles.csv", index=False)
    permutation_results.to_csv(
        DATA_DIR / "topology_holonomy_permutation.csv",
        index=False,
    )
    loop_results.to_csv(DATA_DIR / "topology_holonomy_loops.csv", index=False)
    np.savetxt(DATA_DIR / "topology_holonomy_coupling.csv", coupling, delimiter=",")

    make_figure(
        ring_latent,
        disk_latent,
        coupling,
        permutation_results,
        loop_results,
        FIGURE_DIR / "topology_holonomy_test.pdf",
    )

    print("Matched-moment ring versus filled disk")
    print(f"  max mean mismatch:       {mean_error:.3e}")
    print(f"  max covariance mismatch: {covariance_error:.3e}")
    print("\nOperator profiles")
    print(profiles.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print("\nPermutation comparison")
    print(
        permutation_results[
            [
                "encoding",
                "metric",
                "ring_minus_disk",
                "permutation_p_two_sided",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.6g}")
    )
    print("\nLoop controls")
    print(
        "  max gauge error:       "
        f"{loop_results['gauge_phase_error'].max():.3e}"
    )
    print(
        "  max orientation error: "
        f"{loop_results['orientation_reversal_error'].max():.3e}"
    )
    coupled_loops = loop_results[loop_results["encoding"] == "J_partial"]
    print(
        "  coupled phase range:   "
        f"[{coupled_loops['bargmann_phase'].min():.6g}, "
        f"{coupled_loops['bargmann_phase'].max():.6g}] rad"
    )


if __name__ == "__main__":
    main()
