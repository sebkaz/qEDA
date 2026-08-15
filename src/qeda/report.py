"""Compact reports for the three qEDA readings of a declared encoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .engine import Encoding, RealArray, class_operator_from_states, encode_rows, n_qubits, profile_from_density


_SCALAR_KEYS = (
    "purity",
    "dispersion",
    "effective_rank",
    "mass_gap",
    "third_moment",
    "entropy_bits",
    "I",
    "Q",
    "current",
    "mode_participation",
)


@dataclass(frozen=True)
class QEDAReport:
    """The spectral, subsystem, and mode-coherence profile of one encoding."""

    name: str
    n_samples: int
    n_features: int
    n_qubits: int
    profile: dict[str, Any]

    def scalars(self) -> dict[str, float]:
        """Return the scalar quantities suitable for a comparison table."""
        return {key: float(self.profile[key]) for key in _SCALAR_KEYS}

    def to_markdown(self) -> str:
        """Render a human-readable report grouped by the three qEDA readings."""
        scalars = self.scalars()
        spectrum = np.asarray(self.profile["spectrum"])
        occupations = np.asarray(self.profile["occupations"])
        current = np.asarray(self.profile["current_matrix"])
        lines = [
            f"# qEDA report: {self.name}",
            "",
            f"Rows: {self.n_samples}; features/qubits: {self.n_features}/{self.n_qubits}.",
            "",
            "## Spectral reading",
            f"- purity: {scalars['purity']:.6f}",
            f"- dispersion: {scalars['dispersion']:.6f}",
            f"- effective rank: {scalars['effective_rank']:.6f}",
            f"- mass gap: {scalars['mass_gap']:.6f}",
            f"- third moment: {scalars['third_moment']:.6f}",
            "- spectrum: " + np.array2string(spectrum, precision=6),
            "",
            "## Declared-subsystem reading",
            f"- mean mutual information: {scalars['I']:.6f}",
            f"- mean logarithmic negativity: {scalars['Q']:.6f}",
            "",
            "## Feature-mode reading",
            f"- maximum encoded current: {scalars['current']:.6f}",
            f"- mode participation: {scalars['mode_participation']:.6f}",
            "- occupations: " + np.array2string(occupations, precision=6),
            "- current matrix:\n```\n" + np.array2string(current, precision=6) + "\n```",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class MatchedQEDAReport:
    """Two reports and their candidate-minus-control scalar contrasts."""

    control: QEDAReport
    candidate: QEDAReport

    def scalar_contrasts(self) -> dict[str, float]:
        """Return signed candidate-minus-control scalar contrasts."""
        control = self.control.scalars()
        candidate = self.candidate.scalars()
        return {key: candidate[key] - control[key] for key in _SCALAR_KEYS}

    def to_markdown(self) -> str:
        """Render the representation decision table for a matched comparison."""
        control = self.control.scalars()
        candidate = self.candidate.scalars()
        lines = [
            f"# Matched qEDA audit: {self.control.name} vs {self.candidate.name}",
            "",
            "| statistic | control | candidate | candidate - control |",
            "| --- | ---: | ---: | ---: |",
        ]
        for key in _SCALAR_KEYS:
            lines.append(
                f"| {key} | {control[key]:.6f} | {candidate[key]:.6f} | "
                f"{candidate[key] - control[key]:.6f} |"
            )
        return "\n".join(lines)


def audit(data: RealArray, encoding: Encoding, *, name: str = "encoding") -> QEDAReport:
    """Run all three qEDA readings for a declared statevector encoding."""
    rows = np.asarray(data, dtype=float)
    states = encode_rows(rows, encoding)
    rho = class_operator_from_states(states)
    return QEDAReport(
        name=name,
        n_samples=rows.shape[0],
        n_features=rows.shape[1],
        n_qubits=n_qubits(rho),
        profile=profile_from_density(rho),
    )


def matched_audit(
    data: RealArray,
    control_encoding: Encoding,
    candidate_encoding: Encoding,
    *,
    control_name: str = "control",
    candidate_name: str = "candidate",
) -> MatchedQEDAReport:
    """Compare two encodings of exactly the same rows in one audit report."""
    return MatchedQEDAReport(
        control=audit(data, control_encoding, name=control_name),
        candidate=audit(data, candidate_encoding, name=candidate_name),
    )
