"""Generate a readable report comparing classical and d-QSA spectral EDA.

The report treats the Bowles datasets as the object of analysis. Published
accuracies appear only in a final context appendix and are never used as
targets, weights, or inputs to the descriptive statistics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


FAMILY_LABELS = {
    "linearly_separable": "Linearly separable",
    "hidden_manifold": "Hidden manifold",
    "hidden_manifold_diff": "Hidden manifold (difficulty)",
    "two_curves": "Two curves",
    "two_curves_diff": "Two curves (difficulty)",
    "hyperplanes_diff": "Hyperplanes parity",
    "bars_and_stripes": "Bars and stripes",
}

COLORS = {
    family: color
    for family, color in zip(
        FAMILY_LABELS,
        ["#0072B2", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#CC79A7", "#D55E00"],
    )
}


def _footer(figure: plt.Figure, page: int) -> None:
    figure.text(
        0.5,
        0.006,
        f"Bowles datasets: classical and d-QSA spectral EDA  |  {page}",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#666666",
    )


def _title(figure: plt.Figure, title: str, subtitle: str | None = None) -> None:
    figure.suptitle(title, x=0.06, y=0.965, ha="left", fontsize=18, fontweight="bold")
    if subtitle:
        figure.text(0.06, 0.925, subtitle, ha="left", fontsize=10, color="#444444")


def _save(pdf: PdfPages, figure: plt.Figure, page: int) -> None:
    _footer(figure, page)
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def _comparison_table(spectral: pd.DataFrame, classical: pd.DataFrame) -> pd.DataFrame:
    quantum = (
        spectral.groupby(["family", "size"], as_index=False)
        .agg(
            purity_J0=("purity_J0", "mean"),
            purity_J=("purity_J", "mean"),
            delta_purity=("delta_purity", "mean"),
            effective_rank_J0=("effective_rank_J0", "mean"),
            effective_rank_J=("effective_rank_J", "mean"),
            delta_effective_rank=("delta_effective_rank", "mean"),
            entropy_bits_J0=("entropy_bits_J0", "mean"),
            entropy_bits_J=("entropy_bits_J", "mean"),
            delta_entropy_bits=("delta_entropy_bits", "mean"),
            mass_gap_J0=("mass_gap_J0", "mean"),
            mass_gap_J=("mass_gap_J", "mean"),
            delta_mass_gap=("delta_mass_gap", "mean"),
            delta_third_moment=("delta_third_moment", "mean"),
        )
    )
    comparison = classical.merge(quantum, on=["family", "size"], how="left")
    comparison["correlation_rank_fraction"] = (
        comparison["correlation_effective_rank"] / comparison["n_features"]
    )
    return comparison


def _family_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    return (
        comparison.groupby("family", as_index=False)
        .agg(
            variants=("size", "count"),
            median_abs_r=("mean_abs_correlation", "median"),
            median_corr_rank_fraction=("correlation_rank_fraction", "median"),
            median_fisher=("fisher_trace_ratio", "median"),
            median_delta_purity=("delta_purity", "median"),
            median_delta_mass_gap=("delta_mass_gap", "median"),
        )
        .sort_values("family")
    )


def _metric_table(axis: plt.Axes, summary: pd.DataFrame) -> None:
    display = summary.copy()
    display["family"] = display["family"].map(FAMILY_LABELS)
    display.columns = [
        "Dataset family",
        "n",
        "median |r|",
        "corr. rank / d",
        "Fisher ratio",
        "median Δ purity",
        "median Δ gap",
    ]
    for column in display.columns[2:]:
        display[column] = display[column].map(
            lambda value: "-" if pd.isna(value) else f"{value:.3f}"
        )
    table = axis.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.26, 0.06, 0.11, 0.13, 0.11, 0.14, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.8)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#DCEAF4")
            cell.set_text_props(fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F4F6F7")
    axis.axis("off")


def _plot_family_trends(
    axes: np.ndarray,
    data: pd.DataFrame,
    metrics: list[tuple[str, str]],
) -> None:
    for axis, (metric, label) in zip(axes, metrics):
        for family, subset in data.groupby("family"):
            subset = subset.sort_values("size")
            axis.plot(
                subset["size"],
                subset[metric],
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=FAMILY_LABELS[family],
                color=COLORS[family],
            )
        axis.set_xlabel("native dimension or difficulty parameter")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)


def _write_intro(pdf: PdfPages, page: int) -> None:
    figure = plt.figure(figsize=(11.69, 8.27))
    _title(
        figure,
        "Bowles datasets as data",
        "Classical EDA versus d-QSA spectral EDA implemented with PennyLane",
    )
    axis = figure.add_axes([0.06, 0.08, 0.88, 0.80])
    axis.axis("off")
    text = (
        "QUESTION\n"
        "What do standard feature-space diagnostics and the proposed spectral instrument reveal "
        "about the same benchmark datasets?\n\n"
        "STANDARD EDA (native training features)\n"
        "Class balance, absolute correlation, the normalised spectrum and effective rank of the "
        "correlation matrix, and the between/within-class Fisher trace ratio. Eigenvalues are "
        "diagnostics only: no PCA projection is applied.\n\n"
        "d-QSA SPECTRAL EDA (class operators)\n"
        "Each row x is encoded with PennyLane as RY(x/2) - IsingZZ(2J_jk) - RY(x/2). "
        "For each class c, rho_c is the empirical mixture of encoded states. We compare the same "
        "local map at J=0 with the partial-correlation coupled map J. Reported changes therefore "
        "describe how the coupling reorganises the spectrum: purity, entropy, effective rank, "
        "mass gap, and the third spectral moment.\n\n"
        "NOT THE RESEARCH QUESTION\n"
        "No classifier is trained in this pipeline. Published accuracies are retained only in a "
        "separate context appendix. They are not used to optimise J, select datasets, or define "
        "the conclusions.\n\n"
        "SCOPE\n"
        "All preprocessing and precision estimates use the original Bowles training split. Native "
        "features are retained. Full density operators are evaluated up to 10 qubits. The 16-feature "
        "Bars-and-Stripes case receives standard EDA but is explicitly excluded from the current "
        "full-operator analysis; it is not reduced by PCA."
    )
    axis.text(0.0, 0.98, text, va="top", fontsize=12, linespacing=1.45, wrap=True)
    _save(pdf, figure, page)


def _write_summary(pdf: PdfPages, page: int, comparison: pd.DataFrame) -> None:
    figure = plt.figure(figsize=(11.69, 8.27))
    _title(
        figure,
        "One table, two descriptive views",
        "Family medians; Δ denotes coupled J minus the matched J=0 encoding",
    )
    axis = figure.add_axes([0.04, 0.24, 0.92, 0.62])
    _metric_table(axis, _family_summary(comparison))
    figure.text(
        0.06,
        0.15,
        "Reading rule: |r| and corr. rank/d describe redundancy in native features; Fisher ratio "
        "describes labelled centroid separation. Δ purity and Δ gap describe spectral sharpening "
        "caused by the fixed coupling. A dash means that the full density operator exceeded the "
        "10-qubit budget.",
        fontsize=10,
        wrap=True,
    )
    _save(pdf, figure, page)


def _write_classical(pdf: PdfPages, page: int, comparison: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(11.69, 8.27))
    figure.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.25, wspace=0.30)
    _title(
        figure,
        "Standard EDA: structure in the native feature space",
        "Correlation spectra are inspected but the observations are never projected",
    )
    _plot_family_trends(
        axes,
        comparison,
        [
            ("mean_abs_correlation", "mean absolute correlation"),
            ("correlation_rank_fraction", "correlation effective rank / d"),
            ("fisher_trace_ratio", "Fisher trace ratio"),
        ],
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    figure.text(
        0.06,
        0.15,
        "The curves and parity families are highly correlated and low-rank in feature space. "
        "Linearly separable data are nearly full-rank and weakly correlated. Fisher separation is "
        "a different axis and should not be inferred from correlation alone.",
        fontsize=10,
        wrap=True,
    )
    _save(pdf, figure, page)


def _write_quantum(pdf: PdfPages, page: int, comparison: pd.DataFrame) -> None:
    quantum = comparison.dropna(subset=["delta_purity"])
    figure, axes = plt.subplots(1, 3, figsize=(11.69, 8.27))
    figure.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.25, wspace=0.30)
    _title(
        figure,
        "New EDA: response of the class-operator spectrum",
        "Class-average changes induced by the partial-correlation coupling",
    )
    _plot_family_trends(
        axes,
        quantum,
        [
            ("delta_purity", "Δ purity"),
            ("delta_effective_rank", "Δ effective rank"),
            ("delta_mass_gap", "Δ mass gap"),
        ],
    )
    axes[0].axhline(0.0, color="#555555", linewidth=0.8)
    axes[1].axhline(0.0, color="#555555", linewidth=0.8)
    axes[2].axhline(0.0, color="#555555", linewidth=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    figure.text(
        0.06,
        0.15,
        "Positive Δ purity together with negative Δ effective rank means that J concentrates "
        "spectral weight into fewer modes. Positive Δ mass gap means that the leading mode becomes "
        "more isolated. These are properties of encoded class mixtures, not accuracy claims.",
        fontsize=10,
        wrap=True,
    )
    _save(pdf, figure, page)


def _write_joint(pdf: PdfPages, page: int, comparison: pd.DataFrame) -> None:
    data = comparison.dropna(subset=["delta_purity"])
    figure, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
    figure.subplots_adjust(left=0.08, right=0.98, top=0.80, bottom=0.30, wspace=0.25)
    _title(
        figure,
        "What the new view adds",
        "The spectral response is not a relabelling of ordinary correlation or centroid separation",
    )
    pairs = [
        ("mean_abs_correlation", "delta_purity", "mean |r|", "Δ purity"),
        ("fisher_trace_ratio", "delta_mass_gap", "Fisher trace ratio", "Δ mass gap"),
    ]
    for axis, (x_name, y_name, x_label, y_label) in zip(axes, pairs):
        for family, subset in data.groupby("family"):
            axis.scatter(
                subset[x_name],
                subset[y_name],
                s=38,
                alpha=0.78,
                label=FAMILY_LABELS[family],
                color=COLORS[family],
                edgecolor="white",
                linewidth=0.4,
            )
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    figure.text(
        0.06,
        0.14,
        "Examples: linearly separable and hyperplanes-parity variants both show strong spectral "
        "sharpening, although their native correlation structures are very different. The d-QSA "
        "view is therefore complementary. The plots are descriptive and do not establish a causal "
        "relation between classical and spectral metrics.",
        fontsize=10,
        wrap=True,
    )
    _save(pdf, figure, page)


def _write_accuracy_context(
    pdf: PdfPages,
    page: int,
    accuracy: pd.DataFrame,
) -> None:
    summary = (
        accuracy.groupby("family", as_index=False)
        .agg(
            variants=("size", "count"),
            model_count_median=("n_published_models", "median"),
            accuracy_min=("published_accuracy_min", "min"),
            accuracy_median=("published_accuracy_median", "median"),
            accuracy_max=("published_accuracy_max", "max"),
        )
        .sort_values("family")
    )
    summary["family"] = summary["family"].map(FAMILY_LABELS)
    summary.columns = [
        "Dataset family",
        "variants",
        "median model count",
        "minimum",
        "median",
        "maximum",
    ]
    for column in ["minimum", "median", "maximum"]:
        summary[column] = summary[column].map(
            lambda value: "-" if pd.isna(value) else f"{value:.3f}"
        )
    summary["median model count"] = summary["median model count"].map(
        lambda value: f"{value:.0f}"
    )

    figure = plt.figure(figsize=(11.69, 8.27))
    _title(
        figure,
        "Context appendix: published accuracy ranges",
        "Reference metadata only - not an outcome of the EDA pipeline",
    )
    axis = figure.add_axes([0.08, 0.25, 0.84, 0.56])
    table = axis.table(
        cellText=summary.values,
        colLabels=summary.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.32, 0.10, 0.18, 0.11, 0.11, 0.11],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#F3E4D7")
            cell.set_text_props(fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F7F7F7")
    axis.axis("off")
    figure.text(
        0.08,
        0.14,
        "Each range pools the model-family accuracies reported by Bowles et al. over the available "
        "variants in that family. It is included only to identify the original benchmark context. "
        "No accuracy value enters J, rho_c, a spectral statistic, or any plot on the preceding pages.",
        fontsize=10,
        wrap=True,
    )
    _save(pdf, figure, page)


def main() -> None:
    """Build the comparison CSV and the verified multi-page PDF source."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectral", default="spectral_benchmark_pennylane.csv")
    parser.add_argument("--classical", default="classical_eda.csv")
    parser.add_argument("--accuracy", default="accuracy_context.csv")
    parser.add_argument("--comparison-out", default="eda_comparison.csv")
    parser.add_argument("--out", default="results/pdf/spectral_eda_report.pdf")
    args = parser.parse_args()

    spectral = pd.read_csv(args.spectral)
    classical = pd.read_csv(args.classical)
    accuracy = pd.read_csv(args.accuracy)
    comparison = _comparison_table(spectral, classical)
    comparison.to_csv(args.comparison_out, index=False)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        _write_intro(pdf, 1)
        _write_summary(pdf, 2, comparison)
        _write_classical(pdf, 3, comparison)
        _write_quantum(pdf, 4, comparison)
        _write_joint(pdf, 5, comparison)
        _write_accuracy_context(pdf, 6, accuracy)
    print(f"Saved comparison: {args.comparison_out} ({len(comparison)} variants)")
    print(f"Saved report: {output} (6 pages)")


if __name__ == "__main__":
    main()
