"""Characterise Iris cross-validated model-error and boundary cases.

The seven rows highlighted by ``e4_model_errors.py`` are the union of
out-of-fold mistakes from three supervised probes.  That construction makes
them difficult classification cases; it does *not* by itself make them
statistical outliers.  This script keeps the definitions separate and does
not rebrand the classifier-error union as an anomaly detector.

It writes one descriptive row-level table containing:

* the original five-fold paper-error flag and the probes responsible for it;
* repeated cross-validated error rates for six classifiers;
* global Isolation Forest and Local Outlier Factor ranks;
* class-conditional robust Mahalanobis and IQR diagnostics; and
* the fraction of other-class points among the 15 nearest neighbours.

The last three items use all labelled Iris rows and are therefore explanatory
descriptions of this reference dataset, not an out-of-sample anomaly detector.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, rankdata
from sklearn.covariance import MinCovDet
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PAPER_SEED = 17
DEFAULT_SEED = 17


def paper_probes() -> dict[str, object]:
    """Return exactly the three fixed E4 supervised error probes."""
    return {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=PAPER_SEED),
        ),
        "lda": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "rbf_svm": make_pipeline(StandardScaler(), SVC(kernel="rbf")),
    }


def extended_probes(seed: int, forest_trees: int) -> dict[str, object]:
    """Return complementary linear, local, nonlinear, and quadratic probes."""
    return {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=seed),
        ),
        "lda": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "rbf_svm": make_pipeline(StandardScaler(), SVC(kernel="rbf")),
        "knn_9": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=9)),
        "random_forest": RandomForestClassifier(
            n_estimators=forest_trees,
            random_state=seed,
            n_jobs=1,
        ),
        "qda": QuadraticDiscriminantAnalysis(reg_param=0.05),
    }


def cross_validated_errors(
    x: np.ndarray,
    y: np.ndarray,
    probes: dict[str, object],
    splitter: object,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Count out-of-fold errors for every row and probe."""
    counts = {name: np.zeros(len(y), dtype=int) for name in probes}
    tests = np.zeros(len(y), dtype=int)
    for train, test in splitter.split(x, y):
        tests[test] += 1
        for name, probe in probes.items():
            probe.fit(x[train], y[train])
            counts[name][test] += probe.predict(x[test]) != y[test]
    if np.any(tests == 0):
        raise RuntimeError("Cross-validation did not test every row")
    return counts, tests


def density_and_boundary_diagnostics(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    forest_trees: int,
) -> dict[str, np.ndarray]:
    """Compute descriptive global, within-class, and boundary diagnostics."""
    z = StandardScaler().fit_transform(x)
    isolation = IsolationForest(
        n_estimators=forest_trees,
        random_state=seed,
    ).fit(z)
    isolation_score = -isolation.score_samples(z)
    local = LocalOutlierFactor(n_neighbors=20).fit(z)
    local_score = -local.negative_outlier_factor_

    _, neighbours = NearestNeighbors(n_neighbors=16).fit(z).kneighbors(z)
    other_class_fraction = np.array(
        [np.mean(y[row[1:]] != y[index]) for index, row in enumerate(neighbours)]
    )

    mcd_distance = np.full(len(y), np.nan)
    mcd_p_value = np.full(len(y), np.nan)
    iqr_flags = np.zeros(len(y), dtype=int)
    for label in np.unique(y):
        mask = y == label
        class_z = z[mask]
        estimator = MinCovDet(support_fraction=0.75, random_state=seed).fit(class_z)
        distance = estimator.mahalanobis(class_z)
        mcd_distance[mask] = distance
        mcd_p_value[mask] = chi2.sf(distance, df=x.shape[1])

        lower = np.quantile(class_z, 0.25, axis=0)
        upper = np.quantile(class_z, 0.75, axis=0)
        spread = upper - lower
        flags = (class_z < lower - 1.5 * spread) | (class_z > upper + 1.5 * spread)
        iqr_flags[mask] = flags.sum(axis=1)

    return {
        "isolation_score": isolation_score,
        "isolation_rank": rankdata(-isolation_score, method="min").astype(int),
        "lof_score": local_score,
        "lof_rank": rankdata(-local_score, method="min").astype(int),
        "mcd_distance": mcd_distance,
        "mcd_p_value": mcd_p_value,
        "iqr_feature_flags": iqr_flags,
        "other_class_fraction_15nn": other_class_fraction,
    }


def analyse(seed: int, repeats: int, forest_trees: int) -> pd.DataFrame:
    """Build the complete diagnostic table for all 150 Iris observations."""
    dataset = load_iris()
    x = np.asarray(dataset.data, dtype=float)
    y = np.asarray(dataset.target, dtype=int)

    paper_counts, paper_tests = cross_validated_errors(
        x,
        y,
        paper_probes(),
        StratifiedKFold(n_splits=5, shuffle=True, random_state=PAPER_SEED),
    )
    repeated_counts, repeated_tests = cross_validated_errors(
        x,
        y,
        extended_probes(seed, forest_trees),
        RepeatedStratifiedKFold(
            n_splits=5,
            n_repeats=repeats,
            random_state=seed,
        ),
    )
    diagnostics = density_and_boundary_diagnostics(x, y, seed, forest_trees)

    paper_models = [
        ";".join(name for name, errors in paper_counts.items() if errors[index])
        for index in range(len(y))
    ]
    frame = pd.DataFrame(
        {
            "row_zero_based": np.arange(len(y)),
            "row_one_based": np.arange(1, len(y) + 1),
            "species": dataset.target_names[y],
            "sepal_length_cm": x[:, 0],
            "sepal_width_cm": x[:, 1],
            "petal_length_cm": x[:, 2],
            "petal_width_cm": x[:, 3],
            "paper_error_union": np.any(
                np.column_stack(tuple(paper_counts.values())) > 0,
                axis=1,
            ),
            "paper_error_models": paper_models,
            "paper_cv_tests": paper_tests,
        }
    )
    for name, errors in repeated_counts.items():
        frame[f"{name}_error_rate"] = errors / repeated_tests
    for name, value in diagnostics.items():
        frame[name] = value
    return frame


def print_paper_cases(frame: pd.DataFrame) -> None:
    """Print the seven manuscript cases in a compact diagnostic table."""
    cases = frame.loc[frame["paper_error_union"]].copy()
    columns = [
        "row_zero_based",
        "species",
        "paper_error_models",
        "logistic_error_rate",
        "rbf_svm_error_rate",
        "random_forest_error_rate",
        "isolation_rank",
        "lof_rank",
        "mcd_p_value",
        "iqr_feature_flags",
        "other_class_fraction_15nn",
    ]
    print(cases.loc[:, columns].to_string(index=False, float_format="%.3f"))
    boundary = cases["other_class_fraction_15nn"].mean()
    remaining = frame.loc[~frame["paper_error_union"], "other_class_fraction_15nn"].mean()
    print(
        "\nMean other-class fraction among 15 nearest neighbours: "
        f"paper-error rows={boundary:.3f}, remaining rows={remaining:.3f}."
    )


def parse_args() -> argparse.Namespace:
    """Parse reproducibility settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--forest-trees", type=int, default=300)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/data/iris_classification_boundary_cases.csv"),
    )
    return parser.parse_args()


def main() -> None:
    """Run the typology and save its frozen row-level table."""
    args = parse_args()
    frame = analyse(args.seed, args.repeats, args.forest_trees)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print_paper_cases(frame)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
