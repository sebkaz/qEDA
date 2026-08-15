"""Parse published results from the qml-benchmarks repo (paper/results/) and
build the retrodiction targets: per dataset & size,

  * gap_var = IQPVariationalClassifier - SeparableVariationalClassifier
  * gap_ker = IQPKernelClassifier      - SeparableKernelClassifier
  * gap_qbm = QuantumBoltzmannMachine  - QuantumBoltzmannMachineSeparable
  * adv_q   = best quantum model       - best classical (MLP, SVC)

Each entry is the mean test accuracy over the seeds reported in
*_GridSearchCV-best-hyperparams-results.csv. These are model-family contrasts,
not controlled causal estimates of the effect of entanglement. In particular,
the IQP and separable models differ in more than the presence or absence of an
entangling gate.
"""

import re
import glob
import os
import pandas as pd

REPOSITORY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_ROOT = os.path.join(REPOSITORY_ROOT, "paper", "results")

PAIRS = [
    ("IQPVariationalClassifier", "SeparableVariationalClassifier", "gap_var"),
    ("IQPKernelClassifier", "SeparableKernelClassifier", "gap_ker"),
    ("QuantumBoltzmannMachine", "QuantumBoltzmannMachineSeparable", "gap_qbm"),
]
CLASSICAL = ["MLPClassifier", "SVC"]
QUANTUM = ["IQPVariationalClassifier", "IQPKernelClassifier",
           "QuantumBoltzmannMachine", "DataReuploadingClassifier",
           "CircuitCentricClassifier", "DressedQuantumCircuitClassifier",
           "ProjectedQuantumKernel", "QuantumMetricLearner",
           "QuantumKitchenSinks", "TreeTensorClassifier"]

# regex extracting the size parameter from the dataset tag in the filename
SIZE_PATTERNS = {
    "linearly_separable": r"linearly_separable_(\d+)d",
    "hidden_manifold": r"hidden_manifold-6manifold-(\d+)d",
    "hidden_manifold_diff": r"hidden_manifold-10d-(\d+)manifold",
    "two_curves": r"two_curves-5degree-0\.1offset-(\d+)d",
    "two_curves_diff": r"two_curves-10d-(\d+)degree",
    "hyperplanes_diff": r"hyperplanes-10d-from3d-(\d+)n",
    "bars_and_stripes": r"bars_and_stripes_(\d+)_x",
}


def _mean_test_acc(path):
    try:
        df = pd.read_csv(path)
        return float(df["test_acc"].mean())
    except Exception:
        return None


def collect(family, results_root=RESULTS_ROOT):
    """-> DataFrame indexed by size with one column per model."""
    pat = SIZE_PATTERNS[family]
    rows = {}
    fam_dir = os.path.join(results_root, family)
    if not os.path.isdir(fam_dir):
        return pd.DataFrame()
    for model_dir in sorted(os.listdir(fam_dir)):
        for f in glob.glob(os.path.join(fam_dir, model_dir,
                                        "*best-hyperparams-results.csv")):
            m = re.search(pat, os.path.basename(f))
            if not m:
                continue
            size = int(m.group(1))
            acc = _mean_test_acc(f)
            if acc is not None:
                rows.setdefault(size, {})[model_dir] = acc
    return pd.DataFrame(rows).T.sort_index()


def targets(family, results_root=RESULTS_ROOT):
    """-> DataFrame with gap_var / gap_ker / gap_qbm / adv_q per size."""
    df = collect(family, results_root)
    if df.empty:
        return df
    out = pd.DataFrame(index=df.index)
    for ent, sep, name in PAIRS:
        if ent in df.columns and sep in df.columns:
            out[name] = df[ent] - df[sep]
    q = [m for m in QUANTUM if m in df.columns]
    c = [m for m in CLASSICAL if m in df.columns]
    if q and c:
        out["adv_q"] = df[q].max(axis=1) - df[c].max(axis=1)
    out["best_quantum"] = df[q].max(axis=1) if q else None
    out["best_classical"] = df[c].max(axis=1) if c else None
    return out
