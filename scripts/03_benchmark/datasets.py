"""Regenerate the benchmark datasets of Bowles, Ahmed & Schuld,
"Better than classical?" (arXiv:2403.07059), exactly as used in the paper.

The paper's generation scripts (qml-benchmarks/paper/benchmarks/generate_*.py)
seed the *global* numpy RNG once and then loop over dataset sizes, consuming a
single random stream (train_test_split with random_state=None also draws from
the global stream). To reproduce the dataset for a given size exactly, we must
replay the loop from the beginning in the same order. This module does that.

Requires: qml_benchmarks installed (pip install -e ./qml-benchmarks).
"""

import importlib.util
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_ROOT = REPOSITORY_ROOT / "src"


def _load_generator(module_name, function_name):
    """Load one NumPy-only generator without importing the full package.

    The package-level ``qml_benchmarks.data`` initializer imports optional JAX
    datasets that are irrelevant to this experiment. Direct module loading
    keeps the retrodiction pipeline independent of those optional dependencies.
    """
    path = SOURCE_ROOT / "qml_benchmarks" / "data" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(
        f"_dqsa_bowles_{module_name}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load benchmark generator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


generate_linearly_separable = _load_generator(
    "linearly_separable", "generate_linearly_separable"
)
generate_hidden_manifold_model = _load_generator(
    "hidden_manifold", "generate_hidden_manifold_model"
)
generate_two_curves = _load_generator("two_curves", "generate_two_curves")
generate_hyperplanes_parity = _load_generator(
    "hyperplanes", "generate_hyperplanes_parity"
)
generate_bars_and_stripes = _load_generator(
    "bars_and_stripes", "generate_bars_and_stripes"
)




def _split(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)
    return (np.asarray(Xtr, dtype=float), np.asarray(Xte, dtype=float),
            np.asarray(ytr), np.asarray(yte))


def linearly_separable(max_d=10):
    """Paper: seed 42, n_samples=300, margin=0.02*d, d=2..20."""
    np.random.seed(42)
    out = {}
    for d in range(2, 21):
        margin = 0.02 * d
        X, y = generate_linearly_separable(300, d, margin)
        Xtr, Xte, ytr, yte = _split(X, y)
        if d <= max_d:
            out[d] = (Xtr, ytr, Xte, yte)
        if d >= max_d:
            break  # later draws never affect earlier ones
    return out


def hidden_manifold(max_d=10):
    """Paper: seed 3, n_samples=300, manifold_dimension=6, d=2..20."""
    np.random.seed(3)
    out = {}
    for d in range(2, 21):
        X, y = generate_hidden_manifold_model(300, d, 6)
        Xtr, Xte, ytr, yte = _split(X, y)
        if d <= max_d:
            out[d] = (Xtr, ytr, Xte, yte)
        if d >= max_d:
            break
    return out


def two_curves(max_d=10):
    """Paper: seed 3, n_samples=300, degree=5, offset=0.1, noise=0.01, d=2..20."""
    np.random.seed(3)
    out = {}
    for d in range(2, 21):
        X, y = generate_two_curves(300, d, 5, 0.1, 0.01)
        Xtr, Xte, ytr, yte = _split(X, y)
        if d <= max_d:
            out[d] = (Xtr, ytr, Xte, yte)
        if d >= max_d:
            break
    return out


def hyperplanes_diff(max_n=10):
    """Paper: seed 1, n_samples=300, d=10, dim_hyperplanes=3, n_hyperplanes=2..20.

    Difficulty knob is the number of hyperplanes (parity of side labels);
    all datasets are 10-dimensional -> 10 qubits.
    """
    np.random.seed(1)
    out = {}
    for n_hyp in range(2, 21):
        X, y = generate_hyperplanes_parity(300, 10, n_hyp, 3)
        Xtr, Xte, ytr, yte = _split(X, y)
        if n_hyp <= max_n:
            out[n_hyp] = (Xtr, ytr, Xte, yte)
        if n_hyp >= max_n:
            break
    return out


def bars_and_stripes(sizes=(4,)):
    """Paper: seed 42 (re-seeded per size), 1000 train / 200 test, noise 0.5.

    4x4 -> 16 features (16 qubits): pairwise-marginal diagnostics only.
    """
    out = {}
    for size in sizes:
        np.random.seed(42)
        Xtr, ytr = generate_bars_and_stripes(1000, size, size, 0.5)
        Xte, yte = generate_bars_and_stripes(200, size, size, 0.5)
        out[size] = (
            Xtr.reshape(len(Xtr), -1), ytr,
            Xte.reshape(len(Xte), -1), yte,
        )
    return out


def hidden_manifold_diff(max_n=20):
    """Paper: seed 3, n_samples=300, n_features=10, manifold_dim=2..20.

    Difficulty knob is manifold dimension; all datasets are 10-dimensional.
    RNG: must consume the hidden_manifold loop (d=2..20, manifold_dim=6) first
    to replay the same global random stream as the paper's generation script.
    """
    np.random.seed(3)
    for d in range(2, 21):
        X, y = generate_hidden_manifold_model(300, d, 6)
        _split(X, y)
    out = {}
    for m in range(2, 21):
        X, y = generate_hidden_manifold_model(300, 10, m)
        Xtr, Xte, ytr, yte = _split(X, y)
        if m <= max_n:
            out[m] = (Xtr, ytr, Xte, yte)
        if m >= max_n:
            break
    return out


def two_curves_diff(max_n=20):
    """Paper: seed 3, n_samples=300, n_features=10, degree=2..20, offset=1/(2*degree).

    Difficulty knob is polynomial degree; all datasets are 10-dimensional.
    RNG: must consume the two_curves loop (d=2..20, degree=5) first.
    """
    np.random.seed(3)
    for d in range(2, 21):
        X, y = generate_two_curves(300, d, 5, 0.1, 0.01)
        _split(X, y)
    out = {}
    for deg in range(2, 21):
        offset = 1.0 / (2 * deg)
        X, y = generate_two_curves(300, 10, deg, offset, 0.01)
        Xtr, Xte, ytr, yte = _split(X, y)
        if deg <= max_n:
            out[deg] = (Xtr, ytr, Xte, yte)
        if deg >= max_n:
            break
    return out


FAMILIES = {
    "linearly_separable": linearly_separable,
    "hidden_manifold": hidden_manifold,
    "hidden_manifold_diff": hidden_manifold_diff,
    "two_curves": two_curves,
    "two_curves_diff": two_curves_diff,
    "hyperplanes_diff": hyperplanes_diff,
    "bars_and_stripes": bars_and_stripes,
}
