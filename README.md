# qEDA: reproducibility package and audit toolkit

Reference computations for *Beyond the kernel: exploratory analysis of quantum
encodings of tabular data through the class density operator*.

This repository contains exactly the numerical evidence used by the
manuscript: implementation controls, matched-moment topology audit, 55-variant
benchmark audit, Iris sample-level audit, and the held-out credit-card fraud
case study. It makes no quantum-advantage or classical-intractability claim.

It also exposes a small public API for auditing a new declared statevector
encoding. The API reads the same three profiles as the paper: spectral,
declared-subsystem, and feature-mode coherence. It accepts either a direct
``row -> statevector`` function or a PennyLane circuit with signature
``circuit(features, wires)``.

The unit of analysis is `(dataset, encoding)`. The matched pair is the real
product control `J = 0` and the partial-correlation sandwich
`RY(x/2) · ZZ(J) · RY(x/2)`. Coupling scale is fixed before each audit; it is
not fitted per dataset or per result.

## Layout

| Path | Contents |
|---|---|
| `src/qeda/` | Shared density-operator diagnostics and NumPy sandwich implementation. |
| `scripts/01_controls/` | Algebra, complex-Gram, current, and coupling-sign controls. |
| `scripts/02_topology/` | Matched ring-versus-disk relevance audit and loop-phase control. |
| `scripts/03_benchmark/` | Regeneration and qEDA analysis of the 55 Bowles variants. |
| `scripts/04_iris/` | Cross-fitted Iris audit, random-coupling/Haar controls, and sensitivity grid. |
| `scripts/05_fraud/` | One-class held-out credit-card audit. |
| `src/qml_benchmarks/data/` | Vendored NumPy-only benchmark generators; upstream license is retained. |
| `results/` | Frozen CSV tables, publication figures, and computation summary. |
| `data/credit-card/` | Data-placement instructions; the source data are deliberately not versioned. |
| `docs/` | Script which builds the plain-language computation summary. |

## Installation

Python 3.11--3.13 is supported. The verified run used Python 3.13.13,
PennyLane 0.45.1, NumPy 2.x, and scikit-learn 1.x.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[report]'
export MPLBACKEND=Agg MPLCONFIGDIR=/tmp/qeda-mpl XDG_CACHE_HOME=/tmp/qeda-cache
```

## Reproduce the manuscript evidence

Run commands from the repository root. The first four groups need only the
declared Python dependencies; the final fraud audit additionally needs the
external public data table described below.

```sh
# 1. Algebraic and implementation controls
.venv/bin/python scripts/01_controls/step1_statistics.py
.venv/bin/python scripts/01_controls/gram_identity_check.py
.venv/bin/python scripts/01_controls/current_witness_check.py
.venv/bin/python scripts/01_controls/coupling_discrepancy_demo.py
.venv/bin/python scripts/01_controls/operational_abelian_controls.py

# 2. Matched-moment ring/disk audit
.venv/bin/python scripts/02_topology/topology_holonomy_test.py

# 3. Bowles 55-variant benchmark audit
.venv/bin/python scripts/03_benchmark/spectral_benchmark.py \
  --max-size 10 --max-qubits 10 \
  --out results/benchmark/spectral_benchmark_pennylane.csv \
  --classical-out results/benchmark/classical_eda.csv \
  --accuracy-out /tmp/qeda_accuracy_context.csv \
  --coverage-out results/benchmark/spectral_coverage_pennylane.csv
.venv/bin/python scripts/03_benchmark/generate_spectral_eda_report.py \
  --spectral results/benchmark/spectral_benchmark_pennylane.csv \
  --classical results/benchmark/classical_eda.csv \
  --accuracy results/benchmark/accuracy_context.csv \
  --comparison-out results/benchmark/eda_comparison.csv \
  --out results/pdf/spectral_eda_report.pdf

# 4. Iris audit and its controls
.venv/bin/python scripts/04_iris/e4_model_errors.py
.venv/bin/python scripts/04_iris/null_controls.py
.venv/bin/python scripts/04_iris/sensitivity_analysis.py

# 5. Fraud audit
.venv/bin/python scripts/05_fraud/fraud_qeda_case_study.py

# Rebuild the computation guide
.venv/bin/python docs/build_computation_summary.py
```

## Audit a custom PennyLane circuit

The adapter deliberately asks for a state-preparation circuit, not a trained
model. qEDA then constructs the empirical density operator of the encoded rows
and reports all three readings.

```python
import pennylane as qml
from qeda import audit, pennylane_encoding

def circuit(features, wires):
    for value, wire in zip(features, wires, strict=True):
        qml.RY(value, wires=wire)
    qml.CZ(wires=[wires[0], wires[1]])

encoding = pennylane_encoding(circuit, n_qubits=2)
report = audit(data[:, :2], encoding, name="RY--CZ")
print(report.to_markdown())
```

For the matched manuscript control and a layered sandwich, use
``product_ry_circuit`` and ``sandwich_circuit``.  With its default
``rescale_features=True``, a depth-$L$ sandwich receives ``x/L`` in each
block, preserving the total product-angle map at ``J=0`` while allowing the
coupled circuit to be tested at several depths.

```sh
.venv/bin/python examples/iris_pennylane_audit.py
```

The Iris example reports product control versus one- and two-layer sandwich
encodings on the same setosa rows.  It is intentionally an audit of the
representation, not a classifier benchmark.

## External data

The Bowles datasets are regenerated from vendored generators and Iris comes
from scikit-learn. The credit-card table is not redistributed. Obtain the
public *Credit Card Fraud Detection* table and place it at
`data/credit-card/credit.csv`; see [data/credit-card/README.md](data/credit-card/README.md).

## Result boundary

- The topology audit is a negative result: operator enrichment is not a
  certificate of a support hole.
- Across 55 benchmark variants, the spectral response is not a monotone
  relabelling of the selected covariance summaries.
- The Iris audit is a held-out case study, not a classifier theorem.
- On fraud, partial-correlation coupling changes operator statistics but adds
  essentially no anomaly-ranking information beyond the `J = 0` control.

These statements are conditional on the declared preprocessing, encoding,
subsystem convention, diagnostics, and controls. Frozen outputs are supplied
for inspection, not as a substitute for executing the scripts.

## License

New qEDA code is released under the BSD 3-Clause License; see [LICENSE](LICENSE).
The vendored Bowles generator retains its own license at
`third_party/bowles_qml_benchmarks/LICENSE`.
