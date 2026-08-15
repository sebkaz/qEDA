# Credit-card fraud input

The raw transaction table is deliberately not redistributed in this package.
Obtain the public *Credit Card Fraud Detection* dataset released by the Machine
Learning Group of Universite Libre de Bruxelles through Kaggle and place the
CSV at:

```text
data/credit-card/credit.csv
```

Expected columns are `Time`, anonymised PCA coordinates `V1`--`V28`, `Amount`,
and binary label `Class`. The script verifies the computation against the
released representation only. It does not apply another PCA and makes no claim
about the unavailable original transaction variables.

The file used for the manuscript contained 284,807 rows, including 492 frauds.
Its approximate size was 144 MB.
