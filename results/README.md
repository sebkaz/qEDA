# Frozen manuscript results

`data/` contains topology, Iris, and fraud result tables.
In particular, `iris_anomaly_typology.csv` distinguishes the seven
cross-validated classifier-error rows from global and class-conditional
density diagnostics; it does not relabel every model error as an outlier.
It also contains the operational-abelian control table separating commuting
gate generators from compatible preparation and readout.
`benchmark/` contains the 55-variant Bowles summary tables.
`figures/` contains the two figures included in the manuscript.
`pdf/` contains the generated computation summary.

The files are versioned because they support inspection of the submitted
numbers. Regenerating an audit overwrites its corresponding files.
