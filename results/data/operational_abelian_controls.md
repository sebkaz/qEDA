# Operational abelian controls

All rows use Iris setosa (50 rows, four features), the same angle scaling, and the same fixed partial-correlation coupling where applicable.

Only the first row is operationally abelian in the computational basis: the gate generators, input state, and Z readout are jointly compatible. The RY-YY-RY row proves that mutually commuting gate generators alone do not imply a classical collapse. The diagonal RZ-ZZ-RZ row can have gauge-dependent complex Gram entries from sample-wise global phases, but its class operator and Bargmann phases remain real/zero.

| encoding | gate_algebra_abelian | input_compatible | Z_readout_compatible | operationally_abelian | log_negativity | max_current | rho_imag_frobenius | max_bargmann_phase_rad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RZ-ZZ-RZ (computationally diagonal) | True | True | True | True | 0 | 0 | 0 | 3.1101e-16 |
| RY product control | True | False | False | False | 0 | 0 | 0 | 0 |
| RY-YY-RY (commuting generators) | True | False | False | False | 0.482444 | 0.056592 | 0.456541 | 3.10283 |
| RY-ZZ-RY (sandwich) | False | False | False | False | 0.263874 | 0.0345532 | 0.468829 | 1.01338 |
