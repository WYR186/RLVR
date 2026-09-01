# E4 calibration summary — small scale

Reference arm: `R_instruct`. Probe n=4096, layers [4, 12, 22], dtype torch.float32.

| arm | requested dose | achieved dose | max \|Δerank\| | layer |
|---|---|---|---|---|
| N_dose_1e-4 | 1e-04 | 1.0001e-04 | 0.0048% | 12 |
| N_dose_1e-3 | 1e-03 | 1.0001e-03 | 0.0902% | 22 |
| N_dose_3e-3 | 3e-03 | 3.0002e-03 | 0.1479% | 22 |
| N_dose_1e-2 | 1e-02 | 1.0001e-02 | 0.2527% | 22 |
| N_dose_3e-2 | 3e-02 | 3.0002e-02 | 1.1674% | 4 |
| N_dose_1e-1 | 1e-01 | 1.0001e-01 | 6.6633% | 4 |
| R_base | — | — (not a controlled dose) | 11.0991% | 22 |

## Where our Stage-A run falls

- `ckpt-0` aggregate relative dose: **0.0000e+00**
- `ckpt-100` aggregate relative dose: **4.8769e-04**
- `ckpt-200` aggregate relative dose: **6.1851e-04**
- `ckpt-25` aggregate relative dose: **2.6976e-04**
- `ckpt-300` aggregate relative dose: **6.8557e-04**
- `ckpt-400` aggregate relative dose: **7.1271e-04**
- `ckpt-50` aggregate relative dose: **3.6436e-04**
- `ckpt-500` aggregate relative dose: **7.1794e-04**

The largest full-parameter exp1.5 v3 dose is `ckpt-500` at **7.1794e-04**.
On the Arm-N dose axis it lies between **1e-04** (max |Δerank| 0.0048%) and **1e-03** (0.0902%).
The 7B E1 erank reference is not plotted or bracketed here: erank levels and response magnitudes are not compared across scales.

## Reading rules

- Arm R is not a controlled dose. It is an order-of-magnitude reference point between two released checkpoints separated by an undocumented pipeline. No causal language.
- Arm N's noise is isotropic and full-rank; a real update is low-rank and structured. This calibrates the detector, not RLVR.
- erank levels are never compared across scales or dtypes, only arms against their own scale's reference.
