# E4 calibration summary — large scale

Reference arm: `R_instruct`. Probe n=4096, layers [5, 14, 26], dtype torch.bfloat16.

| arm | requested dose | achieved dose | max \|Δerank\| | layer |
|---|---|---|---|---|
| N_dose_1e-4 | 1e-04 | 3.1067e-05 | 0.1272% | 26 |
| N_dose_1e-3 | 1e-03 | 8.5206e-04 | 0.1086% | 26 |
| N_dose_3e-3 | 3e-03 | 3.1691e-03 | 0.1795% | 14 |
| N_dose_1e-2 | 1e-02 | 1.0259e-02 | 0.3990% | 5 |
| N_dose_3e-2 | 3e-02 | 3.0426e-02 | 1.0710% | 5 |
| N_dose_1e-1 | 1e-01 | 1.0128e-01 | 3.6050% | 14 |
| R_base | — | — (not a controlled dose) | 11.8594% | 26 |

## Where our Stage-A run falls

- `ckpt-0` aggregate relative dose: **0.0000e+00**
- `ckpt-100` aggregate relative dose: **5.4596e-04**
- `ckpt-50` aggregate relative dose: **5.0884e-04**

E1 measured the Stage-A LoRA moving erank by at most **0.7303%** (down_in L14) / 0.7227% (resid L16).
On this ladder that sits between the **1e-02** rung (0.3990%) and the **3e-02** rung (1.0710%).

**Resolution floor.** This platform reproduces E1's published ckpt-0 eranks to within **0.174%**, so a change smaller than that is not resolvable here.
2 ladder rung(s) fall at or below it (`N_dose_1e-4`, `N_dose_1e-3`) and must be read as floor, not as a measured response.

## Reading rules

- Arm R is not a controlled dose. It is an order-of-magnitude reference point between two released checkpoints separated by an undocumented pipeline. No causal language.
- Arm N's noise is isotropic and full-rank; a real update is low-rank and structured. This calibrates the detector, not RLVR.
- erank levels are never compared across scales or dtypes, only arms against their own scale's reference.
