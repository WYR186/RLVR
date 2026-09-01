# Result — effective rank has a noise-dose threshold, but real training moves it far more efficiently

**Date:** 2026-09-01
**Owner:** Aaron Wang (early-warning diagnostics)
**Cost:** **0 compute units.** 7B on the M3 Max, 0.5B on the RTX 4070.
**Spec:** `SPEC_E4_DETECTOR_CALIBRATION.md`
**Artifacts:** `outputs/e4_large/`, `outputs/e4_large_seeds/`, `outputs/e4_small/`
**Companion:** `FINDING_E4_SMALL_WIN4070.md` (the 0.5B scale, run on the 4070)

## 1. Bottom line

E1 showed Q does not move across our checkpoints. E4 asks what a change that
*does* move it looks like, and the answer reframes the whole project:

- **The detector is not blind.** A full instruction-tuning pipeline moves 7B
  effective rank by **11.86%** on the same model, probe and contract.
- **It has a threshold.** Below a relative Frobenius weight change of roughly
  **1e-2**, isotropic perturbation produces nothing resolvable. Above it the
  response is monotone: 0.399% → 1.071% → 3.605% at 1.03e-2 / 3.04e-2 / 1.01e-1.
- **Our intervention was ~19x below that threshold.** Arm W measures the 7B
  Stage-A LoRA at **5.4596e-04** (ckpt-100).
- **At 0.5B, update structure matters.** At matched dose `7.1794e-04`, the real
  exp1.5 v3 ckpt-500 moves erank **0.6143%**, versus isotropic noise mean
  **0.0234%** (range 0.0079%--0.0321%): **26.3x** the mean and above every
  measured noise direction.

So the honest claim is not "Q has no dynamic range." It is: **the isotropic
noise ruler has a dose threshold, while a structured gradient update can move Q
substantially below that threshold.** The ruler calibrates magnitude but is a
conservative detector bound for real training at 0.5B.

Dormant fraction is the opposite and E1 already settled it: no range at any
dose, at either scale.

## 2. Correction — seed repeats falsified one of our own claims

The single-seed ladder put the first above-floor rung at **3.17e-03**
(0.1795%), and an earlier draft of this work called that the detection
threshold. **Three noise directions at a slightly larger dose (3.35e-03) give
0.0581%, 0.0759% and 0.1432% — mean 0.0924%, all below the floor.** The
original 0.1795% was a high draw from a single direction.

The corrected evidence is stronger than the claim it replaced. Across a **108x
span of dose**, from 3.11e-05 to 3.35e-03, the response is flat at 0.058–0.180%
with no trend:

| achieved dose | seeds | max abs erank change |
|---:|---:|---|
| 3.11e-05 | 1 | 0.1272% |
| 3.66e-04 | 3 | 0.1214 / 0.1329 / 0.1390% (mean 0.1311, sd 0.0090) |
| 8.52e-04 | 1 | 0.1086% |
| 3.17e-03 | 1 | 0.1795% |
| 3.35e-03 | 3 | 0.0581 / 0.0759 / 0.1432% (mean 0.0924, sd 0.0449) |
| **1.03e-02** | 1 | **0.3990%** |
| 3.04e-02 | 1 | 1.0710% |
| 1.01e-01 | 1 | 3.6050% |

Flatness over two orders of magnitude of dose is the signature of a measurement
floor, not of a weak signal. The threshold therefore lies between **3.4e-03 and
1.03e-02**, not at 3e-03.

**Reporting the single-seed ladder would have overstated the detector.** The
1e-02 and larger rungs remain single-draw and should carry the same caveat
until repeated.

## 3. The floor is measured, not assumed

Gate R2 compares this platform's bare-Instruct eranks against E1's published
A100 values: −0.0244% / +0.0192% / **+0.1739%** at layers 5 / 14 / 26. The
largest, **0.174%**, is the smallest change resolvable here, and every rung in
the table above except three sits under it.

This is why the spec made Gate R2 record-only rather than a pass/fail gate.
E1's `1e-4` reproduction tolerance is a same-hardware, same-kernel statement;
it does not survive a change of accelerator, and pretending otherwise would
have hidden the floor that makes the rest of this table interpretable.

## 4. Arm W — what our interventions actually did to the weights

| run | checkpoint | aggregate relative dose |
|---|---|---:|
| 7B GURU (LoRA r=16) | ckpt-0 | 0.000000e+00 |
| | ckpt-50 | 5.088352e-04 |
| | ckpt-100 | 5.459591e-04 |
| 0.5B exp1.5 v3 (full-parameter) | ckpt-0 | 0.000000e+00 |
| | ckpt-100 | 4.876857e-04 |
| | ckpt-500 | 7.179374e-04 |

Both exact zeros are gates, not observations: LoRA initialises `B = 0`, and the
0.5B pre-update checkpoint must be byte-identical to its base snapshot on the
targeted modules.

Two things fall out. First, **7B ckpt-50 → ckpt-100 moves the weights only
7.3% further** — Stage A had largely saturated in parameter space by update 50,
so those two checkpoints are near-identical models and "Q could not tell them
apart" is partly a statement about the checkpoints. Second, full-parameter
training at 0.5B for 500 updates reaches **7.18e-04**, the same order as a
rank-16 LoRA at 100 updates. The dose ceiling is not a LoRA artifact.

## 5. Arm A closes the structured-vs-noise comparison at 0.5B

The real exp1.5 v3 checkpoints were re-measured on the 4070 under E4-small's
exact 4,096-prompt, float32, layers-4/12/22 contract. Because the run started
from the pinned Qwen2.5-0.5B Base revision, every Arm-A value is measured against
`R_base`; Arm N remains measured against the `R_instruct` model it perturbed.

| real checkpoint | aggregate dose | max abs erank change vs R_base | layer |
|---|---:|---:|---:|
| ckpt-0 | 0.000000e+00 | 0.000000% | 4 |
| ckpt-100 | 4.876857e-04 | 0.869783% | 22 |
| ckpt-500 | 7.179374e-04 | 0.614301% | 22 |

The ckpt-0 exact identity is a hard gate. At ckpt-500's matched dose, three
isotropic directions moved erank 0.0321% / 0.0079% / 0.0301% (mean 0.0234%).
The real response is therefore **26.3x the noise mean** and **19.2x the largest
observed direction**. The non-monotone ckpt-100/500 response also shows that
weight-norm dose alone does not determine the spectral effect.

This closes the comparison only at 0.5B. The equivalent 7B Arm A remains absent
because its mixed bf16-base/fp32-adapter run wedged on MPS. It also does not turn
exp1.5 v3's separately measured −8.56 pp adaptability change into a causal
consequence of dose or erank; the outcome and spectral probes remain distinct.

## 6. Reading rules

- Arm R is an order-of-magnitude reference between two released checkpoints
  separated by an undocumented pipeline. **Not** a controlled dose, and no
  causal language about instruction tuning and plasticity.
- Arm N's noise is isotropic and full-rank. It calibrates *the detector*. It is
  not a model of an RLVR update, and §5 is exactly why that distinction matters.
- Effective-rank **levels** are never compared across scales, dtypes or probes —
  only arms against their own reference, in their own frame.
- The 7B and 0.5B thresholds are each one model, one probe, one task.

## 7. Reproduce

```bash
python "experiment 2/drivers/09_audit_e4_artifacts.py" --dir outputs/e4_large --require-arm-w
python "experiment 2/drivers/10_e4_report.py" --dir outputs/e4_large
python "experiment 2/drivers/09_audit_e4_artifacts.py" --dir outputs/e4_small --require-arm-w
python "experiment 2/drivers/10_e4_report.py" --dir outputs/e4_small --reference R_base
```
