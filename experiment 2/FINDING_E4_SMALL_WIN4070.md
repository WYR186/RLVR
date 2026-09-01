# E4-small on the Windows RTX 4070: 0.5B detector ladder and exp1.5 v3 dose

**Date:** 2026-09-01  
**Machine:** NVIDIA GeForce RTX 4070 Laptop GPU, 8 GiB  
**Status:** complete; strict artifact audit passed

## Scope and contract

This run completes the 0.5B scale of E4 and adds the full-parameter Arm W that
was unavailable on the other machines. Arm W reads all eight preserved exp1.5
v3 Stage-A checkpoints. Arms R and N use the frozen 4,096-prompt E1 probe,
layers `[4, 12, 22]`, float32 model weights, and batch size 8. The six Arm-N
doses are `{1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}`. Direction repeats use
seeds 42/43/44 at the exact ckpt-500 dose and at the first >1% response rung,
`3e-2` (the original `3e-2` rung supplies seed 42 without duplication).

Every R/N arm ran in a fresh Python/CUDA process. The first combined-process
attempts at batch 16 and batch 8 exhausted the 8 GiB device when the Base arm
followed the Instruct arm. Their completed batch-16 Instruct record is retained
under `outputs/e4_small/diagnostics_batch16/`. The runner now supports
`--r-only` and explicitly releases caller-owned model references; no scientific
parameter changed.

## Arm W: exp1.5 v3 full-parameter dose

The dose is the aggregate Frobenius ratio over the same 168 attention/MLP
projection matrices used by Arm N.

| checkpoint | aggregate relative dose |
|---|---:|
| 0 | 0.000000e+00 |
| 25 | 2.697579e-04 |
| 50 | 3.643607e-04 |
| 100 | 4.876857e-04 |
| 200 | 6.185143e-04 |
| 300 | 6.855682e-04 |
| 400 | 7.127086e-04 |
| 500 | 7.179374e-04 |

The exact zero at checkpoint 0 confirms that the saved pre-update checkpoint
and pinned base snapshot are identical on the targeted modules. Dose increases
monotonically and nearly saturates between checkpoints 400 and 500.

## E4-small ladder

All changes below are measured against `R_instruct` on the same machine and
under the same contract.

| arm | achieved dose | max absolute erank change | layer |
|---|---:|---:|---:|
| N 1e-4 | 1.000079e-04 | 0.0048% | 12 |
| N 1e-3 | 1.000079e-03 | 0.0902% | 22 |
| N 3e-3 | 3.000236e-03 | 0.1479% | 22 |
| N 1e-2 | 1.000079e-02 | 0.2527% | 22 |
| N 3e-2 | 3.000237e-02 | 1.1674% | 4 |
| N 1e-1 | 1.000079e-01 | 6.6633% | 4 |
| R Base | uncontrolled | 11.0991% | 22 |

At the exact largest exp1.5 v3 dose, `7.179374e-04`, three independent noise
directions produce maximum absolute erank changes of 0.0321%, 0.0079%, and
0.0301%: mean **0.0234%**, range **[0.0079%, 0.0321%]**. At `3e-2`, the three
directions give mean **1.5499%**, range **[1.1674%, 1.7658%]**. Thus the
matched-dose response is small and direction-dependent, while the larger rung
is consistently above 1%. This calibrates detector sensitivity; it does not
claim that a structured RLVR update behaves like isotropic noise. Arm R is only
an order-of-magnitude reference between released Base and Instruct checkpoints,
not a controlled intervention.

## Arm A: the real exp1.5 v3 update in the same frame

Arm A loads the full-parameter checkpoints directly and compares them against
`R_base`, the exact pinned revision from which exp1.5 v3 was trained. Arm N
continues to use `R_instruct`, its actual perturbation origin.

| checkpoint | aggregate dose | max absolute erank change vs R_base | layer |
|---|---:|---:|---:|
| 0 | 0.000000e+00 | 0.000000% | 4 |
| 100 | 4.876857e-04 | 0.869783% | 22 |
| 500 | 7.179374e-04 | 0.614301% | 22 |

At the matched ckpt-500 dose, the real response is **26.3x** the isotropic-noise
mean and **19.2x** the largest of the three observed noise directions. The real
update therefore has a much larger spectral effect per unit weight norm under
this contract. The decline from ckpt-100 to ckpt-500 despite increasing dose is
additional evidence that dose magnitude alone does not determine the response.

This does not connect the separately measured −8.56 pp adaptability drop to
either dose or erank: that outcome used another probe and remains an independent
observation.

The 7B E1 response is deliberately not used to bracket this small-scale result.
Erank levels and response magnitudes are not compared across model scales.

## Acceptance evidence

`09_audit_e4_artifacts.py --require-arm-w` passed with:

- probe n=4,096, ID hash `1e61252e7b54793e`, no truncation relative to E1;
- tokenizer identity gate passed;
- gated-MLP hook reconstruction maximum error exactly 0.0;
- all 11 unique `(requested dose, noise seed)` cells within 0.01% of their
  requests over all 168 target modules, with no duplicate dose/seed pair;
- all eight Arm-W checkpoints present and checkpoint 0 exactly zero.
- all three Arm-A checkpoints share the contract, with `A_ckpt0` reproducing
  `R_base` at exactly 0.000000%.

The complete machine-readable output is in `outputs/e4_small/`; the portable
archive and checksum are recorded in `experiment 2/artifacts/README.md`.
