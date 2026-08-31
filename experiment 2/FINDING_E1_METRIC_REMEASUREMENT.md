# Result — E1 finds no usable Q dynamic range under any tested operationalization

**Date:** 2026-08-31
**Status:** COMPLETE — reference gate, V1a, V2–V4, V5a–c and V6a–c complete.
**Run:** `exp2_colab_guru_math7b_instruct_group8_e33527592dd9`
**Scope:** metric re-measurement only; no training and no new Stage-B claim.
**Artifacts:**
`eaaj-pilot/outputs/exp2_colab_guru_math7b_instruct_group8_e33527592dd9/measurements/e1_sweep/`

## 1. Bottom line

E1 rules out the main measurement-choice explanations for the flat 7B Q
trajectory. Across alternative probe distributions, dormancy poolings,
dormancy tensors, thresholds, layers, probe sizes, token poolings and
effective-rank tensors, **no comparable Q series develops useful
across-checkpoint variation**.

- The largest ckpt-0 → ckpt-100 effective-rank change in the main sweep is
  **+0.7303%** (`down_in`, layer 14).
- The corrected all-28-layer residual profile peaks at **+0.7227%** (layer 16).
- V1a, measured on frozen ckpt-0 continuations, is flatter still: the largest
  effective-rank change is **+0.0695%**.
- The largest absolute shift anywhere on the prompt-only dormancy-fraction
  curves is **0.285 percentage points**; for V1a it is **0.523 points**. These
  maxima occur at high exploratory thresholds, not at the registered
  thresholds.

The strongest defensible conclusion is therefore about this detector in this
regime: **activation Q has no usable dynamic range for distinguishing these
three rank-16 LoRA checkpoints.** It is not evidence that RLVR reduces—or does
not reduce—the model's ability to learn, and it is not a general verdict on Q
under full-parameter RLVR or a larger intervention.

## 2. Reference gate passed exactly

The unchanged registered arm was rerun before both sweep sessions. All nine
effective-rank values reproduced within the pre-specified `1e-4` tolerance,
and registered dormant fraction reproduced as exactly 0.0.

| layer | ckpt-0 | ckpt-50 | ckpt-100 | Δ 0→100 |
|---:|---:|---:|---:|---:|
| 5 | 1127.4155 | 1128.2271 | 1128.1812 | +0.0679% |
| 14 | 1281.0450 | 1287.8799 | 1287.8093 | +0.5280% |
| 26 | 1426.0597 | 1433.8730 | 1432.5480 | +0.4550% |

This clears the environment/provenance gate: config hash `e33527592dd9`, all
four split hashes, frozen 4096-prompt probe, eval mode, bf16 base/fp32 adapter,
float32 accumulation and float64 centered SVD.

## 3. V5a correction: all 28 layers remain flat

The first E1 execution did **not** actually run V5a: `--all-layers` was omitted,
and the implementation also retained residual spectra only at reference layers
`[5, 14, 26]`. The corrected supplement explicitly retained
`resid/last` at every decoder block while keeping the expensive V2/V3
cross-product at the registered layers.

| depth region | largest absolute Δ 0→100 | layer |
|---|---:|---:|
| layers 0–9 | 0.0679% | 5 |
| layers 10–18 | **0.7227%** | 16 |
| layers 19–27 | 0.5910% | 27 |

No layer changes by 1%. The registered three-layer selection slightly
understated a mid-depth bump, but it did not miss a qualitatively different
trajectory. The largest values are:

| layer | ckpt-0 | ckpt-50 | ckpt-100 | Δ 0→100 |
|---:|---:|---:|---:|---:|
| 14 | 1281.0450 | 1287.8799 | 1287.8093 | +0.5280% |
| 16 | 1367.2377 | 1374.7996 | 1377.1188 | **+0.7227%** |
| 17 | 1381.8101 | 1389.3352 | 1391.2941 | +0.6863% |
| 27 | 1302.9899 | 1312.3676 | 1310.6910 | +0.5910% |

## 4. V6: changing the effective-rank tensor does not rescue the signal

This was the most important alternative because the 0.5B pilot's MLP spectrum
had moved when its residual spectrum did not. It does not reproduce here.

| tensor / layer | ckpt-0 | ckpt-50 | ckpt-100 | Δ 0→100 |
|---|---:|---:|---:|---:|
| residual / 14 | 1281.05 | 1287.88 | 1287.81 | +0.5280% |
| down-in / 14 | 1698.66 | 1710.08 | 1711.06 | **+0.7303%** |
| gate-post / 14 | 1501.46 | 1508.72 | 1509.18 | +0.5140% |
| down-in / 26 | 1591.87 | 1604.12 | 1602.78 | +0.6850% |
| gate-post / 26 | 1710.92 | 1724.47 | 1723.07 | +0.7105% |

The MLP tensors are 18,944-dimensional with only 4096 probes, so their erank
levels are sample-truncated and must not be compared with the 3584-dimensional
residual levels. The within-variant checkpoint changes above are comparable;
all remain below 0.74%.

## 5. V2–V4: dormancy is threshold-sensitive but checkpoint-insensitive

Changing mean-over-token pooling to per-token, max-over-token or per-prompt
median; changing the measured tensor to gate-pre, gate-post or up; and sweeping
26 thresholds from `1e-4` to `1.0` produces different **levels**, as expected.
It does not produce a checkpoint trajectory.

Under the registered mean/down-in definition, dormant fraction remains 0.0 at
τ=0.025 and τ=0.1 for all three registered layers and checkpoints. Some
alternative tensors have tiny nonzero levels—for example gate-post layer 26 is
`0.000211` at τ=0.1—but those values are identical at ckpt-0/50/100.

Across the entire prompt-only V2×V3×V4 grid, the largest ckpt-0→100 absolute
curve change is 0.002851 (0.285 percentage points), at an exploratory threshold.
That is threshold calibration without early-warning separation: moving τ can
make the number nonzero, but it does not make it vary with Stage-A checkpoint.

## 6. V1a: completion-token probing is even flatter

V1a freezes greedy ckpt-0 continuations for the first 512 probe prompts and
measures every checkpoint on exactly those continuation tokens. Its largest
effective-rank change is only +0.0695% (gate-post, layer 26); residual changes
are between −0.0193% and +0.0067%.

Because 512 < hidden dimension 3584, these erank **levels** are
sample-truncated and are not comparable to the 4096-prompt reference levels.
The across-checkpoint V1a comparison is valid because sequences are frozen.
V1b was intentionally not run: the spec makes it conditional on V1a showing
movement, and V1a did not. V1b would also change the input distribution by
checkpoint and therefore would not be a comparable early-warning series.

## 7. V5b/V5c: two interpretation caveats, neither changes the result

V5b shows that erank level is still strongly sample-size-dependent even for the
residual stream. At ckpt-0, residual layer-14 erank rises from 313.54 at n=512
to 548.48, 903.28 and 1281.05 at n=1024/2048/4096. Thus `n > d` removes the
hard rank ceiling but does **not** demonstrate estimator saturation. This
strengthens the warning against comparing V1 levels with the reference arm.
The nested-prefix across-checkpoint changes nevertheless remain below 0.74%.

V5c changes residual pooling from last token to mean over tokens. Its largest
ckpt-0→100 change is only +0.1245% (layer 26), smaller than the registered
last-token result.

## 8. Consequence for the fixed-budget research question

Stage B's pre-registered outcome is fixed-budget adaptability on Simulation at
30 GRPO updates. Its ΔR values are +0.1033, +0.0967 and +0.0900 for Stage-A
checkpoints 0/50/100; their 1.33-point range is below the arm-to-arm resolution.
E1 now shows that the predictor side remains flat under every defensible
operationalization tested.

Therefore this three-checkpoint run **cannot test whether Q predicts a later
adaptation stall**: neither Q nor fixed-budget adaptability has enough resolved
variation to correlate. The correct report is a clean null at a named task,
budget and LoRA dose—not a claim that RLVR changes a general ability to learn.

## 9. Artifact audit and correction record

Local strict audit passes:

- reference gate: 9/9 within `1e-4`, dormant fraction 0.0;
- provenance and measurement contracts present at every checkpoint;
- V5a residual spectra cover all layers 0–27;
- all 258 V5a per-unit vectors exist, are finite, shape `(18944,)`, and match
  their JSON summaries;
- V5a filenames are variant-scoped, preventing cross-variant overwrite.

One legacy defect is preserved and disclosed: the original base and V1a runs
used the same 108 score-vector filenames, so V1a overwrote those base `.npy`
files. The base JSON summaries and all spectral results were unaffected. The
V5a supplement regenerated the prompt-only dormancy vectors under `V5a_...`
names, including all base tensors/poolings at layers 5/14/26; those are the
canonical prompt-only vectors going forward. Do not cite the unprefixed base
`.npy` files.

## 10. What this does and does not close

E1 closes the inexpensive “we chose the wrong hook/pooling/τ/layer/probe” audit
for this run. It supports the mechanistic interpretation already registered:
a rank-16 adapter that moved only modestly atop frozen 7B base weights does not
substantially rearrange these high-dimensional activation spectra.

It does not close the broader question. A future detector test needs a regime
where the intervention and fixed-budget outcome both have measurable dynamic
range: more checkpoints and seeds, a larger Stage-A dose or less constrained
parameter update, and a pre-powered held-out adaptation evaluation.
