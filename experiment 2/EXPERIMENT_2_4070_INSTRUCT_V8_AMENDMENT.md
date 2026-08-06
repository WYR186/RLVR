# Experiment 2 — final RTX 4070 Instruct v8 engineering amendment

**Config:** `exp2_config_4070_instruct_v8.json`  
**Status:** Phase-0 two-stage CUDA smoke passed on 2026-08-04

## Frozen synthesis

V8 combines only two components already independently validated in immutable
smokes:

1. V5's three-generation geometry (`prompt<=512`, completion 1280) produced
   4.1667% and 8.3333% clipping on two updates, both under the 10% gate.
2. V7's registered Math reward (`exact GURU + 0.1 balanced boxed format`)
   produced nonzero reward standard deviation and gradients on both updates.

V7's four-generation geometry failed because its second update clipped 18.75%.
V5's exact-only reward failed because both updates were zero. V8 adds no new
reward term, threshold, prompt instruction, dataset choice, or optimizer change.

## Registered recipe

- Model: pinned `Qwen/Qwen2.5-0.5B-Instruct`.
- Stage A: 54,251 Math prompts at <=512 tokens; group/device batch 3;
  accumulation 8; eight unique prompts per update; completion 1280; no prompt
  truncation; exact reward plus 0.1 for a non-empty balanced boxed answer.
- Stage B: unchanged exact-reward CodeIO short-context split, 1,132 train and
  300 eval, group 8, `640+384` tokens.
- Stage-A maximum sequence is 1,792 tokens and claims must name the three-sample
  GRPO group and shaped reward.

The deterministic preflight must retain combined-reward variance. Both CUDA
smoke stages must save two checkpoints. Every Stage-A update must clip <=10%,
and at least one must have positive reward standard deviation and nonzero
gradient. Existing five-update zero-variance/clipping safety stops remain.

This is the final 4070 engineering recipe. It is not an equivalent run of the
original Base/full-CodeIO exp2 and cannot be used for that efficacy claim.

## Frozen Phase-0 result

The deterministic Stage-A preflight recorded 5 exact-correct and 24 valid-box
outputs across 8 prompts x 3 generations, with one group having registered
reward variance. The two Stage-A CUDA updates recorded:

| update | clip ratio | reward std | grad norm | seconds |
|---:|---:|---:|---:|---:|
| 1 | 0.041667 | 0.020412 | 0.611589 | 146.41 |
| 2 | 0.083333 | 0.207818 | 1.111593 | 172.01 |

Stage B retained exact reward, clipped 0 on both updates, and recorded gradient
norms 3.120619 and 0.830134. Both stages saved `checkpoint-2`; the promotion
marker is `data/exp2_4070_instruct_v8_phase0_smoke_complete.json`. GPU telemetry
is `logs_4070/gpu_20260804_104122_exp2_4070_smoke.csv` and the wrapper transcript
is `logs_4070/run_20260804_104122_exp2_exp2_config_4070_instruct_v8_smoke.log`.
