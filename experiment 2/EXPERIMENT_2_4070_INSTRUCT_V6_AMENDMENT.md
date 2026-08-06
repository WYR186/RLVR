# Experiment 2 — RTX 4070 Instruct v6 short-Math amendment

**Parent:** `EXPERIMENT_2_4070_INSTRUCT_V5_AMENDMENT.md`  
**Config:** `exp2_config_4070_instruct_v6.json`  
**Status:** pre-registered before v6 preflight and CUDA updates

## Trigger and frozen decision

V5 reduced Stage-A clipping to 4.1667% and 8.3333%, but both smoke updates had
zero reward variance and zero gradient. Three generations per prompt made the
exact-reward estimator too sparse for promotion. V5 is retained as a length
diagnostic and may not start formal training.

V6 combines the two useful observations already obtained without changing the
prompt or verifier: v2's `<=256` Math population produced nonzero updates, and
v5 showed that a 1280-token completion cap clears the clipping gate. V6 freezes
exactly **52,755** Math examples (97.2% of the pinned Math table), uses four
generations/device batch 4, gradient accumulation 8, and `256 + 1280 = 1536`
maximum tokens. This is the same maximum per-sequence budget as v4 but shifts
256 tokens from prompt to completion.

Stage B remains the previously validated 1,132-train/300-eval short-context
CodeIO recipe. No prompt suffix, shaping reward, answer leakage, or truncation
is introduced.

## Strengthened smoke gate

In addition to exact split, deterministic verifier signal, checkpoint, finite
metric, and per-update clipping `<=10%`, Stage A must record at least one of two
smoke updates with both positive reward standard deviation and nonzero gradient
norm. This closes the gap exposed by v5; a length-safe but update-free smoke is
not considered launchable.

All results must name the Instruct model, short-Math population, four-sample
GRPO groups, 1536-token Stage-A budget, and short-context CodeIO population.
They are not equivalent to the original Base/full-CodeIO exp2.
