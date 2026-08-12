# Experiment 2 — RTX 4070 Instruct v4 group-size amendment

**Parent:** `EXPERIMENT_2_4070_INSTRUCT_AMENDMENT.md`  
**Config:** `exp2_config_4070_instruct_v4.json`  
**Hardware:** RTX 4070 Laptop GPU, 8 GiB  
**Status:** pre-registered before v4 preflight and CUDA updates

## Evidence motivating v4

The original Instruct 4070 smoke was memory-safe but clipped 71.875% of both
Stage-A update batches at 512 completion tokens. V2 increased the completion
cap to 768 inside the same 1024-token total budget, but still clipped 23.4375%
and 34.3750%; its two updates took 319.21 and 439.60 seconds. V3 instead added
a uniform concise-answer instruction, but the released verifier found no
within-group reward variance in its 8-by-8 no-update preflight. V3 stopped
before constructing a trainer.

These diagnostics are preserved under the three corresponding
`smoke_outputs_4070_instruct*` directories. None is a formal exp2 run.

## Frozen hardware trade

V4 makes no prompt or reward change. It restores the full `<=512`-token Math
population (54,251 examples) and changes only Stage-A generation geometry:

| field | prior Instruct recipe | v4 |
|---|---:|---:|
| generations per prompt | 8 | 4 |
| device batch | 8 | 4 |
| gradient accumulation | 8 | 8 |
| unique prompts per update | 8 | 8 |
| maximum completion | 512 | 1024 |
| maximum prompt + completion | 1024 | 1536 |

Four simultaneous generations lower activation pressure enough to test a
longer completion cap on the 8 GiB GPU. The number of unique Math prompts per
optimizer update remains eight, while completions and the within-prompt GRPO
sample size are halved. Stage B is unchanged from the validated Instruct
short-context recipe.

The preflight now explicitly seeds Python/Transformers/Torch generation before
sampling. Earlier diagnostics recorded the prompt-selection seed but did not
reset the Torch generation RNG; their files remain immutable historical
evidence and are not silently regenerated.

## Hard gates and interpretation

Formal Stage A may start only after the frozen split, released-verifier signal,
both two-update CUDA smokes, finite loss/gradient, checkpoint/log growth, and
`<=10%` Stage-A completion clipping at **each** smoke update all pass. OOM,
constant reward, any over-limit smoke update, or missing checkpoint is STOP.

V4 changes GRPO group size and Stage-A sequence budget, so it cannot support a
model-controlled comparison to the original Base/full-CodeIO exp2. Its claim
boundary is 4070-feasible four-sample-group training on the pinned Instruct
checkpoint and the frozen short-context CodeIO stratum.
