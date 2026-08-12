# Experiment 2 — RTX 4070 Instruct v3 concise-output amendment

**Parent:** `EXPERIMENT_2_4070_INSTRUCT_AMENDMENT.md`  
**Config:** `exp2_config_4070_instruct_v3.json`  
**Hardware stratum:** RTX 4070 Laptop GPU, 8 GiB  
**Status:** pre-registered before v3 sparse-reward and CUDA smoke gates

## Why a third 4070 recipe is necessary

The first Instruct 4070 recipe fit in memory and completed both CUDA smoke
stages. Stage A nevertheless clipped 71.875% of completions at both updates
with a 512-token completion cap. Starting the registered 200-update run would
therefore have predictably tripped the five-update `>10%` clipping stop.

The v2 diagnostic reallocated the unchanged 1024-token sequence budget from
`512 prompt + 512 completion` to `256 + 768`. It also remained unsuitable:

| v2 Stage-A update | clipped completions | step time |
|---:|---:|---:|
| 1 | 23.4375% | 319.21 s |
| 2 | 34.3750% | 439.60 s |

The v2 experiment is retained as a diagnostic and must not be promoted to a
formal run. Increasing the completion cap further would reduce the retained
Math population and make the already poor 4070 throughput worse.

## Frozen v3 change

V3 returns Stage A to `512 prompt + 512 completion` and appends this exact
instruction to every Math user message **before** chat rendering and length
filtering:

> Answer with concise reasoning. Put the final answer inside `\boxed{}` and
> finish within 384 new tokens.

No reward shaping, answer leakage, prompt truncation, dataset substitution, or
optimizer change is introduced. The suffix is uniform and contains no
question-specific information. Stable IDs and lengths are recomputed from the
rendered prompts.

At the pinned model and dataset revisions this retains 54,227 Math examples at
`<=512` tokens. Stage B is unchanged: 1,432 eligible CodeIO prompts, frozen as
1,132 train and 300 eval, with `640 prompt + 384 completion`.

## Gates and claim boundary

V3 may start formal Stage A only if all of the following hold:

1. the recomputed split exactly matches the frozen v3 split;
2. the released GURU verifier finds within-group reward variance in the
   eight-prompt, eight-generation preflight;
3. both two-update CUDA smoke stages finish with finite loss/gradient and no
   immediate error;
4. Stage-A smoke completion clipping is `<=10%` at each update.

If gate 4 fails, STOP and retain the result as a hardware-feasibility finding;
do not tune the instruction after observing the sampled questions. Results
from this recipe support only claims about the pinned Instruct checkpoint,
the registered concise Math prompt, and the frozen short-context CodeIO
population. They are not equivalent to the original Base/full-CodeIO exp2.
