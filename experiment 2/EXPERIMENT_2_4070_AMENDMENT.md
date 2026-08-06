# Experiment 2 — RTX 4070 short-context amendment

**Status:** pre-registered local feasibility variant, not the original full-distribution Experiment 2  
**Parent:** `EXPERIMENT_2_PLAN.md` at commit `935d05b`  
**Dataset revision:** `2e2790a962a3c099bfb5ea61389cbf98a5ea439b`  
**Reasoning360 verifier revision:** `13158341d2a0dfe5f3bb80e7126ff21de0d16676`

## Trigger

The original Phase-0 Gate 0a stopped correctly: CodeIO p95 was 1407 tokens,
above the registered 1024-token limit. No original-exp2 training started. The
owner then explicitly requested a separately recorded RTX 4070 version.

## Frozen change

This variant defines Stage B as the **short-context CodeIO subset** whose full
Qwen chat-rendered prompt is at most 640 tokens. Prompts are never truncated.
At the pinned revisions this retains 1493/3730 examples: 300 seeded held-out
eval examples and 1193 train examples. Stage A similarly removes the 147 Math
examples above its existing 512-token contract, retaining 54257/54404.

Stage-B completion length is reduced from 512 to 384. Therefore the maximum
prompt-plus-completion budget remains 1024 tokens, matching the geometry that
already ran on this machine. Batch size, gradient accumulation, eight
generations, model, optimizer, learning rates, checkpoints, seeds, outcomes,
and safety gates remain unchanged.

## Scientific boundary

This is not a substitute for the original Experiment 2. Length filtering
changes the task population and selects an easier slice: Qwen2.5-7B's released
mean pass rate is 0.2150 on the <=640 subset, versus a lower mean on the full
CodeIO distribution. Every result must say **short-context CodeIO**. It cannot
support a claim about full CodeIO or the original GURU Simulation domain.

The original `exp2_config.json`, Gate 0a diagnostics, and requested L4 path are
preserved. A future L4 run must use the original plan and config, not this one.

## Phase-0 gates for this variant

1. Recompute token lengths with the pinned tokenizer and chat template.
2. Require exact eligible counts: Math 54257; CodeIO 1493.
3. Freeze IDs in `data/exp2_4070_splits.json`; never regenerate after training.
4. Require zero overlap between Stage-B train and eval and exactly 300 eval.
5. Run real released-verifier sparse-reward preflight on both stages.
6. Run two CUDA updates on each stage in throwaway directories, with telemetry.
7. STOP on constant reward groups, non-finite loss/gradient, OOM, missing
   checkpoint/dashboard, or safety-stop. Do not automatically tune around it.

## Verifier portability deviation

Math and CodeIO scoring are vendored from Reasoning360 at the revision above.
The upstream Math verifier uses POSIX `SIGALRM` and imports a veRL timeout
context, neither available in this Windows/TRl runner. The scoring and
normalization logic is otherwise retained; the Windows port replaces those
timeouts with bounded-input best-effort execution and converts verifier
exceptions to zero reward. This portability patch is code-tested before smoke.

## Claim language

Never say this run measures general loss of learning ability. Report only
fixed-budget Stage-B adaptability on the frozen short-context CodeIO subset,
relative to ckpt-0 and conditional on transfer `T_t`, with the 50-update budget
and three-seed uncertainty attached.
