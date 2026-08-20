# Experiment 2 (Colab) — MVP scope amendment

**Config:** `exp2_colab_config_mvp.json`
**Amends:** `EXPERIMENT_2_COLAB_PLAN.md` (scope only — the plan is not retracted)
**Registered:** 2026-08-16, before any Phase 0 run
**Owner:** Aaron Wang
**Status:** Phase 0 not started

## Why this amendment exists

The abstract is due **2026-08-23**, seven days out, and as of today **neither
experiment-2 track has produced a single complete Math → Simulation curve**:

- The **WIN4070 track** (Qwen2.5-0.5B-Instruct, full-parameter, group 3) safety-stopped
  Stage A at update 110 of 200; 52 of those 110 updates had zero within-group reward
  variance and `grad_norm` exactly 0.0. Its post-stop Stage-B grid completed cells at
  ckpt 0 and 50 but the ckpt-100 cell safety-stopped at update 23 — so the ΔR curve is
  missing its endpoint.
- The **Colab/7B track** has never successfully run Phase 0. The pipeline, notebooks, and
  config are written and pushed; zero GPU-hours of real execution exist.

Because zero throughput data exists for LoRA GRPO at any scale in this project, the
full-scope schedule in `EXPERIMENT_2_COLAB_PLAN.md` §4 cannot be defended as feasible —
§4 says so itself ("no number in this section is measured yet"). This amendment commits
to the smallest scope that still produces a **complete** ΔR curve rather than a second
truncated one.

## What is NOT de-scoped

**The model.** `Qwen/Qwen2.5-7B` is unchanged. Tommy's "stronger than Qwen2.5-7B" note
is absorbed entirely on the scope axis, not the model axis. This was an explicit operator
decision on 2026-08-16 when the alternative (drop to 1.5B as a pipeline-proving run) was
offered and declined.

Also not cut: Stage-A `max_completion_length` (1280), the 300-question frozen Stage-B eval
set, the group-8 geometry, and every runtime safety stop.

## The three scope changes

### 1. Stage A: 200 → 100 updates, checkpoints [0,100,200] → [0,50,100]

Not a new deviation — this is exactly the `de_scope_fallback_max_steps` /
`de_scope_fallback_checkpoint_steps` already pre-registered in the full-scope config, and
de-scope priority item 2 in plan §4. Still three checkpoints (Tommy's stated floor), still
ckpt-0 as the stage-2-alone baseline.

**Cost:** less stage-1 exposure per checkpoint, i.e. a narrower dose range on the x-axis of
any Q-vs-adaptability relationship. If the effect is dose-dependent and small, 100 updates
may not span enough of it to see anything. Report the null accordingly if it happens.

### 2. Stage B: 50 → 30 updates, eval at [0,10,20,30]

Endpoint and AUC both remain readable. Applied uniformly to all three cells — a
mixed-budget grid is not a fixed-budget experiment.

### 3. Stage B `max_completion_length`: 384 → 640 — **DEVIATION**

This is the only genuinely new deviation in this amendment.

**Cause.** The WIN4070 v8 post-stop Stage-B cell at ckpt100/seed42 safety-stopped at update
23 with `completions/clipped_ratio` rising 0.1094 → 0.1719 → 0.2344 against the
">10% clipping for 5 consecutive updates" stop. That is a recipe-length failure, not a
training failure: CodeIO's structured-JSON answers were being cut off mid-output. 640
matches Stage-B `max_prompt_length` and the committed token audit
(`data/token_length_audit.json`).

**The stop threshold is not relaxed.** The fix is to give the model room to finish, not to
stop noticing that it cannot.

**Consequence.** This changes the fixed-budget recipe. All three cells must run under 640,
and no cell from this run may be compared against a 384-token cell from any earlier run.

## Correction to the measurement contract (not a scope change)

The full-scope config declared `measurement.model_dtype: "float32"`. That field **was never
read by any code path**: `pipeline.measure_checkpoint_q()` calls `build_peft_model()`, which
unconditionally loads the base in bfloat16 and upcasts only LoRA parameters. Declaring fp32
was also infeasible at this scale — fp32 weights for a 7B model are ~28 GB and do not fit an
L4 at all.

The MVP config records what the code actually does (`bfloat16_base_float32_lora_adapter`).
All checkpoints are measured under identical dtype, so within-run comparability — the
property the project's comparability rule actually requires — holds.

**Q values from this run are not numerically comparable to the WIN4070 track's float32
measurements.** This was already true before the correction; the correction only makes it
visible.

## Promotion gate (decide once, at the end of Phase 0)

If Phase 0's **measured** s/update implies the full-scope schedule (Stage A 200 updates,
Stage B 50 updates × 3 cells) finishes before 2026-08-23 with ≥30% calendar margin, run the
full scope from `exp2_colab_config.json` and discard this fork unused.

Decide once, at the end of Phase 0. Never mid-run — switching scope mid-flight produces a
grid that is neither.

## Claim boundary

The deliverable is one complete fixed-budget ΔR curve at three Stage-A checkpoints, one
seed, plus T_t and Q(t) at the same three points.

Honest limits, to be stated in the abstract rather than discovered by a reader:
n = 3 checkpoints, 1 seed, a LoRA-adapter subspace rather than full parameter space, and a
Stage-A learning rate (2e-5) that no dose-response study supports. **Any Q-vs-adaptability
relationship over three points is an exploratory description carrying no inferential claim.**
Do not report a rank correlation over three points as evidence.
