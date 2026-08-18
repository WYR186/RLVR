# Finding — Stage A (7B) stopped at update 7 on the completion-clipping gate

**Date:** 2026-08-17/18
**Run:** `exp2_colab_guru_math7b_group8_fc243e587296/stage_a`, Qwen2.5-7B base +
LoRA, group 8, `max_completion_length=1280`, Colab A100-SXM4-80GB
**Stop:** `LocalSafetyCallback` — *five consecutive updates exceeded 10%
completion clipping* — at step 7 of a 100-update budget.
`fixed_budget_completion` then correctly refused the run:
`RuntimeError: incomplete: requested 100, got 7`.

---

## 1. The per-update record

| step | `completions/clipped_ratio` | `completions/mean_length` | reward | reward_std | grad_norm |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0781 | 615.4 | 0.1047 | 0.2278 | 0.0451 |
| 2 | 0.0781 | 640.6 | 0.1000 | 0.1869 | 0.0539 |
| 3 | **0.1875** | 716.9 | 0.0984 | 0.2292 | 0.0626 |
| 4 | **0.1250** | 632.8 | 0.1109 | 0.2262 | 0.0459 |
| 5 | **0.1563** | 695.0 | 0.0641 | 0.0484 | 0.0395 |
| 6 | **0.1406** | 641.2 | 0.0844 | 0.1371 | 0.0468 |
| 7 | **0.1563** | 709.2 | 0.1656 | 0.3062 | 0.0416 |

Steps 3–7 are the five consecutive breaches. **The gate did exactly what it was
registered to do.** Nothing here is a bug in the harness, the data, or the
recipe plumbing.

## 2. This falsifies the Phase-0 inference — including mine

`FINDING_7B_PHASE0_COMPLETE.md` §5 concluded, from a 2-update smoke that cleared
the gate, that:

> "Longer, better completions from the larger model apparently finish inside the
> 1280 cap often enough to stay under the clipping gate. This weakens the
> urgency of the v10 amendment: the truncation problem looks
> model-scale-dependent, not a property of the recipe."

**That is wrong, and 7 real updates are what showed it.** The 7B numbers are
essentially the 0.5B numbers:

| | 0.5B v9 (2 smoke updates) | 7B (7 real updates) |
|---|---|---|
| mean completion length | 753.2, 718.6 | 615–717 |
| clip ratio | 0.0938, 0.1406 | 0.078–0.188 |

Same cap, same distribution, same failure. Truncation on this Math population is
a property of **the recipe and the data**, not of model scale. A 2-update smoke
was simply too short to see a five-update streak — it *cannot* fail this gate by
construction, which is worth remembering the next time a smoke is read as
evidence that a streak-based gate will pass.

§5 of that document has been corrected in place.

## 3. Consequence for the team: v10 is now on the critical path for BOTH tracks

`EXPERIMENT_2_4070_INSTRUCT_V10_AMENDMENT_DRAFT.md` was written as a 0.5B-track
question and explicitly labelled "the 7B deliverable does not need it". That is
no longer true. The 7B MVP arm — the arm that answers Tommy's ">=7B" ask —
cannot produce a single Stage-A checkpoint beyond ckpt-0 at `1280`.

## 4. What was NOT done

- The gate was **not** relaxed. `max_clip_ratio=0.10` and `signal_patience=5`
  are untouched.
- `max_completion_length` was **not** raised. It is a pre-registered Stage-A
  variable; changing it is exactly what the v10 amendment exists to authorize.
- No shaping reward was added, no model variant switched.

## 5. What WAS done, and why it needs no sign-off

The v10 draft names its own prerequisite:

> "Before registering a number, run a **generation-only measurement** (no
> optimizer step, therefore cheap and not a training run) ... and record the
> empirical completion-length distribution."

That measurement was run on the frozen Stage-A population (32 prompts x 8
generations, registered temperature 0.7 / top-p 1.0, cap raised to 3072),
because it trains nothing and changes no registered variable. Its output feeds
the draft's own mechanical sizing rule — *smallest cap in {1536, 1792, 2048,
2560} with measured truncation <= 5%* — so the resulting number is derived, not
chosen. Results: `data/completion_length_measurement.json`.

**Caveat that must travel with the number:** the measurement uses the base
policy (ckpt-0), because the step-7 policy was never checkpointed. Observed mean
length grew 615 -> 709 over 7 updates (~+15%), so a cap sized on the base policy
is an *underestimate* of what the trained policy will need. The rule's 5% target
— half the 10% gate — is the margin that absorbs this, and it should not be
spent on anything else.

## 6. Cost, and a schedule correction

7 updates in 19m07s = **162.7 s/update**, not the ~120 s/update that Phase 0's
2-update smoke implied. At 1280, a 100-update Stage A is **~4.5 h, not ~3.5 h**.
Any raised cap makes it longer. The MVP schedule to 2026-08-23 needs re-planning
around 4.5 h as the floor.
