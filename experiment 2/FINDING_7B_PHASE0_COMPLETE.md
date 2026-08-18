# Result — 7B MVP Phase 0 passed end to end (first complete Phase 0 in exp2)

**Date:** 2026-08-16
**Config:** `exp2_colab_config_mvp.json` — Qwen2.5-7B **base**, LoRA, group 8,
Stage A 100 updates, checkpoints [0, 50, 100]
**Hardware:** Colab **A100-SXM4-80GB** (High-RAM), bf16
**Notebook:** `experiment 2/colab/00_phase0_selfcontained.ipynb`
**Status:** every Phase-0 gate PASSED. Nothing stopped. This is the first time
either exp2 track has cleared Phase 0.

---

## 1. Gate results

| Gate | Result |
|---|---|
| GPU gate | PASS — A100-SXM4-80GB, cap 8.0, 80.0 GiB |
| GATE 0a (token audit) | PASS on the eligible population; raw audit still matches the WIN4070 reference exactly |
| Splits freeze | completed (see §4 — the `stage_b_train=0` bug found earlier today is fixed) |
| Gate C0 (memory) | **PASS** — peak 40.14 GiB / 79.25 GiB, headroom 49.3% |
| GATE 0b (sparse-reward preflight) | **PASS on both stages** |
| Smoke (2 updates/stage) | **PASS** — both stages completed 2/2 |

## 2. Gate C0 — first real 7B + LoRA + group-8 memory number

```json
{"peak_allocated_gib": 40.14378309249878,
 "total_device_gib": 79.250732421875,
 "headroom_pct": 49.345852251810626,
 "gate_pass": true}
```

**A 40 GB A100 is not enough.** Peak allocation is 40.14 GiB, which exceeds the
usable capacity of the 40 GB part before any headroom. This run happened to draw
the 80 GB SKU; an earlier attempt today drew the 40 GB one. Colab hands out both
under the same "A100" label, so **the runtime must be checked, not assumed** —
if Phase 1 lands on a 40 GB A100 it will OOM.

For contrast, the 0.5B track at the identical group-8 geometry needed ~27 GiB
and OOM'd on a 22 GiB L4. Scaling 0.5B -> 7B cost only ~13 GiB more, because
LoRA keeps the optimizer state tiny; the bulk is activations and the logits
tensor, which are geometry-driven, not parameter-driven.

## 3. GATE 0b — the 7B reward signal is qualitatively better than 0.5B's

```
Stage A (Math)      : combined-variable groups 12/16, exact-variable groups 4/16, has_grpo_signal=True
Stage B (Simulation): combined-variable groups  5/8,  exact-variable groups 5/8, has_grpo_signal=True
GATE 0b: PASS on both stages
```

The column that matters is **exact-variable**, not combined.
`FINDING_GROUP_SIZE_REWARD_VARIANCE.md` §9 warned that combined variance
overstates the group-8 gain because most of it is the 0.1 boxed-format bonus —
formatting signal, not reasoning signal. On the 0.5B track that warning was
borne out: n_boxed=107 against n_exact_correct=6, i.e. the recovered variance
was almost entirely format-shaping.

At 7B, Stage A has 4/16 groups with genuine exact-match variance, and Stage B
has **5/8 — every variable group is variable on the exact channel**. So the
group-8 gradient here carries real reasoning signal, not just format noise.
This is the strongest argument so far that the >=7B ask was worth honouring.

## 4. Base vs Instruct — the contingency does NOT need to fire

`model_variant_contingency` in both configs says: if Phase 0's preflight shows
persistent `\boxed{}` failures on the base model, switch to Qwen2.5-7B-Instruct
and log the deviation. **It does not need to fire.** `has_grpo_signal=True` on
both stages, and the exact channel is live, so format-following has not
collapsed at this scale. The config author's reasoning ("7B >> 0.5B and
format-following is expected to be far less of a problem at this scale") is now
verified rather than assumed.

Practical consequence: the pre-authorisation we were going to ask Tommy for
(switch to Instruct if the preflight failed) is **no longer needed**. One fewer
open question.

## 5. Smoke — no clipping stop

```
stage A smoke completed 2/2 smoke updates OK    (loss 0.085340 -> 0.005556)
stage B smoke completed 2/2 smoke updates OK
```

The 0.5B v9 run died exactly here, on
`stage_a_smoke_completion_clipping_exceeded_limit` (14.06% vs a 10% gate) at the
same `max_completion_length=1280`. **The 7B run cleared it.** Longer, better
completions from the larger model apparently finish inside the 1280 cap often
enough to stay under the clipping gate.

**CORRECTED 2026-08-18 — this inference was wrong.** A 2-update smoke cannot
fail a gate that requires a *five*-update streak, so clearing it was never
evidence about the streak. Given 7 real updates, Stage A stopped on exactly this
gate with clip ratios 0.078-0.188 and mean completion lengths 615-717 - i.e.
statistically the same distribution as the 0.5B track. Truncation is a property
of the recipe and the Math population, **not** of model scale, and v10 is on the
critical path for the 7B deliverable too. See
[`FINDING_7B_STAGE_A_CLIPPING_STOP.md`](FINDING_7B_STAGE_A_CLIPPING_STOP.md).

## 6. Measured wall times (A100 80 GB)

| Phase-0 step | Wall time |
|---|---|
| Gate C0 memory probe | ~5 min |
| GATE 0b preflight (16 Stage-A + 8 Stage-B prompts x 8 gens) | **17 min** |
| Smoke, 2 updates x 2 stages | **8 min** |

The smoke figure is the one that matters for scheduling: 2 Stage-A updates and
2 Stage-B updates in 8 minutes. Taken naively that is ~2 min/update, so Stage A
at 100 updates is roughly **3.5 h**, and the three Stage-B cells at 30 updates
each roughly **3 h**, plus Phase 2/2b. That fits the MVP scope inside the
2026-08-23 deadline **only if runtimes stop being reclaimed** (see §8).

Treat these as lower bounds: the smoke runs 2 updates including warm-up, so
extrapolation has wide error bars, and Stage A's completions grow as the policy
learns to emit longer reasoning.

## 7. Artifacts

The frozen splits and audits were written to `/content/RLVR/experiment 2/data`
inside the runtime. The persistence cell offered Drive; Drive access was
**declined** (granting OAuth is the operator's call, not the assistant's), so it
fell back to emitting gzip+base64 into the cell output. That blob could not be
transcribed because Colab virtualises long outputs — only the viewport-adjacent
portion is in the DOM.

This is recoverable and not blocking: `build_exp2_splits` is seeded
(`seed=42`) and deterministic, so re-running Phase 0 reproduces byte-identical
splits. If the splits are wanted in the repo without a re-run, mount Drive on
the next run and let the first branch of the persistence cell take it.

## 8. The real remaining risk is operational, not scientific

Six runs were lost today to Colab reclaiming the runtime for inactivity. The run
that finally completed did so because the operator kept the tab in the
foreground and interacted with it. Phase 1 needs **multi-hour** sessions, far
longer than any Phase-0 step. Before committing to the MVP schedule, decide how
those sessions will be kept alive — Colab Pro+ background execution is the
obvious answer and is likely cheaper than repeatedly losing A100 hours.

## 9. What this unblocks

- The 7B track is **cleared to start Phase 1 (Stage A)** — this is the arm that
  answers Tommy's ">= 7B" ask.
- The `formal_stage_a_eligible: false` flag in the v9 4070 contract is about the
  0.5B variant chain and does not gate this track.
- The MVP promotion gate can now be evaluated: measured s/update says full scope
  (200 updates) would be ~7 h for Stage A alone, so **stay on MVP**.
