# Phase 1 — Stage A (Math GRPO, Qwen2.5-7B + LoRA, group 8) — live run log

**Started:** 2026-08-17 ~23:21 (Colab clock)
**Config:** `exp2_colab_config_mvp.json` (hash `fc243e587296`) — 100 updates,
checkpoints [0, 50, 100], group 8, `exact_plus_boxed_format_0.1`, lr 2e-5 (LoRA only)
**Hardware:** Colab A100-SXM4-80GB High-RAM — the same warm runtime that passed
Phase 0 (`FINDING_7B_PHASE0_COMPLETE.md`)
**Run dir (in-runtime):** `/content/RLVR/eaaj-pilot/outputs/exp2_colab_guru_math7b_group8_fc243e587296/stage_a`
**Driver:** `/content/run_stage_a.py`, launched as a detached subprocess, log at
`/content/stage_a.log`

---

## 1. Why this deviates from notebook 01, and why the deviation is the cheaper default

Notebook 01 is the pre-registered Stage-A notebook. It was **not** used. Three
reasons, in order of force:

1. **It cannot run.** Its first cell clones the private repo using the Colab
   `GITHUB_TOKEN` secret. That PAT is confirmed broken (HTTP 403 —
   `exp2_both_tracks_blocked_at_phase0`). There is no path through notebook 01
   today.
2. **A fresh runtime is a gamble we already know how to lose.** Gate C0 measured
   a 40.14 GiB peak; Colab serves both the 40 GB and 80 GB A100 SKUs under one
   label, and an earlier session on 2026-08-16 drew the 40 GB part. The warm
   runtime is a *known* 80 GB card with deps installed, 7B weights cached and
   splits frozen. Restarting throws that away and re-rolls the dice.
3. **The science is identical.** The driver script calls
   `pipeline.run_stage_a_grpo` with exactly the arguments notebook 01 cell 5
   passes, all read from the same pre-registered config. No recipe value was
   retyped, tuned, or defaulted. Same seed (42), same run-dir hash.

**Subprocess rather than an in-cell call** is the one structural change. In a
cell, the kernel is blocked for the whole ~3.5 h, so nothing can inspect
progress or rescue an artifact off `/content` (ephemeral) until the run ends —
and if the runtime is reclaimed mid-run, everything is lost with no warning.
As a subprocess the kernel stays free, so a read-only monitor cell can be
re-run at any time. This is also what the v9 4070 probe already did (every
stage in a subprocess), so it is an established pattern here, not a new one.

## 2. `drive_backup_dir=None` — stated plainly

Notebook 01 passes a Drive mirror. This run passes `None`, because mounting
Drive is an OAuth grant that is the operator's call, not the assistant's, and it
was declined during Phase 0. **Consequence, stated rather than buried: if the
Colab runtime is reclaimed mid-run, the local run directory is wiped and the
completed updates are lost.** The mitigation is the free kernel — checkpoints
are pulled out as they land rather than only at the end. If the operator grants
Drive, the better protection is available immediately and should be taken.

## 3. Timeline

| Local time | Elapsed | State |
|---|---|---|
| 23:21 | 0:00 | launched, pid 21875. `VRAM free 76.4 / 79.3 GiB` after freeing the Phase-0 model |
| 23:24 | 0:03 | `ckpt-0` written (identity adapter, the stage-2-alone baseline). 54257 stage-A train rows loaded. Base weights loading. |
| 23:40 | 0:19 | **STOPPED at update 7/100.** `LocalSafetyCallback`: five consecutive updates >10% completion clipping (steps 3-7 = 0.188, 0.125, 0.156, 0.141, 0.156). `fixed_budget_completion` correctly refused the partial run: `RuntimeError: incomplete: requested 100, got 7`. |
| 23:55 | — | Launched the generation-only completion-length measurement the v10 draft prescribes (32 prompts x 8 gens, cap 3072). Trains nothing; changes no registered variable. |

| 00:35 | — | Measurement #1 done. **No candidate cap reaches the v10 rule's 5% target** (1536: 8.98%, 2048: 6.25%, 2560: 5.08%). Distribution is bimodal — p50 532, p95 2805, and 4.69% still running at 3072. Raising the cap is not the fix. See [`FINDING_COMPLETION_LENGTH_MEASUREMENT_7B.md`](FINDING_COMPLETION_LENGTH_MEASUREMENT_7B.md). |
| 00:40 | — | Launched measurement #2 on **Qwen2.5-7B-Instruct**, identical protocol, to quantify the base-vs-Instruct question rather than decide it. |

| 01:05 | — | Measurement #2 (Instruct) done: p95 **1338** vs base 2805; 1536 -> **2.34%** truncation. The v10 sizing rule now returns a qualifying candidate. |
| 01:20 | — | **RESTARTED** as `exp2_colab_guru_math7b_instruct_group8_e33527592dd9` — Qwen2.5-7B-Instruct, cap 1536, every gate unchanged. Config hash asserted in-runtime. Splits re-frozen to `exp2_colab_splits_instruct.json` (ids change with the chat template). Registration: [`EXPERIMENT_2_COLAB_7B_INSTRUCT_AMENDMENT.md`](EXPERIMENT_2_COLAB_7B_INSTRUCT_AMENDMENT.md). |

**Outcome of the BASE run: ckpt-0 only.** Full analysis and the correction it
forces on `FINDING_7B_PHASE0_COMPLETE.md` §5 are in
[`FINDING_7B_STAGE_A_CLIPPING_STOP.md`](FINDING_7B_STAGE_A_CLIPPING_STOP.md).

Two numbers to carry forward regardless of what happens next:
- **162.7 s/update**, so 100 updates is **~4.5 h**, not the ~3.5 h Phase 0's
  2-update smoke implied. The MVP schedule needs re-planning around 4.5 h.
- The 7B completion-length distribution is **the same as 0.5B's** (mean 615-717
  vs 719-753). Model scale did not help; the "7B clears the clipping gate"
  reading of Phase 0 was an artifact of a 2-update smoke being unable to fail a
  five-update streak gate.

## 4. What to watch for

- **`clip_ratio` / completion clipping.** The 0.5B v9 smoke died on this at the
  identical `max_completion_length=1280`; 7B cleared it over 2 updates, but
  Stage A completions grow as the policy learns to emit longer reasoning, so
  the margin can erode over 100 updates. This is the most likely scientific
  failure mode, not an infrastructure one.
- **Reward actually moving.** GATE 0b said 4/16 Stage-A groups vary on the
  exact channel. Low but non-zero. If mean reward is flat at update 50 the run
  still completes — that is a *result*, not a bug — but it changes what Stage B
  can show.
- **s/update drift.** Phase 0 extrapolated ~2 min/update from a 2-update smoke.
  If the true rate is materially worse, the MVP schedule to 2026-08-23 needs
  re-planning, and it is better to know at update 10 than at update 90.

---

## 5. Instruct run — periodic checks

Run `exp2_colab_guru_math7b_instruct_group8_e33527592dd9`, cap 1536, gates unchanged.

| updates | clip range (last 8) | max consecutive >10% | mean_len | reward | s/update |
|---:|---|---:|---:|---:|---:|
| 9 | 0.000 – 0.266 | 1 | 709 – 1038 | ~0.26 | 197 |
| 22 | 0.047 – 0.172 | 1 | 746 – 936 | ~0.25 | 198 |
| 31 | 0.016 – 0.141 | 1 | 774 – 1032 | ~0.24 | 203 |
| 40 | 0.016 – 0.125 | 1 | 733 – 982 | ~0.20 | 207 |
| 49 | 0.016 – 0.156 | 1 | 658 – 1008 | ~0.23 | 203 |
| 52 | 0.000 – 0.156 | 1 | 633 – 981 | ~0.28 | 201 |

Steady. Clipping stays mostly under the 10% gate and the consecutive-breach
streak has never exceeded 1 against a patience of 5. Mean completion length is
oscillating around ~875 with no upward trend since the step-6 peak of 1038.
Reward holds ~0.24–0.26, about 2.5x the base run's ~0.10.

At ~203 s/update, 100 updates is **~5.6 h**. `ckpt-50` is the next artifact.

### ckpt-50 landed (2026-08-18, ~04:45)

`checkpoints: ['ckpt-0', 'ckpt-50']` — **the first mid-training checkpoint any
exp2 track has produced.** Sizes on the ephemeral disk:

```
ckpt-0    155 MB
ckpt-50   165 MB
run dir   320 MB   (~485 MB once ckpt-100 lands)
```

**These exist only on Colab's ephemeral disk.** This is not a "survive until the
run finishes" problem — Phases 2/3 need these adapters *after* the run, so they
have to leave the runtime either way. `drive_backup_dir=None` was chosen because
mounting Drive is an OAuth grant that is the operator's call; that decision now
needs revisiting, and it is blocking on one click, not on any technical
obstacle.

## 6. Attempt 2 (after the reclaim) — periodic checks

Fresh 80 GB A100, identical config `e33527592dd9`, polling every 15 min.

| updates | clip (last few) | mean_len | reward | s/update | Drive |
|---:|---|---:|---:|---:|---|
| 4 | .016 .031 .047 .156 | 689 – 968 | .13 – .41 | 199 | NOT MOUNTED |
| 9 | .156 .047 .188 0 0 .047 | 726 – 1034 | .16 – .52 | 201 | NOT MOUNTED |
| 15 | .109 .016 .125 0 0 .172 | 676 – 945 | .13 – .41 | 197 | NOT MOUNTED |
| 20 | .172 .125 .031 .047 .063 .063 | 780 – 945 | .14 – .45 | 192 | NOT MOUNTED |
| 25 | .063 .141 .063 0 .063 .031 | 806 – 927 | .14 – .25 | 201 | NOT MOUNTED |
| 29 | .063 .031 .031 **.125 .109 .156** | 819 – 1036 | .14 – .34 | 199 | NOT MOUNTED |
| 32 | .156 .063 .031 .141 | 799 – 1036 | .10 – .28 | 202 | NOT MOUNTED |
| 37 | .141 .203 .078 .031 .047 .094 | 789 – 1000 | .10 – .27 | 207 | NOT MOUNTED |
| 42 | .094 .078 .031 .047 .063 .188 | 699 – 872 | .12 – .33 | 201 | NOT MOUNTED |
| 47 | .188 0 .094 .094 .047 .016 | 650 – 1004 | .13 – .33 | 196 | NOT MOUNTED |
| 51 **ckpt-50** | .047 .016 0 .063 .188 .063 | 708 – 997 | .22 – .33 | 197 | NOT MOUNTED |
| 56 | .063 .016 .141 0 .156 .078 | 684 – 977 | .21 – .40 | 199 | NOT MOUNTED |
| 61 | .078 0 0 .078 .156 0 | 668 – 1008 | .17 – .43 | 190 | NOT MOUNTED |
| 66 | 0 .063 .094 .078 .016 0 | 643 – 944 | .11 – .43 | 199 | NOT MOUNTED |
| 70 | .016 0 .016 .063 0 .125 | 730 – 866 | .18 – .27 | 197 | NOT MOUNTED |
| 76 | .078 .047 .063 .063 .016 0 | 717 – 911 | .15 – .30 | 201 | NOT MOUNTED |
| 80 | .016 0 .078 .016 .094 .203 | 717 – 1000 | .16 – .27 | 196 | NOT MOUNTED |

Reproducibility spot-check against attempt 1: step-3 `mean_len` is 953.06 here
against 953.2 there, and step-4 clip 0.156 against 0.156. The seed is doing its
job; the residual difference is ordinary sampling/CUDA nondeterminism, not a
different recipe. At 9 updates the match is stronger still: the mean-length
transient peaks at step 6 in both runs (1034 here, 1038 there) and collapses at
step 7 in both (726 vs 731). The step-6 length spike is a **reproducible feature
of this recipe**, not a one-off.

### 29/100 — first 3-update breach streak

Steps 27, 28, 29 all exceeded the 10% clipping limit (.125, .109, .156) with
mean length climbing 890 -> 928 -> 1036. **Streak 3 against a patience of 5** —
two more consecutive breaches stop the run.

No intervention. The cap is not being raised, the gate is not being touched, and
no third variable is moving. If it stops, that is the registered protection
doing its job and it is the reportable outcome; chasing it would mean an
unbounded sequence of untracked changes.

Worth noting the shape is familiar: attempt 1 showed the same mean-length
transient (peak 1038 at step 6) and recovered from it without the streak
completing. Whether this one recovers is the thing to watch, not to act on.

**Resolved at step 30 — the streak broke without intervention.** Clip fell to
.063, resetting the counter, and mean length came back 1036 -> 854 -> 799. This
is the second time the recipe has pulled itself out of a rising-length excursion
on its own. The gate's five-update patience is doing exactly what patience is
for: tolerating transients while still catching a genuine trend.

### 51/100 — ckpt-50 written (second time)

`checkpoints: ['ckpt-0', 'ckpt-50']`. Reward over updates 46-51 averages ~0.27,
above the ~0.20 of the middle stretch and well above the base run's ~0.10 — so
the Instruct arm is not just surviving the clipping gate, it is earning more
reward as it goes.

Two updates from here the run passes update 53, where attempt 1 was lost.

**Passed it at 18:55.** Everything from update 54 onward is ground attempt 1
never reached.

### 70/100 — the length distribution is narrowing

Through the middle of the run `mean_len` swung 650-1040; over updates 65-70 the
band is 730-866 with no 1000+ excursions, while the learning rate has decayed to
6.8e-6. Clipping follows: one breach in six updates. The policy is settling
rather than drifting, which is the opposite of the failure mode the base run
showed and is the strongest sign yet that the cap at 1536 is genuinely adequate
for this variant rather than marginally so.
