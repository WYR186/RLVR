# Analysis handoff — exp2 7B Colab MVP (Person 4 deliverable)

Copy everything below the line into the analyst's context. It is self-contained
except for the repo itself.

---

## Your task

Analyse a completed multi-stage RLVR experiment and write the results + limitations
sections for the team Research Doc. **All measurement is finished; do not run any
training.** Everything you need is committed in the repo.

The project ("Seeing the Stall Coming") asks whether activation-based plasticity
metrics Q — effective rank and dormant-neuron fraction — measured during RL stage A
give early warning that a later RL stage B will stall, with more lead time than
dashboard signals (reward slope, grad norm, entropy).

**The honest headline you are analysing is a null plus a scoping failure**, and your
job is to characterise both precisely rather than rescue a positive finding. Read
§"What the run actually shows" before forming a plan.

## The experiment as run

Qwen2.5-**7B-Instruct**, LoRA (r=16, alpha=32, all attn+MLP projections), GRPO via
TRL, group size 8, bf16 base + fp32 adapter, seed 42 only.

- **Stage A** — GURU-RL-92k **Math**, 100 updates, `max_completion_length=1536`,
  reward = exact-match + 0.1 boxed-format bonus, beta=0 (no KL term).
  Checkpoints saved at updates 0 / 50 / 100.
- **Phase 2** — at each checkpoint: zero-shot transfer score T_t on the frozen
  Simulation eval set, and Q on a frozen 4096-prompt probe set at layers [5, 14, 26].
- **Stage B** — GURU **Simulation** (CodeI/O), fixed budget of **30 GRPO updates**
  from each checkpoint, `max_completion_length=2048`, 1132 train / 300 eval questions,
  frozen splits. Outcome is Delta-R = acc_after - acc_before.

## Where the data is

Run dir: `eaaj-pilot/outputs/exp2_colab_guru_math7b_instruct_group8_e33527592dd9/`

```
stage_a/dashboard.jsonl          101 rows, per-update reward/entropy/grad_norm/
                                 loss/clipped_ratio/completion lengths/step_time
stage_a/summary.json             completion status, wall time
stage_a/update_sentinel.jsonl    proof the LoRA adapter actually moved
analysis/transfer_T.json         zero-shot T_t per checkpoint
measurements/metrics_ckpt{0,50,100}.json
                                 per-layer erank, erank_norm, participation_ratio,
                                 dormant_frac at tau 0.025 and 0.1, anisotropy,
                                 topK variance shares, plus whole-model weight norms
measurements/metrics_ckpt0_recheck.json
                                 an independent re-measurement of ckpt-0
stage_b_v2/ckpt-{0,50,100}/
   summary.json                  acc_before, acc_after, delta_acc, wall_seconds
   dashboard.jsonl               30 rows per arm, same schema as stage A
   stageb_eval_curve.jsonl       step-30 eval: accuracy, n_correct, eval_seconds
   update_sentinel.jsonl         updates_effective flag + relative param change
   baseline.json                 acc_before
```

Pre-registered recipes, both hash-verified in-run:
`experiment 2/exp2_colab_config_mvp_instruct.json` (`e33527592dd9`) and
`exp2_colab_config_mvp_instruct_stageb_v2.json` (`bd99ddd2817f`).

Context you should read before analysing, in this order:
`FINDING_STAGE_B_DELTA_R.md`, `FINDING_STAGE_B_CAP_SIZING.md`,
`FINDING_Q_METRICS_7B_INSTRUCT.md`, `FINDING_TRANSFER_T_7B_INSTRUCT.md`,
`EXPERIMENT_2_COLAB_7B_INSTRUCT_AMENDMENT.md`.

## What the run actually shows

**Delta-R (the outcome):**

```
ckpt-0    52 -> 83 /300   Delta-R +0.1033
ckpt-50   57 -> 86 /300   Delta-R +0.0967
ckpt-100  56 -> 83 /300   Delta-R +0.0900
```

All three arms completed 30/30 updates. Full Delta-R range is 1.33 pp against an
arm-to-arm SE of roughly 3.0-3.9 pp. **Delta-R is flat.**

**Q (the predictor) barely moves.** Across ckpt 0 -> 50 -> 100, effective rank
changes by <=0.55% at every measured layer (e.g. layer 26: 1426.1 / 1433.9 / 1432.5),
and dormant fraction is **exactly 0.0000 at both thresholds, at every layer, at every
checkpoint**.

**So neither side of RQ1's correlation varies.** This run cannot test whether Q
predicts a stall — there is no stall, and no variance in Q. It is equally consistent
with the hypothesis and with its negation. That is a limitation to state plainly, not
a negative result about Q.

## Hard constraints on what you write

1. **Never claim "RLVR reduces the model's ability to learn."** Every claim must be
   about *fixed-budget adaptability* on a *named* held-out task family at a *named*
   budget, versus a *named* baseline. Here: Simulation/CodeI-O, 30 GRPO updates,
   ckpt-0 baseline.
2. **Do not report the monotone ordering as a trend.** The Delta-R point estimates
   fall evenly (+31, +29, +27 net questions). Three points go monotone by chance one
   time in three; the range is inside noise; and `acc_after` is itself *not* monotone
   (83, 86, 83) — the monotone look comes entirely from subtracting a non-monotone
   `acc_before`. Treat it as a hypothesis for a powered run, nothing more.
3. **Leakage rule:** any feature computed at step t may use only information
   available at step t. No post-hoc normalisation across a whole run.
4. **n=1 seed.** Seed 42 only; seeds 43/44 were registered as stretch goals and not
   run. Every interval you report must reflect that.
5. Four deviations from the pre-registered spec were implemented, logged, and flagged
   to the team lead but **not authorised by him**: base -> Instruct, Stage-A cap
   1280 -> 1536, Stage-B cap 640 -> 2048, and Stage-B eval points
   [0,10,20,30] -> [0,30]. Say so in the limitations; if any is rejected the affected
   arms are discarded.

## Known validity checks that already passed — verify, do not redo

- All three Stage-B `acc_before` values reproduce their `transfer_T.json` entries
  **exactly** (0.1733 / 0.1900 / 0.1867), measured on different VMs either side of a
  runtime recycle.
- `metrics_ckpt0_recheck.json` reproduces `metrics_ckpt0.json` exactly.
- `update_sentinel.jsonl` reports `updates_effective: true` on all three Stage-B arms
  (1.15% / 1.31% / 1.23% relative parameter change), excluding the "learning rate too
  small, every update rounds to zero" failure mode by measurement.
- The registered >10%-completion-clipping-for-5-consecutive-updates stop never fired:
  across 90 Stage-B updates the streak never exceeded 1.

## Deliverables

1. **Independent recomputation.** Rebuild every headline number from the raw
   artifacts. Report any discrepancy rather than reconciling it silently.
2. **Correct uncertainty on Delta-R.** Note the blocker first: the eval saved only
   aggregate `n_correct`, not per-question outcomes, so a paired test (McNemar) is
   **not computable post-hoc**. State the bound you can compute, and record
   "pass `return_details=True` in `guru_greedy_accuracy`" as a required fix for the
   next run.
3. **Dashboard signals vs Q.** This is the owner's own assignment. Over Stage A's 101
   rows, do reward slope, entropy, grad norm, or completion-length statistics move at
   all while Q is flat? A dashboard signal that moves when Q does not is informative
   about instrumentation sensitivity even with no stall to predict. Note beta=0, so
   there is no KL column.
4. **Power analysis for a run that could actually test RQ1.** Given the measured
   effect sizes and variances, state concretely what is needed: how many eval
   questions, how many checkpoints, how many seeds, and — critically — what training
   regime would produce *any* variance in Q, since 100 LoRA updates at this scale
   produced none.
5. **Limitations section**, covering: n=1 seed; Q's near-zero variance; the four
   unauthorised deviations; the 512-token cap inside `guru_greedy_accuracy` that
   truncates roughly 10-15% of this domain's eval answers (left at the registered
   default deliberately — changing it redefines R); and the missing per-question
   outcomes.
6. **One figure**: Delta-R by checkpoint with error bars, with Q on a secondary axis.
   It should make the "nothing varies on either axis" point visually obvious rather
   than inviting the reader to see a trend.

## Out of scope

The parallel Qwen2.5-0.5B WIN4070 track is a separate lineage, still paused on its
own amendment, and is **not** comparable to these numbers — different model,
different completion caps. Do not pool them.
