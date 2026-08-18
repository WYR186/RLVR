# Experiment 1 — Results Explained

**Author:** Aaron Wang
**Audit date:** 2026-07-23  
**Companion data:** `exp1_result_aaron.zip` → `pilot_evidence_pack/`  
**Scope:** Experiment 1 only
**Framing:** fixed-budget future adaptability on a specified task and budget; this document does
not claim that RLVR changes a model's general or intrinsic “ability to learn.”

---

**Bottom line:** the pipeline produced useful diagnostic evidence, but Experiment 1 does not
establish effective rank as a predictor and does not show a reproducible loss of fixed-budget
SVAMP adaptability relative to checkpoint 0. RQ1 remains open.

---

## What was run

Stage A used the pinned base-model revision
`Qwen/Qwen2.5-0.5B@060db6499f32faf8b98477b0a26969ef7d8b9987`, 512 frozen GSM8K
training questions, binary exact-answer reward, and 200 GRPO updates. Checkpoints were saved at
0/25/50/100/200. At every checkpoint, Q was measured on a frozen 512-prompt probe at layers
4/12/22, followed by a nominal 50-update GRPO adaptation on 256 frozen SVAMP training questions
and greedy exact-answer evaluation on 100 frozen held-out SVAMP questions.

The primary registered analysis was:

`Spearman rho(erank_L12, SVAMP accuracy after adaptation − accuracy before adaptation), n = 5`.

### Execution strata and repeats


| Evidence folder              | Stage A                             | Stage B seed | Execution profile                                        | Status                                       |
| ---------------------------- | ----------------------------------- | ------------ | -------------------------------------------------------- | -------------------------------------------- |
| `mac_run/`                   | separate seed-42 execution          | 42           | CPU, float32                                             | 4 valid 50-step cells; ckpt-25 stopped at 30 |
| `win_run/adaptation_seed42/` | one Windows seed-42 trajectory      | 42           | RTX 4070, fp32 master + bf16 autocast, paged AdamW 8-bit | 5/5 manually audited complete                |
| `win_run/adaptation_seed43/` | reuses the same Windows checkpoints | 43           | same Windows profile                                     | 5/5 strict-validator complete                |
| `win_run/adaptation_seed44/` | reuses the same Windows checkpoints | 44           | same Windows profile                                     | 1/5 complete                                 |


Mac and Windows are **separate execution strata**, not interchangeable observations and not
independent statistical replicates: they share the same seed, model revision, and frozen logical
splits, but differ in hardware, precision, micro-batch geometry, and optimizer implementation.
All Windows Stage-B seeds reuse one Stage-A trajectory. That Windows trajectory also crossed
repaired interruption boundaries where optimizer moments and RNG state were not restored, so it
is not equivalent to an uninterrupted run.

### Registration and provenance audit

- The core model, task, seed, LR, beta, temperature, checkpoint schedule, and nominal adaptation
budget are recorded in `recipe/pilot_config.json`.
- Package versions and tracked-input hashes are in each run's `manifest.json`; execution overrides
are in each run's `config.json`. The split-file hashes differ across Mac and Windows only because
Windows wrote CRLF line endings; after newline normalization, the logical files match.
- The exact split files and source files are **not included in the Slack zip**; the zip contains
their hashes. A teammate needs the repository to verify split membership from those hashes.
- Mac's manifest has `git_sha: null`, so it has file-level hashes but weaker commit-level
provenance than Windows.
- Model weights and optimizer states are intentionally outside the evidence pack. The pack is
sufficient to audit reported tables and completion logs, not to reproduce inference from the
zip alone.
- Registered Q dtype was `float16`; actual `metrics_ckpt*.json` files record `torch.float32`
model evaluation, float32 activation accumulation, and float64 SVD. Both strata used the same
actual protocol, but the override was not mirrored into the shared recipe file.

---

## Result 1 — Stage A changed the model

Verify with `*/dashboard.jsonl`, `*/gsm8k_eval.jsonl`,
`*/sparse_reward_preflight.json`, and Windows `phase1_update_sentinel.jsonl`.


| Stratum | Reward mean, first 10 → last 10 updates | Entropy, update 1 → 200 | GSM8K accuracy, step 0 → 200 | Sparse-reward preflight  |
| ------- | --------------------------------------- | ----------------------- | ---------------------------- | ------------------------ |
| Mac     | 0.364 → 0.653                           | 0.253 → 0.137           | 0.438 → 0.516                | 7/8 prompt groups varied |
| Windows | 0.334 → 0.588                           | 0.247 → 0.116           | 0.359 → 0.422                | 8/8 prompt groups varied |


The held-out accuracy curves are not monotone, but both endpoints exceed checkpoint 0. On Windows,
all eight 25-update sentinel windows passed; relative weight change per window ranged from
`6.58e-6` down to `3.23e-7`, above the `1e-8` warning threshold. This supports the narrow statement
that training updates were effective. It does not make the two trajectories equivalent.

---

## Result 2 — Full-budget Stage-B audit

Endpoint SVAMP accuracy change is `delta = accuracy_after − accuracy_before`. Baseline evaluation
is stored separately; the learning-curve files contain evaluations at updates 10/20/30/40/50.


| Stage-A checkpoint | Mac s42    | Windows s42 | Windows s43 | Windows s44 |
| ------------------ | ---------- | ----------- | ----------- | ----------- |
| 0                  | +0.05      | +0.06       | +0.04       | +0.00       |
| 25                 | **+0.13†** | +0.11       | +0.08       | —           |
| 50                 | +0.08      | +0.03       | +0.09       | —           |
| 100                | +0.02      | +0.05       | +0.06       | —           |
| 200                | +0.04      | +0.12       | +0.05       | —           |


† **Invalid for the fixed-50 comparison:** the Mac checkpoint-25 dashboard ends at update 30,
`svamp_eval_curve.jsonl` has only steps 10/20/30, and the last trainer checkpoint is 30. The
legacy summary's `budget_updates: 50` is the requested budget, not proof that 50 updates occurred.

Two counts should therefore be kept separate:

- **All endpoint summaries present in the pack:** 16 total = 15 positive, 1 flat, 0 negative.
- **Auditable 50-update cells:** 15 total = 14 positive, 1 flat, 0 negative.

Among the valid cells there is no negative endpoint delta. More importantly, there is no
checkpoint that shows a reproducible loss of delta relative to checkpoint 0 across execution
strata and Stage-B seeds. This is evidence of **no reproducible fixed-budget adaptability loss in
this pilot**, not proof that plasticity collapse is absent in general.

The evidence pack's `phase3_complete.json` and README should not be treated as completion proof by
themselves: both incorrectly accept the truncated Mac cell. The strict completion validator was
added for the later seed-repeat path; legacy seed-42 cells require artifact-level manual audit.

---

## Result 3 — Dormant fraction is constant; effective rank moves

Verify with `*/measurements/metrics_ckpt*.json`.

### Dormant fraction

Dormant fraction is exactly `0.0` at every checkpoint, at layers 4/12/22, in both strata, for both
registered thresholds (`tau = 0.025` and `0.1`). A constant feature has undefined Spearman
correlation, which explains the blank rows in `spearman_table.csv`. This metric has no usable
dynamic range in Experiment 1.

### Layer-22 effective rank


| Checkpoint  | 0     | 25    | 50             | 100   | 200            |
| ----------- | ----- | ----- | -------------- | ----- | -------------- |
| Mac L22     | 354.2 | 321.9 | 306.3 (−13.5%) | 311.4 | 314.7 (−11.1%) |
| Windows L22 | 354.2 | 321.3 | 325.7          | 317.0 | 324.9 (−8.3%)  |


The late-layer spectral change is a reproducible **representation diagnostic**, not evidence by
itself of impaired future adaptability.

The 2048-prompt sensitivity check confirms that absolute erank depends strongly on probe size,
while the checkpoint-0 to checkpoint-200 relative L22 change is similar:


| Stratum | 512-prompt L22 change | 2048-prompt L22 change |
| ------- | --------------------- | ---------------------- |
| Mac     | −11.15%               | −12.12%                |
| Windows | −8.26%                | −7.71%                 |


Absolute erank values are comparable only under a like-for-like measurement contract: same probe
membership and size, layer/site, pooling, dtype, and preprocessing. The sample-size cap does not
make all cross-machine comparison meaningless; the nearly identical checkpoint-0 values are a
useful implementation-consistency check because the actual protocol and source model were aligned.

At checkpoint 0, `erank_L12` is 231.7567 on both machines to four decimals. This shows that the
two environments computed the same quantity on the same source model. It does **not** validate
effective rank as a predictor, nor does it erase later trajectory differences.

---

## Result 4 — The registered predictor is not reproducible


| Analysis        | Spearman rho(erank_L12, delta) | p-value                                    | Audit status                               |
| --------------- | ------------------------------ | ------------------------------------------ | ------------------------------------------ |
| Mac seed 42     | −0.60                          | 0.285                                      | **Legacy only; invalid fixed-budget grid** |
| Windows seed 42 | +0.50                          | 0.391                                      | Valid five-point descriptive estimate      |
| Windows seed 43 | −0.50                          | not preregistered in the original analyzer | Valid seed-repeat descriptive estimate     |
| Windows seed 44 | —                              | —                                          | Incomplete: only checkpoint 0              |


The strongest evidence is the Windows seed comparison because it holds the five Stage-A
checkpoints fixed and changes only Stage-B stochasticity. The best endpoint moves from checkpoint
200 at seed 42 (`+0.12`) to checkpoint 50 at seed 43 (`+0.09`), and rho changes from `+0.50` to
`−0.50`.

With only five checkpoints, a 100-question fixed eval set, and one Stage-B trajectory per
checkpoint per seed, the checkpoint ranking is unstable. The current evidence therefore neither
establishes effective rank as a reliable predictor nor establishes a reproducible plasticity
collapse.

---

## Side finding — the pure-bf16 no-op run

An earlier Windows v1 execution used pure bf16 parameters at `lr = 1e-6`. Its maximum
checkpoint-0 to checkpoint-200 relative change in any logged parameter-group norm was only
`2.45e-8`, reward was effectively flat, and Q barely moved. Those observations are consistent
with the overwhelming majority of low-magnitude updates being swallowed by bf16 quantization.

The safe wording is **“effectively a no-op execution”**, not “literally every scalar update was
zero.” It is useful as an accidental execution diagnostic, but it was not a randomized,
pre-registered scientific negative control and should not be used as evidence for RQ1.

The Colab notebook has already been repaired: it now loads float32 master weights and uses bf16
autocast, with an update-effectiveness sentinel. The action item is to preserve and test that
numerical contract, not to claim the notebook is still unfixed.

---

## Interpretation and next steps

What Experiment 1 supports:

- The end-to-end instrumentation and artifact path is usable.
- Layer-22 effective rank changes under Stage A in both execution strata.
- Dormant fraction, as implemented at the registered thresholds, is uninformative here.
- The single-seed, five-checkpoint endpoint correlation is not stable to Stage-B seed.

What it does not support:

- A general claim that RLVR reduces a model's ability to learn.
- A valid Mac five-point fixed-budget correlation.
- A reliable effective-rank predictor.
- A reproducible loss of fixed-budget SVAMP adaptability relative to checkpoint 0.
- A formal stall-detector result; this pilot has no proposal-level stall labels or dashboard
bake-off.

Priority fixes:

1. Finish seed 44 and report per-checkpoint uncertainty across Stage-B seeds.
2. Use a completion-gated analysis table that rejects any cell without 50 actual updates and the
  full 10/20/30/40/50 curve.
3. Add a less endpoint-sensitive outcome, such as a registered learning-curve AUC and/or pass@k.
4. Increase checkpoint density before treating a rank correlation as inferential evidence.
5. Freeze one team-wide Q protocol, including activation site, pooling unit, probe size, and dtype.
6. Either repair and rerun Mac checkpoint 25 or exclude it permanently from fixed-budget analysis.

---

## Teammate-safe summary

Experiment 1 successfully exercised the training, Q-measurement, and Stage-B adaptation pipeline,
but a file-level audit found that one Mac cell (checkpoint 25) stopped at 30/50 adaptation updates
despite being marked complete. Excluding that cell leaves 15 auditable 50-update adaptations:
14 positive endpoint deltas, 1 flat, and 0 negative. Layer-22 effective rank fell about 8–14% and
dormant fraction stayed exactly zero, but the valid Windows primary correlation changed from
+0.50 at Stage-B seed 42 to −0.50 at seed 43 on the same checkpoints. Therefore RQ1 remains open:
the current evidence does not show a reproducible fixed-budget adaptability loss and does not
validate effective rank as a predictor.

## External technical cross-checks

- The pinned Qwen config reports 24 hidden layers and hidden size 896:
[https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/060db6499f32faf8b98477b0a26969ef7d8b9987/config.json](https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/060db6499f32faf8b98477b0a26969ef7d8b9987/config.json)
- TRL's GRPO documentation confirms the meaning of `num_generations` and the logged reward,
entropy, length, and zero-reward-variance fields:
[https://huggingface.co/docs/trl/grpo_trainer](https://huggingface.co/docs/trl/grpo_trainer)
- PyTorch documents `torch.finfo(...).eps` as the spacing from 1.0 to the next representable
floating-point value; the run-specific no-op diagnosis still rests on the local weight/reward/Q
artifacts:
[https://docs.pytorch.org/docs/stable/type_info.html](https://docs.pytorch.org/docs/stable/type_info.html)

