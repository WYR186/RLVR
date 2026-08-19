# Analysis report — exp2 7B Colab MVP

**Analyst pass over the completed run.** Written against
[`ANALYSIS_HANDOFF_PROMPT.md`](ANALYSIS_HANDOFF_PROMPT.md); every number below was
recomputed from the raw artifacts rather than copied from the finding docs.

**Run:** `eaaj-pilot/outputs/exp2_colab_guru_math7b_instruct_group8_e33527592dd9/`
**Recipes:** `exp2_colab_config_mvp_instruct.json` (`e33527592dd9`),
`exp2_colab_config_mvp_instruct_stageb_v2.json` (`bd99ddd2817f`)
**Model:** Qwen2.5-7B-Instruct + LoRA r=16 / alpha=32 / dropout 0.05, all attn+MLP
projections, bf16 base + fp32 adapter, **seed 42 only**
**Figure:** `outputs/.../analysis/delta_r_vs_q.png` (`.svg` alongside)

---

## 0. Headline

The experiment is **internally clean and externally uninformative about RQ1**, and
the reason is sharper than "the metrics didn't move":

> Across 100 Stage-A GRPO updates, **no measured quantity in the run changed
> detectably** — not the plasticity metrics Q, not any dashboard signal, not the
> zero-shot transfer score, and not fixed-budget adaptability. The full-layer weight
> perturbation delivered by Stage A is ~3 x 10^-7 in relative Frobenius norm.

RQ1 asks whether Q predicts a later stall. This run contains **no stall and no
variance in Q**, so it cannot test the question in either direction. That is a
statement about the *dose of RLVR administered*, not about Q's merit as a predictor.

What the run does establish, and establishes well: the pipeline is correct, the
outcome measure Delta-R is real and significantly positive on every arm, and we now
have measured constants to size a run that could actually test RQ1.

---

## 1. Independent recomputation — all headline numbers reproduce

Recomputed from `summary.json`, `baseline.json`, `stageb_eval_curve.jsonl`,
`update_sentinel.jsonl`, `dashboard.jsonl`, `analysis/transfer_T.json`,
`measurements/metrics_ckpt*.json`. **No discrepancies found.**

| arm | acc_before | acc_after | Delta-R | hits | updates | wall |
|---|---|---|---|---|---|---|
| ckpt-0 | 0.173333 | 0.276667 | **+0.103333** | 52 → 83 / 300 | 30/30 | 11 976 s |
| ckpt-50 | 0.190000 | 0.286667 | **+0.096667** | 57 → 86 / 300 | 30/30 | 12 184 s |
| ckpt-100 | 0.186667 | 0.276667 | **+0.090000** | 56 → 83 / 300 | 30/30 | 11 853 s |

Checks performed:

- `acc_after - acc_before == delta_acc` to floating-point exactness, all arms.
- All six accuracies are exact multiples of 1/300 — no averaging or resampling
  contaminated the counts.
- `baseline.json` `acc_before` equals `summary.json` `acc_before`, all arms.
- The step-30 eval-curve accuracy equals `acc_after` **exactly** on all three arms.
  These are two separate invocations of a greedy (`do_sample=False`) decode on the
  same rows and the same model state, so exact equality is the expected result and
  confirms the eval is deterministic.
- `completion_status: complete`, `actual_updates == requested_updates == 30`, all arms.
- All three `acc_before` reproduce `analysis/transfer_T.json` **exactly**
  (0.173333 / 0.190000 / 0.186667), despite being measured on a different VM either
  side of a runtime recycle. This is the strongest end-to-end check in the run: it
  validates split restoration, adapter restoration, and the eval path simultaneously.
- `measurements/metrics_ckpt0_recheck.json` is **bit-identical** to
  `metrics_ckpt0.json` across all 12 metrics x 3 layers.
- Safety gate: across all 90 Stage-B updates, the ">10% completion clipping for 5
  consecutive updates" streak reached a **maximum of 1**. The gate never came close
  to firing.
- `update_sentinel.jsonl` reports `updates_effective: true` on all three arms, with
  relative LoRA-parameter change 1.15% / 1.31% / 1.23%.

---

## 2. Uncertainty on Delta-R — a worst-case bound that closes the gap

### 2.1 The blocker, stated first

`pipeline.guru_greedy_accuracy` was called without `return_details=True`, so only the
aggregate `n_correct` was persisted. **A McNemar test is therefore not computable
post-hoc.** Writing `return_details=True` into the next run is a required fix, listed
again in §7.

### 2.2 What can still be computed — and it is enough

The design is paired: the same 300 questions before and after. Let `b` = correct
before / wrong after, `c` = wrong before / correct after. Only `c - b` (the net
gain) is observable. But `b` is bounded: `0 <= b <= before_hits`. McNemar's
statistic `z = (c-b)/sqrt(b+c) = net/sqrt(2b+net)` is **monotonically decreasing in
`b`**, so `b = before_hits` gives the worst case and the whole feasible interval can
be enumerated.

**Within-arm — is Stage B learning at all?**

| arm | net | feasible n_discordant | z range | worst-case two-sided p |
|---|---|---|---|---|
| ckpt-0 | +31 | 31 – 135 | 2.668 – 5.568 | **0.0076** |
| ckpt-50 | +29 | 29 – 143 | 2.425 – 5.385 | **0.0153** |
| ckpt-100 | +27 | 27 – 139 | 2.290 – 5.196 | **0.0220** |

**Every arm's Delta-R is significantly positive at p < 0.05 for every feasible value
of the unobserved discordance.** Fixed-budget Stage B genuinely learns on all three
arms; that conclusion does not depend on the missing data at all.

**Between-arm — is Delta-R flat?** Treating arms as independent is conservative here
(they share the eval set, and positive correlation would only shrink the SE of the
difference).

| contrast | difference | SE(diff) range | \|z\| range |
|---|---|---|---|
| ckpt-0 − ckpt-100 | +1.33 pp (+4 q) | 2.54 – 5.52 pp | **0.242 – 0.525** |
| ckpt-0 − ckpt-50 | +0.67 pp (+2 q) | 2.58 – 5.56 pp | 0.120 – 0.258 |
| ckpt-50 − ckpt-100 | +0.67 pp (+2 q) | 2.49 – 5.60 pp | 0.119 – 0.267 |

`|z|` never exceeds ~0.53 **anywhere** in the feasible range. **The flatness
conclusion is robust to the missing per-question data** — the bound is loose enough
to be honest and tight enough to be decisive.

Cross-check on `acc_after` alone (83 / 86 / 83, unpaired): largest pairwise gap is
3 questions = 1.00 pp against SE(diff) = 3.65 pp, z = 0.274. Same verdict.

### 2.3 The monotone ordering is not a trend

Delta-R falls evenly: +31, +29, +27 net questions, i.e. −2 per 50 Stage-A updates.
Three reasons it must not be reported as a finding:

1. The whole range (1.33 pp) sits inside noise, as §2.2 shows.
2. With three exchangeable points, a monotone sequence in either direction arises by
   chance **1 time in 3** (2/3! = 0.333).
3. **`acc_after` is not monotone** — 83, 86, 83. The monotone appearance is
   manufactured entirely by subtracting a non-monotone `acc_before` (52, 57, 56).

Point 3 is the decisive one: the quantity that would carry a real effect (where the
arms *end up* under a fixed budget) shows no ordering at all.

---

## 3. Dashboard signals versus Q — the assignment, and the finding

This is Person 4's own deliverable: do dashboard signals move when Q does not?

**Nothing moves.** OLS slope of each Stage-A signal on update index, n = 100
(descriptive over the whole run — explicitly *not* a leakage-safe online feature):

| signal | mean | sd | slope / update | t | moves? |
|---|---|---|---|---|---|
| reward | 0.2381 | 0.0804 | −4.6e−5 | −0.16 | no |
| reward_std | 0.3383 | 0.0841 | +9.4e−5 | 0.32 | no |
| frac_reward_zero_std | 0.4163 | 0.1786 | −8.2e−5 | −0.13 | no |
| entropy | 0.0975 | 0.0229 | −6.8e−5 | −0.85 | no |
| grad_norm | 0.0284 | 0.0053 | +2.5e−5 | 1.35 | no |
| loss | 0.0369 | 0.0354 | −1.3e−4 | −1.09 | no |
| completions/mean_length | 836.9 | 99.8 | −0.013 | −0.04 | no |
| completions/clipped_ratio | 0.0652 | 0.0578 | −1.4e−4 | −0.71 | no |
| step_time | 194.3 | 14.8 | +0.053 | 1.03 | no |
| *num_tokens* | — | — | +61 582 | 1025 | *(cumulative counter — not a signal)* |

`beta = 0.0`, so there is no KL column to test.

First-third versus last-third confirms it is not a nonlinearity being hidden by a
linear fit: reward 0.2414 → 0.2342, entropy 0.0997 → 0.0950, grad_norm 0.0269 →
0.0288, mean completion length 853.9 → 839.3. All within one standard deviation.

**Interpretation.** The intended comparison — "Q gives more lead time than the
dashboard" — is unanswerable *in this run*, because there is no stall and the
dashboard is as flat as Q. The instrumentation is not shown to be insensitive; it is
shown to have been pointed at a run in which nothing happened.

**But the team-level position is stronger than "unanswerable", and should be written
that way.** See [`CROSS_RUN_NOTE_7B_VS_05B.md`](CROSS_RUN_NOTE_7B_VS_05B.md): in a
separate 0.5B GSM8K→SVAMP pilot — the one run in this project where something
actually happened — dashboard signals fired by update ~35 while effective rank was
still reading healthy at checkpoint 150, more than 100 updates after the model had
already reached 0% accuracy. There, Q gave **negative** lead time. That evidence is
second-hand and unverified (no artifacts supplied), so it is not folded into this
report's own conclusions, but it means the headline claim is *contradicted* where it
could be tested, not merely untested.

---

## 4. Why nothing moved — the dose was tiny

This is the most useful thing in the run, and it is not in the existing finding docs.

`metrics_ckpt*.json` carries whole-model `weight_norms` over 87 tracked tensors.
Comparing ckpt-0 with ckpt-100:

- **largest relative change of any tensor: +3.0 x 10^-5 %**, i.e. ~3 x 10^-7 relative
- 56 of 87 tensors register any change at all; the rest are bit-identical

Meanwhile the **LoRA adapter's own** relative change over Stage A is 1.92%, and the
per-window series decays monotonically: 1.38% → 0.84% → 0.51% → 0.17%.

Both facts are consistent and together they explain the run: a rank-16 adapter at
lr = 2e-5 over 100 updates perturbs a 7B model's weights by parts in ten million.
There was **no plasticity change for Q to detect**, because there was almost no
functional change at all. Reading the flat Q as "the metric is uninformative" would
be a straightforward misreading.

**A tension worth flagging rather than resolving.** Stage B, at the *same* learning
rate and only 30 updates, moved accuracy by +9 to +10 pp. A perturbation of this
magnitude therefore *can* produce a large behavioural change. Two readings are
compatible with the artifacts and cannot be separated from them:

- Simulation had far more headroom (17% → 28%) than Math did under this recipe; or
- part of Stage B's gain is answer-format adaptation to the exact-match verifier
  rather than reasoning improvement.

Distinguishing them needs per-completion outputs, which were not saved (§2.1). It
should be an explicit check in the next run, not an assumption in either direction.

---

## 5. Q metrics — full table

Effective rank is reported against hidden dim 3584 (`erank_norm = erank / 3584`).

| layer | metric | ckpt-0 | ckpt-50 | ckpt-100 | rel. change 0→100 |
|---|---|---|---|---|---|
| 5 | erank | 1127.415 | 1128.227 | 1128.181 | +0.068% |
| 5 | dormant_frac (tau .025 / .1) | 0.0000 | 0.0000 | 0.0000 | — |
| 14 | erank | 1281.045 | 1287.880 | 1287.809 | +0.528% |
| 14 | dormant_frac (tau .025 / .1) | 0.0000 | 0.0000 | 0.0000 | — |
| 26 | erank | 1426.060 | 1433.873 | 1432.548 | +0.455% |
| 26 | dormant_frac (tau .025 / .1) | 0.0000 | 0.0000 | 0.0000 | — |

Secondary metrics behave the same way. The largest relative move anywhere in the
table is `anisotropy_centered` at layer 14 (+8.8%), but its absolute value is
0.00207 → 0.00225 — a large relative change on a near-zero quantity, and the same
layer's `anisotropy_uncentered` moves −0.06%. Nothing here supports a claim of
change. `participation_ratio` at layer 26 moves −1.77%, the only other metric above
1%, and in the opposite direction to erank.

**Both dormant-neuron thresholds return exactly 0.0000 at every layer and every
checkpoint.** `dormant_score_min` never falls below 0.1596, i.e. the least-active
unit measured is still ~6x above the tau = 0.025 threshold. Dormancy is not merely
flat here — this model/probe combination is nowhere near the regime the metric was
designed to detect, so it carries no information in this run at any dose.

Measurement contract (identical for all checkpoints, from the artifacts): eval mode,
bf16, last-non-padding-token pooling for hidden states, mean-abs-over-non-padding for
dormancy, 512 max prompt tokens, float32 accumulator, float64 SVD, n_probe = 4096,
layers [5, 14, 26]. Comparability across checkpoints is satisfied.

---

## 6. Power — what a run that could test RQ1 needs

**Eval-set size** (unpaired normal approximation on `acc_after`, p ≈ 0.28; a paired
test would be cheaper but is not computable here):

| detect a Delta-R difference of | questions per arm |
|---|---|
| 1.33 pp *(this run's observed gap)* | ~17 800 |
| 2 pp | ~7 900 |
| 3 pp | ~3 500 |
| 5 pp | ~1 300 |
| 8 pp | ~490 |

n = 300 detects ~10.2 pp. **The eval set is roughly 4x too small even to detect a
5 pp effect**, and the observed gap is 1.33 pp. Chasing the observed ordering with
this eval set is not viable at any number of seeds.

**Checkpoint count** for a Q-vs-Delta-R correlation (Pearson, alpha .05, power .80):

| to detect | checkpoints needed |
|---|---|
| r = 0.9 | 7 |
| r = 0.8 | 10 |
| r = 0.7 | 14 |
| r = 0.5 | 30 |

This run had **3**. Even a near-deterministic relationship is undetectable at k = 3.

**Stage-A reward slope sensitivity.** SE(slope) ≈ 2.9e−4 per update, so the smallest
detectable *total* reward change over 100 updates is **~5.8 pp** (observed: 0.46 pp).
Stage A's training reward is consistent with anything in ±5.8 pp. Note this is
training reward on 8 fresh prompts x 8 generations per update — the noisiest
quantity in the run, and not a held-out eval (§7).

**The binding constraint is none of the above.** Before any of these numbers matter,
the next run must first deliver a Stage-A dose large enough to move *something*
(§4). Concretely, one or more of: a materially higher LoRA learning rate; a higher
adapter rank; full-parameter rather than LoRA training; many more Stage-A updates; or
a task where the reward has room to climb. **Recommendation: gate the next run on a
cheap pre-check** — run Stage A and require a detectable move in at least one
dashboard signal *before* spending on checkpoints, Q measurement, and three Stage-B
arms. This run spent ~11 h of A100 on Stage B to measure differences between three
models that were, functionally, nearly the same model.

---

## 7. Limitations

1. **n = 1 seed.** Seed 42 only; 43/44 were registered as stretch goals and not run.
   Every interval above reflects within-run sampling error only, and none of it
   captures seed-to-seed variability in either Delta-R or Q.
2. **Q has essentially no variance** (§5), and neither does anything else (§3), so
   RQ1 is untestable here. This is a scoping outcome, not evidence about Q.
3. **Per-question outcomes were not saved** (§2.1), so no paired test, no error
   analysis, and no way to check whether Stage B's gain is reasoning or formatting
   (§4). Fix: `return_details=True`.
4. **No in-domain Math evaluation exists at any checkpoint.** `analysis/transfer_T.json`
   scores Simulation only. We therefore cannot state whether Stage A improved at its
   own training task — only that its noisy training reward shows no detectable trend.
   This is a genuine instrumentation gap and the single cheapest thing to add.
5. **The 512-token eval cap.** `guru_greedy_accuracy` uses `max_new_tokens=512`,
   which truncates roughly 10–15% of this domain's answers (measured p90 ≈ 620) and
   scores them wrong. Left at the registered default deliberately, since changing it
   redefines R. It is bounded, not eliminated: completion lengths drift *upward*
   ~20–35% within each Stage-B run (ckpt-0 338 → 385, ckpt-50 297 → 410, ckpt-100
   302 → 404) and stay ~30% below the cap throughout, so the truncation mechanism
   would have to act in the opposite direction to the observed drift to explain
   Delta-R. It does affect the absolute value of R.
6. **Four deviations implemented, logged, and flagged to the team lead but not
   authorised by him**: base → Instruct; Stage-A cap 1280 → 1536; Stage-B cap
   640 → 2048; Stage-B eval points [0,10,20,30] → [0,30]. If any is rejected, the
   affected arms are discarded. The 640 → 2048 change is the best-evidenced of the
   four (the registered 640 measured 9.38% truncation *before training started* and
   killed the prior attempt at update 26).
7. **Sequential-arm confound is absent but unverified in one respect.** The three
   Stage-B arms ran in one process. GPU reserved memory plateaued at 76 512 MiB and
   did not creep, and each arm reloads the base model and its own adapter, so no
   cross-arm state is expected. Nothing in the artifacts contradicts this, but it was
   not tested directly.
8. **The 0.5B WIN4070 track is a separate lineage** — different model, different
   caps, still paused on its own amendment. Do not pool.

---

## 8. What to report, in one paragraph

> On Qwen2.5-7B-Instruct with LoRA, 100 updates of GRPO on GURU Math produced no
> detectable change in any measured quantity: effective rank moved ≤0.55% and
> dormant-neuron fraction was exactly zero at every layer and checkpoint; no
> dashboard signal (reward, entropy, gradient norm, loss, completion length) had a
> significant slope over 100 updates; zero-shot transfer to GURU Simulation moved
> +1.3 to +1.7 pp against a ±6 pp detection floor; and fixed-budget adaptability was
> flat — Delta-R over a 30-update budget was +10.33, +9.67 and +9.00 pp from
> checkpoints 0, 50 and 100, a range of 1.33 pp against an arm-to-arm SE of 2.5–5.5
> pp. Each arm's Delta-R is individually significant (worst-case McNemar p < 0.05 for
> every feasible discordance), so the adaptation itself is real; what is absent is
> any dependence on prior RLVR. The whole-model weight perturbation delivered by
> Stage A was ~3 x 10^-7 in relative Frobenius norm, which is the most parsimonious
> explanation for the uniform flatness. **We therefore report this as a null result
> about the recipe's dose rather than a test of RQ1: with no variance on either the
> predictor or the outcome, the run is equally consistent with the plasticity
> hypothesis and with its negation.** (Read alongside
> `CROSS_RUN_NOTE_7B_VS_05B.md`: a separate 0.5B pilot, unverified here, is the one
> run in which anything happened, and in it the plasticity metrics *trailed* the
> dashboard signals by 100+ updates.)
