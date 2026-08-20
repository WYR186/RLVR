# exp2 — Math → Simulation: summary and conclusions

**Aaron Wang · 2026-08-19**
Companion to `aaron_exp2_7b_mvp_2026-08-19.zip`. Every number here is recomputed from
the artifacts in that zip; running `python3 VERIFY.py` inside it re-derives all 45
claims from the raw files and currently passes 45/45.

---

## 1. The assignment, and what was delivered

Tommy's 2026-08-02 protocol, executed end to end on **Qwen2.5-7B-Instruct + LoRA**,
seed 42, on Colab A100-80GB.

| Tommy's requirement | Status |
|---|---|
| Take a stage 1 distinct from the others; Math → Simulation | done (confirmed 2026-08-05) |
| `LLM360/guru-RL-92k`; Simulation = CodeI/O, Math = OR1 / DAPO / DeepScaler | done, dataset revision pinned, splits frozen as id lists and hash-checked |
| Save stage-1 checkpoints at 0, TOTAL/n, TOTAL | done at updates 0 / 50 / 100 — **but in updates, not epochs. See §4.** |
| Baseline of training directly with stage 2 | done — the ckpt-0 arm |
| Train stage 2 from every checkpoint; measure the plasticity metric | done, all three arms completed their 30-update fixed budget |
| `T_t = Score_B(M_{A,t}) − Score_B(M_0)` | done |
| "choose a model that is stronger than Qwen2.5-7B" | **partially — Instruct, not a larger model. Needs your call. See §7.** |

This is the first complete Delta-R curve in the project. Two earlier attempts died on
the pre-registered completion-clipping stop at updates 23 and 26.

---

## 2. Results

### 2.1 Your plasticity metric is zero

Computed exactly as defined: reward achievable with stage 2 alone, minus reward
achievable with stage 2 following stage 1.

```
P_50  = Score_B(ckpt-0) − Score_B(ckpt-50)  = 0.2767 − 0.2867 = −0.0100   (−3 of 300)
P_100 = Score_B(ckpt-0) − Score_B(ckpt-100) = 0.2767 − 0.2767 = +0.0000   ( 0 of 300)
```

**P_100 is exactly zero**, and P_50 has the *opposite* sign to plasticity loss.

### 2.2 Transfer is flat

```
T_50  = +0.0167   (+5 of 300)
T_100 = +0.0133   (+4 of 300)
```

By your criterion (T ≈ 0): **little transfer** from Math to Simulation, neither
helping nor hurting. Note the detection floor at n = 300 is about ±6 pp, so this is
"no difference measurable", not "no difference".

### 2.3 Each arm does learn — the pipeline works

| arm | before | after | **Delta-R** | questions |
|---|---|---|---|---|
| ckpt-0 (stage 2 alone) | 0.1733 | 0.2767 | **+0.1033** | 52 → 83 / 300 |
| ckpt-50 | 0.1900 | 0.2867 | **+0.0967** | 57 → 86 / 300 |
| ckpt-100 | 0.1867 | 0.2767 | **+0.0900** | 56 → 83 / 300 |

All three completed 30/30 updates with `completion_status: complete`.

The per-question outcomes were not saved, so a paired McNemar test is not computable
after the fact. It can still be **bounded**: `b` (correct→wrong) is at most the
before-hits count, and `z = net/√(2b+net)` decreases in `b`, so the whole feasible
interval can be enumerated. Result: **every arm's Delta-R is significantly positive
for every feasible value of the missing data** (worst-case p = 0.0076 / 0.0153 /
0.0220). Fixed-budget adaptation is real.

Between arms, the same bound gives **|z| ≤ 0.53 everywhere in the feasible range**.
The flatness conclusion does not depend on the data we failed to save.

### 2.4 Q — the predictor — has no variance either

| | ckpt-0 | ckpt-50 | ckpt-100 | change |
|---|---|---|---|---|
| effective rank, layer 26 | 1426.06 | 1433.87 | 1432.55 | +0.46% |
| effective rank, layer 14 | 1281.05 | 1287.88 | 1287.81 | +0.53% |
| effective rank, layer 5 | 1127.42 | 1128.23 | 1128.18 | +0.07% |
| dormant fraction, τ ∈ {0.025, 0.1} | 0.0000 | 0.0000 | 0.0000 | — |

Dormant fraction is **exactly zero at every layer, checkpoint and threshold**, and the
least-active unit anywhere in the run scores 0.1596 — **6.4× above the τ = 0.025
threshold**. The metric has no dynamic range in this setting. (Mechanism, from the
Task-3 review: Qwen2's SiLU-gated MLP computes `act_fn(gate) · up`, which essentially
never reaches zero.)

An independent re-measurement of ckpt-0 reproduces the original **bit for bit**, and
all three `acc_before` values reproduce the separately measured transfer curve
**exactly** (0.1733 / 0.1900 / 0.1867) across a runtime recycle — so the measurement
path itself is verified.

---

## 3. What this does and does not show

**It cannot test RQ1.** Both sides of the correlation are flat: Q barely moves, and
fixed-budget adaptability does not move. The run is equally consistent with the
plasticity hypothesis and with its negation. That is a limitation to report plainly,
not a negative result about Q.

**Stated within the framing constraint** (never "RLVR reduces the ability to learn"):
at a fixed budget of 30 GRPO updates on the held-out Simulation family, **100 updates
of Math RLVR produced no detectable change in fixed-budget adaptability.**

**Do not report the monotone ordering as a trend.** Delta-R falls evenly (+31, +29,
+27 net questions). Three reasons it is not a finding: the range is inside noise;
three points go monotone by chance one time in three; and `acc_after` is itself **not**
monotone (83, 86, 83). The ordering is produced by subtracting a non-monotone
`acc_before` — i.e. it is the ceiling confound (§6.3), not adaptability.

---

## 4. Why nothing moved: Stage A ran ~70× short of a single epoch

This is the most useful result in the run, and it is a spec-versus-compute problem the
team needs to decide on.

Your protocol sets checkpoints in **epochs**. Stage A ran 100 updates × 8 prompts =
**800 of 54,251 training rows = 1.47% of one epoch.**

```
one full Stage-A epoch = 6,781 updates
                       × 198 s/update measured
                       ≈ 373 A100-hours
                       ≈ 2,525 Colab compute units
```

That is **more than 20× the compute budget I have**, for one epoch of one stage of
one arm. Reaching one epoch would take 68× the updates I ran.

Three independent measurements agree that almost nothing happened:

- **Full-layer weight norms moved by 2.9 × 10⁻⁵ percent** (≈3 × 10⁻⁷ relative) from
  ckpt-0 to ckpt-100, across 87 tracked tensors. The LoRA adapter's own norm moved
  1.92%, with per-window change decaying monotonically (1.38 → 0.84 → 0.51 → 0.17%).
- **No dashboard signal has a significant slope** over 100 updates. Reward t = −0.16,
  entropy −0.85, grad norm +1.35, loss −1.09, mean completion length −0.04, clipped
  ratio −0.71. Max |t| = 1.35.
- Q as above.

**Cross-check:** Stage B, at the *same* learning rate but 21.3% of an epoch — 14× the
dose — moved accuracy by +9 to +10 pp. The pipeline works; the Stage-A dose does not.

**Consequence for RQ1:** Q is not shown to be insensitive here. It is shown to have
been pointed at a run in which there was almost nothing to detect. That is a very
different conclusion, and the actionable one.

---

## 5. What the safety instrumentation bought

The pre-registered stop is >10% completion clipping for 5 consecutive updates. The
registered Stage-B cap of 640 was justified as "matches `stage_b max_prompt_length`" —
a prompt-side setting used as a completion-side bound, never measured.

Measured on the two actual Stage-B starting policies (24 frozen prompts × 8
generations, probe cap 2048): **truncation at 640 is 9.38% before training starts.**
The run began pressed against its own stop, which is why the earlier attempt died at
update 26. It was the recipe, not the training.

The sizing rule adopted — smallest candidate cap whose worst-arm truncation is ≤2.34%,
where 2.34% is the init-policy figure of a recipe observed to survive 100/100 updates,
carrying a measured 2.79× init→training inflation — selects **2048**. Across all 90
Stage-B updates at that cap, the >10% streak **never exceeded 1**.

One property of this domain is not fixable by a larger cap: the completion
distribution is bimodal, with a median of 271–285 tokens and **~1.5% of generations
that never terminate**. 1.56% is a floor, not a price.

---

## 6. Combined with Jason's three 0.5B runs

His raw artifacts were supplied and independently recomputed (`derived/cross_run_derived.csv`
and `docs/CROSS_RUN_NOTE_7B_VS_05B.md` §7). Setup differs — 0.5B, GSM8K → SVAMP, the
proposal's original pilot protocol rather than the GURU spec — so these are not a
second stage-1 arm of the same study, but they bear directly on the metric question.

### 6.1 Four runs bracket the target regime; none lands in it

| run | dose | outcome |
|---|---|---|
| 7B (this) | 0.015 epochs | nothing moved — nothing to erode |
| 0.5B run 1, lr 5e-6 cosine→0 | 1.56 epochs | trained fine, nothing eroded |
| 0.5B run 2, lr 1e-5 constant | 3.52 epochs | destroyed by ~update 50 |
| 0.5B run 3, lr 5e-6 constant | 0.86 epochs | collapsed ~update 90 |

The hypothesis needs stage A to keep training successfully *while* capacity quietly
erodes. Four attempts, three learning rates, a 14× model-size range, β = 0 throughout:
**nobody has produced that state.** Each of Jason's updates carries **53× the dose** of
one of mine, measured from the epoch counters.

Note the tension inside his set: **run 1 survived because its learning rate decayed to
2.5e-8, and it is also the run in which nothing eroded.** Its clipping ratio was
already climbing back to 0.13–0.19 and its completion length from 154 to 300 over the
last 100 updates — it was drifting the same way and was rescued by its own schedule.
The mechanism that keeps stage A healthy is the mechanism that prevents erosion.

### 6.2 On the headline claim, the evidence is now negative, not merely absent

My run cannot test "Q warns earlier than the dashboard" because both are flat. His
run 2 is the one run in which something happened, and there the timeline is:

| update | event |
|---|---|
| **20** | **entropy crosses 2× its baseline** (0.2195 → 0.5760) |
| 30 | reward crosses the critical group pass rate p\* ≈ 0.083 |
| 35 | zero-variance groups reach 0.90; **grad norm becomes exactly 0.000**; clipping 0.463 |
| 50 | first post-failure checkpoint — GSM8K already 0.4531 → 0.0469 |
| 150 | effective rank (MLP) still within 1% of baseline, model at 0% accuracy |
| 300 | effective rank (MLP) finally collapses, −71.5% |

**No Q variant achieved positive lead time.** The MLP variant gave roughly **−250
updates**. The residual-stream variant had already moved +46% by checkpoint 50 — but
the model was already dead at checkpoint 50, so it moved *concurrently with* the
visible collapse, not before it. And the two variants move in **opposite directions
during the same collapse** (MLP −71%, residual +539%), so the metric's sign depends on
which variant you pick. A single measurement contract is required before any of our Q
numbers are pooled.

**Entropy is the one signal that led** — by 10 updates over reward and 15 over the
gradient going to zero. It is not yet reliable: run 3 shows the relationship running
the other way, and my run has entropy flat. Worth promoting to a first-class candidate,
not yet worth claiming.

**Also free:** `grad_norm` logged as exactly 0.000 is an unambiguous "training is dead"
flag already in every dashboard we produce, and `frac_reward_zero_std` has a
theoretical threshold (p\* ≈ 0.083 at `num_generations = 8`). Every arm of my run
stayed well above that line; his run 2 crossed it at update 30 and never returned.

### 6.3 The outcome measure is confounded by the starting point — three datasets

Delta-accuracy correlates negatively with pre-adaptation accuracy:

| dataset | ρ | n |
|---|---|---|
| exp1.5 v3 (0.5B, mine) | −0.66 | 6 |
| Jason run 1 (recomputed) | −0.611 | 5 |
| **this 7B run** | **−0.756** | 3 |

Same sign, same magnitude, three independent datasets. A large part of what "Δ
accuracy" measures is **how much room was left to improve**. His run 3 shows it
directly: the damaged ckpt-50 had the lowest SVAMP start (0.2467) and the largest gain
(+0.1933) in its set. **Pre-adaptation accuracy must be a covariate in anything we
report.**

### 6.4 What each side contributes to a fix

The proximate cause of both 0.5B collapses is length explosion into the token cap —
completions saturate, reward starves, group variance goes to zero, gradient goes to
zero. That is exactly what my cap-sizing procedure and clipping stop control. Applied
to his run 2, the gate would have fired around **update 40**; the run instead continued
to 450, with every logged update from 235 onward showing reward 0.0000, grad norm
0.000, clipping 1.000 and mean length pinned at exactly 384.0. **About 410 of 450
updates — roughly 1.8 hours of GPU — were spent on a dead model.**

Neither of us has the full recipe:

| ingredient | from | why |
|---|---|---|
| shrink the stage-A set so epochs accumulate at constant compute | 7B run | 0.015 epochs cannot erode anything; this buys ~100× the dose for the same money |
| **KL anchor, β ≈ 0.02–0.04** | 0.5B runs | the proposal's own stability mechanism, never once exercised; without it, pressure destroys the model |
| **measured completion cap + clipping stop** | 7B run | removes the identified proximate cause of collapse |
| in-domain eval every 25–50 updates | both | I never measured Math at all; he spent 215 updates on a dead model |
| early-abort guard | 0.5B run 3 | saved half a session there |
| entropy deviation + grad-norm-zero as first-class signals | 0.5B run 2 | the only signals that have ever led a failure |

---

## 7. Two decisions I need from you

1. **Model.** You said "choose a model that is stronger than Qwen2.5-7B". I ran
   Qwen2.5-7B-**Instruct**. That was forced: the base model has a non-terminating
   completion tail — 4.69% of generations still running at a 3072-token cap — so no
   cap in {1536, 1792, 2048, 2560} met the ≤5% truncation rule, and base died on the
   clipping stop at update 7. Instruct measures 0.39% on the same test. **Is Instruct
   what you meant, or do you want a genuinely larger model?** It changes the compute
   plan by an order of magnitude.

2. **Scale.** Given that 1.47% of an epoch produces no measurable change in anything,
   what is the smallest stage-A dose you would accept as a real RLVR treatment? My
   recommendation: **gate the next run on a cheap pre-check** — run stage A and require
   a detectable move in at least one dashboard signal *before* spending on
   checkpoints, Q measurement and three stage-B arms. This run spent ~11 hours of A100
   on stage B to measure differences between three models that were, functionally,
   nearly the same model.

Also flagged: four deviations implemented, logged, and not authorised by you — base →
Instruct; Stage-A cap 1280 → 1536; Stage-B cap 640 → 2048; Stage-B eval points
[0,10,20,30] → [0,30]. Each is argued in `docs/`; the 640 → 2048 change is the
best-evidenced (the registered value measured 9.38% truncation before training began).

---

## 8. Limitations

1. **n = 1 seed** (42). Seeds 43 and 44 were registered as stretch goals and not run.
   No interval here captures seed-to-seed variability.
2. **Q has essentially no variance**, and neither does anything else, so RQ1 is
   untestable in this run.
3. **Per-question eval outcomes were not saved** — no paired test, no error analysis,
   and no way to check whether Stage B's gain is reasoning or answer formatting. Fix:
   `return_details=True` in `guru_greedy_accuracy`.
4. **No in-domain Math evaluation at any checkpoint.** Only Simulation transfer was
   measured, so we cannot state whether Stage A improved at its own training task —
   only that its (very noisy) training reward shows no detectable trend, with a ±5.8 pp
   detection floor. Cheapest gap to close.
5. **The eval uses `max_new_tokens = 512`**, which truncates roughly 10–15% of this
   domain's answers (measured p90 ≈ 620) and scores them wrong. Left at the registered
   default deliberately — changing it redefines R. It is bounded rather than
   eliminated: completion lengths drift *upward* 20–35% within each Stage-B run and
   stay ~30% below the cap throughout, so the truncation mechanism would have to act in
   the opposite direction to the observed drift to explain Delta-R. It does affect the
   absolute value of R.
6. **Four unauthorised deviations** (§7).
7. **Jason's runs use a different protocol** (GSM8K → SVAMP, 0.5B) and are not pooled
   with mine; §6 treats them as separate evidence about the metric, not as a second arm.
8. **Unresolved and not assumed either way:** whether 7B collapses at all under
   sustained KL-free GRPO. This run never applied meaningful pressure, so the
   "razor-thin regime" conclusion from the 0.5B runs may or may not transfer to scale.

---

## 9. Power — what a run that could test RQ1 needs

| target | requirement | this run |
|---|---|---|
| detect a 5 pp Delta-R difference between arms | ~1,300 eval questions per arm | 300 |
| detect a 3 pp difference | ~3,500 | 300 |
| detect a Q↔Delta-R correlation of r = 0.8 | 10 checkpoints | 3 |
| detect r = 0.9 | 7 checkpoints | 3 |

n = 300 detects about 10.2 pp; the observed gap is 1.33 pp. **Chasing the observed
ordering with this eval set is not viable at any number of seeds.**

None of these is the binding constraint. The binding constraint is §4: until stage A
delivers a dose that moves *something*, more eval questions and more checkpoints only
measure the same non-event more precisely.

---

## 10. Package contents

`aaron_exp2_7b_mvp_2026-08-19.zip` (1.6 MB, 68 files). `MANIFEST.sha256` covers
every file; `shasum -a 256 -c MANIFEST.sha256` verifies the lot.

| | |
|---|---|
| `VERIFY.py` | recomputes all 45 claims from `data/`; reads no summary or report; no dependencies; **currently 45/45 pass** |
| `data/` | Stage-A dashboard and sentinel, frozen splits, Q at three checkpoints plus an independent re-measurement, the transfer curve, and full Stage-B artifacts for all three arms |
| `code/` | `src/` (the modules that ran), `drivers/` (the scripts that called them), pinned `requirements.txt`, unit test |
| `configs/` | both pre-registered recipes, hash-verified in-run (`e33527592dd9`, `bd99ddd2817f`) |
| `docs/` | analysis report, cross-run note, findings; `docs/operational/` has the plans, amendments, run log, and the expensive gotchas |
| `figures/` | `delta_r_vs_q` and `collapse_timeline_run2`, each `.png` + `.svg` |
| `derived/` | `headline_numbers.csv` (every value with its source file), `cross_run_derived.csv` |
| `compute/` | per-session GPU accounting |
| `WEIGHTS.md` | where the LoRA adapters live and how to load them |

**Two things deliberately excluded.**

*LoRA adapter weights* (154 MB each) — nothing in `VERIFY.py` or any figure needs
them. They are in Drive; `WEIGHTS.md` has the details.

*The `colab/01–04` notebooks* — they were written for the base-model config
(`fc243e587296`), reference the wrong splits file, clone through a PAT that is
confirmed broken, and have **zero cell outputs; none was ever executed.** Shipping
them as "the code that produced this" would misstate the provenance. `code/drivers/`
is what actually ran, and its README says which files are verbatim and which are
reconstructed.

*Jason's raw run files* are also not redistributed — they are his to share.
`derived/cross_run_derived.csv` carries the quantities §6 depends on so they can be
checked against his originals.

---

## 11. Two things this package does not satisfy

1. **GPU-dashboard screenshots were never taken**, for any exp2 session, and cannot
   be recovered after the fact. The compute-accounting requirement is therefore only
   partly met: dates, GPU types, durations and phases are recorded throughout, and
   unit readings exist for the final session only. Noted at the end of
   `compute/compute_log.md`. Screenshots start with the next run.
2. **Unit deltas for the three earlier sessions are permanently missing** — Colab
   shows a live balance and keeps no history, and no reading was taken at the time.
   Those entries say so rather than carrying a placeholder.
