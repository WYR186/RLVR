# Cross-run note — the 7B GURU run against the 0.5B GSM8K→SVAMP three-run pilot

**Date:** 2026-08-19
**Status:** comparison of this owner's completed 7B run against a second party's
0.5B pilot, supplied as two PDF write-ups (`three_run_summary`, `run2_analysis`).

**Provenance — UPGRADED 2026-08-19.** The raw artifacts were subsequently supplied
(`jason run/run {1,2,3}.zip`, each containing `stageA_log_history.json` and
`summary.csv`). **Every claim below has now been recomputed from those files**, and
§7 records which of the original write-up's claims survived verification and which
needed correcting. The 7B numbers are recomputed from this repo's artifacts (see
`ANALYSIS_REPORT_7B_MVP.md`).

**Prior-work caveat, and a correction.** Most of what §2 below reports as
"agreement" was **already established in this owner's own `AARON_COLLATING_PAGE.md`
(2026-08-02)**, before the 7B run existed. That page already states "Dose, not the
dataset pair, is the current blocker", already predicts "if four people pick four new
stage-1/stage-2 pairs and all run at safe learning rates, we should expect four more
null results", already cross-checks Jason's three runs in its §3.3, and already
explains *why* dormant fraction is dead (§5.2: Qwen2's SiLU-gated MLP computes
`act_fn(gate) * up`, which essentially never reaches zero — a mechanism, not just an
observation). The 7B run **confirms and quantifies that prediction at 14x the model
scale**; it does not discover it. Any write-up should say so.

**Protocol caveat.** The 0.5B runs use **GSM8K → SVAMP**, which is the *proposal's
original pilot* protocol, not Tommy's 2026-08-02 GURU spec (Math / Table →
Simulation). They are therefore not a second "stage 1" arm of the same study. They
are still directly informative about the *metric* question, which is what this note
is about.

---

## 1. Side by side

| | **7B (this run)** | **0.5B run 1** | **0.5B run 2** | **0.5B run 3** |
|---|---|---|---|---|
| model | Qwen2.5-7B-Instruct | Qwen2.5-0.5B-Instruct | " | " |
| stage A → B | GURU Math → Simulation | GSM8K → SVAMP | " | " |
| LR | 2e-5 | 5e-6, cosine→0 | 1e-5, constant | 5e-6, constant |
| KL β | **0** | **0** | **0** | **0** |
| updates | 100 | 200 / 200 | 450 / 450 | 450 / **110 aborted** |
| **dose (epochs)** | **0.01** | ~3.1 | ~7.0 | ~1.7 |
| stage-A outcome | nothing moved | trained, no erosion | destroyed by ~ckpt 50 | collapsed ~step 90 |
| in-domain acc | *(not measured)* | 0.45 → 0.39 | 0.45 → 0 | 0.45 → 0.23 |
| effective rank | +0.46%, flat | flat (±1%) | **1708 → 492** (after death) | flat over survivors |
| dormant fraction | **0.0000 everywhere** | **0 everywhere** | **0 → 0.0002** (trivial) | **0 everywhere** |
| entropy | flat (t = −0.85) | 0.37 → 0.10 | spiked 0.59, then crashed | 0.32 → 0.16, then collapse |
| stage-B gain | +10.3 / +9.7 / +9.0 pp | +.06/+.07 → −.05/−.07 | +.12/+.17 → 0/0/0 | +.16/+.19 |
| verdict | dose too small | "too cold" | "too hot" | "still collapses" |

Epoch figures for the 0.5B rows assume the proposal's 512-question GSM8K set; if the
actual set differs the numbers scale, but not the conclusion that those runs
accumulated **several epochs** while this one accumulated **0.015**.

---

## 2. What the two agree on

### 2.1 Four data points bracket the target regime and none lands in it

The hypothesis needs a state where **stage A keeps training successfully while
capacity quietly erodes**. Nobody has produced it:

- 7B, 0.01 epochs — nothing eroded because nothing happened at all
- 0.5B run 1, 3.1 epochs, decayed LR — trained fine, nothing eroded
- 0.5B runs 2 and 3, constant LR — the model was destroyed before erosion could be
  observed

That is a miss from three different directions across a 14x model-size range. It is
a stronger team-level statement than either party's result alone, and it is a direct
answer to Tommy's 2026-08-02 framing ("first find *when* plasticity loss can be
identified"): **so far, not at any setting either of us has run.**

### 2.2 Dormant fraction does not work on LLMs — two independent kinds of evidence

- **7B:** exactly 0.0000 at every layer, checkpoint and threshold. The least-active
  unit measured scores 0.1596, i.e. **6.4x above the τ = 0.025 threshold**. There is
  no dynamic range to move in.
- **0.5B run 2:** the model was reduced to a deterministic 384-token repetition loop
  at 0% accuracy, and dormant fraction still only reached **0.0002**.

The first says the metric has no headroom; the second says it **does not respond even
to total collapse**. Together they are close to decisive: dormant-neuron fraction, as
currently operationalised, should be demoted from the metric set rather than carried
into another run.

### 2.3 β = 0 in every run

All four are KL-free. The proposal's own stability mechanism was never exercised.

### 2.4 The ceiling confound reappears at 7B

`AARON_COLLATING_PAGE.md` §3.2 documents that Δ-accuracy correlates negatively with
pre-adaptation accuracy on two independent datasets — ρ = −0.66 (exp1.5 v3, n=6) and
ρ = −0.667 (recomputed on Jason's run 1, n=5). Much of what "Δ accuracy" measures is
**how much room was left to improve**, not adaptability.

Recomputed on the 7B run: **ρ(acc_before, Delta-R) = −0.756** (n=3). Same sign, same
magnitude, third independent dataset. The mechanism is visible directly in the data:
`acc_after` has an SD of 0.47 pp across arms while `acc_before` has 0.72 pp, so
Delta-R here is close to `constant − acc_before` by construction.

This subsumes an observation made independently in `ANALYSIS_REPORT_7B_MVP.md` §2.3
("the monotone ordering is manufactured by subtracting a non-monotone acc_before").
That is the ceiling confound, which already had a name and a prior estimate. n=3 makes
the 7B ρ descriptive only, but it belongs in the limitations with the other two.

### 2.5 A zero-compute check that explains why this run did not collapse

The Task-3 review (`lit review/TASK3_FOUR_PAPER_REVIEW.md`) records a critical group
pass rate from arXiv:2606.18487: at `num_generations = 8`, below **p\* ≈ 0.083** most
groups carry no reward variance and therefore no gradient. `frac_reward_zero_std` is
logged in every run this project has done.

| | mean reward | updates below p\* | mean frac_zero_std |
|---|---|---|---|
| 7B Stage A *(includes a 0.1 format bonus — not directly comparable)* | 0.2381 | 0 / 100 | 0.416 |
| 7B Stage B ckpt-0 *(pure exact-match = pass rate)* | 0.2812 | 2 / 30 | 0.542 |
| 7B Stage B ckpt-50 | 0.3016 | 1 / 30 | 0.542 |
| 7B Stage B ckpt-100 | 0.2708 | 1 / 30 | 0.575 |

Every arm stayed **well above** the critical line throughout. Against Jason's run 2,
which crossed 0.083 at step 30 and never recovered, this is an independent and
free explanation of why these runs stayed healthy — and a candidate early-warning
signal that is reward-based (what Tommy asked for), has a theoretical threshold, and
needs no new compute.

---

## 3. Where the 0.5B run says something this run could not

**This is the finding that changes our conclusion.**

`ANALYSIS_REPORT_7B_MVP.md` §3 concludes the "Q beats the dashboard" comparison is
*unanswerable* here, because Q and every dashboard signal were equally flat. The
0.5B run 2 is the case where something *did* happen, and there the answer is not
"unanswerable" — it is **negative**:

> Dashboard signals (collapsing reward, exploding clip ratio, zero-gradient fraction
> rising to 0.9) fired by update ~35. Effective rank was still reading a healthy
> ~1708 at checkpoint 150 — **more than 100 updates after the model was already at
> 0% accuracy.** Q trailed the failure; it gave *negative* lead time.

So the team's position on the headline claim should be stated as: **untestable in the
7B run, and contradicted in the one run where anything happened.** Writing it as a
pure "we couldn't test it" would understate what is known.

**Entropy is the live thread.** Run 2 shows an entropy *anomaly* (a spike to ~0.59)
preceding visible collapse — the earliest warning any signal gave. But run 3 shows
the entropy→learnability relationship running the opposite direction, and the 7B run
has entropy flat (t = −0.85), consistent with nothing happening. Entropy deviation is
worth promoting to a first-class candidate predictor; it is not yet a reliable one.

---

## 4. Where this run says something the 0.5B run could not

**The mechanism that destroyed runs 2 and 3 is the one this run measured and
controlled.**

The 0.5B write-up identifies the proximate cause of collapse as **length explosion**:
"reward fell exactly as completions ballooned to the token cap (99–100% clipped),
starving GRPO of gradient."

That is precisely the failure this run's instrumentation targets:

- a pre-registered stop at >10% completion clipping for 5 consecutive updates;
- completion caps **measured** rather than assumed — the sizing rule in
  `FINDING_STAGE_B_CAP_SIZING.md` (smallest cap whose worst-arm truncation ≤ 2.34%,
  anchored on a recipe observed to survive 100/100 updates, carrying a measured
  2.79x init→training inflation);
- the registered Stage-B cap of 640 measured **9.38% truncation before training
  started** and killed an earlier attempt at update 26 — the same mechanism, caught.

Across 90 Stage-B updates at the measured cap, the clipping streak never exceeded 1.

**Transferable conclusion:** the length-explosion collapse is not an inevitability of
KL-free GRPO — it is at least partly a cap-sizing failure, and this run has a
measured procedure for avoiding it. Runs 2 and 3 would very likely have survived
longer with a measured cap plus the clipping guard. That is a concrete contribution
this owner can hand the group, and it is independent of whether the plasticity
hypothesis survives.

---

## 5. What this changes about the plan

### 5.1 A correction to a suggestion made before seeing these runs

Earlier advice from this owner's own analysis was: shrink the Stage-A training set so
epochs accumulate at constant compute, raise the learning rate, and deliberately
induce plasticity loss. **The 0.5B data makes that path, as stated, unlikely to
work.** Three learning rates spanning 2x, all KL-free, all collapsed. Increasing
pressure without a stability anchor reproduces run 2, not the target regime.

Note also the uncomfortable pattern inside the 0.5B set: **run 1 survived because its
LR decayed to zero, and it is also the run in which nothing eroded.** The mechanism
that keeps stage A healthy is the same one that prevents erosion. That tension sits
directly under the proposal's premise and should be named in the write-up rather than
discovered later.

### 5.2 The amended proposal — each party has half of it

| ingredient | from | why |
|---|---|---|
| shrink stage-A set so epochs accumulate | 7B analysis | 0.01 epochs cannot erode anything; this buys ~100x dose at constant compute |
| **KL anchor, β ≈ 0.02–0.04** | 0.5B analysis | the missing stability mechanism; without it, pressure destroys the model |
| **measured completion cap + clipping stop** | 7B analysis | removes length explosion, the identified proximate cause of collapse |
| mid-run in-domain eval every ~25–50 updates | both | 7B never measured in-domain Math at all; 0.5B run 2 wasted 215 updates on a dead model |
| early-abort guard | 0.5B analysis | run 3 saved half a session with one |
| entropy tracked as a first-class predictor | 0.5B analysis | the only signal that has ever led a failure |

Neither party has this recipe alone. It is worth proposing to Tommy as a single joint
run rather than two more independent ones.

### 5.3 Open question this comparison cannot settle

Does 7B collapse at all under sustained KL-free GRPO? This run never approached
collapse, but it also never applied meaningful pressure. Scale may change the
picture: larger models may be more robust, in which case the 0.5B collapse is a
small-model artifact and the "razor-thin regime" conclusion does not transfer. **This
should be stated as unknown, not assumed in either direction.**

---

## 6. One-paragraph version for the group

> Two independent runs, 0.5B GSM8K→SVAMP and 7B GURU Math→Simulation, bracket the
> regime the proposal needs and neither lands in it: the 7B run applied too small a
> dose for anything to erode (0.015 epochs; whole-model weight norms moved 3e-7), and
> the 0.5B runs collapsed at three different learning rates before quiet erosion
> could appear. Both agree that **dormant-neuron fraction is unusable here** — it is
> exactly 0.0000 across the 7B run with 6.4x headroom to its threshold, and it
> reaches only 0.0002 in a 0.5B model reduced to 0% accuracy and deterministic
> repetition. On the headline claim, the 7B run is uninformative (Q and every
> dashboard signal equally flat), while the one run where anything happened shows Q
> **trailing** the dashboard by 100+ updates — negative lead time. Entropy anomaly is
> the only signal that has ever led a failure, and it is unstable across runs. The
> two runs also supply complementary halves of a fix: KL anchoring (β ≈ 0.02–0.04)
> for stability, and measured completion caps with a clipping guard for the length
> explosion that proximately caused the collapses.


---

## 7. Verification against the raw artifacts (2026-08-19)

`stageA_log_history.json` logs every 5th update; `summary.csv` carries the
per-checkpoint Q, GSM8K and SVAMP numbers. Completion cap is **384** in all three runs
(`completions/max_length` saturates there exactly).

### 7.1 Dose, from real epoch counters rather than an assumed dataset size

| run | steps | epochs | updates / epoch | runtime |
|---|---|---|---|---|
| Jason 1 | 200 | 1.5625 | **128** | 3 311 s |
| Jason 2 | 450 | 3.5156 | **128** | 7 279 s |
| Jason 3 | 110 | 0.8594 | **128** | 1 993 s |
| **7B Stage A** | 100 | **0.0147** | **6 781** | 19 783 s |

**Each of Jason's updates carries 53x the dose of one of ours**, in epochs. The
earlier estimate in §1 (assuming a 512-question set) was close but is now superseded.

### 7.2 The collapse timeline — entropy DID lead, by 10–15 updates

Run 2, the only complete collapse. Baseline entropy = 0.2195 (step 10).

| step | reward | entropy | ×baseline | clip | zero-var groups | grad norm | first firing |
|---|---|---|---|---|---|---|---|
| 15 | 0.1750 | 0.3841 | 1.75 | 0.019 | 0.40 | 4.42 | |
| **20** | 0.3125 | **0.5760** | **2.62** | 0.000 | 0.35 | 4.82 | **entropy > 2× baseline** |
| 25 | 0.1875 | 0.5097 | 2.32 | 0.019 | 0.45 | 2.81 | |
| **30** | **0.0813** | 0.5062 | 2.31 | 0.094 | 0.60 | 4.66 | **reward < p\* = 0.083** |
| **35** | 0.0187 | 0.6260 | 2.85 | **0.463** | **0.90** | **0.000** | zero-variance ≥90%, **grad norm exactly 0**, clip > 10% |
| 40 | 0.0250 | 1.1817 | 5.38 | 0.869 | 0.85 | 0.000 | |

**This corrects the original write-up in the project's favour on one point and
against it on another.**

- **Correction 1 (favourable):** the write-up called entropy "the exception, and the
  most promising thread". The artifacts are stronger than that — entropy crossed 2×
  its baseline at **step 20**, ten updates before reward crossed the critical group
  pass rate and fifteen before the gradient hit exactly zero. **That is a genuine
  positive lead time, and it is the only one anywhere in this dataset.**
- **Correction 2 (unfavourable):** `grad_norm` is reported as **exactly 0.000** from
  step 35 onward — a hard, unambiguous "training is dead" flag available for free in
  the existing dashboard. Any lead-time claim for Q must beat *that*, not a soft
  reward trend.

### 7.3 Q is variant-dependent, and neither variant led

| checkpoint | erank **MLP** | vs ckpt-0 | erank **residual** | vs ckpt-0 | GSM8K |
|---|---|---|---|---|---|
| 0 | 1724.0 | — | 78.1 | — | 0.4531 |
| 50 | 1666.6 | **−3.3%** | 114.3 | **+46.3%** | 0.0469 |
| 150 | 1708.3 | −0.9% | 499.5 | **+539.2%** | 0.0000 |
| 300 | 492.0 | **−71.5%** | 361.3 | +362.3% | 0.0000 |
| 450 | 495.3 | −71.3% | 361.8 | +363.0% | 0.0000 |

The two variants move in **opposite directions during the same collapse** —
`AARON_COLLATING_PAGE.md` §3.3 flagged exactly this and required a single measurement
contract before pooling. The artifacts confirm it.

On lead time, the honest statement is narrower than the write-up's:

- Checkpoints exist only at 0 / 50 / 150 / 300 / 450, so Q's time resolution is 50
  updates at best.
- By the **first** post-failure checkpoint (50) the model is already dead
  (GSM8K 0.4531 → 0.0469), so even the residual variant, which had already moved
  +46%, moved **concurrently with the visible collapse, not before it**.
- The MLP variant was still within 1% of baseline at checkpoint 150 and only collapsed
  by 300 — roughly **−250 updates of lead time**.

So: **no Q variant achieved positive lead time; one achieved strongly negative lead
time; and the metric's sign depends on which variant you pick.** That is a sharper
and more defensible claim than "Q lagged".

### 7.4 The ceiling confound, recomputed

`AARON_COLLATING_PAGE.md` §3.2 records ρ = −0.667 on Jason's run 1. Recomputing from
`run1/summary.csv` (`svamp_acc_start` vs `svamp_improvement`, n=5) gives **ρ = −0.611**
— same sign, same magnitude; the small difference is presumably a variant of the
statistic. Run 3 shows the mechanism directly: its damaged ckpt-50 has the lowest
SVAMP start (0.2467) **and** the largest improvement (+0.1933) in the set.

Together with the 7B value (**ρ = −0.756**, n=3, §2.4), that is **three independent
datasets, same sign, similar magnitude**. Pre-adaptation accuracy must be a covariate
in anything we report.

### 7.5 What this owner's safety gate would have bought

The registered stop — >10% completion clipping for 5 consecutive updates — applied to
Jason's artifacts:

- **Run 2:** clip goes 0.094 (step 30) → 0.463 (35) → 0.869 (40) → 0.894 (45) → 0.812
  (50). The gate fires around **update 40** (≤55 even on the 5-step-sampled log). The
  run instead continued to 450. From step 235 onward every logged update has reward
  0.0000, grad norm 0.000, clip 1.000 and mean length exactly 384.0 — a dead model
  spinning on the GPU. **~410 of 450 updates, roughly 1.8 h of GPU, were spent after
  the gate would have stopped it.**
- **Run 3:** fires around update 90–105, close to where its manual guard fired
  anyway — so ~5 updates saved. Its own abort guard did the job.

This is the concrete transferable piece: the gate is worth about 90% of one wasted run,
and the completion-cap sizing procedure in `FINDING_STAGE_B_CAP_SIZING.md` addresses
the *cause* rather than the symptom.

### 7.6 One further observation the write-ups do not make

**Run 1 was also drifting toward length explosion and was rescued by its LR schedule.**
Its clip ratio falls to 0.013–0.025 mid-run, then climbs back to 0.13–0.19 over steps
105–200 while mean completion length grows 154 → 300. Meanwhile its cosine LR decays to
2.5e-8, so the last third of the run is barely training at all.

That sharpens the tension already noted in §5.1: run 1 is the run in which nothing
eroded, and it is also the run whose learning rate went to zero before anything could.
The two facts are not independent.