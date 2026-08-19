# Cross-run note — the 7B GURU run against the 0.5B GSM8K→SVAMP three-run pilot

**Date:** 2026-08-19
**Status:** comparison of this owner's completed 7B run against a second party's
0.5B pilot, supplied as two PDF write-ups (`three_run_summary`, `run2_analysis`).

**Provenance caveat, stated first.** The 0.5B numbers below are **transcribed from
those write-ups and have not been independently verified** — no dashboards, configs,
or artifacts were supplied, and nothing matching them exists in this repo. Every
claim attributed to that run should carry that caveat until its artifacts are
available. The 7B numbers are recomputed from artifacts (see
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
