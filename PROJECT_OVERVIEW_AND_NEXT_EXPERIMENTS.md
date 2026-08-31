# Project overview, evidence ledger, and the next three experiments

**Author:** Aaron Wang (early-warning diagnostics)
**Date:** 2026-08-30
**Status:** proposal — §7 lists what needs a team decision before execution.

This document exists because the project is at a turning point. Four training runs
are complete across two machines and two model scales, the first complete Δ-R curve
is in, and the result is that **the study as originally framed cannot be answered
with the data we have**. This is not a failure, but it does mean the next compute
spend and the paper's framing have to be chosen deliberately rather than by
continuing the existing plan.

Sections 1–4 are the macro picture and how we got here. Section 5 is the concrete
experiment design. Sections 6–8 are what we deliberately skip, what needs a
decision, and how each experiment maps onto a paper section.

Every number in this document is recomputed from artifacts in this repository or
from the raw logs in `jason run/`. Where a previously circulated figure was wrong,
it is corrected inline and marked.

---

## 1. What the project is

**Question.** During multi-stage RLVR training of reasoning LLMs, do
activation-based *plasticity* metrics **Q** — effective rank and dormant-neuron
fraction, measured on a frozen probe set during RL stage A — give **earlier warning**
that a later RL stage B will stall than the signals practitioners already watch
(reward slope, KL, gradient norm, entropy)?

**The state the question presupposes.** For the question to even be askable, there
must exist a regime where **stage A keeps training successfully while capacity
quietly erodes underneath**. If capacity never erodes, there is nothing for Q to
detect. If the model breaks loudly, the ordinary dashboard already caught it and Q
has nothing to add. The hypothesis lives in the narrow band between those two.

**Framing constraint (mentor feedback, Madhur).** We never claim "RLVR reduces the
model's ability to learn." The measurable claim is about **fixed-budget future
adaptability** on a held-out task family, relative to a checkpoint-0 baseline, with
task / budget / baseline all fixed in advance. Every outcome in this document is
stated in those terms.

**Two supporting definitions used throughout.**

- **Δ-R** (fixed-budget adaptability): `Score_B(after) − Score_B(before)` for a
  stage-B adaptation run of a *pre-registered, identical* update budget started from
  a given stage-A checkpoint.
- **T_t** (transfer): `Score_B(M_{A,t}) − Score_B(M_0)` — how much stage-A training
  by itself moved held-out stage-B performance, before any stage-B adaptation.

**Leakage rule (proposal §6).** Any feature computed at step *t* uses only
information available at step *t*. No post-hoc normalization across a run.

**Comparability contract.** All Q values are measured on a frozen probe set, in
eval mode, at fixed layers and fixed dtype, so they are comparable across
checkpoints. The exact contract in force for the 7B run is recorded in
`experiment 2/FINDING_Q_METRICS_7B_INSTRUCT.md` and reproduced in §5.1 below.

---

## 2. Everything that has been run

Two parallel tracks, same underlying question, different protocols. They are **not
two arms of one study** — different models, task pairs, and dose by two orders of
magnitude.

| | **7B GURU** (this owner) | **0.5B run 1** | **0.5B run 2** | **0.5B run 3** |
|---|---|---|---|---|
| model | Qwen2.5-7B-Instruct + LoRA | Qwen2.5-0.5B-Instruct | " | " |
| stage A → B | GURU Math → Simulation | GSM8K → SVAMP | " | " |
| LR | 2e-5 | 5e-6, cosine→0 | 1e-5, constant | 5e-6, constant |
| KL β | **0** | **0** | **0** | **0** |
| updates | 100 | 200 / 200 | 450 / 450 | 450 / **110 aborted** |
| **dose (epochs)** | **0.0147** | **1.5625** | **3.5156** | **0.8594** |
| stage-A outcome | nothing moved | trained, no erosion | destroyed by ~ckpt 50 | collapsed ~step 90 |
| in-domain acc | *(not measured)* | 0.45 → 0.39 | 0.45 → 0 | 0.45 → 0.23 |
| effective rank | +0.55%, flat | flat (±1%) | **variant-dependent, see §3.1 C** | flat over survivors |
| dormant fraction | **0.0000 everywhere** | **0 everywhere** | **0 → 0.0002** (trivial) | **0 everywhere** |
| entropy | flat (t = −0.85) | 0.37 → 0.10 | spiked 0.59, then crashed | 0.32 → 0.16, then collapse |
| Δ-R | +10.3 / +9.7 / +9.0 pp | +.06/+.07 → −.05/−.07 * | +.12/+.17 → 0/0/0 | +.16/+.19 |
| verdict | dose far too small | "too cold" | "too hot" | "still collapses" |

\* Run 1's negative SVAMP improvements were later traced partly to a greedy-eval
truncation artifact, fixed in runs 2–3. That column's magnitudes are not
trustworthy and the eval definition is not constant across the three runs — this
must be stated wherever run 1's Δ-R appears.

**Stale table in the cross-run note — fix before anything is copied out of it.**
`experiment 2/CROSS_RUN_NOTE_7B_VS_05B.md` §1 still reports the 0.5B doses as
~3.1 / ~7.0 / ~1.7 epochs, reconstructed from an assumed 512-question set at 8
prompts/update. Its own §7.1 already supersedes those with the values logged
directly in `jason run/run{1,2,3}.zip → stageA_log_history.json` — `epoch` is a
recorded field, 0.0078125 per update, i.e. 128 updates/epoch, giving **1.5625 /
3.5156 / 0.8594**, exactly half the §1 values. So the note is internally
inconsistent, not simply wrong, and the stale §1 table is the one most likely to be
copied into the paper. The table above uses the logged values throughout. The
7B-to-run-2 dose ratio is **238×**, and each of the 0.5B updates carries **53×** the
epoch-dose of one 7B update.

Incidentally, 128 updates/epoch implies an effective batch of **4 prompts/update**
if the GSM8K set is the proposal's 512 questions. That needs confirming with the
run owner; it does not affect the dose figures, which come straight from the log.

---

## 3. Evidence ledger — what we can and cannot claim

The single most important discipline for the write-up is separating **"not
detectable"** from **"is zero."** Most of what we have is the former.

### 3.1 Claims that are decisive

**(A) The dormant-neuron fraction has no dynamic range on gated-MLP LLMs.**

Two independent kinds of evidence plus a mechanism:

- *No headroom.* On the 7B run, `dormant_frac` is **exactly 0.0 at every layer,
  every checkpoint, and both thresholds** (τ = 0.025, 0.1). The minimum **normalized**
  dormancy score observed is 0.1604 / 0.4148 / 0.1606 at layers 5 / 14 / 26 — the
  least-active unit in the network sits **1.6× above even the loose threshold**. No
  unit could have been counted dormant regardless of what training did. Note this
  score is already the ReDo (Sokar et al. 2023) *normalized* form,
  `s_i = E|h_i| / mean_j E|h_j|`, so this is not a threshold-calibration artifact.
- *No response even to total collapse.* On 0.5B run 2, the model was reduced to a
  deterministic 384-token repetition loop at 0% GSM8K accuracy, and dormant fraction
  still only reached **0.0002**.
- *Mechanism.* Qwen2's gated MLP computes `act_fn(gate_proj(x)) · up_proj(x)`. Unlike
  the ReLU networks dormancy was defined for, this quantity essentially never reaches
  zero, and a near-zero gate can be rescued by a large `up`.

This is the strongest thing the project has. §5.1 (**E1**) is designed to make it
bulletproof rather than merely asserted.

**(B) The ceiling confound: much of what "Δ-accuracy" measures is headroom, not
adaptability.**

`ρ(acc_before, Δ-accuracy)` on three independent datasets:

| dataset | n | ρ |
|---|---|---|
| exp1.5 v3 | 6 | −0.66 |
| 0.5B run 1 (recomputed) | 5 | −0.667 |
| 7B GURU Δ-R | 3 | −0.756 |

Same sign, same magnitude, three times. The mechanism is visible directly in the 7B
data: `acc_after` has SD 0.47 pp across arms while `acc_before` has SD 0.72 pp, so
Δ-R is close to `constant − acc_before` by construction. This is a methodological
critique of how fixed-budget adaptability is measured across this literature, and it
is arguably more valuable than the original hypothesis. n is small in each case —
report all three together, never one alone.

### 3.2 Claims that are suggestive but not decisive

**(C) Effective rank is operationalization-dependent — the two variants move in
*opposite directions* during the same collapse — and neither variant leads.**

This is the sharpest result in the 0.5B data and it belongs in §3.1 alongside (A),
not here, once E1 confirms it at 7B. Run 2, the only complete collapse:

| ckpt | erank **MLP** | vs ckpt-0 | erank **residual** | vs ckpt-0 | GSM8K |
|---|---:|---:|---:|---:|---:|
| 0 | 1724.0 | — | 78.1 | — | 0.4531 |
| 50 | 1666.6 | **−3.3%** | 114.3 | **+46.3%** | 0.0469 |
| 150 | 1708.3 | −0.9% | 499.5 | **+539.2%** | 0.0000 |
| 300 | 492.0 | **−71.5%** | 361.3 | +362.3% | 0.0000 |

Same checkpoints, same collapse, one variant down 71% and the other up 362%. A
metric whose *sign* depends on which tensor you read is not yet a detector. This is
why E1's variant sweep is the priority, and note the 7B run only ever measured the
**residual** variant for erank — the one that went *up*.

**On lead time, state it narrowly.** Checkpoints exist only at 0 / 50 / 150 / 300 /
450, so Q's time resolution here is 50 updates at best. By the first post-failure
checkpoint (50) the model is already dead (GSM8K 0.4531 → 0.0469), so the residual
variant's +46% moved **concurrently with the visible collapse, not before it**; the
MLP variant did not move meaningfully until ckpt 300. So the defensible claim is
"neither variant led," not a quantified negative lead time.

**And the bar Q has to clear is higher than a soft reward trend.** `grad_norm` is
logged as **exactly 0.000** from step 35 onward — a free, unambiguous "training is
dead" flag already on the dashboard. Any lead-time claim for Q must beat that.

**(D) The collapse pathway is length explosion → clip saturation → gradient
starvation.** In runs 2 and 3, reward fell exactly as completions ballooned into the
384-token cap (99–100% clipped) and `frac_reward_zero_std` rose, leaving GRPO with
no reward variance and therefore no gradient. The Task-3 review records a critical
group pass rate p\* ≈ 0.083 at `num_generations = 8` below which most groups carry no
gradient; `frac_reward_zero_std` is logged in every run this project has done, so
this is checkable at zero compute cost.

**(E) The target regime was not reached at any of four settings.** Three directions,
14× model scale, 238× dose range, and none landed in "trains healthily while capacity
erodes." This is the honest team-level headline, and it is stronger than either
track's result alone. It is a statement about *our four settings*, not about the
regime's existence in general — the write-up must not overreach here.

### 3.3 Claims we must NOT make

- **Not** "RLVR reduces the ability to learn" — violates the framing constraint.
- **Not** the monotone Δ-R ordering (+31, +29, +27 net questions). The range is
  inside noise; three points go monotone by chance one time in three; and
  `acc_after` is itself **not** monotone (83, 86, 83). The apparent trend is
  manufactured by subtracting a non-monotone `acc_before` — it is (B), the ceiling
  confound, not adaptability.
- **Not** "entropy is the early-warning signal" — but do not undersell it either.
  Verified from run 2's artifacts, entropy crossed **2× its baseline at step 20**,
  ten updates before reward crossed the critical group pass rate p\* and fifteen
  before `grad_norm` hit exactly zero. **That is a genuine positive lead time, and
  the only one anywhere in this dataset.** What blocks promoting it to the headline
  is that it does not replicate: run 3's surviving checkpoints show the
  entropy→learnability relation running the *opposite* direction, and the 7B run's
  entropy is flat (t = −0.85). Report it as a dashboard baseline with one confirmed
  lead and two non-replications — which is also exactly the bake-off the proposal
  commits to, with Q currently losing it.
- **Not** "Q is uncorrelated with adaptability." The 7B run cannot test this: **both
  sides of the correlation are flat.** Q moved ≤0.55%, dormant fraction sat at 0.0,
  and Δ-R varied 1.33 pp against a ~3.0–3.9 pp SE (z ≈ 0.4). The run is equally
  consistent with the hypothesis and with its negation. That is a limitation, not a
  negative result about Q.

---

## 4. Cause and effect — how we got here

The chain is worth stating explicitly, because each of the three proposed
experiments attacks one specific link in it.

**4.1 The 7B run under-dosed by two orders of magnitude, for a hard compute reason.**
Stage A ran 100 updates × 8 prompts = 800 of 54,251 rows = **1.47% of one epoch**.
One full epoch would be 6,781 updates × 198 s/update ≈ **373 A100-hours ≈ 2,525
compute units** — more than 20× the entire budget, for one epoch of one stage of one
arm. Three independent measurements confirm almost nothing happened: full-layer
weight norms moved 2.9 × 10⁻⁵ percent, no dashboard signal has a significant slope
over the 100 updates, and Q moved ≤0.55%. **This is not fixable by rerunning; it is
arithmetically closed at 7B on this budget.**

**4.2 The 0.5B runs reached real dose but could not stay alive.** Three learning
rates spanning 2× produced: trains-but-self-extinguishes (5e-6 decayed to zero),
collapse at ~step 90 (5e-6 constant), and collapse by ~update 35 (1e-5 constant).
The LR axis is effectively exhausted — the only setting that survived is the one
whose schedule drives the learning rate to zero, which is why nothing eroded.

**4.3 Therefore the missing lever is not learning rate.** All four runs are **β = 0**.
The proposal's own stability mechanism — the KL anchor — has never been switched on.
Combined with a length control targeting the collapse pathway in (D), that is the
one untested direction with a plausible path into the target regime.

**4.4 Meanwhile, "Q did not move" has an unexamined alternative explanation.** Q is
currently measured on **prompt-only forward passes**: the probe set is 4,096 frozen
prompts, the model is run with `model(**enc)` on those prompts, effective rank is
taken over the last-non-padding-token hidden state, and dormancy over MLP
post-activations at prompt token positions. **RLVR does not update the model on
prompts — it updates it on generated completions.** If stage A changed the model's
generation behaviour while leaving its representation of *prompt* tokens intact, our
probe would read flat by construction. We have never tested this. It is cheap to
test, it uses checkpoints we already have, and it is squarely this owner's assignment.

That gives three experiments, in priority order.

---

## 5. The next three experiments

Budget context: **~115 compute units** as of the 2026-08-19 reading (re-check before
planning). A100-80GB bills at **~6.77 units/hour**; 0.5B work does not need an A100
and should run on L4/T4 per the standing guideline. All 7B stage-A adapters
(ckpt-0/50/100), configs, and frozen splits are preserved in Drive at
`MyDrive/eaaj-exp2-checkpoints`, hash-verified — so E1 requires **no retraining**.

### 5.1 E1 — Metric re-measurement sweep on existing 7B checkpoints

**Owner:** this owner (early-warning diagnostics — directly the assigned role).
**Depends on:** nothing. No team decision, no other person, no new training.
**Cost:** ~1–2 A100-hours ≈ **7–14 units**.
**Full runnable spec:** `experiment 2/SPEC_E1_METRIC_REMEASUREMENT.md`.

**The question.** Our strongest claim is that Q — especially dormant fraction — has
no usable dynamic range in this setting. A reviewer's first move is to ask whether we
simply operationalized it badly. Right now we cannot answer that. E1 answers it by
measuring the same checkpoints under a grid of defensible alternative
operationalizations, all against the same frozen probe and the same comparability
contract.

**Current contract (held fixed as the reference arm):** `model_eval=True`,
`dtype=bfloat16`, `hidden_pooling=last_non_padding_token`,
`dormant_pooling=mean_abs_over_all_non_padding_tokens`, `max_prompt_tokens=512`,
activation accumulator float32, SVD float64, n_probe = 4096, layers [5, 14, 26],
batch 16. Two independent passes over ckpt-0 agree **bit for bit**, which is what
makes small differences interpretable.

**The five variants.**

| # | Variant | What it tests | Why it is not already answered |
|---|---|---|---|
| V1 | **Probe distribution: prompt-only → prompt + fixed continuation** | whether Q is blind to the part of the distribution RLVR actually trains on | the current probe never sees a generated token; §4.4 |
| V2 | **Dormancy pooling: mean-over-tokens → per-token and max-over-tokens** | whether averaging hides units that are dormant on most inputs but active on a few | the current statistic collapses every token position into one scalar per unit |
| V3 | **Dormancy tensor: `act_fn(gate)·up` → `act_fn(gate)` alone** | whether the gate has range that the product destroys (a large `up` rescues a near-zero gate) | only the `down_proj` input has ever been measured |
| V4 | **τ sweep: {0.025, 0.1} → log grid 1e-4 … 1.0, plus the full score histogram** | where the metric *would* start reporting, and how far the distribution sits from it | two points cannot show a curve; "no headroom" is currently an assertion about two thresholds |
| V5 | **Sensitivity: all 28 layers; n_probe ∈ {512, 1024, 2048, 4096}; last-token vs mean-over-positions** | that the flat reading is not an artifact of layer choice, probe size, or pooling | only 3 of 28 layers were measured; n=4096 > d=3584 is asserted but not demonstrated to be non-binding |

**V1 needs care and is specified precisely in the spec.** Two sub-arms:

- **V1a (comparable):** continuations generated **once from ckpt-0** and then held
  fixed for all three checkpoints. The probe stays frozen, so the comparability
  contract survives and V1a values are directly comparable across checkpoints.
- **V1b (on-policy, not strictly comparable):** each checkpoint generates its own
  continuations. This is the behaviourally relevant probe but the input distribution
  differs per checkpoint, so it violates the frozen-probe contract. It is reported
  as a separate, explicitly-labelled measurement — never mixed into the V1a series.

Generation is the expensive part, so V1 subsamples the probe to 512 prompts at 256
new tokens. V1a needs one generation pass total; V1b needs three.

**What each outcome means — both are publishable.**

- *Every variant still reads flat / zero.* The claim becomes "we tested k
  operationalizations spanning pooling, tensor choice, threshold, layer, probe size,
  and probe distribution, and the metric has no dynamic range under any of them."
  That is a decisive negative result about a metric the plasticity literature imports
  into LLMs, with a mechanism.
- *Some variant shows range* — most plausibly V1 (the probe was blind to generation)
  or V3 (the gate has range the product destroys). Then we have a **positive**
  finding: a corrected operationalization of an existing metric, plus a
  demonstration that the standard one fails on gated-MLP LLMs. That is a better
  paper than the negative version.

There is no outcome in which E1 is wasted, and it is the only proposed experiment
that depends on nobody else.

**Acceptance gates.** (i) The reference arm must reproduce the recorded
`FINDING_Q_METRICS_7B_INSTRUCT.md` values **exactly** — any drift invalidates the
sweep and must be investigated before results are trusted. (ii) Every variant writes
the full measurement contract into its output JSON. (iii) Per-unit score vectors are
saved, not just summary fractions, so the histograms in V4 are reproducible without
re-running.

### 5.2 E2 — Seed repeats at 0.5B

**Owner:** 0.5B pipeline (Jason's track) — coordinate before running.
**Cost:** ~3–6 L4-hours ≈ **6–12 units** (run 1's stage A was 3,311 s ≈ 55 min).

**The question.** Every "flat," "no difference," and "not detectable" statement in
§3 currently rests on a **single seed**. Without error bars, none of them is
defensible at review, and the ceiling-confound ρ values in (B) have no interval
either.

**Design.** Whichever 0.5B configuration becomes the paper's primary figure, run it
at 3 seeds end to end (stage A + the checkpoint Q measurements + the SVAMP probe),
with everything else pre-registered and identical. Report mean ± SD, and give the
ceiling-confound ρ a bootstrap interval across seeds.

**Explicitly out of scope: reseeding the 7B arms.** Each is ~11 A100-hours ≈ 75
units; three seeds would consume roughly twice the remaining budget for a run we
already know is under-dosed by 68×.

### 5.3 E3 — Run 4: the KL-anchored 0.5B run

**Owner:** 0.5B pipeline (Jason's track). This is the run his own run-2 analysis
recommends; it should not be started in parallel by two people.
**Cost:** ~2–3 L4-hours ≈ **4–6 units**.

**The question.** Is the target regime reachable at all once the proposal's own
stability mechanism is switched on? Per §4.2–4.3, the LR axis is exhausted; β is
untouched.

**Design.** β ≈ 0.02–0.04, LR 5e-6 constant, **plus a length control in the reward**
to target the collapse pathway in (D) directly. Keep run 3's two safeguards: a
mid-run GSM8K health eval every ~50 updates so we can see stage A staying alive, and
the early-abort guard so a dead run does not consume the session. Q measured at the
same checkpoints under the same contract as every other run.

**What each outcome means — again, both help.**

- *Lands in the regime* (stage A healthy at real dose while Q moves): we can test
  RQ1 for the first time. This is the outcome that would justify aiming higher than a
  workshop.
- *Still collapses*: claim (E) upgrades from "three learning rates" to "four
  settings including the proposal's own stabilizer," which is a much harder result to
  wave away, and (D)'s collapse pathway gains a fourth confirmation.

### 5.4 Two operational fixes that cost nothing and must not be skipped

1. **Save per-question outcomes on every evaluation.** The 7B Δ-R arms did not, so a
   paired McNemar test is not computable after the fact and had to be replaced by a
   feasible-interval bound. The bound happened to be sufficient (every arm's Δ-R is
   significantly positive for every feasible value of the missing data; worst-case
   p = 0.0076 / 0.0153 / 0.0220), but that was luck. Writing per-question correctness
   into the output JSON costs nothing.
2. **Take the GPU-dashboard screenshots.** `eaaj-pilot/compute_log.md` records that
   screenshots were taken for **no exp2 session** and cannot be recovered after the
   fact — a standing violation of a hard compute-accounting constraint. Start with
   the next session. Also: disconnect the runtime the moment a run finishes; 5.6
   units were burned idle after the 2026-08-19 run completed.

---

## 6. What we deliberately do NOT run, and why

- **Any further 7B stage-A training.** Reaching one epoch needs 68× the updates ≈
  373 A100-hours ≈ 2,525 units against ~115 available. Arithmetically closed. A
  repeat at the same dose would reproduce "nothing moved."
- **A fourth 0.5B learning rate.** Three LRs spanning 2× gave three collapses; §4.2.
  The axis is exhausted and another point on it buys nothing.
- **A new stage-1/stage-2 task pair.** `AARON_COLLATING_PAGE.md` (2026-08-02) already
  predicted that four people picking four new pairs at safe learning rates would
  produce four more null results, and the 7B run confirmed that prediction at 14×
  scale. Dose and stability are the blockers, not the dataset pair.
- **Chasing the entropy signal as the new headline.** See §3.3 — it is unstable in
  sign across the runs we have. It should be logged as a first-class dashboard
  baseline that Q must beat, not promoted to hero metric on one run's evidence.

---

## 7. Decisions needed from the team

These are scope and framing calls, not this owner's to make unilaterally. Each is
stated with a recommendation so it can be answered yes/no rather than reopened.

1. **Reframe the paper's RQ?** The current RQ1 ("does Q warn earlier than the
   dashboard") is unanswerable with our data and the available evidence points the
   other way (§3.2 C). *Recommended:* reframe to "**is the presumed regime reachable,
   and do the proposed detectors have the dynamic range to operate in it?**" — which
   our four runs plus E1 do answer. This changes a pre-registered proposal and
   therefore needs Tommy and Madhur.
2. **Report both tracks as one bracketing study?** *Recommended: yes* — §3.2 (E) is
   a team-level result neither track can claim alone. Requires agreeing the shared
   Method section and a common symbol table.
3. **Target venue.** *Recommended:* draft for a workshop and upgrade if E3 lands in
   the regime. Method / Experiments / Limitations are identical either way; only the
   Introduction's contribution paragraph and the length differ, so starting now costs
   nothing under either answer.
4. **Who runs E2 and E3.** Both are on the 0.5B pipeline. Needs an explicit owner so
   the same run is not done twice.
5. **Still open from the 7B run** (unchanged, listed for completeness): the
   `max_new_tokens=512` truncation on `guru_greedy_accuracy` (~10–15% of that
   domain's eval answers, bounded but not eliminated), the GATE 0a population, the
   Instruct/1536 amendment, the 640→2048 cap, and the eval-point de-scope.

---

## 8. How each experiment maps onto the paper

The program's paper-writing guide splits this into **Method** (procedural, one
method, no notation not introduced in Background) and **Experiments** (model choice,
parameters, datasets, baselines). That split resolves the "whose results do we use"
question structurally: **Method is singular and shared; Experiments is plural.**

| Paper section | Content | Source |
|---|---|---|
| Background | symbol table: Q, erank, dormant fraction, Δ-R, T_t, dose, fixed budget | this doc §1 |
| Method | the shared measurement protocol — frozen probe, eval mode, fixed layers/dtype, Δ-R and T_t definitions, leakage rule | §1, §5.1 contract |
| Experiments 1 | 0.5B GSM8K→SVAMP, LR/dose sweep + E2 seeds + E3 | 0.5B track; setup written by its owner |
| Experiments 2 | 7B GURU Math→Simulation, fixed-budget Δ-R | this repo |
| Experiments 3 | E1 metric operationalization sweep | this repo |
| Results | (A) dormancy has no dynamic range; (B) ceiling confound; (E) the bracketing | §3.1, §3.2 |
| Limitations | cannot test RQ1; single seed; n=300 → ±6 pp floor; ceiling confound; 512-token truncation; β=0 throughout; 7B dose | §3.3, §4 |

For \*ACL venues the Limitations section is separate and **does not count toward the
page limit**, which is where most of §3.3 and §4 belongs — it costs no page budget
and pre-empts the obvious reviews.

Writing is not blocked on any of these experiments. Background, Method, the
Experiments configuration tables, and Limitations can all be drafted now; E1–E3
affect only Results and the Introduction's contribution paragraph.

---

## 9. Execution order

```
now        E1  (this owner, ~7–14 units, no dependencies)
in parallel    Background + Method + Experiments tables + Limitations draft
after §7.4     E2 seeds, then E3 run 4  (0.5B track, ~10–18 units combined)
```

Total proposed spend: **~17–32 units of ~115**, leaving headroom for a second E3
attempt if the first tells us something about where β should sit.

E1 goes first because it is the cheapest, it depends on nobody, it uses checkpoints
we already hold, it is precisely this owner's assigned role, and it determines
whether the project's strongest claim survives review.
