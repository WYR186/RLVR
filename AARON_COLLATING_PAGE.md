# Aaron Wang (Person 4) — Collating Page

Last updated: 2026-08-02. Every number below comes from committed run artifacts and
has been re-checked against the raw JSON/CSV. Values computed for this page rather
than taken from a committed analysis file are marked **[descriptive]**.

---

## 1. Status

Three experiments completed since exp-1 (exp1.5, exp1.5.1, exp1.6 phase 1) plus Task 3.
The blocker for RQ1 is unchanged and now well characterised: **the thing we want to warn
about — a persistent, fixed-budget adaptability loss — has never been produced at a dose
the model survives.** Task 3 is finished. Next milestone: the exp1.6 Q measurement
(~20 min), which decides whether lr 3e-6 is a usable dose.

---

## 2. Experiments

| Run | Stage-A dose | Outcome | Key numbers |
|---|---|---|---|
| **exp-1 pilot** (2 strata) | lr 1e-6 × 200 | Healthy, both platforms | 16 adaptations: 15 positive, 1 flat, 0 negative. Primary ρ(erank_L12, Δ) unstable: −0.60 (Mac) / +0.50 (Win) / −0.50 (seed 43). Dormant ≡ 0. ckpt-0 erank agrees to 4 decimals across machines. |
| **exp1.5 v1** | lr 1e-5 | Safety-stopped, update 7 | 5 consecutive updates over the 10% completion-clipping line |
| **exp1.5 v2** | lr 1e-5 | **Policy collapse, update 55** | Clipping 0.8 → 1.0; entropy 0.20 → 0.05; 5 consecutive zero-variance updates → zero gradient. Full per-step telemetry preserved. |
| **exp1.5 v3** | lr 1e-6 × 500 | Healthy, saturates ~300 | erank_L12 endpoint **−0.98%** → pre-registered **MC1 fail**. MC2 pass (ckpt-50 −8.6pp vs ckpt-0), but see §3.2. ρ(erank_L12, Δ) = **+0.60**, n=6, 3 seeds consistent in sign (+0.43/+0.71/+0.90). |
| **exp1.6** | lr 3e-6 × 500 | **Healthy, phase 1 complete** | Training reward 0.51 → 0.78; held-out GSM8K **35.9% → 57.8%**; entropy 0.14 → 0.07; no safety stop. **Phases 2–4 pending.** |

Noise engineering (exp1.5 onward): 3 adaptation seeds × 300 eval questions. Measured
variance components — between-checkpoint **0.00202**, within-checkpoint (seed) **0.00049**.
Seed-to-seed Δ ranking correlation improved from −0.50 (pilot) to +0.55…+0.71. These are
directly usable for power calculations.

Dose–response map so far (0.5B, GRPO, β=0):

| Dose | Stability | Q endpoint shift | Fixed-budget adaptation |
|---|---|---|---|
| 1e-6 × 200 | healthy | L12 −7.4% … +0.7% (run-to-run spread) | all non-negative |
| 1e-6 × 500 | healthy | L12 −0.98% | V-shaped transient, then all positive |
| **3e-6 × 500** | **healthy, still improving at 500** | **pending** | **pending** |
| 1e-5 | **collapse ≤ 55 updates** | not sampled in time | never reached |

---

## 3. Findings that affect the whole team

### 3.1 Dose, not the dataset pair, is the current blocker

1e-6 and 3e-6 train the model well and erode nothing. 1e-5 kills the policy in under 60
updates. If four people pick four new stage-1/stage-2 pairs and all run at safe learning
rates, we should expect four more null results. **Whatever pair each of us picks, the dose
has to be one that has been shown to move something without killing the model — and right
now nobody knows what that dose is.** The exp1.6 measurement is the cheapest available
answer, so I will post it as soon as it runs.

### 3.2 Our outcome measure is confounded by the starting point

Δ-accuracy correlates negatively with pre-adaptation accuracy:

- **ρ = −0.66** in exp1.5 v3 (n=6) **[descriptive]**
- **ρ = −0.667** when the same statistic is recomputed on Jason's run 1 (n=5) **[descriptive]**

Two independent datasets, same answer: a large part of what "Δ accuracy" measures is *how
much room was left to improve*, not adaptability. In exp1.5 the worst-adapting checkpoint
(ckpt-50, −2.9pp) was also the one with the highest zero-shot SVAMP start (62.3%, +9.3pp
above ckpt-0) — positive GSM8K→SVAMP transfer ate its headroom. Jason's run 3 shows the
mirror image: his damaged ckpt-50 (SVAMP start 24.7%) posted the largest gain in his whole
set, +19.3pp.

**Consequence:** this supports switching the stage-B family, but a swap alone does not fix
it. The new pair needs a starting-point drift check, and any correlation we report needs
the pre-adaptation accuracy as a covariate.

### 3.3 Cross-checks against Jason's three runs

| Point | Mine | Jason's |
|---|---|---|
| Collapse at lr 1e-5 | onset ~update 37, terminal 55 | onset ~step 30, dead by 50 |
| Collapse mechanism | length explosion → token cap → reward starves → zero group variance → zero gradient | identical |
| Dormant fraction | ≡ 0 everywhere | ≡ 0 everywhere |
| Q at safe dose | endpoint −0.98% | flat ±1% |
| Ceiling confound | ρ = −0.66 | ρ = −0.667 (my recomputation) |

Two differences worth noting rather than averaging away:

- **My 3e-6 run does not collapse**, while his constant-LR 5e-6 run collapsed at ~step 90.
  Different model variant (base vs Instruct) and different token cap (512 vs 384). Since
  both of us identify length explosion as the proximate cause, the tighter cap is a
  plausible reason his stability boundary sits lower. Combined, the untested band narrows
  to roughly 3e-6 – 5e-6.
- **Metric definitions are not yet comparable.** In his run 2 the two effective-rank
  variants moved in opposite directions during the same collapse (MLP −71%, residual
  +539%). Before any of our results are pooled, we need one measurement contract.

---

## 4. Task 3 — four-paper review

Full document: `lit review/TASK3_FOUR_PAPER_REVIEW.md` (comparison table in §4, split into
four panels covering training transition, plasticity definition, metrics, future-learning
evaluation, prediction, interventions, released setup). PDFs and extracted text in
`lit review/task3_core_papers/`.

1. **The four papers do not share a definition of plasticity, and only one shares ours.**
   Plasticine defines it through a six-metric panel; paper 2 defines it as a term in the
   GRPO loss; paper 3 as a disposition of a checkpoint; paper 4 as "degradation in a
   model's ability to improve on a target distribution under a fixed training budget" —
   the definition we use.
2. **No paper predicts future fixed-budget adaptability from an activation metric.** That
   gap is exactly RQ1: good for novelty, bad for having a method to copy.
3. **Paper 4 reports the relevant negative.** Across 8 model sizes it tested dormant-unit
   fraction and parameter magnitude against fixed-budget adaptability and found neither
   tracks it. It did **not** test effective rank — that is our opening — but our null
   hypothesis deserves real respect, and the write-up should be prepared to report a clean
   null rather than fish.
4. **Release status:** only Plasticine has an official repository (MIT, verified via the
   GitHub API). Papers 2–4 released nothing. Paper 2 additionally cross-references
   appendices B–L that **do not exist** in the arXiv version, so its setup, algorithm and
   both proofs are unverifiable.
5. **Pipeline recommendation: stay on TRL 1.6 / our own `eaaj-pilot`** (scored 49/50, vs
   verl 32 and OpenRLHF 27). The parts that are hard — checkpoint branching, independent
   stage-B launches, frozen probe sets, in-process activation hooks — do not exist in any
   framework and we have already written them. Borrow formulas from Plasticine (MIT) and
   the discarded-copy probe protocol + AUC outcome from paper 4.

### One paper not on the list that may matter more than the four

**arXiv:2606.18487** runs the multi-seed, statistically analysed version of paper 3's
design and reports **pre-RL entropy predicting GRPO outcome at ρ = +0.69** — directly
relevant to Jason's entropy result. It also derives a critical group pass rate: with
`num_generations = 8`, below **p\* ≈ 0.083** most groups carry no reward variance and
therefore no gradient.

Checking that against Jason's run 2 **[descriptive]**: training reward crossed 0.083 at
step 30 — the same step its zero-variance-group fraction jumped to 0.60 — and it never
recovered. Run 3 crossed the same line between steps 90 and 95.

**We already log this quantity (`frac_reward_zero_std`) in every run we have done.** It is
reward-related in the sense Tommy asked for, it has a theoretical threshold, it rises
before the terminal event, and it is computable on existing data with no new compute.

*Caveat: I have read this paper's abstract page only, not its full text. Everything
attributed to it must be verified before it is cited.*

---

## 5. Known limitations of my own measurements

1. **Probe set is smaller than the hidden dimension.** The primary Q measurement uses 512
   probe prompts against a hidden dim of 896, so the activation matrix is 512×896 and
   effective-rank *magnitudes* are sample-truncated and n-dependent. Plasticine's code
   asserts `n ≥ d` for exactly this reason. Direction is confirmed at n=2048 (Windows
   ckpt-200 vs 0: L12 −7.2%, L22 −7.7%), so the shape findings hold, but absolute values
   should not be quoted or compared to literature until re-measured. Fix is a config
   change and one re-measurement pass; I will do it in the same run as the exp1.6
   measurement.
2. **Dormant-neuron fraction has no resolution in this setting.** It is identically 0.0 at
   every layer, checkpoint and threshold. The cause is structural: Qwen2's SiLU-gated MLP
   computes `act_fn(gate) * up`, which essentially never reaches exactly zero, and the
   minimum normalised score observed is 0.13. This must be reported as *a metric with no
   resolution*, never as evidence that plasticity is preserved.
3. **n = 5–6 checkpoints, single stage-A trajectory.** Any correlation over this many
   points is descriptive. The pilot's primary ρ flipped sign across execution strata and
   across stage-B seeds alone; none of those ρ values should be cited as evidence for RQ1.

---

## 6. Next steps and dates

| Date | Item |
|---|---|
| 8/3 | exp1.6 Q measurement (+ probe-size fix) → dose verdict, posted to Slack |
| 8/5 | Stage-B family screening: zero-shot eval of all 8 exp1.6 checkpoints on SVAMP (control) / PrOntoQA / ProofWriter, no training, to test whether stage-A training moves the stage-B starting point |
| 8/6 | Stage-2 family and dose locked — no changes after this date |
| 8/12 | Adaptation grid complete |
| 8/15 | Analysis and figures |

**Claimed pair:** GSM8K → PrOntoQA (ProofWriter as fallback). Stage 1 is already trained —
8 checkpoints at lr 3e-6 are on disk — so I start at stage 2.

**Open questions for the team** (implemented with the cheaper default, logged, not decided
unilaterally): base vs Instruct; GRPO vs SFT for stage B; KL β>0 arm (β=0.04 stub is ready,
≈8 GPU-h); whether we adopt AUC of the stage-B curve as a co-primary outcome; whether all
existing checkpoints get re-measured at the larger probe size.

---

## 7. Artifact index

| Content | Path |
|---|---|
| Exp-1 full report | `EXP1_TEAM_REPORT.md` |
| Exp-1.5 analysis (ZH) | `EXP1_5_RESULTS_ANALYSIS_ZH.md` |
| Task 3 review | `lit review/TASK3_FOUR_PAPER_REVIEW.md` (+ `.docx`, + PDFs in `task3_core_papers/`) |
| Evidence pack shared on Slack | `experiment 1/pilot_evidence_pack/` |
| Exp-1.5 plans and amendments | `experiment 1.5/EXPERIMENT_1_5_PLAN_ZH.md`, `…AMENDMENT_V2/V3.md` |
| Exp-1.6 plan and gates | `experiment 1.5/EXPERIMENT_1_6_PLAN_ZH.md`, `exp1_6_gate_eval.py` |
| Run artifacts | `eaaj-pilot/outputs/` |
| Compute ledger | `eaaj-pilot/compute_log.md` |
