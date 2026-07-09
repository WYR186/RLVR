# Agent Briefing — First Experiment Prep (RLVR Plasticity Collapse Project)

**Prepared for:** coding agent assisting Aaron Wang (Person 4)
**Date context:** early July 2026. Slides due ~Jul 5; first experiment starts right after. Hard team cadence: progress updates 2x/week in Slack.
**Attached alongside this file:** (1) `Formal_Research_Proposal` (PDF, "Seeing the Stall Coming", Version 3) — the authoritative spec; (2) `eaaj-research.docx` — the team Research Doc (collating place; anything we produce must eventually land there).

---

## 1. Project in one paragraph

The team studies **multi-stage RLVR training of reasoning LLMs** (RL stage A → RL stage B). A common failure is that a later stage **stalls** (stops improving) and trainers only find out after the compute is spent. The proposal's bet: a **plasticity quantity Q** read from the model's internal activations during stage A — primarily **effective rank** and **dormant-neuron fraction** — is an **early-warning signal** for whether stage B will stall, and warns **earlier than dashboard signals** (reward slope, KL accumulation, gradient norm, entropy). Headline deliverable: a validated, calibrated stall detector benchmarked head-to-head against those baselines on **lead time**. Full details, metric definitions, and success criteria are in proposal §5–§9. Read §7 (Metrics) before writing any measurement code.

## 2. Where the team is right now

- Lit review is done (Person 1–4 sections) and being merged into the Research Doc / proposal. Slides for the PI meeting are being made (4 sections; Aaron is involved in "Related Works & Methods" and experiment sanity-checking).
- **Tommy (team lead) has proposed the first experiment** (below) and asked the team to sanity-check it and start on **Google Colab** (budget: **$20–30, roughly 300 compute units per 2-person pair**, reimbursable). Jason, Arnav, Aaron, and Shwara are tagged as implementers.
- **Mentor feedback (Madhur) that constrains our framing and code:** do NOT claim "RLVR reduces the model's ability to learn" — plasticity has no single accepted scalar. The clean question is: **does RLVR reduce fixed-budget future adaptability on held-out task families, compared to the base model (checkpoint 0) or a KL-regularized checkpoint?** So every outcome we log must be a *fixed-budget* quantity (adaptation speed, final accuracy after N updates, pass@k), with the future task, budget, and baseline checkpoints defined up front. The experiment below is designed exactly that way — keep it that way in code.

## 3. The first experiment (Tommy's spec, verbatim intent)

> Run **GRPO on Qwen2.5-0.5B** with **512 GSM8K questions** and **exact-answer reward**. Save checkpoints at **0, 25, 50, 100, and 200 updates**.
> At each checkpoint, measure **dormant-neuron fraction** and **effective rank**. Then run the **same short SVAMP adaptation test from each checkpoint**: e.g., **256 SVAMP training questions, 100 SVAMP eval questions, 50-update budget**.

Hyperparameters and splits are explicitly *suggestions* — we may adjust at implementation time, but any change must be logged and justified in the Research Doc.

**What this is, in proposal terms:** a scaled-down pilot of Task 1 + Task 2 + a miniature future-learning probe (Task 4's outcome variable). Stage A = GSM8K GRPO; the "stage B" proxy = fixed-budget SVAMP adaptation. The scientific target of the pilot is **RQ1 (existence/correlation)**: does Q measured at checkpoint t correlate with fixed-budget SVAMP adaptability from checkpoint t? Note the proposal's danger zone is 1.5B full-FT; 0.5B is a Colab-budget compromise — treat it as a feasibility pilot, not the real corpus.

## 4. Aaron's specific responsibility (what the agent should optimize for)

Aaron is **Person 4: early-warning diagnostics — dashboard baselines, representation metrics, LLC/devinterp**. In this pilot that means:

1. **Own the instrumentation**: the Q metrics (effective rank, dormant fraction) and — equally important — the **dashboard signal logging** (reward mean/std per step, KL if enabled, gradient norm, policy entropy, response length, per-difficulty accuracy). Even though the pilot only *needs* Q, logging the dashboard signals now costs nothing and makes the later detector bake-off possible on this same data. Do not skip this.
2. Produce clean per-checkpoint measurement artifacts (JSON/CSV) + plots that can go straight into slides.
3. Keep compute accounting evidence (see §8).

## 5. Metric implementation spec (from proposal §7 + Person 4's lit review)

Measure on a **fixed probe set** in **eval mode** (dropout off, fixed dtype, fixed layer choices) so values are comparable across checkpoints:

- **Probe set:** 512 fixed prompts. Use GSM8K *test-split* questions (never trained on) or a held-out slice of train; freeze the exact list in a committed file. Forward pass prompt-only (no sampling needed for Q).
- **Effective rank (primary Q):** collect last-token hidden states at 3 fixed layers (early / middle / late, e.g. layers 4, 12, 22 of 24 for Qwen2.5-0.5B) → matrix A (512 × d) → center → SVD → `erank = exp(−Σ p_i log p_i)`, `p_i = σ_i / Σσ_j`. Also log the **normalized** variant (erank/d), **participation ratio** `(Σλ)²/Σλ²`, and **top-k variance share** (k ∈ {1,8,32}) — free once you have the spectrum. Report **centered and uncentered anisotropy** (mean pairwise cosine) as a pair to rule out mean-shift artifacts.
- **Dormant-neuron fraction (secondary Q):** on MLP post-activation units at the same layers, score `s_i = E_x|h_i(x)| / ((1/H) Σ_j E_x|h_j(x)|)`; neuron dormant iff `s_i < τ_d`; report at **τ_d ∈ {0.025, 0.1}** (both, per proposal).
- **Cheap extras (one line each, always log):** weight-norm growth per layer group; gradient norm per update (from trainer logs).
- **Known gotcha (from the Tracing paper, NeurIPS 2025):** spectral metrics are sample-size sensitive; that paper used ~10K probe samples. Since we use 512, run a quick **probe-size sensitivity check** once (512 vs 2048) at checkpoint 0 and 200 and note the deviation. Cheap, and the PI will ask.
- **Leakage rule (proposal §6):** any feature at step t must use only information available at t. No post-hoc normalization across the whole run.

## 6. Suggested pipeline (phased, Colab-friendly)

**Phase 0 — dry run (CPU/T4, <1 unit):** dataset loaders (GSM8K via HF `openai/gsm8k`, SVAMP via HF `ChilleD/SVAMP` or equivalent — verify fields), reward function (parse `#### <number>` / final-number extraction; exact match → 1.0 else 0.0), probe-set freeze, metric functions unit-tested on the base model with 8 prompts.

**Phase 1 — GRPO training (A100 or L4):**
- TRL `GRPOTrainer` (verl is overkill for Colab), Qwen2.5-0.5B (use the **base** model, not Instruct, unless the team says otherwise — confirm in Slack; base matches the proposal's setting but Instruct formats answers more reliably; log the choice).
- Full fine-tune, bf16. Suggested: `num_generations` 8, modest prompt/completion lengths (GSM8K completions ≤ 512 tok), fixed sampling temperature (proposal requires fixed T across runs), **β(KL) = 0** for the danger-zone flavor (default in recent TRL) — but log whatever is used.
- Save full checkpoints at **0 (= base), 25, 50, 100, 200** updates. 0.5B fp32 state ≈ small enough for Drive; save bf16 weights only (no optimizer state) for the measurement/adaptation steps.
- Log per-update: reward mean/std, grad norm, entropy, KL (if any), response length, and eval accuracy on a small fixed GSM8K eval slice every 25 updates.

**Phase 2 — Q measurement:** load each checkpoint, run probe set, dump `metrics_ckpt{N}.json` with all quantities from §5.

**Phase 3 — SVAMP fixed-budget adaptation (the outcome):** from **each** of the 5 checkpoints, run the *identical* adaptation recipe: 256 fixed SVAMP train questions, 50 updates, same optimizer/LR/seed, eval on the same fixed 100 SVAMP questions **before and after**. Record: final accuracy, Δaccuracy, and accuracy-vs-update curve (adaptation speed). Decide GRPO-vs-SFT for the adaptation phase with the team — GRPO keeps it apples-to-apples with the proposal's "stage B = RL" framing and is the better default; note whichever is chosen. 5 runs × 50 updates is cheap.

**Phase 4 — analysis:** one table (checkpoint × {erank per layer, dormant fraction per τ, GSM8K acc, SVAMP Δacc, SVAMP final acc}) + three plots: (a) Q vs training updates, (b) GSM8K reward curve with Q overlaid on twin axis (the "dashboard flat / Q moving" motif from the proposal), (c) scatter Q(ckpt) vs SVAMP fixed-budget outcome with Spearman ρ. Export PNG + the table as CSV for slides/doc.

## 7. What counts as a good result (for the slides' "Demo & Experiments" section)

- **Good:** Q declines systematically across GSM8K training AND correlates (negative direction for erank decline) with fixed-budget SVAMP adaptability across the 5 checkpoints. Even a clean monotone trend with 5 points is a persuasive pilot.
- **Null/negative:** Q barely moves at 0.5B/200 updates, or no relation to SVAMP outcome. **Pivot options to state up front:** longer training (200 → 500+ updates), enforce danger zone harder (β=0 confirmed, full-FT, smaller LR warmup), scale to Qwen2.5-1.5B, or make stage A/B more dissimilar (GSM8K → ProntoQA instead of SVAMP, which is very close to GSM8K — flag this to the team: SVAMP's similarity to GSM8K may mask capacity effects and mostly measure specialization; that is exactly the confound proposal §9's relearning discriminator worries about).
- Either way, record wall-clock and GPU type per phase to extrapolate the full sweep's compute (that extrapolation is itself a required deliverable for Tommy).

## 8. Compute & logistics constraints (hard requirements)

- Google Colab, **~300 compute units budget for the pair**, $20–30 reimbursable. A100 burns units several times faster than L4/T4 — use L4/T4 for anything that fits, A100 only for the GRPO generation-heavy phase if needed. Check the current units/hour in Colab's resource panel and record it.
- **Keep screenshots of the Colab GPU dashboard during runs** and a `compute_log.md` (date, GPU, duration, units before/after, phase). Tommy explicitly requires this for full-experiment budgeting.
- Everything reproducible: fixed seeds, `requirements.txt` / pinned pip installs at notebook top, all artifacts saved to Drive with checkpoint-named folders.
- Any deviation from Tommy's spec (model variant, adaptation recipe, LRs) → one-line justification in the notebook header, mirrored into the Research Doc later.

## 9. Repo/notebook structure suggestion

```
eaaj-pilot/
  00_setup_and_data.ipynb        # loaders, reward fn, probe freeze, unit tests
  01_grpo_gsm8k.ipynb            # Phase 1 training + dashboard logging
  02_measure_Q.ipynb             # Phase 2, pure measurement, no training
  03_svamp_adaptation.ipynb      # Phase 3, one function run 5x
  04_analysis.ipynb              # table + 3 plots
  src/metrics.py                 # erank/PR/anisotropy/dormant (tested)
  src/reward.py                  # exact-answer reward + parsing tests
  data/probe_set_ids.json        # frozen probe prompts
  compute_log.md
```

## 10. Open questions to surface to the team before/while coding (don't silently decide)

1. Base vs Instruct Qwen2.5-0.5B for stage A?
2. Adaptation phase: GRPO or SFT? (Recommend GRPO for consistency.)
3. Is SVAMP too close to GSM8K to be a meaningful "future task"? (Suggest logging it anyway + proposing ProntoQA as a second probe if budget allows.)
4. KL β confirmed 0 for the pilot, or do we also want one β>0 checkpoint pair as Madhur's "KL-regularized checkpoint" baseline? (His comment explicitly names it — a single extra β>0 run at 100 updates would directly answer him and costs little.)

---

*Source of truth ranking if documents conflict: proposal PDF §5–§9 > Tommy's Slack spec > this briefing. When in doubt, implement the cheaper version, log everything, and flag the question in Slack rather than blocking.*
