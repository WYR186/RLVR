# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this workspace is

This is **not a software project** — it is Aaron Wang's workspace for the Algoverse AI research program. The team project studies **plasticity collapse in multi-stage RLVR training** of reasoning LLMs ("Seeing the Stall Coming"): whether activation-based plasticity metrics Q (effective rank, dormant-neuron fraction) measured during RL stage A give early warning that a later RL stage B will stall, with more lead time than dashboard signals (reward slope, KL, grad norm, entropy).

Aaron is **Person 4: early-warning diagnostics** — he owns metric instrumentation, dashboard-signal logging, and per-checkpoint measurement artifacts.

## Source-of-truth ranking (when documents conflict)

1. `team_doc/proposal_v3.1_formal.docx` — the formal research proposal, §5–§9 are the authoritative spec (read §7 Metrics before writing any measurement code)
2. Tommy's (team lead) Slack spec
3. `agent_doc/AGENT_BRIEFING_first_experiment.md` — **read this first for any experiment work**; it condenses the proposal, the first-experiment spec, metric formulas, pipeline phases, and open questions

`team_doc/eaaj-research (1).docx` is the team Research Doc — anything produced here must eventually land there. `resources/` and `Slides/` are program materials, not project inputs.

## The first experiment (pilot)

GRPO on **Qwen2.5-0.5B** with 512 GSM8K questions, exact-answer reward; checkpoints at **0/25/50/100/200 updates**; measure effective rank + dormant fraction at each checkpoint; then an **identical fixed-budget SVAMP adaptation** (256 train / 100 eval questions, 50 updates) from every checkpoint. Target: does Q(checkpoint) correlate with fixed-budget SVAMP adaptability (RQ1)?

Code lives in `eaaj-pilot/` (Colab-oriented notebooks + `src/` modules). Training runs on **Google Colab** (~300 compute-unit budget for the pair; L4/T4 preferred, A100 only for generation-heavy GRPO). Local machine (Apple Silicon, no CUDA) is only for Phase-0 development: unit tests, data loaders, metric functions on tiny inputs.

### Commands

```bash
cd eaaj-pilot
python -m pytest tests/ -v          # run all unit tests (metrics, reward parsing)
python -m pytest tests/test_reward.py -v   # single test file
```

## Hard constraints (violating these invalidates results)

- **Framing (mentor feedback, Madhur):** never claim "RLVR reduces the model's ability to learn." The measurable question is whether RLVR reduces **fixed-budget future adaptability** on held-out task families vs. checkpoint-0 / KL-regularized baselines. Every outcome logged must be a fixed-budget quantity with task, budget, and baseline defined up front.
- **Leakage rule (proposal §6):** any feature computed at step t may use only information available at step t. No post-hoc normalization across a whole run.
- **Comparability:** all Q metrics are measured on a frozen probe set (`eaaj-pilot/data/probe_set_ids.json`), in eval mode, fixed dtype and fixed layers, so values are comparable across checkpoints.
- **Compute accounting:** every Colab run needs GPU-dashboard screenshots and an entry in `eaaj-pilot/compute_log.md` (date, GPU, duration, units before/after, phase). Tommy requires this for full-experiment budgeting.
- **Deviations:** any change from Tommy's spec (model variant, LRs, adaptation recipe) gets a one-line justification in the notebook header, mirrored to the Research Doc.
- **Reproducibility:** fixed seeds everywhere, pinned pip installs at notebook top, artifacts saved to Drive in checkpoint-named folders.
- Don't silently decide open team questions (base vs Instruct model, GRPO vs SFT for adaptation, SVAMP-too-close-to-GSM8K, KL β>0 baseline run) — implement the cheaper default, log it, and flag the question for Slack.
