# eaaj-pilot — RLVR plasticity pilot (first experiment)

Scaled-down pilot of "Seeing the Stall Coming" Tasks 1+2+4-outcome, targeting
**RQ1**: does a plasticity quantity Q (effective rank, dormant-neuron fraction)
measured at GRPO checkpoint t correlate with fixed-budget SVAMP adaptability
from checkpoint t?

Spec sources: `../agent_doc/AGENT_BRIEFING_first_experiment.md` (condensed) ·
proposal §5–§9 (authoritative) · Tommy's Slack spec.

## Run order (Colab)

Upload this whole folder to Drive as `MyDrive/eaaj-pilot`, then run in order:

| Notebook | Phase | GPU | What it does |
|---|---|---|---|
| `00_setup_and_data.ipynb` | 0 | CPU/T4 | loaders, field checks, split freeze, unit tests, 8-prompt metric dry run |
| `01_grpo_gsm8k.ipynb` | 1 | A100 | GRPO on 512 GSM8K questions, 200 updates, ckpts at 0/25/50/100/200, full dashboard logging |
| `02_measure_Q.ipynb` | 2 | T4/L4 | Q metrics in fixed fp16 per checkpoint on the frozen 512-prompt probe (+512-vs-2048 sensitivity at ckpt 0 & 200) |
| `03_svamp_adaptation.ipynb` | 3 | L4/A100 | identical fixed-budget adaptation (256 SVAMP train, 50 updates, 100-question eval) from each checkpoint |
| `04_analysis.ipynb` | 4 | CPU | master table + 3 slide-ready plots + Spearman table |

**Every GPU session:** record units before/after + screenshot the Colab GPU
panel into `compute_log.md` (hard requirement from Tommy — the compute
extrapolation for the full sweep is itself a deliverable).

## Local development (no GPU needed)

```bash
python3 -m venv --system-site-packages .venv   # reuses system torch
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -v           # 45 tests: contract + metrics + reward + MPS-patch equivalence
.venv/bin/python scripts/smoke_test_grpo.py     # 1 tiny CPU update; TRL API contract
.venv/bin/python scripts/dry_run_metrics.py 8  # Phase-0 dry run on base model
```

Local full-pilot runner (resumable): `scripts/run_local_pipeline.py --phase {1..4,all}
[--backend cpu|mps]`. Default cpu resumes `outputs/local_grpo_gsm8k_eac028bfcc87`.
`--backend mps` exists and is validated but measured at CPU parity — not
recommended for scientific runs (LOCAL_EXPERIMENT_PLAN.md, 2026-07-08 section).

## Layout

```
src/metrics.py      erank / participation ratio / top-k shares / anisotropy pair /
                    dormant fraction / weight norms + activation collectors (hooked)
src/reward.py       exact-answer reward (TRL GRPOTrainer-compatible)
src/data.py         loaders, prompt template, frozen splits (seed 42)
src/callbacks.py    dashboard JSONL logger, exact-step checkpointing, periodic eval
src/adaptation.py   the one fixed-budget adaptation function (run 5x)
src/evaluate.py     greedy exact-answer accuracy
data/               frozen probe set + splits (committed — do not regenerate)
tests/              unit tests (run before spending any GPU units)
pilot_config.json   single pre-registered recipe shared by all notebooks
outputs/
  ACTIVE_RUN.txt    explicit run selected by notebooks 02–04
  grpo_gsm8k_*/     manifest + checkpoints + dashboard/eval logs
    measurements/   per-checkpoint Q JSON
    adaptation/     fixed-budget outcomes, one folder per source checkpoint
    analysis/       table, Spearman results, and slide-ready PNGs
```

## Decisions already logged (mirror into Research Doc)

- Base Qwen2.5-0.5B (not Instruct) — briefing default, **open question #1**
- Adaptation algorithm = GRPO (apples-to-apples with "stage B = RL") — **open question #2**
- β(KL) = 0 (danger-zone flavor) — **open question #4** (a single β>0 run at
  100 updates would answer Madhur's KL-regularized-baseline comment cheaply)
- Probe set = 512 GSM8K test-split prompts; the 2048 sensitivity superset tops
  up from held-out train because the test split is too small (see `src/data.py`)
- Prompt template = plain QA with explicit `####` answer convention (`src/data.py`)
- SVAMP-too-close-to-GSM8K concern (**open question #3**): flag to team; ProntoQA
  is the proposed second probe if budget allows

The exact-reward gate in notebook 01 intentionally aborts before update 1 if
eight sampled groups have no within-group reward variance. That means GRPO has
zero advantage signal under the chosen base-model/exact-reward recipe; take the
saved `sparse_reward_preflight.json` to the team rather than silently switching
models or rewards. A ready-to-send note is in `TEAM_DECISIONS_NEEDED.md`.

## Interpreting the pilot (briefing §7)

- **Good:** Q declines systematically across training AND correlates (erank:
  positive ρ with Δacc; dormant: negative) with fixed-budget SVAMP outcome.
- **Null:** Q barely moves at 0.5B/200 updates → pivots: longer training (500+),
  harder danger zone, Qwen2.5-1.5B, or ProntoQA as stage B.
- Either way: wall-clock + GPU type per phase (in `wall_clock.json` /
  `compute_log.md`) → extrapolate the full sweep for Tommy.
