# Pilot experiment status

Last updated: 2026-07-08

## Phase 1 (local CPU stratum) — interrupted, resumable

- Run `outputs/local_grpo_gsm8k_eac028bfcc87`: reached step ~67/200 before the
  process was killed (~5.8 h wall on 2026-07-07, no crash/safety marker; likely
  terminal closed). Reward was healthy (~0.48–0.59 around step 63–67).
- Scientific checkpoints ckpt-0/25/50 saved; trainer checkpoint-50 has full
  optimizer/RNG state. Resume with:
  `caffeinate -dimsu .venv/bin/python scripts/run_local_pipeline.py --phase 1`
  (steps 51–67 replay deterministically; ~12.5 h to step 200 at ~300 s/update).

## 2026-07-08 MPS investigation — measured, not adopted

MPS is available (earlier "unavailable" note corrected) but the real TRL
update runs at parity with CPU (~265–320 s/update after fixes) because of
per-token host↔device sync overhead in the generation loop. Full data and
root cause: LOCAL_EXPERIMENT_PLAN.md §"2026-07-08 MPS investigation". The
`selective_log_softmax` fix lives in `src/mps_compat.py` (equivalence-tested)
and the runner now takes `--backend {cpu,mps}`; cpu remains the default and
resumes the existing run. Fast path remains the pre-registered Colab recipe.

## Completed locally (Phase 0)

- Frozen GSM8K/SVAMP splits and 512/2048 prompt sets validated.
- 37 offline contract, metric, and exact-reward tests pass.
- One-step tiny GRPO contract smoke passes with TRL 1.6.0, including dashboard
  logging, step-0/step-1 eval, and weights-only checkpoint save.
- Qwen2.5-0.5B 8-prompt activation dry run passes for layers 4/12/22.
- The selective three-layer hooks are elementwise identical to Transformers'
  full `output_hidden_states` path on a two-prompt check (max error 0.0).

These are implementation checks, not scientific results.

## Pending on authenticated Colab GPU

1. `00_setup_and_data.ipynb` — repeat Phase-0 checks in the Colab environment.
2. `01_grpo_gsm8k.ipynb` — run sparse-reward preflight, then 200 updates only
   if at least one prompt group has within-group exact-reward variance.
3. `02_measure_Q.ipynb` — measure all five checkpoints in fixed fp16.
4. `03_svamp_adaptation.ipynb` — five identical 50-update adaptation runs.
5. `04_analysis.ipynb` — generate the run-scoped CSVs and three figures.

For every GPU session, fill `compute_log.md` and save a matching resource-panel
screenshot. No formal Stage-A or SVAMP result has been claimed yet.

## Team gate

Send `TEAM_DECISIONS_NEEDED.md` to Slack. In particular, a failed sparse-reward
preflight requires a team decision on base vs Instruct / reward shaping before
spending the 200-update budget.
