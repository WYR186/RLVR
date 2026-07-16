# Experiment 1.5 — Stage-A dose escalation + noise fix

Read `EXPERIMENT_1_5_PLAN_ZH.md` first: what changes vs. the pilot, why each
change is justified by pilot evidence, and the pre-registered analysis
(manipulation checks MC1/MC2 + rho on mean-of-3-seed deltas).

Recipe: `exp1_5_config.json` (pre-registered — do not edit once runs start).
Code reuses `eaaj-pilot/src` unmodified; only the runner, the adaptation
eval material (300-question SVAMP eval + legacy-100 curve) and the analysis
are new, all inside this folder.

## Smoke test (any machine, ~15–30 min on CPU)

Plumbing-only end-to-end validation into `smoke_outputs/` (gitignored;
dummy reward + tiny budgets — never a result):

```bash
cd /path/to/algoverse
eaaj-pilot/.venv/bin/python "experiment 1.5/run_exp1_5.py" --phase all --backend cpu --smoke
```

## Real run (Windows RTX 4070 machine only)

Same conda/venv as the pilot v2 runs. From `D:\algoverse`:

```powershell
# Phase 1 — GRPO 500 updates (~13 h; resumable, can be split across evenings)
python "experiment 1.5\run_exp1_5.py" --phase 1 --backend cuda

# KILL-GATE (same as pilot v2 rerun guide): after step 25, check
# update_sentinel.jsonl in the run dir — rel_change_window must be well
# above 1e-8 or you stop and investigate before burning 13 hours.

# Phase 2 — Q metrics, all 8 checkpoints (<15 min)
python "experiment 1.5\run_exp1_5.py" --phase 2 --backend cuda

# Phase 3 — 18 adaptations (~21 h total; each (ckpt, seed) is atomic and
# resumable; pre-registered order 0 -> 500 -> 200 -> 100 -> 50 -> 300)
python "experiment 1.5\run_exp1_5.py" --phase 3 --backend cuda
#   granular resume, e.g. only ckpt 500 / seed 43:
python "experiment 1.5\run_exp1_5.py" --phase 3 --backend cuda --adapt-checkpoint 500 --adapt-seed 43

# Phase 4 — analysis (manipulation checks + primary rho)
python "experiment 1.5\run_exp1_5.py" --phase 4 --backend cuda
```

Run dir: `eaaj-pilot\outputs\exp15_cuda_grpo_gsm8k_<hash>\`.

After every session: append to `eaaj-pilot/compute_log.md`, commit run
artifacts (`git add eaaj-pilot/outputs/exp15_... && git commit`), attach GPU
telemetry CSVs like the pilot did.

Disk: needs ≥30 GiB free (8 fp32 checkpoints + rolling trainer state).
Adaptation trainer dirs are auto-deleted after validation
(`--keep-trainer-dirs` to keep them).
