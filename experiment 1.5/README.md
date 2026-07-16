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

**Follow `WIN4070_EXP15_GUIDE.md`** — the step-by-step guide for the agent
on the Windows box (gates, kill-gate thresholds, session plan, recording
and commit protocol). Short version:

```powershell
python "experiment 1.5\exp15_gates.py" rundir                                    # Gate 0: pre-registered hash
powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp15.ps1" -Phase all -Smoke   # first contact
powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp15.ps1" -Phase 1  # ~13 h; sentinel gate at step 25
python "experiment 1.5\exp15_gates.py" sentinel                                  # Gate 1
powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp15.ps1" -Phase 2  # <15 min
python "experiment 1.5\exp15_gates.py" ckpt0                                     # Gate 3: measurement identity
powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp15.ps1" -Phase 3  # ~21 h, resumable per (ckpt, seed)
python "experiment 1.5\exp15_gates.py" bridge                                    # Gate 4: legacy-100 bridge
powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp15.ps1" -Phase 4  # analysis + MC1/MC2 verdicts
```

The wrapper adds keep-awake, nvidia-smi telemetry CSVs, full transcripts,
and native-stderr tolerance (judges by exit code). Run dir:
`eaaj-pilot\outputs\exp15_cuda_grpo_gsm8k_e73704296e47\`.

Disk: needs ≥30 GiB free (8 fp32 checkpoints + rolling trainer state).
Adaptation trainer dirs are auto-deleted after validation
(`-KeepTrainerDirs` to keep them).
