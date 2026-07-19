# Exp 1.5.1 + Exp 1.6 run guide — for the agent on the Windows RTX 4070 box

This is the operating manual for the TWO pre-registered follow-up experiments.
It reuses the conventions of `WIN4070_EXP15_GUIDE.md` (environment §1,
recording §9, commit protocol §10, troubleshooting §12) — read that first if
this is a fresh session; only the deltas are spelled out here.

Authoritative plans (frozen; do not deviate without a written amendment):

- `EXPERIMENT_1_5_1_PLAN_ZH.md` — stall forensics, 3 replicates, ~6–8 GPU h.
  **May start immediately** (a Slack heads-up is sent by Aaron; you do not
  need to wait for a reply).
- `EXPERIMENT_1_6_PLAN_ZH.md` — 3e-6 dose probe, gated, ~20 GPU h before the
  expansion gate. **HARD PRECONDITION: team ack (plan §6). Do not launch a
  formal (non-smoke) exp1.6 phase unless the operator confirms the ack and
  the launch commit message carries a `team-ack:` line.** Smoke is always
  allowed.

Run order when both are cleared: **1.5.1 first** (cheap, unblocked,
self-contained), then 1.6.

---

## 0. Sync, environment, smoke

```powershell
git pull
# same env as exp1.5 (setup_win4070.ps1 / conda env eaaj-win4070)
```

Runner code changed since v3 (terminal-stop tolerance, HardCapStop, phase-3/4
guards). Validate plumbing once on this box before anything formal:

```powershell
powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp15.ps1" -Phase 1 -Smoke -ConfigPath "experiment 1.5\exp1_5_1_config_seed42.json"
powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp15.ps1" -Phase 2 -Smoke -ConfigPath "experiment 1.5\exp1_5_1_config_seed42.json"
powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp15.ps1" -Phase 1 -Smoke -ConfigPath "experiment 1.5\exp1_6_config.json"
```

All three smokes were already green on the mac (cpu); on this box they mostly
re-verify the CUDA path. A smoke failure = STOP, report, do not "fix" science
files.

The wrapper is unchanged (`run_exp15.ps1` + `-ConfigPath`): keep-awake,
GPU CSV telemetry, transcripts to `experiment 1.5\logs\` all apply to every
command below. After every sitting: copy telemetry into the run dir's
`telemetry\`, append `eaaj-pilot/compute_log.md`, commit (old guide §9–10).

---

## 1. Exp 1.5.1 — stall forensics

### 1.0 Phase 0: measure v2's existing checkpoints (~10 min, do this first)

```powershell
powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp15.ps1" -Phase 2 -ConfigPath "experiment 1.5\exp1_5_config_v2.json"
python "experiment 1.5\exp15_gates.py" ckpt0 --run-dir "eaaj-pilot\outputs\exp15_cuda_grpo_gsm8k_dd5f54a0e2b7"
```

Expected: measurements for ckpt-0/25/50 written into the archived v2 run dir
(additive; nothing overwritten — the skip of missing ckpts is the new
terminal-stop tolerance working as designed, `phase2_complete.json` will list
`skipped_missing`). Gate must print PASS (fp32 vs pilot reference). STOP on
anything else. Commit as `exp1.5.1: phase 0 v2 float32 measurements`.

### 1.1 Replicates — one at a time, A → B → C

| Replicate | Config | Expected run dir |
|---|---|---|
| A (seed 42) | `exp1_5_1_config_seed42.json` | `exp15_cuda_grpo_gsm8k_fc2941d83cbe` |
| B (seed 43) | `exp1_5_1_config_seed43.json` | `exp15_cuda_grpo_gsm8k_5ffdb56fc613` |
| C (seed 44) | `exp1_5_1_config_seed44.json` | `exp15_cuda_grpo_gsm8k_a9dc95cbc2e8` |

Per replicate:

```powershell
# gate 0 — run-dir identity (config drift guard)
python "experiment 1.5\exp15_gates.py" rundir --config "experiment 1.5\exp1_5_1_config_seed42.json"

# disk: >= 45 GiB free required (17 fp32 ckpts + rolling trainer), else clean up first

# phase 1 — expect EITHER a safety stop around update ~55 (the observation!)
# OR the step-80 hard cap. Both end with "Phase 1 ... pre-registered terminal state"
# or a normal completion — the runner exits 0 either way. A nonzero exit = real failure.
powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp15.ps1" -Phase 1 -ConfigPath "experiment 1.5\exp1_5_1_config_seed42.json"

# sentinel check at the step-25 window (~40 min in), lr-scaled thresholds:
python "experiment 1.5\exp15_gates.py" sentinel --config "experiment 1.5\exp1_5_1_config_seed42.json"

# phase 2 + identity gate
powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp15.ps1" -Phase 2 -ConfigPath "experiment 1.5\exp1_5_1_config_seed42.json"
python "experiment 1.5\exp15_gates.py" ckpt0 --config "experiment 1.5\exp1_5_1_config_seed42.json"
```

What is NORMAL for this experiment (do not treat as failure, do not retry):

- `safety_stop.json` with "zero group reward variance" around step 40–70;
- clipping ratios near 1.0 in the dashboard from early steps;
- entropy collapsing below 0.1 shortly before the stop;
- `phase1_complete.json` containing `terminated_by` + `missing_checkpoints`;
- `phase2_complete.json` listing `skipped_missing`.

What is NOT normal (STOP + report, preserve everything): non-finite loss
stops, timing/memory stops, sentinel STOP verdict, ckpt0 gate ≠ PASS,
a safety stop before step 10 (grid too coarse — report before replicate B).

Commit per replicate: `exp1.5.1: replicate A (seed 42) phase 1+2 — stopped at update NN`.

**Disk protocol between replicates**: after a replicate's phase 2 + ckpt0
gate PASS + commit, delete `model.safetensors` inside its `ckpt-*` dirs
(KEEP every json/tokenizer file, dashboards, measurements). Keep replicate
A's weights entirely if ≥ 45 GiB still free. Never touch v1/v2/v3 dirs.

### 1.2 Analysis (after all three replicates)

```powershell
python "experiment 1.5\analysis_exp1_5_1.py" "eaaj-pilot\outputs\exp15_cuda_grpo_gsm8k_fc2941d83cbe" "eaaj-pilot\outputs\exp15_cuda_grpo_gsm8k_5ffdb56fc613" "eaaj-pilot\outputs\exp15_cuda_grpo_gsm8k_a9dc95cbc2e8"
```

Writes `forensics.json` per run dir + prints the cross-replicate SC1/SC2/SC3
verdicts. Commit as `exp1.5.1: forensics analysis`. Do NOT run runner
`--phase 3` or `--phase 4` for 1.5.1 configs (the runner refuses; that is
intentional, not a bug).

---

## 2. Exp 1.6 — 3e-6 dose probe (TEAM ACK REQUIRED)

Preconditions checklist (all hard):

1. operator confirms Tommy's ack of the 3e-6 probe path;
2. `git log` will carry `team-ack: <who/when/link>` in the launch commit;
3. exp1.5.1 committed (its result goes into the same Slack thread);
4. ≥ 30 GiB free disk.

Expected run dir (both configs resolve here — that is by design):
`exp15_cuda_grpo_gsm8k_caebbcc73461`.

```powershell
python "experiment 1.5\exp15_gates.py" rundir --config "experiment 1.5\exp1_6_config.json"

# phase 1 (~13 h, resumable; sentinel gates at 25/50 as with v3):
powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp15.ps1" -Phase 1 -ConfigPath "experiment 1.5\exp1_6_config.json"
python "experiment 1.5\exp15_gates.py" sentinel --config "experiment 1.5\exp1_6_config.json"

# phase 2 + identity gate:
powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp15.ps1" -Phase 2 -ConfigPath "experiment 1.5\exp1_6_config.json"
python "experiment 1.5\exp15_gates.py" ckpt0 --config "experiment 1.5\exp1_6_config.json"

# endpoint probe — 6 cells, one process per cell (v3 OOM lesson):
# ckpt 0 then 500; per ckpt seeds 42 -> 43 -> 44
powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp15.ps1" -Phase 3 -AdaptCheckpoint 0 -AdaptSeed 42 -ConfigPath "experiment 1.5\exp1_6_config.json"
python "experiment 1.5\exp15_gates.py" bridge --config "experiment 1.5\exp1_6_config.json"   # once, after the first ckpt-0 cell
# ... repeat -AdaptCheckpoint/-AdaptSeed for the remaining 5 cells ...

# expansion gate:
python "experiment 1.5\exp1_6_gate_eval.py"
```

- `VERDICT: EXPAND` → run the remaining 12 cells with
  `-ConfigPath "experiment 1.5\exp1_6_config_fullgrid.json"` (completed cells
  auto-skip; same run dir), then `-Phase 4` with the fullgrid config.
- `VERDICT: STOP` → do NOT expand, do NOT run phase 4 (the runner refuses on
  the endpoint config anyway). Commit everything incl. `exp16_gate_eval.json`;
  the "still subtherapeutic" dose-response point is the result.
- If phase 1 hits a safety stop: that is the strict contract for 1.6 (unlike
  1.5.1) — the runner raises, artifacts are preserved. Do not amend, do not
  retry; commit + report. Phase 2 may still be run afterwards to measure the
  existing checkpoints (terminal-stop tolerance), but only after reporting.

If a phase-3 cell OOMs: preserve the cell dir as `ckpt-N_oom_<date>`, rerun
that single cell in a fresh process (`expandable_segments:True` is already
set by the wrapper) — the v3 recovery precedent, old guide §12.

---

## 3. Do NOT (both experiments)

- change any value inside a config, plan, or threshold (a mismatch is a STOP,
  never an edit);
- silently rerun a safety-stopped training phase;
- delete or overwrite anything in v1/v2/v3 run dirs (Phase 0 ADDS
  measurements to v2 — that is the only sanctioned write, and only via
  `--phase 2`);
- run 1.5.1 phases 3/4, or 1.6 phase 4 before EXPAND + fullgrid completion;
- launch formal 1.6 without the team-ack line;
- run anything formal on cpu/mps, or two trainings concurrently on this GPU;
- skip the compute-log/telemetry/commit routine (old guide §9–10).

## 4. Success criteria

- 1.5.1: three replicate dirs each with phase1/phase2 markers + measurements,
  v2 Phase-0 measurements, `forensics.json` × 3, cross-replicate verdict
  printed and committed, compute log rows, no unexplained gate verdicts.
- 1.6: either (EXPAND path) 18 validated cells + phase 4 outputs, or (STOP
  path) endpoint probe + `exp16_gate_eval.json` verdict committed; in both
  cases sentinel/ckpt0/bridge gates PASS and the dose-response point is
  reported back to the team.
