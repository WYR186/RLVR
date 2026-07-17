# Experiment 1.5 run guide — for the agent on the Windows RTX 4070 box

Audience: the coding agent (or human) driving the RTX 4070 Laptop machine.
Mission: execute experiment 1.5 (Stage-A dose escalation + noise fix) end to
end on the 8 GiB 4070, prove within the first 25 updates that the hotter
learning rate is behaving, and record everything so the run is auditable.

给 Aaron 的一句话：这是 pilot 的"加大剂量 + 降噪"版——lr 提到 1e-5、训 500 步、
8 个 checkpoint、SVAMP 评估扩到 300 题、每个 checkpoint 跑 3 个适应种子。
wrapper 和 kill-gate 沿用 v2 的全部教训,跑之前先读第 0 节的差异表。

You already ran the pilot v2 and the Stage-B seed repeats on this machine.
This experiment reuses that exact execution profile and all of its guards.
What is new is the science dose, not the plumbing.

---

## 0. What this is (read before running)

Design + rationale: `EXPERIMENT_1_5_PLAN_ZH.md` (same folder). Pre-registered
recipe: `exp1_5_config.json` — **read-only once runs start**.

| | pilot v2 (done) | experiment 1.5 (this) |
|---|---|---|
| Stage-A lr | 1e-6 | **1e-5** (10×) |
| Stage-A updates | 200 | **500** |
| Checkpoints | 0/25/50/100/200 | **0/25/50/100/200/300/400/500** |
| SVAMP eval | 100 questions | **300 questions** (pilot's 100 ⊂ 300, logged as legacy bridge) |
| Adaptation | 1 seed × 5 ckpts | **3 seeds (42/43/44) × 6 ckpts (0/50/100/200/300/500)** |
| Adaptation recipe | lr 1e-6, 50 updates, 256 train q | **unchanged** (only the eval set widened) |
| Execution profile | fp32 master + bf16 autocast + paged_adamw_8bit + grad ckpt + 4×16 | **identical, imported from the pilot's code** |
| Run dir | `local_cuda_grpo_gsm8k_e9b0b52aab6c` | **`exp15_cuda_grpo_gsm8k_e73704296e47`** |

Two structural differences from the pilot tooling:

1. This runner **never reads or writes `outputs/ACTIVE_RUN.txt`**. The run
   dir is derived from the config hash. Never point the pilot's
   `run_pipeline.ps1` / `run_local_pipeline.py` at an exp1.5 dir or vice versa.
2. Adaptation lives under `adaptation_seed42/`, `adaptation_seed43/`,
   `adaptation_seed44/` inside the run dir (pilot v2 used `adaptation/`).

Total budget: ≈13 h (phase 1) + ≈15 min (phase 2) + ≈21 h (phase 3, 18 runs
≈70 min each) + minutes (phase 4). Plan 4–6 evening sittings; everything is
resumable.

## 1. Sync and environment

```powershell
cd D:\algoverse                # never OneDrive
git pull --rebase origin main
```

Same Python environment as the pilot v2 runs (`EAAJ_PYTHON` > active conda >
`.conda\envs\eaaj-win4070` > win4070 venv — the wrapper resolves it the same
way `run_pipeline.ps1` did). **No new dependencies.** Quick sanity:

```powershell
python -c "import torch, trl, transformers, bitsandbytes; print(torch.__version__, trl.__version__, transformers.__version__, bitsandbytes.__version__)"
```

Versions must match the pilot v2 run's `manifest.json` (package drift would
make strata incomparable). If anything upgraded itself, STOP and flag.

If this is a fresh clone, model + dataset caches may be missing; run
`python eaaj-pilot-win4070\scripts\prefetch_assets.py` first (the runner
loads the model with `local_files_only=True` by design).

## 2. Gate 0 — pre-registered run dir (guards against code/config drift)

```powershell
python "experiment 1.5\exp15_gates.py" rundir
```

Expected: `VERDICT: PASS` with `exp15_cuda_grpo_gsm8k_e73704296e47`.
Any other hash = the recipe or execution profile drifted since
pre-registration. Do not train; report to Aaron.

## 3. First contact — smoke on the 4070 (~3–6 min)

```powershell
powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp15.ps1" -Phase all -Smoke
```

Runs the whole four-phase pipeline with tiny sizes into
`experiment 1.5\smoke_outputs\` (gitignored; dummy reward — never a result).
Exit code 0 and an `analysis/analysis_summary.json` inside the smoke run dir
= plumbing is good on this box. Already validated on the mac (CPU); this
repeats it against your CUDA stack.

## 4. VRAM/throughput probe (reuse the pilot's — geometry is identical)

```powershell
cd eaaj-pilot-win4070
python scripts\win_preflight.py
python scripts\win_preflight.py --grpo-probe-small
python scripts\win_preflight.py --grpo-probe
cd ..
```

Same gates as v2: peak reserved **< 7.3 GiB**, **< 120 s/update**. The lr
change does not alter memory or step time, so v2's measured numbers should
reproduce. (v2 event on record: the *full-geometry probe* once reported
10.8 GiB reserved — an allocator/accounting deviation, logged, training
itself stayed within budget. If you see that again, log it the same way;
training OOM is what matters, see §12.)

Disk: the runner refuses to start under 30 GiB free. 8 fp32 checkpoints
≈15 GiB + rolling trainer state ≈4 GiB; adaptation trainer dirs are
auto-deleted after each run validates.

## 5. Phase 1 — Stage A, 500 updates (~13 h)

```powershell
powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp15.ps1" -Phase 1
```

What happens: preflight (sparse-reward check on the base model), ckpt-0
saved, then 500 GRPO updates with checkpoints at 25/50/100/200/300/400/500,
GSM8K-64 eval every 25 steps, sentinel every 25 steps, safety callback armed.

### Gate 1 — the step-25 kill-gate. Do not walk away before it.

First sentinel row lands ~40 min in. Check it:

```powershell
python "experiment 1.5\exp15_gates.py" sentinel
```

- **PASS** (`rel_change_window` in the 1e-6..1e-4 band): keep going. At lr
  1e-5 expect roughly 10× the v2 window movement.
- **INVESTIGATE** (1e-7..1e-6): an order of magnitude under expectation —
  finish the current window, then pause and ask before continuing.
- **STOP** (< 1e-7 or `updates_effective: false`): v1 no-op territory.
  Ctrl+C now, keep every artifact, report.

Re-run the same gate at any later point; it prints all windows so far.

### Gate 2 — the new-regime watch (lr 1e-5 is deliberately hot)

Around steps 50–100, skim the tail of `dashboard.jsonl` in the run dir:

- Reward mean should move **up faster** than v2's did (v2: 0.354→0.443 over
  the first 100 steps). Reward reaching a plateau *and then staying there
  for hundreds of steps* is the desired danger zone, **not** a failure.
- If `safety_stop.json` appears (the callback stops on 5 consecutive
  zero-variance-reward updates, non-finite loss, or repeated >8-min steps):
  **that is data, not a bug.** Do NOT delete, retry, lower the lr, or edit
  the config. Commit everything including `safety_stop.json`, append a
  compute_log row, and report which step it stopped at. The pre-registered
  plan (§4 of the plan doc) covers truncated runs.

Interruptions (power, sleep, crash): re-run the same command — the trainer
resumes from its rolling checkpoint. With `paged_adamw_8bit`,
optimizer moments are NOT restored across a resume (known v2 limitation);
record every interruption boundary step in compute_log notes, same as v2 did.

## 6. Phase 2 — Q measurement, 8 checkpoints (<15 min)

```powershell
powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp15.ps1" -Phase 2
```

### Gate 3 — ckpt-0 measurement identity (comparability anchor)

```powershell
python "experiment 1.5\exp15_gates.py" ckpt0
```

ckpt-0 is the same base model, same frozen probe set, same dtype, same
machine as pilot v2 — its effective ranks must reproduce the pilot's
committed values (L4 225.14, L12 231.76, L22 354.19) within 0.01.
**PASS** → measurements are comparable across experiments; continue.
**STOP** → something in the measurement path drifted; phase 3 would be
wasted compute. Report before proceeding.

### 2026-07-17 v3 float32 measurement recovery

The first v3 Phase-2 run used float16 while the pilot reference used float32.
That STOP is preserved as evidence. Do not delete the float16 metrics and do
not retrain Stage A. Use the gated recovery entry point:

```powershell
powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp15_v3_float32_recovery.ps1"
```

It archives the float16 results, reruns Phase 2 in float32, requires the
ckpt-0 gate to PASS, runs the Phase-3 bridge cell and gate, then resumes the
remaining Phase-3 grid. It never runs Phase 4. Add `-Phase2Only` to stop after
the corrected measurement gate.

## 7. Phase 3 — 18 fixed-budget adaptations (~21 h total, split as you like)

```powershell
powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp15.ps1" -Phase 3
```

The runner executes the **pre-registered order** automatically:
checkpoints 0 → 500 → 200 → 100 → 50 → 300, seeds 42 → 43 → 44 within each.
Endpoints first, so the earliest possible read on "did adaptability degrade
at all" (MC2) arrives ≈7 h in, after ckpt-0 and ckpt-500 finish.

Each (checkpoint, seed) is atomic (~70 min: baseline eval on 300 q → 50 GRPO
updates with sentinel + legacy-100 curve every 10 → final eval on 300 q).
Ctrl+C between runs is always safe. Per-sitting resume is just re-running
the same command; to target one cell:

```powershell
powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp15.ps1" -Phase 3 -AdaptCheckpoint 500 -AdaptSeed 43
```

Suggested sittings (≈3.5 h each = one checkpoint × 3 seeds):
evening 1 ckpt-0, evening 2 ckpt-500, evening 3 ckpt-200, evening 4
ckpt-100, evening 5 ckpt-50, evening 6 ckpt-300. Post the ckpt-0 + ckpt-500
mean deltas to Slack after evening 2 — that is the early MC2 signal.

### Gate 4 — legacy bridge, once, after the first ckpt-0 run

```powershell
python "experiment 1.5\exp15_gates.py" bridge
```

The ckpt-0 legacy-100 sub-score should land within ±0.02 of the pilot's
0.53 baseline. INVESTIGATE = log it and flag; not a hard stop.

Built-in cross-seed guard: seeds 43/44 must reproduce seed 42's greedy
baseline **exactly** (the runner enforces it; a mismatch writes
`baseline_mismatch.json` and aborts). If that fires, do not rerun; report.

Failed attempt protocol (same as v2's ckpt-50 OOM event): if a run dies
mid-training (OOM, crash), the runner refuses to reuse the dirty directory.
Rename it in place —

```powershell
Rename-Item <run_dir>\adaptation_seed43\ckpt-500 ckpt-500_failed_run_20260718_2130
```

— keep it forever as evidence, then re-run the same cell fresh. Never edit
or delete a failed attempt.

## 8. Phase 4 — analysis

```powershell
powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp15.ps1" -Phase 4
```

Writes `analysis/` into the run dir: results tables (long + wide),
spearman table, three figures, and `analysis_summary.json` containing the
pre-registered readout — MC1 (did the dose move erank_L12 ≥10%?), MC2 (did
any later checkpoint's mean-of-3-seed delta fall ≥0.05 below ckpt-0's?),
the primary rho over mean deltas (n=6), per-seed rhos, seed-pair rank
correlations, and the variance decomposition.

**Report the `interpretation` field verbatim to the team.** Whether the
primary rho may be read as an RQ1 result is decided by MC2
(`rq1_primary_interpretable`), not by how the scatter looks. No
editorializing on the Windows side.

## 9. Recording requirements (every sitting, no exceptions)

1. **compute_log.md** — append to `eaaj-pilot/compute_log.md`, new section
   `## 2026-07-XX exp1.5 (Windows RTX 4070)`:
   `| Phase 1 (updates 0-212 of 500) | RTX 4070 Laptop (CUDA fp32-master/bf16-autocast) | 5.5 h | sentinel healthy (~2e-6/window); telemetry telemetry/gpu_..._exp15_phase1.csv | `
2. **GPU telemetry** — the wrapper writes
   `experiment 1.5\logs\gpu_<stamp>_exp15_phaseN....csv` (gitignored).
   After each sitting copy it into the run dir so it gets committed:
   ```powershell
   New-Item -ItemType Directory -Force "eaaj-pilot\outputs\exp15_cuda_grpo_gsm8k_e73704296e47\telemetry" | Out-Null
   Copy-Item "experiment 1.5\logs\gpu_*_exp15_*.csv" "eaaj-pilot\outputs\exp15_cuda_grpo_gsm8k_e73704296e47\telemetry\"
   ```
3. **Console transcripts** — `experiment 1.5\logs\run_<stamp>_*.log` stay
   local unless something went wrong; on any failure, copy the relevant
   transcript into the run dir's `telemetry\` too.
4. **Deviations** — anything manual (interruption, rename, driver event,
   VRAM ladder rung) = one line in compute_log notes.

## 10. Commit & push protocol

After each sitting (at minimum after phases 1, 3-per-checkpoint, 4):

```powershell
git pull --rebase origin main
git add "eaaj-pilot\outputs\exp15_cuda_grpo_gsm8k_e73704296e47" eaaj-pilot\compute_log.md
git commit -m "exp1.5: phase 1 artifacts"          # or: "exp1.5: record ckpt 500 seeds 42-44"
git push origin main
```

Weights (`*.safetensors`) are auto-ignored; JSON/JSONL/CSV/PNG artifacts are
what lands in git. Never add `outputs/ACTIVE_RUN.txt`, never force-push.
Artifact paths are disjoint from every pilot run, so rebases should not
conflict; if docs conflict, keep both sides and ask.

## 11. Success criteria (what "done" looks like)

- Phase 1: 8/8 checkpoints on disk; every sentinel window
  `updates_effective: true`; `phase1_complete.json` present; if instead a
  safety stop fired — diagnostics committed and reported (that is a valid,
  pre-registered outcome).
- Phase 2: 8/8 `metrics_ckpt*.json` + sensitivity blocks at 0 and 500;
  Gate 3 PASS.
- Phase 3: 18/18 `summary.json` with `completion_status: complete`,
  validator-clean (50/50 updates, full curves, sentinel effective,
  cross-seed baselines identical).
- Phase 4: `analysis_summary.json` with MC1/MC2 verdicts + the 2×2
  interpretation, posted to Slack as-is.
- compute_log rows + telemetry CSVs committed for every sitting.

## 12. Troubleshooting

| Symptom | Action |
|---|---|
| CUDA OOM mid-run | Same as v2's ckpt-50 event: keep the failed attempt (rename `*_failed_run_<stamp>`), close other GPU apps (browser/iGPU display helps), re-run the same cell fresh. Two OOMs in a row → stop, report; do NOT change batch geometry unilaterally (it is part of the stratum definition). |
| PowerShell shows red `NativeCommandError` lines but training continues | Expected — triton/bitsandbytes warnings on Windows. The wrapper judges by exit code (2026-07-14 lesson). Ignore unless the exit code is non-zero. |
| bitsandbytes hang at a save | Should not occur (`save_only_model=True` everywhere, the v2 mitigation). If a save hangs >10 min anyway: note the step, kill, re-run (resume), report. |
| `runner.lock` refusal | Another runner is (or died while) holding the run dir. If no python process is alive, the lock is stale and is reclaimed automatically on the next start; never delete it while a trainer runs. One GPU = one trainer, ever. |
| `baseline_mismatch.json` | Greedy eval stopped being deterministic across seeds — a real problem. Do not rerun; commit the mismatch file and report. |
| Disk check refusal | Free ≥30 GiB (old smoke outputs, recycle bin, HF cache dupes). Do not delete anything inside pilot run dirs. |
| Sentinel INVESTIGATE verdict | Finish the current 25-step window, post the gate output to Slack, wait for a go/no-go. |
| Windows Update reboot killed phase 1 | Re-run `-Phase 1` (resumes from rolling trainer ckpt); add the boundary step to compute_log notes; consider pausing updates for the week. |

## 13. Do NOT

- Edit `exp1_5_config.json`, the plan, seeds, lr, lengths, checkpoint steps,
  splits, or the reward — pre-registered science.
- Touch any pilot run dir: `local_cuda_grpo_gsm8k_6a075c15808e` (v1 negative
  control), `local_cuda_grpo_gsm8k_e9b0b52aab6c` (v2 + its
  `adaptation_repeats/`, where seed-44 is still pending), or the mac dirs.
- Run the pilot's `run_pipeline.ps1` / `run_local_pipeline.py` against
  exp1.5 dirs, or this wrapper against pilot dirs.
- Decide open team questions (base-vs-Instruct, GRPO-vs-SFT, SVAMP
  distance, β>0 arm — the `optional_klreg_arm` stub stays `enabled: false`
  until the team says otherwise in Slack).
- Continue past a STOP verdict "just to see".
- Run two trainers at once on this GPU (including pilot seed-44 repeats —
  coordinate: finish one job before starting the other).
