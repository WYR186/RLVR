# Compute log (hard requirement — Tommy)

Record EVERY Colab GPU session. Screenshot the GPU/resources panel during runs
and save to `compute_screenshots/` with matching date-phase names.
Check the current units/hour rate in Colab's resource panel and record it.

| Date | Phase / notebook | GPU | Units/hr rate | Units before | Units after | Wall time | Notes |
|------|------------------|-----|---------------|--------------|-------------|-----------|-------|
|      |                  |     |               |              |             |           |       |

Budget: ~300 compute units for the pair ($20–30, reimbursable).
Guideline: L4/T4 for anything that fits; A100 only for generation-heavy GRPO.

## Local (free) runs

| Date | Phase | Hardware | Wall time | Notes |
|------|-------|----------|-----------|-------|
| 2026-07-07 | Phase 0: 37 unit/contract tests + 1-step tiny GRPO smoke + 8-prompt Q dry run | MacBook (CPU) | <15 sec after model cache warm-up | no Colab units spent; all passed |
| 2026-07-07 | Phase 1 (partial): GSM8K GRPO steps 0→~67/200, ckpt-0/25/50 saved | MacBook M3 Max (CPU fp32) | ~5.8 h (12:52–18:41) | interrupted, resumable from trainer checkpoint-50; ~300 s/update |
| 2026-07-08 | MPS feasibility investigation: standalone benchmarks + 4 instrumented real GRPO updates + profiler runs | MacBook M3 Max (MPS fp32) | ~1.5 h total | outcome: parity with CPU, not adopted — see LOCAL_EXPERIMENT_PLAN.md; sls patch kept in src/mps_compat.py |

## 2026-07-09 Windows RTX 4070 local CUDA run

Run dir: `outputs/local_cuda_grpo_gsm8k_6a075c15808e`.

| Phase | Hardware | Wall time | Notes |
|-------|----------|-----------|-------|
| Phase 1: GSM8K GRPO 200 updates | RTX 4070 Laptop (CUDA bf16) | 3.09 h | checkpoints 0/25/50/100/200 saved; telemetry `eaaj-pilot-win4070/logs/gpu_20260709_000620_phase1.csv`; micro-batch 4 x grad-accum 16 |
| Phase 2: Q metrics | RTX 4070 Laptop (CUDA bf16) | 0.9 min | all five checkpoint metrics saved under `measurements/`; telemetry `eaaj-pilot-win4070/logs/gpu_20260709_031435_phase2.csv` |
| Phase 3: fixed-budget SVAMP adaptation | RTX 4070 Laptop (CUDA bf16) | 4.23 h | all five checkpoint adaptation summaries saved under `adaptation/`; telemetry `eaaj-pilot-win4070/logs/gpu_20260709_031539_phase3.csv` |
| Phase 4: analysis | RTX 4070 Laptop (CUDA bf16) | <1 min | tables and figures saved under `analysis/`; telemetry `eaaj-pilot-win4070/logs/gpu_20260709_072930_phase4.csv` |

## 2026-07-10 Windows RTX 4070 local CUDA v2 rerun

Run dir: `outputs/local_cuda_grpo_gsm8k_e9b0b52aab6c`.

| Phase | Hardware | Wall time | Notes |
|-------|----------|-----------|-------|
| Probes: small + full-geometry GRPO | RTX 4070 Laptop (CUDA fp32-master/bf16-autocast, paged_adamw_8bit) | <5 min | small probe passed; full probe completed without OOM at 86.45 s/update, but PyTorch reported 10.809 GiB peak reserved, above the 7.3 GiB go/no-go threshold; accepted as allocator/accounting deviation on the 8 GiB WDDM stack and logged in `telemetry/probe_results.jsonl` |
| Phase 1: GSM8K GRPO 200 updates | RTX 4070 Laptop (CUDA fp32-master/bf16-autocast, paged_adamw_8bit) | 5.18 h active wrapper time across resumes; final resume 2.92 h | run ckpts 0/25/50/100/200 saved; trainer final checkpoint-200 saved; all sentinel windows healthy (`rel_change_window`: 6.58e-06, 5.82e-06, 4.72e-06, 3.89e-06, 2.26e-06, 1.55e-06, 9.36e-07, 3.23e-07); eval step200 accuracy 0.4219; telemetry copied under `outputs/local_cuda_grpo_gsm8k_e9b0b52aab6c/telemetry/` |
| Phase 2: Q metrics | RTX 4070 Laptop (CUDA fp32-master/bf16-autocast) | 3.9 min | metrics saved for ckpts 0/25/50/100/200 under `measurements/`; telemetry `outputs/local_cuda_grpo_gsm8k_e9b0b52aab6c/telemetry/gpu_20260710_123718_phase2.csv` |
| Phase 3: fixed-budget SVAMP adaptation | RTX 4070 Laptop (CUDA fp32-master/bf16-autocast, paged_adamw_8bit) | 5.26 h active wrapper time across three runs | all five summaries saved under `adaptation/`; before -> after accuracy: ckpt-0 0.53 -> 0.59, ckpt-25 0.51 -> 0.62, ckpt-50 0.56 -> 0.59, ckpt-100 0.55 -> 0.60, ckpt-200 0.54 -> 0.66; every sentinel window healthy; telemetry `gpu_20260710_124220_phase3.csv`, `gpu_20260710_215303_phase3.csv`, and `gpu_20260710_225526_phase3.csv` copied under the run telemetry directory |
| Phase 4: analysis | RTX 4070 Laptop (CPU analysis) | 1.3 min | tables and three figures saved under `analysis/`; primary `erank_L12` vs. `svamp_delta` Spearman rho = 0.50 (p = 0.391, n = 5); telemetry `outputs/local_cuda_grpo_gsm8k_e9b0b52aab6c/telemetry/gpu_20260711_005628_phase4.csv` |

Notes:
- bitsandbytes `paged_adamw_8bit` optimizer checkpointing hung or produced corrupt optimizer state on Windows. For this optimizer, local CUDA training now uses `save_only_model=True`; phase-3 adaptation uses the same guard.
- Interrupted checkpoint repairs were required at steps 25, 50, and 75. Repairs restored model/trainer resume metadata and scheduler position, but optimizer moments and RNG state at those resume boundaries are not identical to an uninterrupted run.
- Phase 3 ckpt-50 hit a transient CUDA OOM after its first update in the original multi-checkpoint process. The incomplete attempt was preserved locally, and ckpt-50 was rerun from scratch with the identical preregistered geometry in a fresh process; the retry and remaining checkpoints completed without changing scientific knobs.
- `run_pipeline.ps1` GPU telemetry now samples once per minute and appends each row immediately, avoiding empty CSVs when a long-running `nvidia-smi -l` job is stopped.

## 2026-07-13 Stage-B seed-repeat hardening

| Phase | Hardware | Wall time | Notes |
|-------|----------|-----------|-------|
| 2-update Stage-B repeat smoke | RTX 4070 Laptop (CUDA fp32-master/bf16-autocast, paged_adamw_8bit) | 6.4 min | completion contract passed with actual/requested updates 2/2; baseline 0.53; sentinel windows effective at steps 1/2; telemetry `outputs/smoke_stageb_repeat_20260713_215451/gpu_20260713_215451_stageb_smoke.csv` |
| 2026-07-14 00:45 | Stage-B repeat seed 43 ckpt 0 | RTX 4070 Laptop | 0.2 min | failed; telemetry `gpu_20260714_004537_stageb_seed43_ckpt0.csv` |
| 2026-07-14 02:01 | Stage-B repeat seed 43 ckpt 0 | RTX 4070 Laptop | 72.2 min | complete; telemetry `gpu_20260714_004930_stageb_seed43_ckpt0.csv` |
| 2026-07-14 03:01 | Stage-B repeat seed 43 ckpt 200 | RTX 4070 Laptop | 56.3 min | complete; telemetry `gpu_20260714_020451_stageb_seed43_ckpt200.csv` |
| 2026-07-14 04:17 | Stage-B repeat seed 43 ckpt 25 | RTX 4070 Laptop | 74.4 min | complete; telemetry `gpu_20260714_030316_stageb_seed43_ckpt25.csv` |
| 2026-07-14 05:12 | Stage-B repeat seed 43 ckpt 50 | RTX 4070 Laptop | 54.2 min | complete; telemetry `gpu_20260714_041835_stageb_seed43_ckpt50.csv` |
| 2026-07-14 05:51 | Stage-B repeat seed 43 ckpt 100 | RTX 4070 Laptop | 38.3 min | complete; telemetry `gpu_20260714_051339_stageb_seed43_ckpt100.csv` |
| 2026-07-14 07:05 | Stage-B repeat seed 44 ckpt 0 | RTX 4070 Laptop | 72.2 min | complete; telemetry `gpu_20260714_055316_stageb_seed44_ckpt0.csv` |

## 2026-07-16 Experiment 1.5 Windows RTX 4070

| Phase | Hardware | Wall time | Notes |
|-------|----------|-----------|-------|
| Gate 0 + CUDA environment | RTX 4070 Laptop | <2 min | pre-registered run dir `exp15_cuda_grpo_gsm8k_e73704296e47`; CUDA available; smoke completed before formal run |
| Phase 1 | RTX 4070 Laptop (CUDA fp32-master/bf16-autocast, paged_adamw_8bit) | stopped at update 7 | formal run started at 2026-07-16 01:58 and triggered the pre-registered safety stop: five consecutive updates exceeded 10% completion clipping; telemetry copied to `outputs/exp15_cuda_grpo_gsm8k_e73704296e47/telemetry/`; no retry or parameter change |
| Execution amendment v2 | RTX 4070 Laptop | 4.1 min smoke | amendment `EXPERIMENT_1_5_AMENDMENT_V2.md`; new frozen run dir `exp15_cuda_grpo_gsm8k_dd5f54a0e2b7`; Stage-A completion clipping remains logged but is diagnostic-only because it was not a declared hard stop; all scientific settings and 512-token geometry unchanged; v2 Phase-1 smoke completed 2/2 updates with clipping ratio 1.0 and no safety stop; local telemetry `experiment 1.5/logs/gpu_20260716_094322_exp15_phase1_smoke_exp1_5_config_v2.csv` |
| Phase 1 v2 restart | RTX 4070 Laptop (CUDA fp32-master/bf16-autocast, paged_adamw_8bit) | stopped at update 55 | started 2026-07-16 09:49 in independent run dir `outputs/exp15_cuda_grpo_gsm8k_dd5f54a0e2b7`; step-25 and step-50 sentinel windows passed; stopped at update 55 after five consecutive updates with zero group reward variance (`safety_stop.json`); clipping remained diagnostic-only and was not the stopping cause; telemetry and launcher logs copied under `outputs/exp15_cuda_grpo_gsm8k_dd5f54a0e2b7/telemetry/`; no further retry or scientific-parameter change made |
| Scientific amendment v3 | RTX 4070 Laptop | static gates | amendment `EXPERIMENT_1_5_AMENDMENT_V3.md`; v2 collapse diagnosed from length/clipping/entropy/reward trajectory; Stage-A learning rate changed only from `1e-5` to the previously completed-pilot value `1e-6`; `beta=0.0` and all other settings unchanged; new independent run dir `exp15_cuda_grpo_gsm8k_c7cc7a1d02d9`; Gate 0 and single-variable config-diff checks passed |
| Phase 1 v3 smoke | RTX 4070 Laptop (CUDA fp32-master/bf16-autocast, paged_adamw_8bit) | 4.1 min | plumbing-only smoke completed 2/2 updates in `experiment 1.5/smoke_outputs/exp15_cuda_grpo_gsm8k_07c2d14b15dd`; checkpoint, completion marker, dashboard, safety callback, GPU telemetry, and transcript verified; telemetry copied under the smoke run's `telemetry/` directory |
| Phase 1 v3 formal | RTX 4070 Laptop (CUDA fp32-master/bf16-autocast, paged_adamw_8bit) | in progress | Gate 0 passed; one launcher attempt at 2026-07-16 21:27 did not start training because a path containing `experiment 1.5` was not quoted, and its logs were preserved; corrected launcher started the independent formal run at 21:28 in `outputs/exp15_cuda_grpo_gsm8k_c7cc7a1d02d9`; exact-reward preflight passed with 8/8 groups showing reward variance; update 1 had reward mean/std 0.469/0.503, entropy 0.247, clipping 0, finite loss/gradient, and no safety stop; step-25 sentinel passed (`rel_change_window=6.367e-06`, effective); step-50 sentinel passed (`4.468e-06`, effective), with reward variance, entropy, length, loss, and gradient healthy; step-75 (`4.288e-06`) and step-100 (`3.875e-06`) sentinel windows also passed; `ckpt-100` saved, recent clipping remained 0-4.7%, no zero-signal streak or safety stop; unattended continuation authorized |
| Phase 1 v3 milestone: ckpt-200 | RTX 4070 Laptop (CUDA fp32-master/bf16-autocast, paged_adamw_8bit) | continuing | step-200 sentinel passed (`rel_change_window=3.021e-06`, effective); `ckpt-200` saved; update 200 had reward mean/std 0.656/0.479, entropy 0.124, clipping 0, and finite loss/gradient; no safety stop |
| Phase 1 v3 milestone: ckpt-300 | RTX 4070 Laptop (CUDA fp32-master/bf16-autocast, paged_adamw_8bit) | continuing | step-300 sentinel passed (`rel_change_window=2.086e-06`, effective); `ckpt-300` saved; update 300 retained nonzero reward variance and finite gradients, with no safety stop |
| Phase 1 v3 milestone: ckpt-400 | RTX 4070 Laptop (CUDA fp32-master/bf16-autocast, paged_adamw_8bit) | continuing | step-400 sentinel passed (`rel_change_window=1.075e-06`, effective); `ckpt-400` saved; subsequent updates retained nonzero reward variance and finite gradients, with clipping at 0–1.6% and no safety stop |
| Phase 1 v3 completion | RTX 4070 Laptop (CUDA fp32-master/bf16-autocast, paged_adamw_8bit) | 9.89 h training | completed 500/500 updates at 2026-07-17 07:22; `ckpt-500` and `phase1_complete.json` saved; final step-500 sentinel passed (`rel_change_window=1.268e-07`, effective); reward variance remained nonzero and gradients finite through the final update; no safety stop; GPU telemetry and complete launcher/wrapper transcripts copied under `outputs/exp15_cuda_grpo_gsm8k_c7cc7a1d02d9/telemetry/` |
| Phase 2 v3 ckpt-0 comparability gate | RTX 4070 Laptop (CUDA float16 measurement) | hard stop for Phase 3 | `metrics_ckpt0.json` written; Gate `ckpt0` returned STOP because effective-rank deltas versus the committed pilot were layer4 0.3910, layer12 0.2936, and layer22 0.0305, exceeding the 0.01 tolerance; contract audit found v3 measured in `torch.float16` while the pilot reference used `torch.float32`; grouped weight norms matched exactly, so this is a measurement-contract mismatch rather than Stage-A checkpoint drift; Phase 2 measurements are retained as evidence |
| Phase 2 v3 measurement | RTX 4070 Laptop (CUDA float16 measurement) | 1.4 min | all eight checkpoint metric files (`0,25,50,100,200,300,400,500`) and `phase2_complete.json` saved; GPU telemetry and launcher/wrapper transcripts copied under `outputs/exp15_cuda_grpo_gsm8k_c7cc7a1d02d9/telemetry/`; Phase 3 deliberately not launched because the recorded ckpt-0 comparability gate was STOP |
| v3 float32 measurement recovery design | RTX 4070 Laptop | prepared, not started | recovery amendment `EXPERIMENT_1_5_MEASUREMENT_RECOVERY.md`; config `exp1_5_config_v3_float32_measurement.json` changes only Phase-2 model dtype from float16 to the pilot's float32 while retaining the existing v3 Stage-A run hash/checkpoints; one-command script preserves the float16 results, reruns Phase 2, requires ckpt-0 and bridge gates before Phase 3, records telemetry/events, and never invokes Phase 4 |
| v3 float32 recovery launch correction | RTX 4070 Laptop | active | first launch exited before artifact changes because `nvidia-smi` reported the Codex desktop process with `[N/A]` memory as a false busy GPU process; the script now ignores `[N/A]` rows, and the restarted process (Python PID 19832) archived the original float16 measurements and began float32 Phase 2; no training checkpoint was changed |
| v3 float32 recovery Phase 3 OOM | RTX 4070 Laptop (CUDA) | stopped after 9 completed cells | the recovery loop completed all cells then reached the newly observed pre-registered `adaptation_seed42/ckpt-100` cell; after step 1, CUDA OOM occurred during backward (requested 1.16 GiB; no safety_stop and nonzero reward variance). The partial cell was preserved as `adaptation_seed42/ckpt-100_oom_20260718`; no prior result was overwritten. A fresh-process single-cell rerun is required before Phase 3 can be marked complete. |
| v3 float32 recovery Phase 3 OOM rerun | RTX 4070 Laptop (CUDA) | active | first launcher retry was command-quoting-only and produced no Python run; its stderr is preserved. Corrected fresh-process rerun started at 2026-07-18 05:32 with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, targeting only `adaptation_seed42/ckpt-100`; stdout/stderr are preserved under the run telemetry directory. |
| v3 float32 recovery Phase 3 OOM rerun complete | RTX 4070 Laptop (CUDA) | 2.1 h | fresh-process seed42/ckpt-100 rerun completed 50/50 and wrote `summary.json`; reward variance remained nonzero, final clipping was 0, and no safety_stop was written. The original OOM directory remains preserved. Remaining missing preregistered cells are checkpoint 50 and 300 for seeds 42, 43, and 44; they will be run as isolated fresh processes to avoid the long-loop allocator failure. |
| v3 float32 recovery isolated-cell launcher correction | RTX 4070 Laptop | <1 min | first ckpt50 launch used a nonexistent config filename and exited before training; stderr preserved. No run directory was modified. Subsequent launches use the exact frozen config `exp1_5_config_v3_float32_measurement.json`. |
| v3 float32 recovery isolated cells | RTX 4070 Laptop (CUDA) | continuing | seed42/ckpt50 completed 50/50 in a fresh process with nonzero reward variance, final clipping 1.56%, and no safety_stop; seed43/ckpt50 launched next as an isolated process. |
| v3 float32 recovery validator audit | RTX 4070 Laptop | partial | project validator confirmed 13/18 expected Phase 3 cells complete; five summaries were still missing: seed43/ckpt100, seed44/ckpt100, and ckpt300 for seeds 42, 43, and 44. The preserved OOM directory is excluded from the expected matrix. |
| v3 float32 recovery Phase 3 completion | RTX 4070 Laptop (CUDA) | completed 2026-07-18 16:59 | all 18 pre-registered cells (`ckpt 0,50,100,200,300,500 × seeds 42,43,44`) passed the project validator; no cell has `safety_stop.json`; final seed44/ckpt300 completed 50/50. Five missing cells were completed as isolated fresh-process reruns with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` after the original loop hit CUDA OOM at seed42/ckpt100. OOM and launcher-correction directories/logs remain preserved. `phase3_complete.json` written. Phase 4 not run. |
| v3 float32 Phase 4 analysis | RTX 4070 Laptop (CPU analysis) | 1.3 min, completed 2026-07-20 00:10 | formal Phase 4 ran against the completed v3 float32 run; generated `analysis/analysis_summary.json`, result tables, Spearman tables, and three figures. MC1 failed (`erank_L12` relative change -0.00976 vs 0.10 threshold); MC2 passed (maximum mean-delta drop 0.08556 vs 0.05 threshold); primary Spearman rho=0.6, p=0.208; `rq1_primary_interpretable=true`. GPU telemetry and transcript copied under the run `telemetry/` directory. |

## 2026-07-19--21 Experiment 1.5.1 Windows RTX 4070

Run dir: `outputs/exp15_cuda_grpo_gsm8k_fc2941d83cbe` (replicate A, seed 42).

| Phase | Hardware | Wall time | Notes |
|-------|----------|-----------|-------|
| Phase 0 v2 checkpoint measurement | RTX 4070 Laptop (CUDA float32 measurement) | <2 min | measured the archived v2 checkpoints 0/25/50 and skipped the pre-registered missing later checkpoints after its safety stop; an initial float16 measurement was archived after ckpt-0 gate STOP, then the required float32 recovery reproduced pilot ckpt-0 exactly (max absolute effective-rank delta 0.0000); telemetry and transcripts copied under `outputs/exp15_cuda_grpo_gsm8k_dd5f54a0e2b7/telemetry/` |
| Replicate A Phase 1 | RTX 4070 Laptop (CUDA fp32-master/bf16-autocast, paged_adamw_8bit) | 2.4 h | completed the pre-registered terminal state at update 80 via `hard_cap_stop`; sentinel windows at steps 25/50/75 were effective (`6.469e-05`, `4.298e-05`, `3.735e-05`); telemetry `telemetry/gpu_20260719_153902_exp15_phase1_exp1_5_1_config_seed42.csv` |
| Replicate A Phase 2 + ckpt-0 gate | RTX 4070 Laptop (CUDA float32 measurement) | 6.0 min | measured all 17 checkpoints (0 through 80 every 5 updates); ckpt-0 identity gate PASS with max absolute effective-rank delta 0.0000; telemetry `telemetry/gpu_20260721_084945_exp15_phase2_exp1_5_1_config_seed42.csv` |
| Replicate B Phase 1 | RTX 4070 Laptop (CUDA fp32-master/bf16-autocast, paged_adamw_8bit) | 1.9 h | completed the pre-registered terminal state at update 80 via `hard_cap_stop`; no safety stop; sentinel windows at steps 25/50/75 were effective (`6.518e-05`, `4.381e-05`, `3.783e-05`); telemetry `telemetry/gpu_20260721_085929_exp15_phase1_exp1_5_1_config_seed43.csv` |
| Replicate B Phase 2 + ckpt-0 gate | RTX 4070 Laptop (CUDA float32 measurement) | 6.0 min | measured all 17 checkpoints (0 through 80 every 5 updates); ckpt-0 identity gate PASS with max absolute effective-rank delta 0.0000; telemetry `telemetry/gpu_20260721_131106_exp15_phase2_exp1_5_1_config_seed43.csv` |
| Replicate C Phase 1 | RTX 4070 Laptop (CUDA fp32-master/bf16-autocast, paged_adamw_8bit) | 2.0 h | completed the pre-registered terminal state at update 80 via `hard_cap_stop`; no safety stop; sentinel windows at steps 25/50/75 were effective (`6.412e-05`, `4.418e-05`, `3.855e-05`); endpoint completion clipping was 0.9844; telemetry `telemetry/gpu_20260721_131855_exp15_phase1_exp1_5_1_config_seed44.csv` |
| Replicate C Phase 2 + ckpt-0 gate | RTX 4070 Laptop (CUDA float32 measurement) | 6.6 min | measured all 17 checkpoints (0 through 80 every 5 updates); ckpt-0 identity gate PASS with max absolute effective-rank delta 0.0000; telemetry `telemetry/gpu_20260721_203706_exp15_phase2_exp1_5_1_config_seed44.csv` |
| Three-replicate forensics analysis | RTX 4070 Laptop (CPU analysis) | 1.2 sec | all three replicates reached the step-80 hard cap without the pre-registered collapse event and are right-censored; `n_collapsed=0`, so SC1-SC3 are not evaluable; dormant fraction remained zero at all measured layers/thresholds; the >=2-censored decision branch indicates collapse-hazard dependence on trajectory randomness |

## 2026-07-22--24 Experiment 1.6 Windows RTX 4070

Run dir: `outputs/exp15_cuda_grpo_gsm8k_caebbcc73461`.

| Phase | Hardware | Wall time | Notes |
|-------|----------|-----------|-------|
| Phase 1 | RTX 4070 Laptop (CUDA fp32-master/bf16-autocast, paged_adamw_8bit) | 11.76 h | completed 500/500 GSM8K GRPO updates at lr=3e-6; all eight checkpoints saved; every sentinel window effective; no safety stop |
| Phase 2 + ckpt-0 gate | RTX 4070 Laptop (CUDA float32 measurement) | 4.6 min | measured checkpoints 0/25/50/100/200/300/400/500; ckpt-0 reproduced the pilot effective ranks exactly; identity gate PASS |
| Phase 3 endpoint probe | RTX 4070 Laptop (CUDA) | 7.74 h | completed ckpt {0,500} × seed {42,43,44}, six cells total, all 50/50 updates; no OOM or safety stop |
| Expansion gate | CPU analysis | <1 min | G-A failed (+4.06% late-window erank_L12 displacement vs 7.5% threshold); G-B failed (-0.0089 endpoint mean-delta drop vs +0.05 threshold); verdict STOP, so full grid and Phase 4 were not run |

## 2026-08-16 Experiment 2 v9 Phase-0 feasibility probe (Colab A100)

Notebook: `exp2_v9_colab_probe_en.ipynb` — Drive-only self-contained probe
(carries `experiment 2/` source as a gzip+base64 blob; no repo clone, no token).
Config: `exp2_config_4070_instruct_v9.json`, unmodified. Phase 0 only; Stage A
never launched. Findings: `experiment 2/FINDING_V9_PHASE0_COLAB_A100.md`.

| Phase | Hardware | Wall time | Notes |
|-------|----------|-----------|-------|
| Phase 0 `contract` | Colab A100-SXM4-80GB High-RAM | 0.3 min | exit 0; config verified pristine at group 8 / device batch 8 / accumulation 8 |
| Phase 0 `prepare` | Colab A100-SXM4-80GB High-RAM | 0.4 min | exit 0; `geometry_and_split_gate_pass`; v9 split IDs `exact_id_match` vs frozen v8 splits |
| Phase 0 `smoke` | Colab A100-SXM4-80GB High-RAM | 11.5 min | exit 1, not a crash — both stages trained 2/2 updates with healthy gradients, then Stage A stopped on the pre-registered gate `stage_a_smoke_completion_clipping_exceeded_limit` (clip 0.09375 / 0.140625 vs 0.1 limit). Preflight gate PASS: 7/16 combined variance groups, reproducing the 4070 reference exactly. `step_times` 59.481 / 59.506 s → Stage A 200 updates ≈ 3.31 h. VRAM peak NOT captured (the probe's Phase-0 cell has lost its `nvidia-smi` sampler thread). |
| Whole-notebook run | Colab A100-SXM4-80GB High-RAM | ~13 min incl. pip install + HF downloads | runtime disconnected immediately after the result was read, to stop unit consumption |

Earlier same-day attempts on a Colab **L4 (22 GiB)** are part of this accounting:
group 8 OOM'd there (`Tried to allocate 5.80 GiB. GPU 0 has a total capacity of
22.03 GiB of which 791.12 MiB is free`), i.e. the recipe needs ≈27 GiB. Several
L4 sessions were also spent on notebook-plumbing failures (missing HF artifact
pre-download, and one attempted batch-geometry change that the contract
correctly refused in 0.3 min).

**Compute units consumed: _____ (to be filled in by hand — not readable from the
notebook; check the Colab usage panel before/after).** Deviation to disclose:
the v9 amendment anticipated an L4 "only after the hardware move is approved";
this probe used an A100 80 GB, which was neither the anticipated device nor
pre-approved. Recorded rather than left implicit.

## 2026-08-16 Experiment 2 Colab/7B MVP Phase-0 attempt (Colab A100)

Notebook: `experiment 2/colab/00_phase0_selfcontained.ipynb` (self-contained; the
Colab Secrets PAT is broken — verified today, clone returns HTTP 403 "Write
access to repository not granted"). Config: `exp2_colab_config_mvp.json`,
Qwen2.5-7B base, MVP scope. Findings:
`experiment 2/FINDING_GATE_0A_MEASURES_THE_WRONG_POPULATION.md`.

| Phase | Hardware | Wall time | Notes |
|-------|----------|-----------|-------|
| PAT diagnostic | Colab CPU | ~2 min | ran notebook 00's clone cell only. Secret exists and notebook access was granted, but the clone fails HTTP 403. Token sanitizer worked — nothing leaked. Establishes that Colab's GitHub OAuth reads the private repo fine and only the PAT path is broken. |
| Phase 0 attempt 1 | Colab A100 (Python 3) | ~8 min incl. deps + 7B download | died at `load_all_records` with `ImportError: cannot import name '_center' from 'numpy._core.umath'` — the pinned `numpy==2.3.5` install needs a kernel restart, which notebook 00 never does. The v9 4070 probe avoided this only because it runs every stage in a subprocess. Latent bug in notebook 00. |
| Phase 0 attempt 2 | Colab A100 (Python 3) | ~20 min | after `Restart session and run all`: config loaded (`..._MVP`, Qwen2.5-7B base, max_steps 100, checkpoints [0,50,100], group 8), data loaded (Math 54404 / Simulation 3730), then **GATE 0a STOP: stage-B p95=1407.0 > 1024**. Reproduced the WIN4070 audit byte-for-byte. No training reached. Runtime disconnected immediately. |

**Compute units consumed: _____ (to be filled in by hand.)** Both A100 sessions
were stopped at their gate rather than left running, and the runtime was
disconnected each time.

### 2026-08-16 (later) — 7B MVP Phase 0, A100 High-RAM, first real measurements

Reached Gate C0 for the first time. Results in
`experiment 2/FINDING_STAGE_B_TRAIN_EMPTY.md`.

| Gate | Result |
|------|--------|
| GATE 0a | **PASS** on the eligible population (p95=628 <= 1024); raw p95=1407 recorded and cross-checked against the WIN4070 reference. Confirms the gate fix. |
| Splits | stage_a_train 54257 / **stage_b_train 0** / stage_b_eval 300 / probe 4096. The zero is a real bug — the probe consumed the whole CodeIO pool. Fixed same day. |
| Gate C0 | **PASS** — peak 40.14 GiB of 79.25 GiB, headroom 49.3%, 2 training steps ran (loss -0.007958, -0.049697). First 7B+LoRA+group-8 memory number in the project. **40 GiB A100 would NOT fit.** |
| GATE 0b + smoke | not reached — Colab reclaimed the runtime for inactivity at the preflight cell. |

**Compute units consumed: _____ (fill in by hand.)** Four A100 sessions today;
this was the only one to produce measurements.

### 2026-08-16 (later) — 7B MVP Phase 0 COMPLETE

| Phase | Hardware | Wall time | Notes |
|-------|----------|-----------|-------|
| Phase 0 full pass | Colab **A100-SXM4-80GB** High-RAM | ~45 min end to end | **Every gate PASSED.** GATE 0a pass; Gate C0 peak 40.14/79.25 GiB (49.3% headroom, ~5 min); GATE 0b 12/16 + 4/16 exact on Stage A, 5/8 + 5/8 on Stage B (~17 min); smoke 2/2 both stages (~8 min). No clipping stop. Full detail: `experiment 2/FINDING_7B_PHASE0_COMPLETE.md`. |

**Compute units consumed: _____ (fill in by hand.)**

Note on the day's total draw: six earlier A100 sessions were lost to Colab
reclaiming the runtime for inactivity before this one completed. Those sessions
produced the Gate C0 number and the GATE 0a / splits bug findings, but each also
drew units. The completed run above is the only one that reached the end.

## 2026-08-17 — exp2 7B Phase 1 (Stage A, Math GRPO) STARTED
- GPU: Colab A100-SXM4-80GB High-RAM (same warm runtime that passed Phase 0 on 08-16)
- Phase: Phase 1 Stage A, 100 updates, group 8, LoRA r=16, config hash fc243e587296
- Launched ~23:21 local as a detached subprocess (pid 21875); expected ~3.5 h
- Units before/after: **not recoverable.** Colab shows only a live balance, no
  history, and no reading was taken at the time. Duration and GPU type below are
  the recoverable part; the unit delta for this entry is permanently missing.
- Note: reused the Phase-0 runtime deliberately — a fresh one risks drawing the
  40 GB A100 SKU, which Gate C0 (peak 40.14 GiB) says will OOM.

## 2026-08-18 — exp2 7B Phase 1, base run STOPPED + Instruct restart
- GPU: Colab A100-SXM4-80GB High-RAM (same session, continuous since 08-17)
- Base Stage A: 7/100 updates in 19m07s, stopped on the clipping gate (162.7 s/update)
- Two generation-only measurements, ~35 min each: base and Instruct completion-length
  distributions at cap 3072, 256 completions each
- Restarted Stage A on Qwen2.5-7B-Instruct, cap 1536, config hash e33527592dd9
- Units before/after: **not recoverable** (no reading taken; Colab keeps no history)

## 2026-08-18 — exp2 7B Phase 1 Stage A COMPLETE (Instruct, cap 1536)
- GPU: Colab A100-SXM4-80GB High-RAM (3rd VM of the day; SKU verified before spending)
- Config: exp2_colab_config_mvp_instruct.json, hash e33527592dd9
- Result: 100/100 updates, completion_status=complete, wall 19783 s = 5 h 29 m
- Throughput ~198 s/update; ckpts 0/50/100 all written and recovered to the Mac
- Same-day cost also includes: base Stage A 7/100 (19 m, stopped on clipping gate),
  two generation-only completion-length measurements (~35 m each), one lost run
  of 53/100 (~2 h 50 m) reclaimed for browser inactivity
- Units before/after: **not recoverable** (no reading taken; Colab keeps no history)

## 2026-08-19 — exp2 7B Stage B v2, all three Delta-R arms (overnight)
- GPU: Colab A100-SXM4-80GB High-RAM. Two VMs: the first was reclaimed ~30 min into
  the run, the second carried the whole overnight session (~10 h continuous).
- Config: `exp2_colab_config_mvp_instruct_stageb_v2.json`, hash `bd99ddd2817f`
  (Stage-B cap 640 -> 2048, eval points [0,10,20,30] -> [0,30])
- Work done: Stage-B completion-length measurement on ckpt-0 and ckpt-100
  (~35 min); environment rebuild + Drive restore after the VM recycle (~15 min);
  then three Stage-B arms at 30/30 updates each — 11 976 s, 12 184 s, 11 853 s.
- **Units, from live readings taken during the session** (Colab exposes only a
  current balance, so these are spot observations, not a ledger):

  | time | reading | note |
  |---|---|---|
  | ~22:00, 08-18 | 94.07 | before the length measurement |
  | ~22:30, 08-18 | 92.38 | at the VM recycle |
  | ~22:35, 08-18 | 91.50 | during environment rebuild |
  | 07:51, 08-19 | 129.44 | after a +100 top-up bought overnight |
  | 09:03, 08-19 | 120.97 | run complete |
  | ~13:00, 08-19 | 115.33 | runtime still connected and idle |

- **Derived draw:** 91.50 − 100 (top-up) + 129.44 → **62.1 units** for the overnight
  work, over ~9.3 h. That implies ~6.7 units/h against the 6.77/h the resources
  panel reported — the two agree, which is the only cross-check available here.
- **5.6 units were wasted** between 09:03 and ~13:00 with the run finished and the
  runtime still attached. Disconnect immediately after `STAGE_B_V2_DONE`.
- Rate quoted by Colab's resources panel throughout: **~6.77 units/hour** on
  A100-SXM4-80GB High-RAM.

### Outstanding against the CLAUDE.md compute-accounting constraint
GPU-dashboard **screenshots were not taken for any exp2 session** and cannot be
recovered after the fact. The constraint is therefore only partly satisfied: dates,
GPU types, durations and phases are recorded throughout, and unit readings exist for
this session only. Screenshots need to start with the next run.
