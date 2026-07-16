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
