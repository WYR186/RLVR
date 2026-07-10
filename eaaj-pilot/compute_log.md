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

Notes:
- bitsandbytes `paged_adamw_8bit` optimizer checkpointing hung or produced corrupt optimizer state on Windows. For this optimizer, local CUDA training now uses `save_only_model=True`; phase-3 adaptation uses the same guard.
- Interrupted checkpoint repairs were required at steps 25, 50, and 75. Repairs restored model/trainer resume metadata and scheduler position, but optimizer moments and RNG state at those resume boundaries are not identical to an uninterrupted run.
- `run_pipeline.ps1` GPU telemetry now samples once per minute and appends each row immediately, avoiding empty CSVs when a long-running `nvidia-smi -l` job is stopped.
