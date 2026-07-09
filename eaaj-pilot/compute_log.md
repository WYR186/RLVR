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
