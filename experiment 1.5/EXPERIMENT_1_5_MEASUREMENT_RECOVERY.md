# Experiment 1.5 v3 measurement recovery

Date: 2026-07-17

## Trigger

The completed v3 Stage-A run reached 500/500 updates with every update-
effectiveness sentinel window valid and no safety stop. The first Phase-2
measurement then stopped at the ckpt-0 comparability gate.

This was a measurement-contract mismatch, not a training failure. The v3
ckpt-0 metric file recorded `model_dtype=torch.float16`, while the committed
pilot reference recorded `model_dtype=torch.float32`. Their complete grouped
weight norms were identical, but their effective-rank values differed by
0.3910 at layer 4, 0.2936 at layer 12, and 0.0305 at layer 22. The 0.01 gate
requires the same machine and dtype, so the original comparison was invalid.

## Recovery amendment

Reuse the completed v3 Stage-A checkpoints without retraining. Preserve the
original float16 Phase-2 directory as
`measurements_float16_v3_gate_stop_20260717/` and preserve its completion
marker as `phase2_complete_float16_v3_gate_stop_20260717.json`. Rerun Phase 2
into the canonical `measurements/` path with `model_dtype=float32`, matching
the pilot reference.

No Stage-A, adaptation, data, seed, probe, layer, or analysis parameter is
changed. The recovery config intentionally keeps the same experiment id and
Stage-A fields, so Gate 0 must resolve to the existing v3 run directory
`exp15_cuda_grpo_gsm8k_c7cc7a1d02d9`.

## Gated continuation

Run:

```powershell
powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp15_v3_float32_recovery.ps1"
```

The script is resumable and performs these steps:

1. require an idle CUDA compute device and validate all eight v3 checkpoints;
2. run Gate 0 before changing any artifact path;
3. archive the float16 measurements and completion marker without deletion;
4. rerun Phase 2 with the float32 recovery config and copy its telemetry;
5. run the ckpt-0 identity gate, now including an explicit dtype check;
6. only after PASS, run ckpt-0/seed-42 adaptation and the bridge gate;
7. only after that gate passes, resume the complete pre-registered Phase-3
   checkpoint/seed grid;
8. stop after Phase 3. Phase 4 is never invoked.

Use `-Phase2Only` to stop immediately after the float32 ckpt-0 gate. Any gate
exit other than PASS stops the script before further adaptation compute.

## Evidence policy

Nothing in v1, v2, or the original v3 float16 measurement is overwritten.
The recovery script writes `postgate_recovery.jsonl` in the v3 run directory,
and copies every wrapper GPU CSV and transcript produced by the recovery into
the run's `telemetry/` directory. Existing completed Phase-3 cells are
validator-checked and skipped on rerun.
