# Experiment 1.5 execution amendment v2

Date: 2026-07-16

## Trigger

The first formal CUDA run, `exp15_cuda_grpo_gsm8k_e73704296e47`, stopped at
update 7. The persisted reason was five consecutive updates above the shared
callback's 10% completion-clipping threshold. The stopped run is preserved in
commit `580f0e1` and is not resumed or overwritten.

## Evidence at the stop

- Loss and gradient norm were finite at every update.
- Every update retained within-group reward variation; the zero-signal gate
  never fired.
- Entropy remained in the observed 0.218-0.295 range.
- Step times were 66-89 seconds, below the 120-second throughput gate and far
  below the 480-second hard limit.
- Completion clipping rose as high as 0.64 and remains an important diagnostic,
  but clipping was not listed as a hard stop in the frozen plan or Windows run
  guide. Those documents defined hard stops for non-finite training, repeated
  very slow steps, and repeated zero within-group reward variance.

## Amendment

For Stage A only, completion clipping is changed from a hard stop to a logged
diagnostic. The dashboard continues to persist `completions/clipped_ratio`,
mean/min/max completion length, and terminated length every update.

No scientific or resource setting changes: model/revision, data and splits,
seed, `lr=1e-5`, 500-update budget, checkpoint grid, exact-answer reward,
temperature/top-p, eight generations, 512-token completion cap, batch geometry,
optimizer, precision profile, measurements, adaptations, and analysis remain
identical.

## Isolation and go/no-go

The amended config is `exp1_5_config_v2.json`. Its experiment id and Stage-A
safety policy produce a new config hash and run directory. The original run is
immutable evidence. The v2 run must still pass:

1. config-hash gate before training;
2. smoke Phase 1;
3. update-effectiveness sentinel at step 25;
4. the original non-finite, zero-signal, timing, memory, and downstream gates.

Any further safety stop is preserved and reported; no additional silent retry
or parameter change is authorized by this amendment.
