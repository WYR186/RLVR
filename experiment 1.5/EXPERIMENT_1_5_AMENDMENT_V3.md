# Experiment 1.5 scientific amendment v3

Date: 2026-07-16

## Trigger and diagnosis

The independent v2 Stage-A run,
`exp15_cuda_grpo_gsm8k_dd5f54a0e2b7`, passed sparse-reward preflight and the
step-25 and step-50 update-effectiveness sentinels, then stopped at update 55.
The persisted hard-stop reason was five consecutive updates with zero
within-group reward variance.

This was a policy-collapse trajectory rather than an infrastructure failure:

- updates 37-47 still produced occasional exact-answer rewards;
- completion clipping then approached 1.0 and mean completion length approached
  the 512-token cap;
- entropy fell from roughly 0.20 to roughly 0.05;
- updates 51-55 had zero reward variance, zero loss, and zero gradient norm;
- GPU, timing, finite-value, and update-effectiveness checks remained healthy.

The closest completed local control used the identical pinned model, reward,
temperature, top-p, eight generations, 512-token completion cap, and batch
geometry at Stage-A `lr=1e-6`. It completed 200 updates without the v2 collapse,
whereas v2 used `lr=1e-5` with no KL penalty.

## Single-variable amendment

For Stage A, lower the learning rate from `1e-5` to the locally validated
`1e-6`. Keep `beta=0.0`. No other scientific or execution setting changes:
model/revision, datasets/splits, seed, exact-answer reward, 500-update budget,
checkpoint grid, temperature/top-p, eight generations, 512-token completion
cap, batch geometry, optimizer/precision, safety rules, measurements,
adaptations, and analysis are unchanged.

The prepared `beta=0.04` arm remains disabled. Enabling KL at the same time as
lowering the learning rate would confound which intervention prevented the
collapse and would add reference-model memory pressure on the 8 GiB GPU.

## Isolation and gates

The v3 config and its hash create a fresh run directory. v1 and v2 remain
immutable evidence and are never resumed or overwritten. Before unattended
continuation, v3 must pass:

1. config-hash Gate 0;
2. the existing CUDA plumbing smoke;
3. exact-reward sparse-signal preflight;
4. step-25 and step-50 update-effectiveness sentinels using learning-rate-scaled
   thresholds;
5. no non-finite, repeated zero-signal, timing, or memory safety stop.

If v3 reaches another safety stop, preserve it and amend again rather than
disabling the zero-signal hard stop.
