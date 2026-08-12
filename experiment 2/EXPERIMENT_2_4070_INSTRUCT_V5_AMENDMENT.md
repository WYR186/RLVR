# Experiment 2 — RTX 4070 Instruct v5 group-3 amendment

**Parent:** `EXPERIMENT_2_4070_INSTRUCT_V4_AMENDMENT.md`  
**Config:** `exp2_config_4070_instruct_v5.json`  
**Status:** pre-registered before v5 preflight and CUDA updates

## Trigger

V4 proved that four-sample GRPO groups with a 1024-token Math completion cap
fit the RTX 4070 and retain exact-reward signal. Its two updates each clipped
15.625% of completions and peaked near the 8 GiB device limit, failing the
pre-registered per-update 10% gate. V4 is preserved as a diagnostic and is not
eligible for formal training. The first gate-report pass also exposed a parser
bug: the dashboard's terminal train-summary row lacks per-update completion
fields. The parser now selects the two update rows by field presence; no
training metric or threshold was changed.

## Frozen v5 geometry

V5 changes no prompts, data, verifier, optimizer, learning rate, checkpoints,
or Stage-B recipe. Stage A uses three generations per Math prompt, device batch
3, and gradient accumulation 8. This still covers eight unique prompts per
optimizer update. The released activation budget is assigned to a 1280-token
completion cap, for a maximum `512 + 1280 = 1792` tokens.

Relative maximum sequence work resident per microbatch falls from v4's
`4 × 1536 = 6144` token-slots to `3 × 1792 = 5376`. The GRPO group estimator is
noisier than both the original eight-sample recipe and v4; all claims must name
this deviation.

## Gates

The deterministic released-verifier preflight must find within-group reward
variance with eight prompts and three generations. Both two-update CUDA smokes
must complete, save checkpoints, show finite loss/gradient and real GPU work,
and every Stage-A smoke update must have completion clipping `<=10%`. Any gate
failure is STOP; do not relax the threshold or regenerate the frozen sample.
