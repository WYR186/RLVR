# Experiment 2 — final RTX 4070 Instruct v2 amendment

**Parent:** `EXPERIMENT_2_4070_INSTRUCT_AMENDMENT.md`  
**Config:** `exp2_config_4070_instruct_v2.json`  
**Status:** frozen after measured CUDA smoke; this is the launch candidate

## Trigger

The first Instruct 4070 smoke completed both stages without OOM. Stage B was
healthy: finite loss/gradient, real reward variance, and 0% completion
clipping. Stage A had real reward variance and a finite first update, but both
smoke updates clipped 71.875% of completions at 512 tokens. Launching the
unchanged 200-step recipe would predictably trigger its registered five-step
clipping safety stop rather than test the intended dose.

## Frozen token-budget reallocation

For Stage A only:

- Complete prompt cap: 512 -> 256 tokens.
- Completion cap: 512 -> 768 tokens.
- Total maximum sequence budget: unchanged at 1024 tokens.
- Eligible Math population: 52755/54404 (96.97%). Prompts are filtered whole,
  never truncated.

Stage B remains exactly the previous Instruct short-context contract: CodeIO
prompt <=640, completion <=384, 1132 train / 300 eval. Model, learning rates,
batch size, gradient accumulation, eight generations, checkpoints, safety
stops, seeds, outcomes, and verifier are unchanged.

The v2 split is independently frozen as
`data/exp2_4070_instruct_v2_splits.json`. Prior Base/Instruct smoke artifacts
remain evidence and are never reused as v2 completion markers.

## Interpretation boundary

This further length-selects Stage A in addition to the already recorded model
and Stage-B deviations. Results apply only to Qwen2.5-0.5B-Instruct trained on
the frozen short-prompt Math population and adapted on frozen short-context
CodeIO. They cannot be reported as the original full-distribution exp2.

The v2 must pass a fresh two-stage CUDA smoke. STOP on OOM, constant reward,
non-finite update, missing artifacts, safety marker, or persistent clipping;
do not tune again without another explicit amendment.
