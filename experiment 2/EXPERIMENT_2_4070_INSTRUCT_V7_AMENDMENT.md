# Experiment 2 — RTX 4070 Instruct v7 shaped-reward engineering amendment

**Config:** `exp2_config_4070_instruct_v7.json`  
**Hardware:** RTX 4070 Laptop GPU, 8 GiB  
**Status:** pre-registered before v7 preflight and CUDA updates

## Why geometry alone is insufficient

The immutable v1–v6 diagnostics establish a hardware/objective conflict. The
exact-reward recipes that produced updates clipped too many Math completions;
the long-completion group-3 recipe passed the length gate but produced two
zero-reward/zero-gradient updates; and the short-Math group-4 recipe had no
exact-reward variance in its deterministic preflight. No original-exp2 formal
training was started.

V7 is therefore an explicitly separate **4070 engineering variant**, not an
equivalent replication. It uses v6's memory geometry (Math prompt <=256,
completion 1280, four generations, device batch 4, accumulation 8) and adds a
small registered Stage-A format reward:

`registered_reward = exact_GURU_reward + 0.1 * balanced_nonempty_boxed_answer`

The format component is Math-only, contains no answer correctness information,
and is at most one tenth of a correct-answer reward. A valid box is recognized
with brace balancing, including nested LaTeX, and must be non-empty. Malformed
or unclosed boxes receive zero. CodeIO/Stage B remains exact GURU reward only.

## Frozen populations and gates

- Stage A: 52,755 Math prompts at <=256 tokens; no truncation.
- Stage B: 1,432 CodeIO prompts at <=640 tokens, split 1,132/300.
- Stage-A maximum sequence: 1,536 tokens; Stage B: 1,024 tokens.
- Deterministic preflight records exact, boxed-format, and combined rewards.
- Both stages must save two CUDA updates; every Stage-A update must clip <=10%.
- At least one Stage-A smoke update must have positive reward standard
  deviation and nonzero gradient norm.

OOM, constant registered reward, two update-free batches, over-limit clipping,
or missing checkpoint is STOP. Results may support only a feasibility claim
for this shaped-reward/short-context 4070 variant; they must not be reported as
the original Base/full-CodeIO Experiment 2.
