# Experiment 2 — independent RTX 4070 Instruct v9 amendment

**Config:** `exp2_config_4070_instruct_v9.json`  
**Status:** candidate registered on 2026-08-04; Phase 0 has not run; formal Stage A is not authorized by this file alone

## Why this is a new run

The immutable v8 Stage-A run safety-stopped at update 110 after five
consecutive updates in which every three-sample prompt group had constant
registered reward. Across updates 1–110, only 88 of 880 prompt groups had
within-group reward variance. The same 52 updates with
`frac_reward_zero_std=1` also had zero gradient norm. There was no OOM,
non-finite loss, or policy-clipping failure.

Post-stop measurements at v8 checkpoints 0/25/50/100 found CodeIO exact
scores of 31/29/28/28 out of 300. On the fixed 2,048-prompt probe, layer-22
effective rank moved 554.91 → 538.01 → 527.33 → 519.27 while layer 4 stayed
near 327 and layer 12 stayed near 344. These are truncated-trajectory
diagnostics, not evidence that v8 completed its registered 200 updates.

V9 is therefore an independent hypothesis test: increase the Stage-A GRPO
group from 3 to 8 so a discrete sparse reward is more likely to vary within a
prompt group. V9 never resumes v8 and never fills in v8 checkpoints 150/200.

## Single training-geometry change

Relative to v8, Stage A changes:

- `num_generations`: 3 → 8;
- `per_device_train_batch_size`: 3 → 8, which is the linked geometry change
  required to keep eight unique prompts per optimizer update with accumulation
  8.

The model and revision, dataset and revision, frozen populations and splits,
seed, exact-plus-0.1-boxed reward, learning rate, optimizer, parameter and
autocast dtypes, prompt/completion limits, beta, temperature, top-p, 200-update
budget, checkpoint grid, measurement probe, and all runtime safety thresholds
remain unchanged. Stage B remains the v8 exact-reward CodeIO contract.

## Phase-0 gates

Phase 0 must run in a new v9 namespace and must pass all of the following
before a formal Stage-A launch is considered:

1. The contract audit confirms that only the linked group/device-batch fields
   differ from v8 on the training recipe, and the newly generated v9 split IDs
   exactly match the frozen v8 split IDs.
2. The deterministic Stage-A preflight uses 16 frozen prompts × 8 generations,
   records exact, boxed-format, and combined within-group variance separately,
   and has at least two groups with variable combined registered reward.
3. Both CUDA smoke stages complete exactly two updates and save
   `checkpoint-2`. Every Stage-A smoke update has clipping ≤10%; at least one
   has positive reward standard deviation and a nonzero gradient.
4. There is no OOM, non-finite metric, safety-stop file, partial checkpoint, or
   missing telemetry. Promotion requires the v9 Phase-0 completion marker and
   manual review of its preflight, dashboard, checkpoints, and logs.

The five-update zero-variance and five-update >10% clipping stops are not
relaxed. If group 8 does not fit the RTX 4070, preserve the failed smoke and run
the exact same recipe on an L4 only after the hardware move is approved. Do not
reduce the group, shorten completions, change the reward, or move another
scientific variable to make the smoke pass.

## Formal-run and failure rules

The formal v9 run must be a fresh config-hashed output containing checkpoints
0/25/50/100/150/200. It may start only after Phase 0 passes and while no other
Algoverse GPU job is active. A safety stop remains terminal for that run.

At a safety stop, v9 additionally saves `safety-stop-weights-step-N` as a
diagnostic-only measurement artifact. It is not a registered checkpoint, is
not evidence of completion, and must never be used as a resume source. Any
OOM, repeated zero variance, repeated clipping, disk shortfall, corrupt
checkpoint, or uncertain recovery is preserved and escalated rather than
worked around.

## Claim boundary

V9 tests whether larger within-prompt groups prevent the reward-signal collapse
seen in the v8 engineering stratum. It is neither a continuation of v8 nor an
equivalent execution of the original Base/full-CodeIO Experiment 2 proposal.
