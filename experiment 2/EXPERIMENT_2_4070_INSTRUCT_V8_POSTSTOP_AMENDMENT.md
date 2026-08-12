# Experiment 2 — RTX 4070 Instruct v8 post-stop amendment

**Status:** pre-registered after the immutable Stage-A safety stop and before
any post-stop Phase 2, Phase 2b, or Stage-B computation.

## Source result

The formal v8 Stage A stopped at update 110 because five consecutive updates
had zero within-prompt-group reward variance. The source run
`exp2_4070_cuda_guru_math_c4a279960232` is immutable. It must not be resumed,
edited, renamed, or used to claim a completed 200-update Stage-A trajectory.

The registered main checkpoints that exist are 0, 25, 50, and 100. The
trainer-only checkpoint at 75 is excluded from the main analysis because it
was not in the registered checkpoint set. Checkpoints 150 and 200 do not
exist.

## Truncated Phase 2 and Phase 2b

Phase 2 evaluates zero-shot CodeIO exact reward on the frozen 300-example
Stage-B evaluation split at checkpoints 0, 25, 50, and 100. Decoding is greedy
and fixed across checkpoints. Transfer is reported as
`T_t = Score_B(M_A,t) - Score_B(M_0)`.

Phase 2b measures effective rank, participation ratio, anisotropy, dormant
fraction, and weight norms at layers 4, 12, and 22 on the already-frozen
2,048-prompt probe set, in float32 and eval mode, for the same four
checkpoints. Because v8 uses the registered Instruct-model deviation, the old
Base-model numerical ckpt-0 reference is not an identity target. Instead,
checkpoint hashes, the frozen probe hash, dtype, layers, pooling, and sample
count form the identity contract; ckpt-0 is the within-v8 baseline.

All outputs go to a new post-stop run directory. Nothing is written beneath
the immutable source run.

## Truncated Stage B

If Phase 2 completes, the fixed-budget Stage-B grid is restricted to
checkpoints 0, 50, and 100, with seeds 42, 43, and 44. Every cell uses the
unchanged registered CodeIO recipe: exact reward, 50 updates, evaluation at
0/10/20/30/40/50, learning rate 1e-6, beta 0, temperature 0.7, top-p 1,
eight generations, 640 prompt tokens, and 384 completion tokens. Each cell
runs in a fresh process and writes to an isolated directory.

This grid is a post-stop truncated trajectory, not the registered five-point
Stage-B experiment. With only three Stage-A checkpoints, rank correlations
are exploratory descriptions and carry no inferential claim. All fixed-budget
outcomes remain conditional on zero-shot transfer.

## Independent v9 boundary

Any attempt to complete a new 200-update Stage A is a new experiment with a
new config hash and output directory. It must repeat deterministic reward
preflight and both CUDA smoke stages. The v8 reward, threshold, learning rate,
or artifacts may not be retroactively changed. The first candidate to test is
a larger Stage-A generation group while holding model, data, reward, learning
rate, sequence limits, and safety thresholds fixed; if that geometry does not
fit the RTX 4070, the candidate moves to an L4 rather than weakening the v8
safety gate.
