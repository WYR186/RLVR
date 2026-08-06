# Experiment 2 — RTX 4070 Instruct rescue amendment

**Parent 4070 amendment:** `EXPERIMENT_2_4070_AMENDMENT.md`  
**Model:** `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775`  
**Status:** pre-registered only after the Base-model sparse-reward gate stopped

## Trigger and frozen decision

The geometry-only 4070 variant passed its memory-side preflight, but the
released-verifier Math preflight for the Base model produced 0 correct answers
across 8 prompts × 8 generations and zero groups with reward variance. No
parameter update occurred. The diagnostic is preserved under
`smoke_outputs_4070/stage_a/sparse_reward_preflight.json`.

The same-size Instruct checkpoint was then tested without training on the exact
same 8 prompt IDs and sampling contract. It produced 1 correct answer and 1
group with within-group reward variance. That diagnostic is preserved as
`data/instruct_candidate_sparse_preflight.json`. This amendment therefore
freezes the Instruct checkpoint; no shaping reward is introduced.

## Recomputed geometry and split

The tokenizer/chat template change alters exact lengths and stable IDs. At the
pinned revisions, the unchanged limits retain:

- Math <=512: 54251 examples.
- CodeIO <=640: 1432 examples, frozen into 1132 train and 300 eval.
- Stage-B completion remains 384; maximum prompt+completion remains 1024.

The Instruct split is independently frozen in
`data/exp2_4070_instruct_splits.json`. It must not reuse the Base split file.

## Scientific boundary

This is a feasibility/rescue experiment on **Qwen2.5-0.5B-Instruct and the
short-context CodeIO subset**. It is not comparable as a model-controlled
continuation of the Base-model pilot, and it does not answer the original
full-CodeIO Experiment 2. Every result must attach both deviations: model
variant and length-selected Stage B.

All remaining hyperparameters, checkpoints, three Stage-B seeds, transfer
control, Q measurements, fixed 50-update budget, safety stops, and conditional
interpretation stay as registered. If either Instruct CUDA smoke stage fails,
STOP; do not select easier prompts or add shaping reward.
