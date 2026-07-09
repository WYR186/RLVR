"""Execution-layer workarounds for GRPO on Apple-silicon MPS.

Motivation (measured 2026-07-08, one real GRPO update, Qwen2.5-0.5B fp32,
8 prompts x 8 generations, 512-token cap, M3 Max):

  stock TRL 1.6 on MPS:   ~335 s/update, of which
    - selective_log_softmax (row-looped torch.logsumexp)  ~95 s
    - backward through that logsumexp graph               ~128 s
    - generate with disable_compile=True                  ~91 s
  reference points: the SAME logits pass through TRL's chunked
  entropy_from_logits in 1.7 s, and the same 64-sequence generate runs at
  341 tok/s standalone — the hardware is fine, two code paths are not.

Both patches below are mathematically identical to what they replace and do
not touch the training recipe (sampler, loss, optimizer, budgets unchanged):

1. `chunked_selective_log_softmax` — log_softmax(x).gather(index) computed in
   row chunks. Identity: log_softmax(x)_i = x_i - logsumexp(x); same numerics,
   different (fused, fast-on-MPS) kernel. Mirrors the memory-bounding pattern
   TRL itself uses in `entropy_from_logits`.
2. Re-enabling compiled decode in generation (TRL pins
   `disable_compile=True`); on this stack the compiled decode path is what the
   standalone 341 tok/s benchmark exercised.

Apply only for MPS runs: `apply_mps_grpo_patches()` before building the
trainer, `enable_compiled_generation(trainer)` after. CPU/CUDA runs are left
on stock TRL so existing strata stay byte-for-byte on their original path.

Outcome for the record (LOCAL_EXPERIMENT_PLAN.md, 2026-07-08): patch 1 saves
~60 s/update on MPS, patch 2 measured neutral, but the end-to-end MPS update
still lands at ~265-320 s/update — parity with the CPU stratum — because
~10 host<->device scalar syncs per decode token inside transformers'
generation loop dominate (aten::_local_scalar_dense, 54% of one profiled
update). MPS therefore remains non-default; these patches matter only if a
future torch/transformers release removes that sync overhead.
"""
from __future__ import annotations

import torch


def chunked_selective_log_softmax(logits, index, chunk_size: int = 256):
    """Drop-in for `trl.trainer.utils.selective_log_softmax`.

    Equivalent to `log_softmax(logits, -1).gather(-1, index)` with peak memory
    bounded by `chunk_size` rows, avoiding the row-looped `torch.logsumexp`
    whose reduction kernel is pathologically slow on MPS.
    """
    squeeze = index.ndim == logits.ndim - 1
    if squeeze:
        index = index.unsqueeze(-1)
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_index = index.reshape(-1, index.size(-1))
    out = []
    for lg, ix in zip(flat_logits.split(chunk_size, dim=0),
                      flat_index.split(chunk_size, dim=0)):
        out.append(torch.log_softmax(lg, dim=-1).gather(-1, ix))
    result = torch.cat(out, dim=0).view(index.shape)
    return result.squeeze(-1) if squeeze else result


def apply_mps_grpo_patches() -> None:
    """Route TRL's GRPO logprob computation through the chunked kernel."""
    import trl.trainer.grpo_trainer as grpo_trainer
    import trl.trainer.utils as trl_utils

    trl_utils.selective_log_softmax = chunked_selective_log_softmax
    grpo_trainer.selective_log_softmax = chunked_selective_log_softmax


def enable_compiled_generation(trainer) -> None:
    """Undo TRL's hard-coded `disable_compile=True` on the generation config."""
    trainer.generation_config.disable_compile = False
