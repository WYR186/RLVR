# Drivers — the code that actually produced the exp2 7B run

**Read this before using anything in here.**

The four notebooks in `experiment 2/colab/01–04` were written for the **base-model**
configuration (`fc243e587296`) before the 2026-08-18 Instruct amendment. They
reference `exp2_colab_splits.json`, clone through a GitHub PAT that is confirmed
broken (HTTP 403), and **have zero cell outputs — none of them was ever executed.**
They did not produce the data in this package and are not shipped with it.

What actually ran: these driver scripts, launched as detached subprocesses from a
copy of `colab/00_phase0_selfcontained.ipynb` (the self-contained notebook, which
carries the repo source as an embedded blob and needs no PAT). Each imports
`experiment 2/src/pipeline.py` and does nothing that `pipeline.py` does not do.

## Provenance of each file

| file | status |
|---|---|
| `01_stage_a.py` | **reconstructed.** Equivalent call, not the byte-identical original. Its `print` statements are reproduced from `stage_a/stage_a_instruct.log`, which is committed, so the structure is checkable line by line. Every parameter comes from the config. |
| `02_measure_stage_b_length.py` | **verbatim** as executed 2026-08-19. |
| `03_stage_b_v2.py` | **verbatim** as executed 2026-08-19. |
| `00_restore_from_drive.py` | **verbatim.** Not part of the science — it rebuilds a runtime after Colab recycles the VM, which happened once mid-run. |

The Phase-2 step (transfer T_t + Q metrics) ran as a similar subprocess calling
`pipeline.run_transfer_T` and `pipeline.measure_checkpoint_q`; its console output is
committed as `phase2.log` in the run directory. It is not reconstructed here because
the log records the full call sequence and both functions take their parameters
directly from the config.

## Why any of this matters for reuse

`CROSS_RUN_NOTE_7B_VS_05B.md` §6.2 says a single measurement contract is needed
before anyone pools Q numbers across runs. That contract is recorded as data in
`measurement_contract` inside every `metrics_ckpt*.json` — eval mode, bf16,
last-non-padding-token pooling, mean-abs-over-non-padding for dormancy, 512 max
prompt tokens, float32 accumulator, float64 SVD, layers [5, 14, 26] — but a contract
you cannot execute is not reusable. `src/pipeline.py:measure_checkpoint_q` is the
implementation.
