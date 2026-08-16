# Finding — v9 Phase 0 on Colab A100: group-8 preflight passes, Stage-A smoke stops on completion clipping

**Status:** standalone Phase-0 measurement finding. v9 Phase 0 **did not pass**;
formal Stage A remains unauthorized. Nothing here promotes v9.
**Date:** 2026-08-16
**Hardware:** Colab `NVIDIA A100-SXM4-80GB` (79.3 GiB, High-RAM runtime)
**Config:** `exp2_config_4070_instruct_v9.json`, unmodified
(`config_sha256 = cdee3eb399a6a9843997aaa32805c9b0b61916bbbd7edcbdab5b6d310208dafd`)
**Runner:** `run_exp2_4070_v9.py`
**Notebook:** `exp2_v9_colab_probe_en.ipynb` — a self-contained probe that carries
the `experiment 2/` source as a gzip+base64 blob, so it needs no repo clone and
no token. It lives in Google Drive only and is **not** in this repository.

---

## 1. What ran

Phase 0 only (`contract -> prepare -> smoke`). Stage A was never launched.

| Stage | Exit | Wall time | Outcome |
|---|---|---|---|
| `contract` | 0 | 0.3 min | config verified pristine: group 8, device batch 8, accumulation 8, 8 unique prompts/update |
| `prepare` | 0 | 0.4 min | `geometry_and_split_gate_pass`; v9 split IDs `exact_id_match` vs frozen v8 splits |
| `smoke` | 1 | 11.5 min | both stages trained 2/2 updates and wrote shards, then Stage A tripped a pre-registered gate |

`smoke` exit 1 is **not** a crash. Training completed; the runner stopped itself.

---

## 2. The group-8 preflight reproduces the 4070 result exactly

```
[PREFLIGHT] gate_pass=True   16 prompts x 8 generations
  combined variance groups = 7   (minimum 2)
  exact variance groups    = 1
  boxed variance groups    = 6
  n_exact_correct=6  n_boxed=107
  reference (same preflight on the 4070): combined 7/16, exact 3/16, boxed 5/16
```

The combined count (7/16) matches the RTX 4070 reference exactly, on different
hardware and a different CUDA/driver stack. This is independent confirmation of
[`FINDING_GROUP_SIZE_REWARD_VARIANCE.md`](FINDING_GROUP_SIZE_REWARD_VARIANCE.md):
group 8 restores usable within-group reward variance, and the recovered variance
is still dominated by the 0.1 boxed-format term (6 boxed vs 1 exact) rather than
by answer correctness. Phase-0 gate 2 of the v9 amendment **passes**.

---

## 3. The blocking result: Stage-A completion clipping exceeds its gate

`smoke_outputs_4070_instruct_v9/stage_a/smoke_gate_failure.json`:

```json
{
  "stage": "a",
  "status": "STOP",
  "reason": "stage_a_smoke_completion_clipping_exceeded_limit",
  "limit": 0.1,
  "clip_ratios": [0.09375, 0.140625],
  "step_times": [59.48098108400063, 59.50586474800002]
}
```

Gate: `phase0_stage_a_smoke_max_clip_ratio_each_update = 0.1`, a frozen gate
inherited unchanged from v8. Update 1 sat just under the limit at **9.375%**;
update 2 exceeded it at **14.0625%**.

Observed completion lengths during the same two updates:

```
{'loss': '0.04867', 'grad_norm': '0.714', 'completions/mean_length': '753.2'}
{'loss': '0.1257',  'grad_norm': '1.042', 'completions/mean_length': '718.6'}
```

With `max_completion_length = 1280` and a mean around 735 tokens, the upper tail
of Math completions is being truncated often enough to breach the gate. A
truncated completion cannot produce a well-formed `\boxed{}` answer, so it
registers as incorrect regardless of the reasoning that preceded it — the
clipping directly contaminates the reward signal the experiment depends on.

Note the gradients were healthy (`grad_norm` 0.714 and 1.042, both nonzero) and
loss was finite, so this is specifically a truncation problem, not the v8
zero-variance problem returning.

**Phase-0 gates 3 and 4 of the v9 amendment therefore fail.** v9 Phase 0 does
not pass, and v9 is not eligible for promotion to formal Stage A.

---

## 4. Feasibility numbers this run establishes

**Seconds per update:** `step_times = [59.481, 59.506]` → **59.49 s/update**.
Extrapolating the registered 200-update Stage-A budget:

```
200 x 59.49 s = 11,898 s = 3.31 h
```

That fits inside a single Colab session with wide margin (the probe's own
thresholds were >12 h "risky", >20 h "refuse"). Caveat carried from the probe:
the smoke runs only 2 updates including warm-up, so this extrapolation has wide
error bars.

**VRAM:** group 8 does **not** fit a 22 GiB L4. An earlier run of this same
notebook on an L4 raised:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 5.80 GiB.
GPU 0 has a total capacity of 22.03 GiB of which 791.12 MiB is free.
```

so the requirement is roughly 27 GiB. The 5.80 GiB single allocation is
accounted for exactly by the training logits tensor:

```
per_device_train_batch_size(8) x (max_completion_length 1280 + 1)
  x vocab(151936) x 4 bytes (fp32)  =  5.80 GiB
```

It runs comfortably on the 80 GiB A100.

**Known gap — the exact peak was not measured.** The report printed
`[MEM] no telemetry: [Errno 2] No such file or directory: '/content/gpu_probe.csv'`.
The probe notebook's Phase-0 cell has lost the `nvidia-smi` sampler thread it
originally carried (no `import csv`/`threading`, no `GPU_CSV`, no `_sample_gpu`),
so no telemetry was written. Only a bound is established, not a number. See §6.

---

## 5. Why this cannot be fixed inside v9

Reducing clipping means raising `max_completion_length`. Two independent
constraints forbid doing that under v9:

1. **The runner refuses it.** `validate_contract` in `run_exp2_4070_v9.py`
   permits only `per_device_train_batch_size` and `num_generations` to differ
   from v8, and then hard-requires both to equal 8:

   ```python
   if _without(cfg["stage_a"], "per_device_train_batch_size", "num_generations") != \
           _without(v8["stage_a"], "per_device_train_batch_size", "num_generations"):
       raise RuntimeError("v9 changes Stage-A fields beyond linked group geometry")
   if cfg["stage_a"]["num_generations"] != 8 or \
           cfg["stage_a"]["per_device_train_batch_size"] != 8:
       raise RuntimeError("v9 Stage-A group and device batch must both be 8")
   ```

   Any edit to `max_completion_length` aborts at `contract` before any GPU work.
   This was verified empirically: an attempt to change the batch geometry in an
   earlier run stopped at `contract` in 0.3 min with exactly this error.

2. **The v9 amendment forbids it in words.**
   [`EXPERIMENT_2_4070_INSTRUCT_V9_AMENDMENT.md`](EXPERIMENT_2_4070_INSTRUCT_V9_AMENDMENT.md)
   states: *"Do not reduce the group, shorten completions, change the reward, or
   move another scientific variable to make the smoke pass."* Lengthening
   completions to clear the gate is the same class of move.

Continuing therefore requires a **new v10 variant with its own amendment**,
which is a change to the pre-registered design and needs team sign-off. A draft
is at [`EXPERIMENT_2_4070_INSTRUCT_V10_AMENDMENT_DRAFT.md`](EXPERIMENT_2_4070_INSTRUCT_V10_AMENDMENT_DRAFT.md).
No v10 config has been created.

---

## 6. Deviations and caveats to disclose

- **Hardware.** The v9 amendment anticipated an L4 if group 8 did not fit the
  4070, "only after the hardware move is approved". This run used a Colab A100
  80 GB, which was not the anticipated device and was not pre-approved. It was
  a feasibility probe, not a formal Stage-A launch, and no result here is
  offered as a Stage-A outcome — but the deviation is recorded here rather than
  left implicit.
- **The config still declares a 4070.** `execution.device` reads
  `"RTX 4070 Laptop 8GB"`. `execution` is a frozen field that must equal v8, so
  it was correctly left untouched — but it means the config-of-record claims a
  4070 while the run happened on an A100. Anyone tracing provenance from
  `config_sha256` alone would be misled. Read this file alongside it.
- **Generation is the bottleneck, and the A100 is underused.** Each update
  generates 8 unique prompts x 8 generations = 64 completions averaging ~735
  tokens, i.e. ~47k tokens in 59.5 s ≈ **790 tokens/s** — modest for a 0.5B
  model on an A100. Generation runs through HuggingFace `generate()`
  (no vLLM: the config has no vLLM/generation-backend settings), so decoding is
  latency-bound and the device idles. A paged-attention backend would plausibly
  cut Stage-A wall time several-fold. **This is not a free speedup**: changing
  the generation backend changes the sampling implementation and could make
  results incomparable to the 4070 track, so it is a scientific-variable
  decision, not a tuning knob. Recorded here as an option, deliberately kept
  out of the v10 draft.
- **`gradient_checkpointing` was already enabled** (`execution` block), so the
  ≈27 GiB requirement is already a compute-for-memory-traded figure, not a naive
  one.
- **Missing VRAM telemetry.** As in §4. Before any rerun, restore the sampler
  thread in the probe's Phase-0 cell, or better, record
  `torch.cuda.max_memory_allocated()` from inside the training process — the
  5-second `nvidia-smi` polling in the original sampler is too coarse to catch
  a backward-pass spike anyway.
- **Ephemeral artifacts.** Everything under `/content` (smoke outputs, logs,
  `smoke_gate_failure.json`) lived in the Colab runtime and is gone; the numbers
  above were transcribed from the notebook output before disconnect. The
  notebook's saved outputs in Drive are the only surviving primary record.
- **Compute accounting.** Recorded in `eaaj-pilot/compute_log.md`. The
  compute-unit delta still needs to be filled in by hand.

---

## 7. Claim boundary

This finding establishes three things and nothing more: that group 8 restores
within-group reward variance reproducibly across hardware; that the v9 recipe
needs roughly 27 GiB and runs at about 59.5 s/update, implying a 3.3 h Stage A;
and that v9's Stage-A smoke breaches its own completion-clipping gate at
`max_completion_length = 1280`. It is not a Stage-A result, not a plasticity
measurement, and not evidence for or against the project's RQ1.
