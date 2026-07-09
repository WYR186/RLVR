# M3 Max local experiment plan and feasibility report

Date: 2026-07-07  
Scope: first pilot only — Qwen2.5-0.5B, GSM8K GRPO checkpoints, Q measurement,
and fixed-budget SVAMP adaptation.

## Decision

**Go on the local CPU path.** This M3 Max has enough memory and disk for the
complete Phase 1–4 pilot. The expected wall time is roughly **40–45 hours**,
dominated by autoregressive rollout generation. Run Phase 1 and each Phase-3
checkpoint as resumable overnight jobs.

Do **not** use PyTorch MPS for the scientific run on this machine today —
but for a different reason than first recorded. **Correction (2026-07-08):**
the earlier claim that MPS "cannot enumerate the GPU on macOS 26.5.1" did not
reproduce; `torch.backends.mps.is_available()` returns True in this exact venv
and allocation/training both work (the earlier probe likely ran in a
restricted environment). MPS was then benchmarked end to end on the real
workload and measured at **parity with CPU** (~265–320 s/update after fixes,
vs ~300 s/update CPU) — see the "2026-07-08 MPS investigation" section below
for the data and root cause. MLX remains unused because porting GRPO would
change the training implementation and weaken comparability with the
pre-registered TRL/Colab recipe.

## Machine audit

| Item | Measured value | Implication |
|---|---:|---|
| Chip | Apple M3 Max, 16 CPU / 40 GPU cores | CPU vector math is strong; GPU currently inaccessible to PyTorch |
| Unified memory | 128 GiB | comfortably exceeds the 31.0 GiB measured peak |
| Free disk | 2.6 TiB | enough for five fp32 checkpoints plus resumable optimizer states |
| OS | macOS 26.5.1 | triggers current PyTorch MPS enumeration bug |
| Runtime | Python 3.13.9 arm64, PyTorch 2.12, TRL 1.6, Transformers 5.13 | one-step real GRPO contract verified |

Raw benchmark evidence is in `outputs/local_benchmarks/`.

## Measured feasibility, not a paper estimate

All GRPO measurements below used the real Qwen2.5-0.5B model, the frozen GSM8K
data, exact-answer reward, 8 generations, effective 64-completion update, full
parameter gradients, and AdamW. Gradient checkpointing was disabled because
128 GiB makes recomputation unnecessary.

| Test | Wall time | Peak RSS | Mean completion | Clip rate | Reward mean |
|---|---:|---:|---:|---:|---:|
| GRPO step, max completion 64 | 37.25 s | 11.73 GiB | 62.7 | 92.2% | 0.0469 |
| GRPO step, max completion 512 | 300.47 s | 31.02 GiB | 179.6 | 1.6% | 0.3125 |

The 512-token result is the load-bearing check:

- no OOM or unsupported CPU operation;
- only 25% of physical memory used at peak;
- base-model exact reward is non-zero and seven of eight prompt groups have
  reward variance, so GRPO has a usable learning signal;
- 512 tokens is justified: the 64-token shortcut clips 92% of generations and
  materially changes reward, so it is not an acceptable formal-run substitute.

## Scientific execution contract

The local run preserves model revision, frozen splits, seed, optimizer/LR,
rollout count, temperature/top-p, β=0, 200 updates, checkpoint steps, 512-token
completion cap, Q layers, and the five identical 50-update SVAMP probes.

One logged deviation is unavoidable:

> Local execution uses full-parameter **float32 CPU** instead of suggested bf16
> CUDA because PyTorch MPS is unavailable on macOS 26. All checkpoints in this
> run use the same backend and dtype; Q is measured in float32. A later Colab
> bf16 run must be treated as a separate backend stratum, never merged silently.

Disabling gradient checkpointing is an execution optimization, not a change to
the objective or update budget. It is logged in the run config and manifest.

## Time and storage budget

| Phase | Local estimate | Basis |
|---|---:|---|
| 1 — 200 GSM8K updates | 16.7 h training + periodic eval/checkpoint overhead; budget 18–20 h | 300.47 s measured per full update |
| 2 — five Q measurements + two 2048 sensitivity checks | 15–30 min | prompt-only forward passes, no generation |
| 3 — 5 × 50 SVAMP updates | 20.9 h training + eval overhead; budget 23–25 h | same rollout/update geometry as Phase 1 |
| 4 — tables and plots | <2 min | CPU dataframe/Spearman/PNG work |
| Total | **approximately 42–45 h** | allow ±20% for thermals and task length |

Expected persistent storage is below 60 GiB even when keeping resumable trainer
state. The 2.6 TiB free disk leaves ample headroom.

## Reliability strategy

- Run on AC power with `caffeinate -dimsu`.
- Scientific weights are saved at 0/25/50/100/200 exactly.
- Trainer checkpoints include optimizer/scheduler state and keep the latest two,
  allowing a long CPU job to resume without restarting the budget.
- JSONL dashboard and eval logs append online; no whole-run normalization.
- Every run records package versions, hardware, config hash, and source/data
  hashes. Local wall time and peak RSS go into the local compute log.
- Never mix a partial retry into a fresh adaptation output directory unless it
  resumes from a valid trainer checkpoint.

## Stop / migrate-to-Colab conditions

Pause the local run and move the same config to Colab if any occurs:

1. three consecutive updates exceed 8 minutes each;
2. resident memory exceeds 96 GiB or the system begins swapping materially;
3. NaN/Inf loss or gradient norm appears;
4. reward has zero within-group variance for five consecutive updates;
5. 512-token clip ratio exceeds 10% for five consecutive updates;
6. a resume-integrity check cannot prove optimizer step continuity.

Thermal slowdown alone is not a scientific failure; it is a scheduling reason
to migrate. Colab remains the fallback, not the default for this first attempt.

## Run order

From `eaaj-pilot/`:

```bash
.venv/bin/python -m pytest tests/ -v
.venv/bin/python scripts/smoke_test_grpo.py

caffeinate -dimsu .venv/bin/python scripts/run_local_pipeline.py --phase 1
caffeinate -dimsu .venv/bin/python scripts/run_local_pipeline.py --phase 2
caffeinate -dimsu .venv/bin/python scripts/run_local_pipeline.py --phase 3
.venv/bin/python scripts/run_local_pipeline.py --phase 4
```

The runner is idempotent at completed checkpoints and resumes valid long-running
trainer state. `outputs/ACTIVE_RUN.txt` selects the same run for all four phases.

## Result interpretation

The valid claim remains narrow: whether Q at a GSM8K RLVR checkpoint predicts
**fixed-budget SVAMP adaptability relative to checkpoint 0**. Local compute does
not authorize the broader claim that RLVR reduces a model's general ability to
learn.

## 2026-07-08 MPS investigation (measured; outcome: not adopted)

Trigger: the Phase-1 CPU run was interrupted at step ~67/200 (~300 s/update)
and the wall-clock was deemed too slow. MPS was re-probed and found available
(contradicting the earlier note, corrected above), so it was benchmarked on
the real workload before any switch.

**Standalone benchmarks looked excellent** (Qwen2.5-0.5B fp32, the real
64-completion geometry, M3 Max 40-core GPU):

| Path | CPU (measured in run) | MPS standalone |
|---|---:|---:|
| 64-sequence sampled generation | ~53 tok/s | 341 tok/s (fp32) / 414 (bf16) |
| fwd+bwd+AdamW, one 64-completion update | ~35–40 s | 23–38 s |

**But the real TRL GRPOTrainer update did not inherit the win:**

| Configuration | s/update (real trainer, measured) |
|---|---:|
| CPU float32 (production run, 67 steps) | ~300 |
| MPS stock TRL 1.6 | 335–535 |
| MPS + both fixes below | 265–322 (5 updates across 3 runs) |

Root causes found by section timing and `torch.profiler`:

1. `trl.trainer.utils.selective_log_softmax` (fp32 branch) computes a
   row-looped `torch.logsumexp` whose reduction kernel is pathologically slow
   on MPS: **~95 s forward + most of ~128 s backward per update**, while TRL's
   own chunked `entropy_from_logits` handles the same logits in 1.7 s.
   → Fixed by `src/mps_compat.py::chunked_selective_log_softmax`
   (mathematically identical `log_softmax+gather` in row chunks; equivalence
   incl. gradients unit-tested in `tests/test_mps_compat.py`). Worth ~60
   s/update on MPS. Applied automatically only when `--backend mps`.
2. The remaining gap is **host↔device synchronization latency**:
   `aten::_local_scalar_dense` = 143 s (54%) over 5,314 calls in one profiled
   update — ~10 scalar syncs per decode token inside the transformers
   generation loop plus TRL's per-micro-batch metric reads. This is
   structural to the transformers/TRL stack on MPS; removing it would mean
   rewriting the decode loop, which is exactly the comparability risk this
   plan already rejects (see MLX note). Shape-recompilation and
   `disable_compile` were tested and ruled out; `PYTORCH_ENABLE_MPS_FALLBACK`
   costs ~90 s/update and is now left unset (no op needed it).

**Decision: stay off MPS for scientific runs** — net parity with CPU does not
justify a new backend stratum. The runner nevertheless gained a validated
`--backend mps` profile (new stratum dir `local_mps_grpo_gsm8k_*`, kernel
patches recorded in its config/manifest) in case a future torch release fixes
the sync overhead; the CPU profile and its run-dir hash are byte-identical to
before, so `--backend cpu` still resumes `local_grpo_gsm8k_eac028bfcc87`
from trainer checkpoint-50.

**Fast-path recommendation:** the pre-registered Colab CUDA recipe (notebooks
01–03, A100/L4 bf16) — projected 30–90 s/update, i.e. Phase 1 ≈ 2–5 h and
Phase 3 ≈ 3–5 h, well inside the ~300-unit pair budget. The local CPU run can
either be resumed to completion as the cpu-fp32 stratum (~12.5 h Phase 1
remaining + ~21 h Phase 3) or parked; its artifacts stay valid either way.

## 2026-07-08 Windows RTX 4070 Laptop stratum (designed, not yet run)

A third execution stratum was designed for Aaron's Windows 4070 laptop:
`--backend cuda` — bfloat16 weights + bf16 autocast (exactly the notebook-01
Colab recipe) plus gradient checkpointing as the execution optimization that
fits GRPO logits/activations in 8 GiB VRAM. Pre-registered run dir:
`outputs/local_cuda_grpo_gsm8k_c1ea6e11b8ca`. Full plan, VRAM budget,
environment setup, preflight probes, runbook, and failure playbook:
`../eaaj-pilot-win4070/WIN4070_EXPERIMENT_PLAN.md`.

Invariants preserved by the change (re-verified after editing the runner):
the cpu profile dict stayed byte-identical, so `--backend cpu` still resumes
`local_grpo_gsm8k_eac028bfcc87`; all 45 unit tests pass. Shared-code fixes
that ride along are Windows-portability only and behavior-preserving on
macOS/Linux: optional `resource` import (POSIX-only module) with a psapi
fallback for peak-RSS, a runner-lock pid probe that no longer uses
`os.kill(pid, 0)` on Windows (it terminates the target process there), and
non-reentrant `gradient_checkpointing_kwargs` wherever checkpointing is
enabled (inert for cpu/mps, which keep it off). `outputs/ACTIVE_RUN.txt` is
now machine-local (untracked) so the two machines cannot fight over the
pointer.
