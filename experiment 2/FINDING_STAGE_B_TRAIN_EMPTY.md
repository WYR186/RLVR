# Finding — the probe consumed the entire Stage-B pool, leaving `stage_b_train` empty

**Status:** blocking bug, fixed 2026-08-16 as an operator decision per the
cheaper-default rule (implement, log, flag) — **flagged to Tommy, not silently
decided**. §5 is the question he should confirm after the fact.
**Observed on:** Colab A100-SXM4-80GB (High-RAM), `exp2_colab_config_mvp.json`,
Qwen2.5-7B base, `experiment 2/colab/00_phase0_selfcontained.ipynb`
**Affects:** the Colab/7B track only. The WIN4070 track's splits are correct.

---

## 1. What the run printed

```
GATE 0a: PASS on the eligible population (p95=628.0 <= 1024; raw p95=1407.0 recorded above, not gated)

stage_a_train: 54257 | stage_b_train: 0 | stage_b_eval: 300 | probe: 4096 / 4096
```

`stage_b_train: 0`. Stage B is the adaptation stage; with no training rows the
fixed-budget ΔR curve — the MVP's entire primary deliverable — cannot be
produced at any checkpoint.

## 2. Cause: allocation order in `build_exp2_splits`

`src/guru_data.py` allocated in the order eval → **probe** → train:

```python
eval_ids  = rng.sample(sim_ids, stage_b_eval_questions)   # 300
remaining_sim_ids = [i for i in sim_ids if i not in eval_set]   # 1432 - 300 = 1132
probe_ids = remaining_sim_ids[:n_probe]                   # n_probe = 4096 -> takes ALL 1132
train_ids = [i for i in remaining_sim_ids if i not in set(probe_ids)]   # -> 0
```

`n_probe` (4096) is larger than the eligible CodeIO pool, so the slice
`[:n_probe]` is not a partial take — it swallows the pool whole and leaves the
train list empty. Nothing errors; the run continues with an empty training set.

## 3. It contradicts the config's own stated intent

`measurement.probe_source`, in both configs:

> "frozen at Phase 0 …: stage-B pool rows **disjoint from stage-B train and
> eval**, topped up from held-out stage-A rows if the CodeI/O pool is too
> small"

"Disjoint from stage-B train" presupposes a stage-B train set to be disjoint
*from*, i.e. train is carved out first and the probe shortfall is made up from
math. The code inverted that.

## 4. The WIN4070 track does it correctly

Its v9 `prepare` output:

```json
"stage_b": { "eligible": 1432, "train": 1132, "eval": 300,
             "max_prompt_tokens": 640 }
```

Same eligible pool, same eval count, and train gets the remaining 1132. The two
tracks' split builders had silently diverged; the Colab one was wrong.

## 5. Fix, and what it costs

Train is now allocated before the probe, and the probe takes its shortfall from
the math top-up path that already existed. Verified offline against the real
eligible counts:

```
eligible sim 1432, eval 300, remaining 1132
OLD: stage_b_train=    0  probe_sim=1132   -> DELTA-R IMPOSSIBLE
NEW: stage_b_train= 1132  probe_sim=0  probe_math_topup=4096  probe_total=4096
```

This reproduces the WIN4070 reference exactly and the probe still reaches its
full 4096.

**Cost, stated plainly:** the probe changes from ~28% CodeIO / ~72% math to
100% math. It was always going to be majority math at `n_probe=4096` against a
1132-row pool, so this is a shift in an already-math-dominated probe, not a new
kind of probe. **Tommy should confirm** this is the intended reading, since the
probe defines what Q (effective rank, dormant fraction) is measured *on*. The
alternative — keeping probe composition and losing stage-B training — makes the
experiment unrunnable, so it is not a real alternative.

`measurement.probe_questions` (4096) is unchanged, and the config's reason for
that number (Qwen2.5-7B hidden dim 3584; a smaller probe makes effective rank
sample-truncated) still holds.

## 6. Everything else in the same run

- **GATE 0a passed** on the eligible population (p95 = 628 ≤ 1024), with the
  raw p95 = 1407 still computed and cross-checked against the WIN4070
  reference. The fix in `FINDING_GATE_0A_MEASURES_THE_WRONG_POPULATION.md`
  behaves exactly as predicted.
- **Gate C0 passed** — the first real 7B + LoRA + group-8 memory measurement
  anywhere in this project:

  ```
  peak_allocated_gib : 40.14
  total_device_gib   : 79.25
  headroom_pct       : 49.35
  gate_pass          : True
  Step 1 loss -0.007958 | Step 2 loss -0.049697
  ```

  **40.14 GiB peak means a 40 GB A100 would not fit either** — this recipe
  needs the 80 GB tier. It also retroactively confirms the L4-skip deviation:
  22 GiB was never going to work.
- The run then died at the GATE 0b preflight cell when Colab reclaimed the
  runtime for inactivity. GATE 0b and the 2-update smoke are still unrun.
