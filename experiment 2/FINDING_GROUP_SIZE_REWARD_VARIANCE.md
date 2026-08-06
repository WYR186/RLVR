# Finding — GRPO group size and reward-variance collapse on sparse binary Math rewards

**Status:** standalone measurement finding. Derived entirely from pre-registered
Phase-0 preflight artifacts; does **not** depend on any completed Stage-A run.
**Date:** 2026-08-05
**Model:** `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae557604adf67be50417f59c2c2f167def9a775`
**Seed:** 42 (both arms)
**Reward:** `exact_plus_boxed_format_0.1` — exact GURU score + 0.1 × non-empty balanced `\boxed{}`

---

## 1. Claim

On this Math population, GRPO group size 3 is **below the threshold at which a
sparse binary reward produces usable within-group variance**. Raising the group
to 8 measurably restores signal, but the majority of the recovered variance
comes from the 0.1 format-shaping term rather than from answer correctness.

Two numbers, on an identical set of 8 frozen prompts:

| | group = 3 | group = 8 |
|---|---|---|
| groups with **any** registered-reward variance | **1 / 8** | **4 / 8** |
| groups with **exact-correctness** variance | **1 / 8** | **2 / 8** |

A group with no within-group variance contributes exactly zero gradient under
GRPO, because the advantage is the within-group normalisation
`(r − mean) / std`. The first row is therefore a direct measure of how much of
each optimizer update is doing any work at all.

---

## 2. Why this comparison is matched

The two preflights draw their prompts with `random.sample(range(N), k)` under
the same seed 42. Because `random.sample` draws sequentially, the group-8
preflight's **16 prompts are a strict superset of the group-3 preflight's 8, in
the same order**. This was verified by ID, not assumed:

```
overlap: 8 / 8 v8 prompt ids also present in v9
c663deb5… 227defa5… 07558e4f… e558d0d4…
552c6988… 4b72c4c4… 45053810… 2b64d870…
```

The group-size comparison in §1 is therefore **paired on identical questions**,
with prompt difficulty held fixed. Both preflights run on the untrained base
model, before any optimizer step, so neither arm is contaminated by training.

---

## 3. Per-prompt result

`correct` counts exact-match answers out of the group; `var` marks within-group
variance of the registered (combined) reward; `exVar` marks variance of the
exact term alone.

| prompt | tokens | G3 correct | G8 correct | G3 var | G8 var | G3 exVar | G8 exVar | regime |
|---|---|---|---|---|---|---|---|---|
| `c663deb5` | 76 | 3/3 | 8/8 | – | – | – | – | saturated correct |
| `227defa5` | 165 | 0/3 | 0/8 | – | – | – | – | saturated wrong |
| `07558e4f` | 58 | 2/3 | 5/8 | ✓ | ✓ | ✓ | ✓ | intermediate |
| `e558d0d4` | 115 | 0/3 | 0/8 | – | – | – | – | saturated wrong |
| `552c6988` | 80 | 0/3 | 0/8 | – | – | – | – | saturated wrong |
| `4b72c4c4` | 79 | 0/3 | 0/8 | – | **✓** | – | – | format-only |
| `45053810` | 200 | 0/3 | 0/8 | – | **✓** | – | – | format-only |
| `2b64d870` | 108 | 0/3 | **2/8** | – | **✓** | – | **✓** | recovered |

Group 8 adds three variable groups. Their character differs:

- **`2b64d870` is the genuine recovery.** At G=3 the model produced three wrong
  answers; at G=8 two of eight were correct. This is signal that G=3 simply
  failed to sample — exactly the effect the group increase was meant to buy.
- **`4b72c4c4` and `45053810` are format noise.** Exact correctness stayed at
  zero in both arms. Their variance comes only from the 0.1 term: with more
  samples drawn, some completions failed to emit a well-formed `\boxed{}` and
  scored 0.0 instead of 0.1.

So of the +3 groups, **one is reasoning signal and two are formatting artefacts.**

---

## 4. The format term does not do what it was intended to do

The 0.1 boxed-format reward was added to lift the all-wrong floor off 0.0 and
create gradient. The preflights show it does not:

| | group 3 (24 completions) | group 8 (128 completions) |
|---|---|---|
| well-formed `\boxed{}` | 24 / 24 = **100 %** | 118 / 128 = **92.2 %** |
| exact correct | 5 / 24 = 20.8 % | 16 / 128 = 12.5 % |

At G=3 every sampled completion already satisfied the format, making the term a
**constant offset with exactly zero variance contribution**. It raised the floor
from 0.0 to 0.1 and produced no gradient whatsoever.

At G=8 it does contribute variance — but only by catching the ~8 % of
completions that fail to emit `\boxed{}`. That gradient trains formatting the
model already satisfies 92 % of the time, not reasoning. Larger groups surface
more format failures as a sampling side effect, so **part of the apparent gain
from G=8 is the shaping term converting sampling noise into gradient.**

This is why the preflight records exact, boxed, and combined variance
separately. Reporting only the combined count would have shown 1 → 4 and hidden
that the correctness channel moved only 1 → 2.

---

## 5. Corroboration from the terminated v8 Stage-A run

The preflight predicts collapse; the training run confirms it. v8 Stage A ran
110 of a registered 200 updates at group 3 before a pre-registered gate stopped
it on five consecutive zero-variance updates.

- **52 of 110 updates (47 %)** had `frac_reward_zero_std = 1.0`, and on every
  one of them `grad_norm` and `loss` were **exactly 0.0**. Nearly half the run
  performed no optimization.
- The rate rose monotonically with training:

  | steps | 1–20 | 21–40 | 41–60 | 61–80 | 81–100 | 101–110 |
  |---|---|---|---|---|---|---|
  | zero-gradient updates | 20 % | 40 % | 40 % | 70 % | 60 % | 60 % |

- Mean reward across the run was **0.1193** against a format floor of 0.100.
  Since each update samples 24 completions, that excess corresponds to **0.46
  correct answers per update — 1.93 % exact-match accuracy**, with no meaningful
  improvement over 110 updates.

The reward is quantised, which makes this legible: 51 of 110 updates landed on
exactly 0.100, i.e. all 24 completions well-formed and all 24 wrong.

### `reward_std` is the wrong thing to watch

Step 109 is worth isolating, because it is a trap for anyone monitoring this
kind of run:

```
step 109   reward 0.2250   reward_std 0.3378
           frac_reward_zero_std 1.0   grad_norm 0.0   loss 0.0
```

A batch-level standard deviation of 0.34 looks perfectly healthy. It is not:
every individual prompt group was internally constant, and the spread exists
only *between* groups — some groups uniformly right, others uniformly wrong.
GRPO normalises within groups, so between-group spread contributes nothing. The
gradient was exactly zero.

Across all 110 updates, **every one of the 52 with
`frac_reward_zero_std = 1.0` had `grad_norm` and `loss` exactly 0.0.** 51 of
those also had `reward_std = 0`; step 109 is the single case where the batch
statistic disagreed with the gradient. Rare, but it establishes the direction of
the implication — `reward_std = 0` always implies zero gradient here, while
`reward_std > 0` guarantees nothing.

The diagnostic that matters is therefore `frac_reward_zero_std`, not
`reward_std`. A dashboard tracking the latter would have shown a run that looked
like it was learning for 3 h 38 m while 47 % of its updates did nothing.

---

## 6. Mechanism

GRPO's advantage is normalised **within** a prompt group. A group whose members
all receive the same reward yields zero advantage and zero gradient, regardless
of how large the group is. Group size therefore only helps on prompts of
**intermediate** difficulty, where sampling can straddle correct and incorrect.

The §3 table shows all three regimes cleanly. Four of eight prompts are
saturated — one always-correct, three always-wrong — and stay dead at both group
sizes. Increasing the group cannot rescue them; only changing the prompt
population's difficulty distribution can.

With per-prompt accuracy `p`, a group of size `G` is dead with probability
`p^G + (1−p)^G`. At the observed `p ≈ 0.02–0.12`, `G = 3` leaves the large
majority of groups dead. This is a property of the population and the group
size, not a training bug.

---

## 7. Feasibility: group 8 does not fit on 8 GB

The group-8 configuration was executed on an RTX 4070 Laptop (8 GB). It cleared
the preflight, then **failed on the first training step's backward pass**:

```
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 5.80 GiB. GPU 0 has a total capacity of 8.00 GiB
of which 0 bytes is free.
```

Peak sampled utilisation was 7933 MiB / 8188 MiB (96.9 %), the same ceiling the
group-3 run sat at — group 3 fits there, group 8 does not. Zero of two smoke
updates completed.

Two engineering notes:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set in the config but
  **is not supported on Windows** (PyTorch emits `expandable_segments not
  supported on this platform`). 4.76 GiB was reserved-but-unallocated at the
  time of failure, so the documented fragmentation mitigation was inactive on
  precisely the platform that needed it.
- The registered amendment forbids reducing the group, shortening completions,
  or altering any other scientific variable to make this fit. The sanctioned
  response is a hardware move.

Group 8 at a 1280-token completion cap should be budgeted at **≥ 24 GB**.

---

## 8. What this finding does not show

- **It does not show that group 8 makes Stage A succeed.** No group-8 optimizer
  step has ever completed. Whether restored variance translates into learning is
  untested.
- **It does not establish an effect size.** n = 8 matched prompts, one seed. The
  1/8 → 4/8 and 1/8 → 2/8 movements are descriptive, not statistically powered.
- **It does not isolate group size from sample size.** The larger group draws
  more completions per prompt; some recovered variance is simply more sampling,
  which is the intended mechanism but is not separable here.
- **It says nothing about the original Experiment-2 claim.** This is an
  engineering-stratum measurement on a shaped reward, a 0.5B Instruct model, and
  a ≤512-token Math population.

---

## 9. Provenance

| artefact | path |
|---|---|
| group-3 preflight | `eaaj-pilot/outputs/exp2_4070_cuda_guru_math_c4a279960232/sparse_reward_preflight.json` |
| group-3 training curve | same directory, `dashboard.jsonl` (110 rows) |
| group-3 stop record | same directory, `safety_stop.json` |
| group-8 preflight | `experiment 2/smoke_outputs_4070_instruct_v9/stage_a/sparse_reward_preflight.json` |
| group-8 OOM transcript | `experiment 2/logs_4070_v9/run_20260805_210856_exp2_config_4070_instruct_v9_smoke.log` |
| group-8 GPU telemetry | `experiment 2/logs_4070_v9/gpu_20260805_210856_exp2_v9_smoke.csv` |

Config hashes — v8 `6b25dbbca876f65a081007e56a749977223f985edf955f7548eecb6e28d5e27c`,
v9 `cdee3eb399a6a9843997aaa32805c9b0b61916bbbd7edcbdab5b6d310208dafd`.

The two arms differ only in `num_generations` (3 → 8) and the linked
`per_device_train_batch_size` (3 → 8); both keep eight unique prompts per
optimizer update at accumulation 8. This is enforced mechanically by
`validate_contract` in `run_exp2_4070_v9.py`, which refuses to run if any other
training field, frozen gate, or `stage_b` value differs from v8.

### A note on the gate that let v8 through

v8's preflight gate was *"STOP if every sampled group has constant registered
reward"* — satisfied by a single variable group. v8's preflight recorded
`groups_with_reward_variance: 1` of 8 and passed. Seven of eight groups were
already dead three minutes into a run that went on to burn 3 h 38 m before
stopping itself.

v9 tightens this to require **≥ 2 of 16** groups with variable combined reward,
and records the exact and boxed channels separately. On the evidence here that
is still a weak floor: the group-8 arm passes it at 7/16 combined, but only 3/16
on the exact channel. A gate defined on the **exact** channel would be the
stronger instrument.
