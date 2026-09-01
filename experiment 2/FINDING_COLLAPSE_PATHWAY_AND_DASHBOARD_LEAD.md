# Result — the collapse pathway, and the lead time Q has to beat

**Date:** 2026-08-31
**Owner:** Aaron Wang (early-warning diagnostics)
**Cost:** zero. Every input is a `dashboard.jsonl` already committed.
**Driver:** `drivers/11_collapse_pathway.py` → `outputs/claim_d/`
**Scope:** claim (D) of `PROJECT_OVERVIEW_AND_NEXT_EXPERIMENTS.md` §3.2, plus the
baseline that §3.2 (C) says any lead-time claim for Q must clear.

## 1. Bottom line

The collapse pathway is confirmed, and — more usefully — one **free dashboard
signal already logged in every run gives a median 42 optimizer steps of warning
at a 2% false-alarm rate**. That is the bar. Q, per §3.2 (C), did not lead at
all.

| signal | detected | missed | median lead | lead range | false alarms |
|---|---:|---:|---:|---:|---:|
| `completions/clipped_ratio` ≥ 0.50 | **3/3** | **0** | **42** | 19–71 | **1/51** |
| `frac_reward_zero_std` ≥ 0.75 | 3/3 | 0 | 50 | 41–68 | **28/51** |
| `grad_norm` == 0.000 | 2/3 | 1 | 43 | 27–43 | 6/51 |
| `entropy` ≤ 50% of baseline | 2/3 | 1 | 64 | 3–64 | 5/51 |
| mean completion length ≥ 90% of cap | 2/3 | 1 | 28 | 3–28 | 1/51 |

**Clip saturation is the detector.** It is the only signal that catches every
collapse and stays quiet on healthy runs.

**`frac_reward_zero_std` is a trap.** It has perfect sensitivity and is the most
mechanistically appealing quantity in the whole pathway — and it fires on
**more than half of the runs that never collapsed**. Sensitivity without a
false-alarm rate is not detection. Reporting it as an early-warning signal on
the strength of the collapse runs alone would have been wrong.

## 2. The comparison is a clean natural experiment

Every 0.5B GRPO run in the repository uses `num_generations = 8`, `beta = 0`,
`max_completion_length = 512`. The only thing that differs is the learning rate,
and it separates the outcomes completely:

| learning rate | runs | collapsed |
|---|---:|---:|
| 1e-5 | 4 | **3** |
| 3e-6 | 1 | 0 |
| 1e-6 | 49 | 0 |

Collapse is defined once, before looking at any signal: reward falls to ≤25% of
its first-5-step baseline **and never returns above that line**. Requiring it to
stay down is what separates a collapse from a dip. The fourth 1e-5 run
(`a9dc95cbc2e8`) is correctly *not* scored as a collapse — it began at 0.234 and
ended at 0.094, never crossing its own threshold — and is scored as healthy,
which is the conservative direction for the false-alarm column.

## 3. The pathway, in order

`outputs/claim_d/claim_d_collapse_pathway.png` traces
`exp15_cuda_grpo_gsm8k_5ffdb56fc613` (lr 1e-5, collapse at step 79):

1. mean completion length climbs off a ~150-token floor from about step 45
2. `clipped_ratio` goes from ≈0 to 0.6 over steps 45–65
3. `frac_reward_zero_std` trends up toward 0.8–1.0
4. `grad_norm` decays from ~4 to ~1
5. reward falls through the same window

That is length explosion → clip saturation → loss of reward variance → gradient
starvation, in that order, as §3.2 (D) proposed.

## 4. The mechanism check, and where it fails

§3.2 (D) cites a critical group pass rate p\* ≈ 0.083 at `num_generations = 8`.
That reproduces exactly: solving `p^8 + (1-p)^8 = 0.5` gives **p\* = 0.0830**.

But the model that produces it does not fit. With group size `G` and a uniform
per-sample success probability `p`, the fraction of groups carrying no gradient
should be `p^G + (1-p)^G`. Using the logged mean reward as `p` and comparing
against the logged `frac_reward_zero_std`, the residual is **positive in all 54
runs**, by +0.17 to +0.44.

The observed degeneracy is far worse than independence predicts, and the reason
is that prompt difficulty is heterogeneous: a uniform-`p` model has no mass on
prompts that are all-pass or all-fail regardless of the policy, and real prompt
sets are full of them. **p\* ≈ 0.083 is therefore an optimistic bound.** Gradient
starvation arrives at a considerably higher mean reward than the uniform model
says, which makes the pathway easier to fall into, not harder.

## 5. What this does and does not license

It **does** license: a stated, measured baseline for early warning — 42 steps of
median lead at a 2% false-alarm rate, free, from a field TRL already logs. Any
claim that an activation metric provides early warning has to beat that, and
has to be scored the same way, on both collapsed and healthy runs.

It **does not** license a general claim about RLVR collapse. There are **three**
collapsed runs, all at one learning rate, one model, one task. The lead times
are a median over three numbers. The right home for the table is Method (how
early warning should be scored) plus a Results paragraph, with the n=3 stated in
the same breath.

It also does not say anything about plasticity. This is a training-stability
pathway, not a capacity measurement, and the two must not be blurred in the
write-up.

## 6. Reproduce

```bash
python "experiment 2/drivers/11_collapse_pathway.py" --out outputs/claim_d
```

No GPU, no model weights, no network. Thresholds are constants at the top of the
driver; they are applied identically to collapsed and healthy runs, and the
false-alarm column is reported whether or not it flatters the signal.
