# Result — zero-shot transfer T_t is FLAT within noise

**Date:** 2026-08-18
**Run:** `exp2_colab_guru_math7b_instruct_group8_e33527592dd9`
**Measurement:** `pipeline.run_transfer_T` — greedy exact-correctness on the
**300 frozen CodeIO (Simulation) eval questions**, at each Stage-A checkpoint,
no adaptation. Wall time 4425 s (~74 min).

---

## 1. The numbers

| checkpoint | Simulation accuracy | hits / 300 | T_t = score − M0 |
|---|---:|---:|---:|
| ckpt-0 (= M0, base Instruct) | 0.1733 | 52 | 0.0 |
| ckpt-50 | 0.1900 | 57 | **+0.0167** |
| ckpt-100 | 0.1867 | 56 | **+0.0133** |

## 2. Both differences are inside one standard error

Binomial SE at n=300 and p≈0.18 is **2.2 pp per point**. For a difference of two
proportions on 300 questions, SE ≈ 3.1 pp.

| comparison | Δ | in questions | Δ / SE |
|---|---:|---:|---:|
| ckpt-50 − ckpt-0 | +1.67 pp | **+5 questions** | **0.53** |
| ckpt-100 − ckpt-0 | +1.33 pp | **+4 questions** | **0.43** |

**Neither difference reaches half a standard error.** The honest statement is:
*100 updates of Math GRPO produced no detectable change in zero-shot Simulation
accuracy, in either direction, at n=300.*

The apparent non-monotonicity (ckpt-50 above ckpt-100) is one question of
difference. It is noise, and it must not be described as an inverted-U or as
"peak transfer at 50 updates".

## 3. Why this is good news for the experiment, not a null result to bury

The primary deliverable is **ΔR**, the *fixed-budget adaptation* curve
R_B(ckpt_t) − R_B(ckpt_0). T_t exists to make ΔR readable: if Math RL had
shifted the zero-shot starting point substantially, every ΔR difference would be
confounded by where each checkpoint started rather than by how well it *adapts*.

T_t being flat means **the three checkpoints start Stage B from statistically
the same place.** Any ΔR difference that shows up is then attributable to
adaptation dynamics, not to a head start. That is a cleaner experimental setup
than a large T_t would have given.

## 4. What this does NOT license

- It does **not** show Math RLVR "preserves" or "damages" transfer. At n=300 the
  95% CI on each point is ±4.3 pp; a real effect of a few points would be
  invisible here. The correct claim is *no detectable effect at this power*, not
  *no effect*.
- It says nothing yet about fixed-budget adaptability, which is what the project
  actually asks about (per the mentor-feedback framing recorded in CLAUDE.md:
  never claim "RLVR reduces the model's ability to learn"; the measurable
  question is fixed-budget adaptability on a held-out family).
- With three checkpoints and one seed, no correlation or trend statistic over
  the T_t points should be reported. The config's own `honest_limits` says this
  explicitly.

## 5. Power, stated up front so nobody has to rediscover it

At 300 eval questions, the smallest difference distinguishable from zero at 95%
confidence is roughly **±6 pp** (≈18 questions). Stage B's ΔR will be measured
on the same 300 questions and inherits exactly this resolution. If ΔR effects
are expected to be a few points, **n=300 is underpowered and the eval set is the
thing to grow**, not the update budget. That is a concrete, actionable finding
for the team, and it is available *before* the Stage-B compute is spent.
