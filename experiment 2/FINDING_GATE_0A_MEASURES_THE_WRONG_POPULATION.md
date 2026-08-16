# Finding — GATE 0a is unsatisfiable by its own remedy, and audits a population that is never trained on

**Status:** blocking specification defect. Both exp2 tracks hit it. Needs a team
decision; deliberately **not** worked around.
**Date:** 2026-08-16
**Observed on:** Colab A100-SXM4-80GB, `exp2_colab_config_mvp.json`,
Qwen2.5-7B base, via `experiment 2/colab/00_phase0_selfcontained.ipynb`
**Prior occurrence:** `data/token_length_audit.json` (WIN4070 track, Qwen2.5-0.5B)

---

## 1. What happened

The 7B Phase-0 token audit stopped on the pre-registered gate:

```
Math (stage A)      : n 54404, p50 110,  p95 217,  p99 333,     max 3325
Simulation (stage B): n 3730,  p50 700,  p95 1407, p99 1746.13, max 1949

SystemExit: GATE 0a STOP: stage-B p95=1407.0 > 1024.
            Escalate GPU tier - do not shrink the batch to force a fit.
```

## 2. It reproduced the WIN4070 audit exactly

`data/token_length_audit.json`, produced months earlier on the 0.5B track:

| | WIN4070 (Qwen2.5-0.5B) | today (Qwen2.5-7B) |
|---|---|---|
| stage A | n 54404, p50 110, p95 217, p99 333, max 3325 | identical |
| stage B | n 3730, p50 700, p95 1407, p99 1747, max 1949 | identical |

That file already records:

```json
"gate_0a": {
  "threshold_stage_b_p95_tokens_max": 1024,
  "observed_stage_b_p95_tokens": 1407,
  "status": "STOP",
  "required_action": "preserve diagnostics and request the L4 24 GB stratum;
                      do not shrink batch, truncate prompts, or start training
                      on the RTX 4070"
}
```

Two things follow. First, this is **not a new failure** — GATE 0a has been in a
STOP state since the WIN4070 audit. Second, the reproduction is exact across a
different model, tokenizer instance, and machine, which is real cross-track
verification that the loader and data contract are correct. The loader is fine.
The gate is the problem.

## 3. The gate's own remedy cannot fix the gate

`required_action` is "escalate GPU tier". But `p95 = 1407` is a property of the
**dataset**, not of the GPU. Tokenizing the same 3730 CodeIO prompts on a bigger
card produces the same 1407.

Today's run escalated all the way to an **A100 80 GB — the largest tier
available** — and the gate fired with the byte-identical number. There is no
hardware on which this gate passes. **As specified, GATE 0a is permanently
unsatisfiable**, and every track that reaches it will stop forever.

## 4. Worse: it audits a population that is never trained on

Notebook 00 runs the audit at Step 4 over **all 3730** stage-B rows. Step 5 then
freezes splits with `stage_b_token_limit = CONFIG['stage_b']['token_filter_max']
= 640`, which discards every row above 640 tokens.

The v9 4070 `prepare` output confirms what survives:

```json
"stage_b": { "eligible": 1432, "train": 1132, "eval": 300,
             "max_prompt_tokens": 640 }
```

So 1432 of 3730 rows are eligible, and the **post-filter maximum is 640 tokens**.
The population that is actually trained and evaluated on therefore has
`p95 <= 640`, comfortably under the 1024 threshold — the gate would pass
trivially on it.

GATE 0a is checking whether the raw pool would fit, when the pipeline never
loads the raw pool. It is measuring the wrong thing.

## 5. The config's note about this gate is factually wrong

Both `exp2_colab_config.json` and `exp2_colab_config_mvp.json` carry:

> `phase0a_note`: "the real token audit (data/token_length_audit.json, WIN4070
> track) already found stage-B max_prompt_tokens=640 - this gate is expected to
> pass comfortably, confirmed rather than assumed"

The audit found no such thing. It found `p95 = 1407, max = 1949, status STOP`.
The number 640 is the config's own `stage_b.token_filter_max` — a filter
setting, not an audit result. Whoever wrote the note conflated the two, which is
why the gate was expected to pass and did not.

## 6. What this does NOT license

Nothing here justifies editing the gate to make the run proceed. The threshold,
the `token_filter_max`, and the "do not shrink batch, truncate prompts" language
are all pre-registered. The v9 amendment's rule — *"Do not reduce the group,
shorten completions, change the reward, or move another scientific variable to
make the smoke pass"* — applies with equal force to gates. **This run is stopped
and left stopped.**

## 7. What the team needs to decide

1. Should GATE 0a audit the **post-filter** stage-B population (the one that is
   trained on) instead of the raw pool? That is the reading under which the gate
   measures what it was clearly meant to measure, and under which it passes.
2. If the raw-pool reading is intended, what is the gate actually protecting
   against, given no GPU tier can satisfy it? If nothing, it should be retired
   rather than left as a permanent stop.
3. Either way `phase0a_note` must be corrected in both configs — it currently
   states an audit result that does not exist.

Until (1) or (2) is answered, **both exp2 tracks are blocked at Phase 0**: the
0.5B track on the v10 completion-clipping amendment
([`FINDING_V9_PHASE0_COLAB_A100.md`](FINDING_V9_PHASE0_COLAB_A100.md)), and the
7B track here.

## 8. Compute spent

Two short A100 sessions, both stopped as soon as their gate fired; no training
was reached on the 7B track. Recorded in `eaaj-pilot/compute_log.md`. The
runtime was disconnected immediately after the STOP rather than left idling.
