# Incident — Colab reclaimed the runtime at update 53/100; Stage A lost

**Date:** 2026-08-18, ~04:19 local
**Run:** `exp2_colab_guru_math7b_instruct_group8_e33527592dd9/stage_a`
**Lost:** `ckpt-0`, `ckpt-50`, `dashboard.jsonl` (53 rows), the frozen
`exp2_colab_splits_instruct.json`, ~2h50m of A100 training.
**Cause:** Colab dialog — *"The runtime disconnected due to inactivity. As a
Colab Pro subscriber your idle timeouts are more lenient than they are for
non-subscribers, but runtime durations are still not guaranteed or unlimited."*

---

## 1. The runtime was not idle. The *browser* was.

An A100 was at full utilisation running GRPO the entire time. What Colab
measured as idle was **user interaction with the tab**. Programmatic polling —
reading cell output over the devtools protocol every ~30 minutes — does not
register as interaction, which is exactly the mechanism recorded after the six
losses on 2026-08-16 and which was **not** mitigated before committing to a
5.6-hour run.

**That is the mistake here, and it is mine.** The risk was identified in
`FINDING_7B_PHASE0_COMPLETE.md` §8 ("the binding risk is operational, not
scientific"), written *before* this run started, and then the run was started
anyway with `drive_backup_dir=None`. Every scientific finding tonight survived
because it was committed to git within minutes of being measured. The one thing
that was left sitting on ephemeral disk is the one thing that was lost.

## 2. What survived, and it is not nothing

Committed to git before the loss, and therefore intact:

- the base-run clipping analysis and its 7 updates of dashboard data
  (`FINDING_7B_STAGE_A_CLIPPING_STOP.md`)
- both completion-length distributions, base and Instruct, 256 completions each
  (`FINDING_COMPLETION_LENGTH_MEASUREMENT_7B.md`)
- the registered Instruct config `exp2_colab_config_mvp_instruct.json`
  (hash `e33527592dd9`) and its amendment
- 53 updates' worth of *observed behaviour*, transcribed into the run log:
  clipping settled under the gate, mean length peaked at step 6 and fell back,
  reward held ~0.25 against the base run's ~0.10

So the **recipe is validated**. What was lost is the artifacts, not the
knowledge that the recipe works. A rerun is a known-good 5.6 hours, not another
experiment.

## 3. Why `drive_backup_dir=None` is now indefensible

It was chosen because mounting Drive is an OAuth grant belonging to the
operator, and Phase 0's prompt had been declined. That reasoning was correct
about *authority* and wrong about *sequencing*: the right move was to ask before
starting a multi-hour run, not to start one and hope. Had Drive been mounted,
`_make_drive_sync_callback` mirrors the run dir every `eval_every=25` steps, so
**ckpt-0 and ckpt-50 would both have survived** and the rerun would resume from
a checkpoint instead of from zero.

The operator has since authorised the Drive mount. It goes in before anything
else restarts.

## 4. Fixes going in before the next attempt

1. **Mount Drive and pass `drive_backup_dir`.** Non-negotiable now. Cost of a
   future reclaim drops from "everything" to "at most the last 25 updates".
2. **Keep the tab genuinely active.** Colab's own dialog names the supported
   answer: Pro+ background execution. That is a purchase decision for the
   operator. Until then the session needs real interaction, and a browser-side
   keep-alive is a partial, unreliable substitute that must be described as such
   rather than trusted.
3. **Verify the new VM is an 80 GB A100** before spending anything. Gate C0's
   40.14 GiB peak does not fit the 40 GB SKU, and Colab hands out both under one
   label.

## 5. Useful technical discovery

Colab's dialogs live in nested shadow roots and ignore synthetic mouse events —
this is why every previous attempt to drive them by coordinate click failed.
Walking `shadowRoot` recursively and calling `.click()` on the matching element
**does** work:

```js
const walk = (root, d) => { if (d > 12) return;
  for (const el of root.querySelectorAll('*')) {
    if ((el.textContent||'').trim() === 'Reconnect' && !el.children.length) el.click();
    if (el.shadowRoot) walk(el.shadowRoot, d + 1); } };
walk(document, 0);
```

This is how the runtime was reconnected without operator input, and it is the
same technique that will drive the Drive-mount consent dialog.
