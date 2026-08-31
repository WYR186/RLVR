# Jason's 0.5B pilot — raw artifacts

**Author: Jason** (not this repo owner). Committed here for reproducibility and
because `CROSS_RUN_NOTE_7B_VS_05B.md` and
`../SPEC_E1_METRIC_REMEASUREMENT.md` cite numbers recomputed from these files.
Anything derived from this data and attributed to "this owner" elsewhere in the
repo is a recomputation *against* these artifacts, not original data collection.

Qwen2.5-0.5B-Instruct, GRPO on GSM8K (stage A), SVAMP learnability probe (stage
C) — the proposal's original pilot protocol, not Tommy's 2026-08-02 GURU spec
that `experiment 2/` otherwise implements. See `CROSS_RUN_NOTE_7B_VS_05B.md`
§"Protocol caveat" for why these are not a second arm of the same study.

## Contents

| Path | Contents | Original filename |
|---|---|---|
| `run1/` | `stageA_log_history.json`, `summary.csv` | `run 1 (1).zip` |
| `run2/` | `stageA_log_history.json`, `summary.csv` | `run 2 (1).zip` |
| `run3/` | `stageA_log_history.json`, `summary.csv` | `run 3  (1).zip` |
| `jason_three_run_summary.pdf` | Jason's write-up comparing all three runs | `three_run_summary (1).pdf` |
| `jason_run2_analysis.pdf` | Jason's write-up on run 2 specifically | `run2_analysis.pdf` |

Extracted from the zips as received on 2026-08-19; contents unmodified, only
unzipped and renamed for a clean path (no spaces / `(1)` suffixes / stray
double space in the original `run 3  (1).zip`).

## Known caveats — read before citing a number from here

- **Run 1's `svamp_improvement` column is not trustworthy.** The PDF write-up
  attributes the negative values partly to a greedy-eval truncation artifact
  that was fixed in runs 2–3, so the eval definition is not held constant
  across all three runs. Flag this wherever run 1's Δ-R appears.
- **`stageA_log_history.json` records `epoch` directly** — use that field, not
  a reconstruction from an assumed dataset size. `PROJECT_OVERVIEW_AND_NEXT_EXPERIMENTS.md`
  §2 documents a case where a reconstructed estimate (~3.1/7.0/1.7 epochs) was
  off by 2× from the logged values (1.5625/3.5156/0.8594), which the
  cross-run note's own §7.1 already corrected — §1 of that note is stale and
  should not be the source copied into the paper.
- **Single seed per run.** No error bars exist on anything computed from these
  three files alone; see `PROJECT_OVERVIEW_AND_NEXT_EXPERIMENTS.md` §5.2 (E2)
  for the proposed fix.
- Completion cap is 384 tokens in all three runs
  (`completions/max_length` saturates there exactly).

## What to ask Jason before writing Experiments setup from this

GSM8K train set size, `num_generations` (group size), per-device batch, SVAMP
train/eval split sizes, the stage-C adaptation budget, and the exact
`erank_mlp_mid` / `erank_resid_mid` layer indices and probe set — needed to
state whether these Q values are numerically comparable to the 7B track's
(`PROJECT_OVERVIEW_AND_NEXT_EXPERIMENTS.md` §7, item under "still open").
