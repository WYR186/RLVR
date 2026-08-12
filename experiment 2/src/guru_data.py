"""GURU (`LLM360/guru-RL-92k`) schema discovery, subset filtering, and frozen
splits for exp2 (Math -> Simulation).

Phase 0 rule (EXPERIMENT_2_PLAN.md / EXPERIMENT_2_COLAB_PLAN.md, both §3 Phase
0 step 1): do NOT assume column names. Every function here that touches the
raw dataset schema either discovers a field or requires the caller to have
already discovered and confirmed it — nothing here hardcodes a guessed field
name as if it were verified.

`audit_schema()` proposes a domain-field candidate by heuristic (scans a
handful of likely column names for values that look like the target subset
labels) and writes it to `data/guru_schema_audit.json` with
`domain_field_manually_confirmed: false`. An agent running Phase 0 must open
that file, verify the candidate against the actual printed examples, and set
the flag to true before anything downstream trusts it — `filter_stage_subset`
and `build_exp2_splits` both refuse to run against an unconfirmed audit.
"""
from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 42
GURU_DATASET_ID = "LLM360/guru-RL-92k"

# Stage 1 (Math) and stage 2 (Simulation) subset labels per Tommy's spec.
# Listed with the spelling variants a real release might use; audit_schema
# matches case-insensitively and substring-wise against whatever field it
# tries, so "DeepScaler" also catches "deepscaler_math" etc.
STAGE_A_SUBSET_NAMES = ("OR1", "DAPO", "DeepScaler")
STAGE_B_SUBSET_NAMES = ("CodeI/O", "CodeIO", "Code I/O", "codeio")

# Columns worth trying as the domain/subset identifier, in priority order.
# This is a search list, not a claim that any one of them is correct.
_CANDIDATE_DOMAIN_FIELDS = (
    "data_source", "source", "subset", "domain", "task", "dataset",
    "ability", "category", "task_source",
)

# Same idea for the prompt-text and gold-answer columns (Phase 0 step 3:
# answer-format determination). First present column wins as the *candidate*;
# the manual-confirmation gate covers these too.
_CANDIDATE_PROMPT_FIELDS = ("prompt", "question", "problem", "query", "input")
_CANDIDATE_ANSWER_FIELDS = (
    "answer", "solution", "ground_truth", "gt", "reward_model", "label", "output",
)


def _load_raw(revision: str | None = None):
    from datasets import load_dataset

    kwargs = {"revision": revision} if revision else {}
    return load_dataset(GURU_DATASET_ID, **kwargs)


def _column_looks_like_domain(dataset, column: str, target_names) -> tuple[bool, dict]:
    """Cheap heuristic: does this column's value set contain something that
    substring-matches (case-insensitive) any of `target_names`?

    Returns (matched, {name: count}) so the audit file records real evidence,
    not just a boolean guess.
    """
    if column not in dataset.column_names:
        return False, {}
    try:
        values = dataset[column]
    except Exception:
        return False, {}
    counts = {name: 0 for name in target_names}
    sample_cap = min(len(values), 20000)  # audit is a discovery step, not a full scan
    for v in values[:sample_cap]:
        v_str = str(v).lower()
        for name in target_names:
            if name.lower().replace("/", "").replace(" ", "") in v_str.replace("/", "").replace(" ", ""):
                counts[name] += 1
    matched = any(c > 0 for c in counts.values())
    return matched, counts


def audit_schema(revision: str | None = None) -> dict:
    """Inspect the real dataset and write data/guru_schema_audit.json.

    Writes, per split: column names, dtypes (via `Dataset.features`), row
    count. Proposes (does not assert) a domain_field candidate and one full
    example row per target subset, if a candidate field was found. Idempotent
    only in the sense that it always re-runs the audit; it never silently
    reuses a stale file, because the whole point is a fresh discovery pass.
    """
    ds = _load_raw(revision=revision)

    split_info = {}
    domain_field_candidate = None
    domain_field_evidence = {}
    prompt_field_candidate = None
    answer_field_candidate = None
    examples_by_subset = {}

    all_target_names = STAGE_A_SUBSET_NAMES + STAGE_B_SUBSET_NAMES

    for split_name, split_ds in ds.items():
        split_info[split_name] = {
            "n_rows": len(split_ds),
            "columns": list(split_ds.column_names),
            "features": {k: str(v) for k, v in split_ds.features.items()},
        }
        if prompt_field_candidate is None:
            prompt_field_candidate = next(
                (c for c in _CANDIDATE_PROMPT_FIELDS if c in split_ds.column_names), None)
        if answer_field_candidate is None:
            answer_field_candidate = next(
                (c for c in _CANDIDATE_ANSWER_FIELDS if c in split_ds.column_names), None)
        if domain_field_candidate is None:
            for col in _CANDIDATE_DOMAIN_FIELDS:
                matched, counts = _column_looks_like_domain(split_ds, col, all_target_names)
                if matched:
                    domain_field_candidate = col
                    domain_field_evidence = {"split": split_name, "match_counts": counts}
                    break

    if domain_field_candidate is not None:
        for split_name, split_ds in ds.items():
            if domain_field_candidate not in split_ds.column_names:
                continue
            for name in all_target_names:
                if name in examples_by_subset:
                    continue
                needle = name.lower().replace("/", "").replace(" ", "")
                for row in split_ds:
                    hay = str(row[domain_field_candidate]).lower().replace("/", "").replace(" ", "")
                    if needle in hay:
                        examples_by_subset[name] = {"split": split_name, "row": row}
                        break

    audit = {
        "dataset_id": GURU_DATASET_ID,
        "revision_requested": revision,
        "splits": split_info,
        "domain_field_candidate": domain_field_candidate,
        "domain_field_evidence": domain_field_evidence,
        "prompt_field_candidate": prompt_field_candidate,
        "answer_field_candidate": answer_field_candidate,
        "domain_field_manually_confirmed": False,
        "examples_by_subset": examples_by_subset,
        "stage_a_subset_names": list(STAGE_A_SUBSET_NAMES),
        "stage_b_subset_names": list(STAGE_B_SUBSET_NAMES),
        "note": (
            "domain/prompt/answer field candidates are heuristic proposals. "
            "An agent must open this file, inspect examples_by_subset against "
            "the actual guru-RL-92k documentation/paper, correct any wrong "
            "candidate, and set domain_field_manually_confirmed=true before "
            "filter_stage_subset or build_exp2_splits will run. The confirmed "
            "flag covers all three fields."
        ),
    }
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "guru_schema_audit.json").write_text(json.dumps(audit, indent=1, default=str))
    return audit


def load_confirmed_audit() -> dict:
    path = DATA_DIR / "guru_schema_audit.json"
    if not path.exists():
        raise RuntimeError("no guru_schema_audit.json — run audit_schema() first (Phase 0 step 1)")
    audit = json.loads(path.read_text())
    if not audit.get("domain_field_manually_confirmed"):
        raise RuntimeError(
            "guru_schema_audit.json exists but domain_field_manually_confirmed "
            "is false — inspect examples_by_subset and confirm before filtering "
            "(Phase 0 step 1: do not assume column names)")
    if not audit.get("domain_field_candidate"):
        raise RuntimeError(
            "no domain_field_candidate was found by the heuristic scan — "
            "identify the correct field manually, add it to the audit file, "
            "and set domain_field_manually_confirmed=true")
    return audit


def filter_stage_subset(split_ds, subset_names, domain_field: str | None = None):
    """Filter a loaded split down to rows whose domain field matches any of
    `subset_names`. `domain_field` defaults to the confirmed audit's field."""
    if domain_field is None:
        domain_field = load_confirmed_audit()["domain_field_candidate"]
    needles = [n.lower().replace("/", "").replace(" ", "") for n in subset_names]

    def _match(row):
        hay = str(row[domain_field]).lower().replace("/", "").replace(" ", "")
        return any(n in hay for n in needles)

    return split_ds.filter(_match)


def token_length_audit(prompts: list[str], tokenizer, label: str) -> dict:
    """p50/p95/p99/max token length of `prompts` under `tokenizer`. Does not
    write to disk itself — callers combine stage-A/stage-B results into one
    data/token_length_audit.json (Phase 0 step 4 / GATE 0a)."""
    import numpy as np

    lengths = [len(tokenizer(p)["input_ids"]) for p in prompts]
    arr = np.asarray(lengths)
    return {
        "label": label,
        "n": len(lengths),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()) if len(arr) else 0,
    }


def build_exp2_splits(n_train_a: int, n_train_b: int, n_eval_b: int,
                      n_probe: int = 4096, seed: int = SEED) -> dict:
    """Freeze stage-A train ids, stage-B train ids, stage-B held-out eval ids,
    and the frozen Q-metric probe set. Writes data/exp2_splits.json.

    Probe set (plan §1 "probe size" row: n_probe must exceed the 7B model's
    3584 hidden dim or effective-rank magnitudes are sample-truncated):
    drawn from stage-B pool rows disjoint from BOTH stage-B train and eval;
    if the pool can't supply n_probe such rows, topped up from held-out
    stage-A rows (disjoint from stage-A train). The top-up mixes domains,
    which is a logged property of the probe — same pattern as eaaj-pilot's
    probe-superset topping up from held-out GSM8K train (src/data.py) — and
    the per-source counts are recorded so the write-up can state it.

    Requires a confirmed schema audit (raises otherwise) — this function is
    the one Phase 0 step 5 refers to, and it must not run on an unverified
    field guess.
    """
    import random

    audit = load_confirmed_audit()
    domain_field = audit["domain_field_candidate"]
    ds = _load_raw(revision=audit.get("revision_requested"))
    rng = random.Random(seed)

    train_split = ds["train"] if "train" in ds else next(iter(ds.values()))
    stage_a = filter_stage_subset(train_split, STAGE_A_SUBSET_NAMES, domain_field)
    stage_b = filter_stage_subset(train_split, STAGE_B_SUBSET_NAMES, domain_field)

    a_idx = rng.sample(range(len(stage_a)), min(n_train_a, len(stage_a)))
    b_perm = list(range(len(stage_b)))
    rng.shuffle(b_perm)
    b_eval_idx = b_perm[:n_eval_b]
    b_train_idx = b_perm[n_eval_b:n_eval_b + n_train_b]
    probe_b_idx = b_perm[n_eval_b + n_train_b:n_eval_b + n_train_b + n_probe]

    need = n_probe - len(probe_b_idx)
    a_heldout = sorted(set(range(len(stage_a))) - set(a_idx))
    probe_a_topup_idx = rng.sample(a_heldout, min(need, len(a_heldout))) if need > 0 else []

    splits = {
        "seed": seed,
        "domain_field": domain_field,
        "prompt_field": audit.get("prompt_field_candidate"),
        "answer_field": audit.get("answer_field_candidate"),
        "dataset_revision": audit.get("revision_requested"),
        "stage_a_subset_names": list(STAGE_A_SUBSET_NAMES),
        "stage_b_subset_names": list(STAGE_B_SUBSET_NAMES),
        "stage_a_pool_size": len(stage_a),
        "stage_b_pool_size": len(stage_b),
        "stage_a_train_idx": sorted(a_idx),
        "stage_b_train_idx": sorted(b_train_idx),
        "stage_b_eval_idx": sorted(b_eval_idx),
        "probe_stage_b_idx": sorted(probe_b_idx),
        "probe_stage_a_topup_idx": sorted(probe_a_topup_idx),
        "probe_requested": n_probe,
        "probe_actual": len(probe_b_idx) + len(probe_a_topup_idx),
    }
    if splits["probe_actual"] < n_probe:
        splits["probe_shortfall_note"] = (
            f"pools could only supply {splits['probe_actual']} of {n_probe} "
            "probe prompts; effective-rank magnitudes may be sample-truncated "
            "— report n_probe alongside every erank value")
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "exp2_splits.json").write_text(json.dumps(splits, indent=1))
    return splits
