"""GURU (`LLM360/guru-RL-92k`) loading for exp2's Colab variant, using the
**confirmed** contract rather than heuristic discovery.

An earlier version of this module discovered the schema by heuristic
(candidate field names, substring matching against subset labels) because
nothing about the real schema was known yet. That discovery has since
happened for real, on the WIN4070 track (`experiment 2/exp2_4070_data.py`,
`data/guru_schema_audit.json`, committed 2026-08-04/05) — the repository has
heterogeneous parquet schemas that `datasets==5.0.0` cannot cast into one
unified `Dataset`, so domain files must be loaded independently by exact
path; the prompt field is a chat-message list, not a string, and must go
through `tokenizer.apply_chat_template`; the answer field is the nested
`reward_model.ground_truth`, not a flat column. This module is adapted
directly from that proven loader (same file paths, same dataset revision, same
stable-ID scheme) so results stay comparable across the team's variants,
generalized to not require `local_files_only=True` (the WIN4070 machine has a
warm HF cache; a fresh Colab session does not) and to take the model id from
the caller's config instead of a hardcoded 0.5B pin.
"""
from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 42

GURU_DATASET_ID = "LLM360/guru-RL-92k"
# Pinned to the revision the WIN4070 track's real audit confirmed and froze
# splits against (data/guru_schema_audit.json, exp2_4070_splits.json).
GURU_DATASET_REVISION = "2e2790a962a3c099bfb5ea61389cbf98a5ea439b"
MATH_FILE = "train/math__combined_54.4k.parquet"
SIM_FILE = "train/simulation__codeio_3.7k.parquet"

# math__deepscaler_preview and math__merged_deduped_dapo_or1_dataset are the
# two data_source values covering OR1+DAPO+DeepScaler — the released parquet
# merges DAPO and OR1 into one value, so a per-origin split is not
# recoverable from the released fields (confirmed in guru_schema_audit.json).
STAGE_A_DATA_SOURCES = ("math__deepscaler_preview", "math__merged_deduped_dapo_or1_dataset")
STAGE_B_DATA_SOURCES = ("simulation__codeio",)


def _snapshot(dataset_revision: str = GURU_DATASET_REVISION) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        GURU_DATASET_ID, repo_type="dataset", revision=dataset_revision))


def _render_prompt(tokenizer, messages: list[dict], prompt_suffix: str | None = None) -> str:
    """Render a GURU `prompt` column (list of role/content messages) through
    the target model's own chat template. `prompt_suffix` optionally appends
    text to the final user turn (e.g. an explicit boxed-answer instruction) —
    a logged recipe choice, not a silent prompt change."""
    messages = deepcopy(messages)
    if prompt_suffix:
        user_messages = [m for m in messages if m.get("role") == "user"]
        if not user_messages or not isinstance(user_messages[-1].get("content"), str):
            raise RuntimeError("prompt suffix requires a final text user message")
        user_messages[-1]["content"] = (
            user_messages[-1]["content"].rstrip() + "\n\n" + prompt_suffix.strip())
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _records(path: Path, tokenizer, prompt_suffix: str | None = None) -> list[dict]:
    """Load one domain's parquet file into row dicts with a rendered prompt,
    token count, and the nested ground-truth/data-source fields flattened out."""
    import pyarrow.parquet as pq

    columns = ["data_source", "prompt", "reward_model", "extra_info"]
    table = pq.read_table(path, columns=columns)
    rows = table.to_pylist()
    texts = [_render_prompt(tokenizer, row["prompt"], prompt_suffix) for row in rows]

    lengths: list[int] = []
    for offset in range(0, len(texts), 256):
        encoded = tokenizer(texts[offset:offset + 256], add_special_tokens=False,
                            truncation=False)["input_ids"]
        lengths.extend(len(ids) for ids in encoded)

    result = []
    for row, text, tokens in zip(rows, texts, lengths, strict=True):
        extra = row["extra_info"] or {}
        raw_id = f"{row['data_source']}:{extra.get('index')}"
        stable_id = hashlib.sha256(
            (raw_id + "\n" + text + "\n" + str(row["reward_model"]["ground_truth"])).encode()
        ).hexdigest()[:24]
        result.append({
            "id": stable_id, "source_id": raw_id, "prompt": text, "prompt_tokens": tokens,
            "ground_truth": row["reward_model"]["ground_truth"],
            "data_source": row["data_source"], "extra_info": extra,
        })
    if len({r["id"] for r in result}) != len(result):
        raise RuntimeError(f"stable ID collision in {path}")
    return result


def token_stats(rows: list[dict]) -> dict:
    """p50/p95/p99/max of `prompt_tokens` over `rows` (from `_records`) — a
    re-verification check against the confirmed `data/token_length_audit.json`
    numbers (GATE 0a), not a discovery step; a different tokenizer (7B vs the
    0.5B track's) could in principle shift these slightly."""
    import numpy as np

    lengths = np.asarray([r["prompt_tokens"] for r in rows])
    return {
        "n": len(rows),
        "p50": float(np.percentile(lengths, 50)) if len(rows) else 0.0,
        "p95": float(np.percentile(lengths, 95)) if len(rows) else 0.0,
        "p99": float(np.percentile(lengths, 99)) if len(rows) else 0.0,
        "max": int(lengths.max()) if len(rows) else 0,
    }


def load_all_records(model_id: str, model_revision: str | None,
                     dataset_revision: str = GURU_DATASET_REVISION,
                     stage_a_prompt_suffix: str | None = None) -> tuple[list[dict], list[dict]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision)
    snapshot = _snapshot(dataset_revision)
    math_rows = _records(snapshot / MATH_FILE, tokenizer, stage_a_prompt_suffix)
    sim_rows = _records(snapshot / SIM_FILE, tokenizer)
    return math_rows, sim_rows


def build_exp2_splits(model_id: str, model_revision: str | None,
                      stage_a_token_limit: int, stage_b_token_limit: int,
                      stage_b_eval_questions: int, n_probe: int,
                      stage_a_prompt_suffix: str | None = None,
                      dataset_revision: str = GURU_DATASET_REVISION,
                      seed: int = SEED, out_name: str = "exp2_colab_splits.json") -> dict:
    """Freeze stage-A train ids, stage-B train/eval ids, and the Q-metric
    probe (drawn from stage-B pool rows disjoint from stage-B train+eval,
    topped up from stage-A rows if needed — plan §1 "probe size": n_probe
    must exceed the model's hidden dim). Writes data/<out_name>.

    Idempotent: an existing file is compared against a fresh recomputation
    and must match exactly, otherwise this raises rather than silently
    diverging from a previously frozen split (same discipline as
    `exp2_4070_data.py:ensure_splits`).
    """
    math_rows, sim_rows = load_all_records(
        model_id, model_revision, dataset_revision, stage_a_prompt_suffix)
    math_ok = [r for r in math_rows if r["prompt_tokens"] <= stage_a_token_limit]
    sim_ok = [r for r in sim_rows if r["prompt_tokens"] <= stage_b_token_limit]

    sim_ids = sorted(r["id"] for r in sim_ok)
    rng = random.Random(seed)
    eval_ids = sorted(rng.sample(sim_ids, min(stage_b_eval_questions, len(sim_ids))))
    eval_set = set(eval_ids)
    remaining_sim_ids = [i for i in sim_ids if i not in eval_set]

    # DEVIATION LOGGED 2026-08-16 (operator decision; see
    # FINDING_STAGE_B_TRAIN_EMPTY.md). Stage-B TRAIN is allocated before the
    # probe, not after. The previous order took `remaining_sim_ids[:n_probe]`
    # first, and since n_probe (4096) exceeds the eligible CodeIO pool (~1132
    # after eval), the probe consumed every remaining row and left
    # stage_b_train EMPTY - observed on the 2026-08-16 7B run
    # (stage_b_train: 0), which makes the Delta-R curve impossible to produce.
    # The config's own probe_source says the probe is "disjoint from stage-B
    # train and eval, topped up from held-out stage-A rows if the CodeI/O pool
    # is too small", i.e. train is meant to be carved out first and the
    # shortfall made up from math - which is what this does. The WIN4070
    # track's splits agree (eligible 1432 -> train 1132 / eval 300).
    # Practical effect is small: with n_probe=4096 the probe was already ~72%
    # math top-up, and is now 100% math.
    train_ids = [i for i in remaining_sim_ids]
    probe_ids = []
    need = n_probe - len(probe_ids)
    math_ids_all = sorted(r["id"] for r in math_ok)
    train_math_pool = math_ids_all  # stage-A train uses ALL eligible math rows
    probe_math_topup_ids = []
    if need > 0:
        # top up from math rows, disjoint from nothing in particular since
        # stage-A trains on the full eligible pool anyway (no held-out
        # math slice exists here) — logged as a cross-domain, in-population
        # top-up, same pattern as the earlier heuristic version.
        topup_pool = [i for i in math_ids_all if i not in set(probe_ids)]
        probe_math_topup_ids = rng.sample(topup_pool, min(need, len(topup_pool)))


    splits = {
        "seed": seed,
        "dataset_revision": dataset_revision,
        "model_id": model_id,
        "model_revision": model_revision,
        "stage_a_prompt_suffix": stage_a_prompt_suffix,
        "stage_a_data_sources": list(STAGE_A_DATA_SOURCES),
        "stage_b_data_sources": list(STAGE_B_DATA_SOURCES),
        "stage_a_token_limit": stage_a_token_limit,
        "stage_b_token_limit": stage_b_token_limit,
        "stage_a_eligible_count": len(math_ok),
        "stage_b_eligible_count": len(sim_ok),
        "stage_a_train_ids": train_math_pool,
        "stage_b_train_ids": sorted(train_ids),
        "stage_b_eval_ids": eval_ids,
        "probe_stage_b_ids": sorted(probe_ids),
        "probe_stage_a_topup_ids": sorted(probe_math_topup_ids),
        "probe_requested": n_probe,
        "probe_actual": len(probe_ids) + len(probe_math_topup_ids),
    }
    if splits["probe_actual"] < n_probe:
        splits["probe_shortfall_note"] = (
            f"pools could only supply {splits['probe_actual']} of {n_probe} "
            "probe prompts; effective-rank magnitudes may be sample-truncated "
            "— report n_probe alongside every erank value")

    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / out_name
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        if existing != splits:
            raise RuntimeError(
                f"frozen {out_name} differs from recomputation; preserve it "
                "and investigate before overwriting")
        return existing
    out_path.write_text(json.dumps(splits, indent=1))
    return splits


def dataset_rows_for(stage: str, split: str, splits: dict,
                     model_id: str, model_revision: str,
                     dataset_revision: str = GURU_DATASET_REVISION) -> list[dict]:
    """Row dicts (id/prompt/ground_truth/data_source/extra_info) for one
    stage/split, filtered to the frozen id set in `splits`. Re-renders from
    the parquet files rather than caching the full row set in the splits
    JSON — the splits file stays small and auditable (ids only)."""
    stage_a_prompt_suffix = splits.get("stage_a_prompt_suffix")
    math_rows, sim_rows = load_all_records(
        model_id, model_revision, dataset_revision, stage_a_prompt_suffix)
    if stage == "a":
        wanted = set(splits["stage_a_train_ids"])
        rows = [r for r in math_rows if r["id"] in wanted]
    elif stage == "b":
        key = f"stage_b_{split}_ids"
        wanted = set(splits[key])
        rows = [r for r in sim_rows if r["id"] in wanted]
    elif stage == "probe":
        wanted_b = set(splits["probe_stage_b_ids"])
        wanted_a = set(splits["probe_stage_a_topup_ids"])
        rows = ([r for r in sim_rows if r["id"] in wanted_b]
               + [r for r in math_rows if r["id"] in wanted_a])
    else:
        raise ValueError("stage must be 'a', 'b', or 'probe'")
    rows.sort(key=lambda r: r["id"])
    return rows


def to_hf_dataset(rows: list[dict]):
    from datasets import Dataset

    return Dataset.from_list(rows)
