"""Pinned GURU data loading and split freezing for the RTX 4070 variant."""
from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "exp2_config_4070.json"
MATH_FILE = "train/math__combined_54.4k.parquet"
SIM_FILE = "train/simulation__codeio_3.7k.parquet"


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def splits_path(cfg: dict) -> Path:
    tag = cfg.get("variant_tag")
    if tag:
        return HERE / "data" / f"exp2_{tag}_splits.json"
    if "instruct_v2" in cfg["experiment"]:
        tag = "4070_instruct_v2"
    elif "Instruct" in cfg["model_id"]:
        tag = "4070_instruct"
    else:
        tag = "4070"
    return HERE / "data" / f"exp2_{tag}_splits.json"


def _tokenizer(cfg: dict):
    return AutoTokenizer.from_pretrained(
        cfg["model_id"], revision=cfg["model_revision"], local_files_only=True
    )


def _snapshot(cfg: dict) -> Path:
    return Path(
        snapshot_download(
            cfg["dataset"]["source"],
            repo_type="dataset",
            revision=cfg["dataset"]["revision"],
            local_files_only=True,
        )
    )


def _render_prompt(tokenizer, messages: list[dict], prompt_suffix: str | None = None) -> str:
    messages = deepcopy(messages)
    if prompt_suffix:
        user_messages = [message for message in messages if message.get("role") == "user"]
        if not user_messages or not isinstance(user_messages[-1].get("content"), str):
            raise RuntimeError("prompt suffix requires a final text user message")
        user_messages[-1]["content"] = (
            user_messages[-1]["content"].rstrip() + "\n\n" + prompt_suffix.strip()
        )
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _records(path: Path, tokenizer, prompt_suffix: str | None = None) -> list[dict]:
    columns = [
        "data_source",
        "prompt",
        "reward_model",
        "extra_info",
        "qwen2.5_7b_pass_rate",
    ]
    rows = pq.read_table(path, columns=columns).to_pylist()
    texts = [_render_prompt(tokenizer, row["prompt"], prompt_suffix) for row in rows]
    lengths: list[int] = []
    for offset in range(0, len(texts), 256):
        encoded = tokenizer(
            texts[offset : offset + 256],
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        lengths.extend(len(ids) for ids in encoded)
    result = []
    for row, text, tokens in zip(rows, texts, lengths, strict=True):
        extra = row["extra_info"] or {}
        raw_id = f"{row['data_source']}:{extra.get('index')}"
        stable_id = hashlib.sha256(
            (raw_id + "\n" + text + "\n" + row["reward_model"]["ground_truth"]).encode()
        ).hexdigest()[:24]
        result.append(
            {
                "id": stable_id,
                "source_id": raw_id,
                "prompt": text,
                "prompt_tokens": tokens,
                "ground_truth": row["reward_model"]["ground_truth"],
                "data_source": row["data_source"],
                "extra_info": extra,
                "qwen2.5_7b_pass_rate": row["qwen2.5_7b_pass_rate"],
            }
        )
    if len({row["id"] for row in result}) != len(result):
        raise RuntimeError(f"stable ID collision in {path}")
    return result


def load_all_records(cfg: dict | None = None) -> tuple[list[dict], list[dict]]:
    cfg = cfg or load_config()
    tokenizer = _tokenizer(cfg)
    snapshot = _snapshot(cfg)
    math_rows = _records(
        snapshot / MATH_FILE,
        tokenizer,
        cfg["stage_a"].get("prompt_suffix"),
    )
    sim_rows = _records(snapshot / SIM_FILE, tokenizer)
    return math_rows, sim_rows


def build_splits(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    math_rows, sim_rows = load_all_records(cfg)
    math_limit = cfg["stage_a"]["token_filter_max"]
    sim_limit = cfg["stage_b"]["token_filter_max"]
    math_ok = [row for row in math_rows if row["prompt_tokens"] <= math_limit]
    sim_ok = [row for row in sim_rows if row["prompt_tokens"] <= sim_limit]
    if len(math_ok) != cfg["stage_a"]["expected_eligible_questions"]:
        raise RuntimeError(f"Math eligible count drift: {len(math_ok)}")
    if len(sim_ok) != cfg["stage_b"]["expected_eligible_questions"]:
        raise RuntimeError(f"CodeIO eligible count drift: {len(sim_ok)}")

    sim_ids = sorted(row["id"] for row in sim_ok)
    rng = random.Random(cfg["seed"])
    eval_ids = sorted(rng.sample(sim_ids, cfg["stage_b"]["eval_questions"]))
    eval_set = set(eval_ids)
    train_ids = [item for item in sim_ids if item not in eval_set]
    if len(train_ids) != cfg["stage_b"]["train_questions"]:
        raise RuntimeError("Stage-B train size does not match config")

    eligible_rates = [
        row["qwen2.5_7b_pass_rate"]
        for row in sim_ok
        if row["qwen2.5_7b_pass_rate"] is not None
    ]
    return {
        "experiment": cfg["experiment"],
        "seed": cfg["seed"],
        "dataset_revision": cfg["dataset"]["revision"],
        "model_revision": cfg["model_revision"],
        "rendering": {
            "chat_template": True,
            "add_generation_prompt": True,
            "add_special_tokens_for_length": False,
            "prompt_truncation": False,
            "stage_a_prompt_suffix": cfg["stage_a"].get("prompt_suffix"),
        },
        "stage_a": {
            "token_filter_max": math_limit,
            "eligible_count": len(math_ok),
            "train_ids": sorted(row["id"] for row in math_ok),
        },
        "stage_b": {
            "token_filter_max": sim_limit,
            "eligible_count": len(sim_ok),
            "train_ids": train_ids,
            "eval_ids": eval_ids,
            "released_qwen2_5_7b_pass_rate_mean": sum(eligible_rates)
            / len(eligible_rates),
        },
    }


def ensure_splits(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    expected = build_splits(cfg)
    path = splits_path(cfg)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != expected:
            raise RuntimeError(
                "frozen exp2_4070_splits.json differs from recomputation; preserve it and investigate"
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(expected, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return expected


def dataset_for(stage: str, split: str = "train", cfg: dict | None = None) -> Dataset:
    cfg = cfg or load_config()
    frozen = ensure_splits(cfg)
    math_rows, sim_rows = load_all_records(cfg)
    if stage == "a":
        if split != "train":
            raise ValueError("Stage A has only the frozen train population")
        wanted = set(frozen["stage_a"]["train_ids"])
        rows = [row for row in math_rows if row["id"] in wanted]
    elif stage == "b":
        wanted = set(frozen["stage_b"][f"{split}_ids"])
        rows = [row for row in sim_rows if row["id"] in wanted]
    else:
        raise ValueError("stage must be 'a' or 'b'")
    rows.sort(key=lambda row: row["id"])
    return Dataset.from_list(rows)
