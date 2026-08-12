#!/usr/bin/env python3
"""Freeze Experiment 2's GURU schema/token audit and enforce Gate 0a.

This is a discovery-only Phase-0 command. It never trains a model and never
creates exp2_splits.json when the geometry gate fails.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
DATASET_ID = "LLM360/guru-RL-92k"
DATASET_REVISION = "2e2790a962a3c099bfb5ea61389cbf98a5ea439b"
MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
VERIFIER_REPOSITORY = "https://github.com/LLM360/Reasoning360"
VERIFIER_REVISION = "13158341d2a0dfe5f3bb80e7126ff21de0d16676"
MATH_FILE = "train/math__combined_54.4k.parquet"
SIM_FILE = "train/simulation__codeio_3.7k.parquet"
GATE_0A_MAX_P95 = 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(sorted_values: list[int], fraction: float) -> int:
    return sorted_values[math.ceil(fraction * len(sorted_values)) - 1]


def prompt_text(tokenizer, messages: list[dict]) -> str:
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return "\n".join(message["content"] for message in messages)


def token_stats(tokenizer, messages: list[list[dict]]) -> dict:
    texts = [prompt_text(tokenizer, item) for item in messages]
    lengths: list[int] = []
    started = time.time()
    for offset in range(0, len(texts), 256):
        encoded = tokenizer(
            texts[offset : offset + 256],
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        lengths.extend(len(ids) for ids in encoded)
    lengths.sort()
    return {
        "n": len(lengths),
        "p50": percentile(lengths, 0.50),
        "p95": percentile(lengths, 0.95),
        "p99": percentile(lengths, 0.99),
        "max": lengths[-1],
        "over_512": sum(value > 512 for value in lengths),
        "over_1024": sum(value > 1024 for value in lengths),
        "tokenization_seconds": round(time.time() - started, 3),
    }


def file_audit(path: Path, wanted_columns: list[str]) -> tuple[dict, list[dict]]:
    parquet = pq.ParquetFile(path)
    table = pq.read_table(path, columns=wanted_columns)
    rows = table.to_pylist()
    source_counts = Counter(row["data_source"] for row in rows)
    examples = {
        source: next(row for row in rows if row["data_source"] == source)
        for source in sorted(source_counts)
    }
    audit = {
        "relative_path": path.as_posix().split("/snapshots/")[-1].split("/", 1)[-1],
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "schema": [
            {"name": field.name, "dtype": str(field.type), "nullable": field.nullable}
            for field in parquet.schema_arrow
        ],
        "data_source_counts": dict(sorted(source_counts.items())),
        "full_examples_by_data_source": examples,
    }
    return audit, rows


def main() -> None:
    snapshot = Path(
        snapshot_download(
            DATASET_ID,
            repo_type="dataset",
            revision=DATASET_REVISION,
            local_files_only=True,
        )
    )
    math_path = snapshot / MATH_FILE
    sim_path = snapshot / SIM_FILE
    for path in (math_path, sim_path):
        if not path.is_file():
            raise FileNotFoundError(f"required cached parquet is missing: {path}")

    common_columns = [
        "data_source",
        "prompt",
        "ability",
        "apply_chat_template",
        "reward_model",
        "extra_info",
    ]
    math_audit, math_rows = file_audit(math_path, common_columns)
    sim_audit, sim_rows = file_audit(sim_path, common_columns)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    lengths = {
        "contract": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "chat_template_applied": bool(tokenizer.chat_template),
            "add_generation_prompt": True,
            "add_special_tokens": False,
            "truncation": False,
        },
        "stage_a_math": token_stats(tokenizer, [row["prompt"] for row in math_rows]),
        "stage_b_simulation": token_stats(
            tokenizer, [row["prompt"] for row in sim_rows]
        ),
    }
    stage_b_p95 = lengths["stage_b_simulation"]["p95"]
    gate_pass = stage_b_p95 <= GATE_0A_MAX_P95
    lengths["gate_0a"] = {
        "threshold_stage_b_p95_tokens_max": GATE_0A_MAX_P95,
        "observed_stage_b_p95_tokens": stage_b_p95,
        "status": "PASS" if gate_pass else "STOP",
        "required_action": (
            "continue Phase 0"
            if gate_pass
            else "preserve diagnostics and request the L4 24 GB stratum; do not shrink batch, truncate prompts, or start training on the RTX 4070"
        ),
    }

    schema = {
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "config": "default",
            "splits": ["train"],
            "loader_note": (
                "The repository contains heterogeneous parquet schemas under train/*.parquet. "
                "datasets==5.0.0 cannot cast them into one unified Dataset, so target domain "
                "files are audited and loaded independently."
            ),
        },
        "field_contract": {
            "domain_field": "data_source",
            "prompt_field": "prompt (list of role/content messages)",
            "answer_field": "reward_model.ground_truth",
            "stage_a_filter": [
                "math__deepscaler_preview",
                "math__merged_deduped_dapo_or1_dataset",
            ],
            "stage_b_filter": ["simulation__codeio"],
            "stage_a_count": len(math_rows),
            "stage_b_count": len(sim_rows),
            "source_limitation": (
                "The released math parquet merges DAPO and OR1 into one data_source value; "
                "the released fields do not permit a defensible per-origin DAPO-vs-OR1 split."
            ),
        },
        "answer_and_verifier_contract": {
            "official_repository": VERIFIER_REPOSITORY,
            "official_revision": VERIFIER_REVISION,
            "router": "verl/utils/reward_score/__init__.py::default_compute_score",
            "math": {
                "format": "final answer inside the last \\boxed{...}",
                "implementation": "verl/utils/reward_score/naive_dapo.py",
                "comparison": "normalized string and mathematical equivalence; 1.0 correct, 0.0 otherwise",
            },
            "simulation": {
                "format": "JSON in a ```json``` code block, normally with input or output wrapper",
                "implementation": "verl/utils/reward_score/codeio.py",
                "comparison": "extract final complete JSON, recursively normalize, then exact structured equality",
            },
        },
        "files": {
            "stage_a_math": math_audit,
            "stage_b_simulation": sim_audit,
        },
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "guru_schema_audit.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (DATA_DIR / "token_length_audit.json").write_text(
        json.dumps(lengths, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    marker = {
        "phase": "0",
        "gate": "0a",
        "status": "STOP" if not gate_pass else "PASS",
        "training_started": False,
        "observed": lengths["gate_0a"],
        "split_freeze_skipped": not gate_pass,
        "smoke_training_skipped": not gate_pass,
        "reason": (
            "Gate 0a fired before split freeze and smoke training."
            if not gate_pass
            else "Gate 0a passed."
        ),
    }
    (DATA_DIR / "phase0_gate_0a.json").write_text(
        json.dumps(marker, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(marker, indent=2, ensure_ascii=False))
    if not gate_pass:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
