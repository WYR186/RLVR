"""Post-stop Phase 2/2b and truncated Stage-B runner for exp2 v8.

The source Stage-A directory is immutable. Every action validates its hashes
and writes only beneath a separate, config-hashed post-stop output directory.
"""
from __future__ import annotations

import argparse
import atexit
import gc
import json
import math
import os
import platform
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PILOT = REPO / "eaaj-pilot"
DEFAULT_CONFIG = HERE / "exp2_config_4070_instruct_v8_poststop.json"
SOURCE_CONFIG = HERE / "exp2_config_4070_instruct_v8.json"
for path in (PILOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from exp2_4070_data import dataset_for, load_config  # noqa: E402
from exp2_4070_reward import guru_reward, score_completion  # noqa: E402
from src.callbacks import (JsonlDashboardLogger, LocalSafetyCallback,  # noqa: E402
                           UpdateEffectivenessSentinel)
from src.metrics import checkpoint_q_metrics  # noqa: E402
from src.repro import config_hash, sha256_file  # noqa: E402
from transformers import TrainerCallback  # noqa: E402


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError:
        if _read_json(path) != value:
            raise RuntimeError(f"refusing to overwrite differing artifact: {path}")


def poststop_contract(cfg: dict) -> dict:
    return {key: value for key, value in cfg.items() if not key.startswith("_")}


def poststop_run_dir(cfg: dict) -> Path:
    digest = config_hash(poststop_contract(cfg))
    return PILOT / "outputs" / f"exp2_4070_v8_poststop_{digest}"


def source_run_dir(cfg: dict) -> Path:
    return PILOT / "outputs" / cfg["source"]["run_name"]


def _assert_sha(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"source hash mismatch for {path}: {actual} != {expected}")


def validate_source(cfg: dict) -> dict:
    """Validate the immutable safety-stopped source without writing to it."""
    source = source_run_dir(cfg)
    if not source.is_dir():
        raise FileNotFoundError(source)
    if (source / "phase1_complete.json").exists():
        raise RuntimeError("source unexpectedly claims a complete Stage A")
    if (source / "runner.lock").exists():
        raise RuntimeError("source runner.lock exists; refuse concurrent post-stop work")

    for relative, expected in cfg["source"]["sha256"].items():
        _assert_sha(source / relative, expected)
    stop = _read_json(source / "safety_stop.json")
    if int(stop.get("step", -1)) != int(cfg["source"]["safety_stop_step"]):
        raise RuntimeError("safety-stop step differs from the post-stop contract")
    if stop.get("reason") != cfg["source"]["safety_stop_reason"]:
        raise RuntimeError("safety-stop reason differs from the post-stop contract")

    manifest = _read_json(source / "manifest.json")
    if manifest.get("config_hash") != cfg["source"]["config_hash"]:
        raise RuntimeError("source config hash differs from the post-stop contract")
    for step in cfg["source"]["required_checkpoints"]:
        checkpoint = source / f"ckpt-{step}"
        for name in ("config.json", "model.safetensors", "tokenizer.json"):
            if not (checkpoint / name).is_file():
                raise FileNotFoundError(checkpoint / name)
    for step in cfg["source"]["forbidden_as_complete"]:
        if (source / f"ckpt-{step}").exists():
            raise RuntimeError(f"unexpected post-stop checkpoint exists: ckpt-{step}")
    return {"source_run": str(source), "safety_stop": stop, "manifest": manifest}


def ensure_poststop_manifest(cfg: dict) -> Path:
    source_info = validate_source(cfg)
    target = poststop_run_dir(cfg)
    manifest = {
        "experiment": cfg["experiment"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "poststop_config_hash": config_hash(poststop_contract(cfg)),
        "poststop_config_sha256": sha256_file(Path(cfg["_config_path"])),
        "amendment_sha256": sha256_file(REPO / cfg["amendment"]),
        "source_run": source_info["source_run"],
        "source_config_hash": cfg["source"]["config_hash"],
        "source_sha256": cfg["source"]["sha256"],
        "python": sys.version,
        "platform": platform.platform(),
        "claim_boundary": cfg["claim_boundary"],
    }
    path = target / "manifest.json"
    if path.exists():
        existing = _read_json(path)
        immutable = ("poststop_config_hash", "poststop_config_sha256",
                     "amendment_sha256", "source_run", "source_config_hash",
                     "source_sha256", "claim_boundary")
        for key in immutable:
            if existing.get(key) != manifest.get(key):
                raise RuntimeError(f"post-stop manifest mismatch for {key}")
    else:
        _write_json_once(path, manifest)
    return target


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for official post-stop computation")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("selected CUDA device does not support bf16 autocast")
    name = torch.cuda.get_device_name(0)
    if "RTX 4070" not in name:
        raise RuntimeError(f"post-stop local stratum requires RTX 4070, found {name}")


def _load_checkpoint(cfg: dict, step: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if step not in cfg["source"]["required_checkpoints"]:
        raise ValueError(f"checkpoint {step} is outside the truncated contract")
    checkpoint = source_run_dir(cfg) / f"ckpt-{step}"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch.float32, local_files_only=True).to("cuda")
    return checkpoint, model, tokenizer


def _release(*objects) -> None:
    import torch

    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _codeio_rows(cfg: dict, split: str) -> list[dict]:
    source_cfg = load_config(SOURCE_CONFIG)
    data = dataset_for("b", split, source_cfg)
    rows = [dict(row) for row in data]
    expected = cfg["stage_b"][f"{split}_questions"]
    if len(rows) != expected:
        raise RuntimeError(f"CodeIO {split} count {len(rows)} != {expected}")
    return rows


def evaluate_codeio(model, tokenizer, rows: list[dict], *, batch_size: int,
                    max_prompt_length: int, max_new_tokens: int) -> dict:
    """Deterministic greedy CodeIO exact-reward evaluation."""
    import torch

    model.eval()
    device = next(model.parameters()).device
    items = []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    started = time.time()
    try:
        with torch.inference_mode():
            for offset in range(0, len(rows), batch_size):
                batch = rows[offset:offset + batch_size]
                prompts = [row["prompt"] for row in batch]
                encoded = tokenizer(
                    prompts, return_tensors="pt", padding=True,
                    truncation=False).to(device)
                lengths = encoded["attention_mask"].sum(dim=1)
                if int(lengths.max()) > max_prompt_length:
                    raise RuntimeError("frozen CodeIO prompt exceeds registered limit")
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    generated = model.generate(
                        **encoded, max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id
                        or tokenizer.eos_token_id)
                suffix = generated[:, encoded["input_ids"].shape[1]:]
                texts = tokenizer.batch_decode(suffix, skip_special_tokens=True)
                for row, text in zip(batch, texts, strict=True):
                    reward = score_completion(
                        text, row["ground_truth"], row["data_source"],
                        row.get("extra_info") or {})
                    items.append({
                        "id": row["id"],
                        "source_id": row["source_id"],
                        "reward": reward,
                        "correct": reward == 1.0,
                        "completion": text,
                    })
    finally:
        tokenizer.padding_side = old_padding_side
    return {
        "score": sum(item["reward"] for item in items) / len(items),
        "n_eval": len(items),
        "n_correct": sum(item["correct"] for item in items),
        "mean_completion_chars": sum(len(item["completion"]) for item in items)
        / len(items),
        "eval_seconds": time.time() - started,
        "items": items,
    }


def run_zero_shot(cfg: dict, step: int) -> dict:
    _require_cuda()
    target = ensure_poststop_manifest(cfg)
    spec = cfg["phase2_zero_shot"]
    if step not in spec["checkpoints"]:
        raise ValueError(f"checkpoint {step} is outside Phase 2")
    out = target / "phase2_zero_shot" / f"ckpt-{step}.json"
    if out.exists():
        result = _read_json(out)
        if result.get("checkpoint") != step or result.get("n_eval") != 300:
            raise RuntimeError(f"invalid existing zero-shot artifact: {out}")
        return result
    checkpoint, model, tokenizer = _load_checkpoint(cfg, step)
    rows = _codeio_rows(cfg, "eval")
    result = evaluate_codeio(
        model, tokenizer, rows, batch_size=spec["batch_size"],
        max_prompt_length=spec["max_prompt_length"],
        max_new_tokens=spec["max_new_tokens"])
    result.update({
        "checkpoint": step,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": cfg["source"]["sha256"][
            f"ckpt-{step}/model.safetensors"],
        "task": spec["task"],
        "split": spec["split"],
        "reward": spec["reward"],
        "decode": spec["decode"],
        "max_prompt_length": spec["max_prompt_length"],
        "max_new_tokens": spec["max_new_tokens"],
    })
    _write_json_once(out, result)
    _release(model)
    return result


def run_q_measurement(cfg: dict, step: int) -> dict:
    _require_cuda()
    target = ensure_poststop_manifest(cfg)
    spec = cfg["phase2b"]
    if step not in spec["checkpoints"]:
        raise ValueError(f"checkpoint {step} is outside Phase 2b")
    out = target / "phase2b" / f"metrics_ckpt{step}.json"
    if out.exists():
        result = _read_json(out)
        if result.get("checkpoint") != step or result.get("n_probe") != 2048:
            raise RuntimeError(f"invalid existing Q artifact: {out}")
        return result

    probe_path = REPO / spec["probe_file"]
    _assert_sha(probe_path, spec["probe_sha256"])
    probe_payload = _read_json(probe_path)
    prompts = probe_payload[spec["probe_key"]]
    if len(prompts) != spec["probe_questions"]:
        raise RuntimeError("frozen probe count differs from the contract")
    checkpoint, model, tokenizer = _load_checkpoint(cfg, step)
    started = time.time()
    result = checkpoint_q_metrics(
        model, tokenizer, prompts, layers=tuple(spec["layers"]),
        batch_size=spec["batch_size"],
        max_length=spec["max_prompt_length"])
    result.update({
        "checkpoint": step,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": cfg["source"]["sha256"][
            f"ckpt-{step}/model.safetensors"],
        "probe_sha256": spec["probe_sha256"],
        "identity_gate": spec["identity_gate"],
        "wall_seconds": time.time() - started,
    })
    _write_json_once(out, result)
    _release(model)
    return result


def transfer_table(cfg: dict, results: dict[int, dict]) -> dict:
    checkpoints = cfg["phase2_zero_shot"]["checkpoints"]
    if sorted(results) != sorted(checkpoints):
        raise ValueError("zero-shot results do not cover the Phase-2 checkpoint set")
    baseline = float(results[0]["score"])
    return {
        "definition": "T_t = Score_B(M_A,t) - Score_B(M_0)",
        "baseline_checkpoint": 0,
        "baseline_score": baseline,
        "trajectory_status": cfg["claim_boundary"]["trajectory"],
        "rows": [
            {
                "checkpoint": step,
                "score_b": float(results[step]["score"]),
                "T_t": float(results[step]["score"]) - baseline,
                "n_eval": int(results[step]["n_eval"]),
            }
            for step in checkpoints
        ],
    }


def finalize_phase2(cfg: dict) -> dict:
    target = ensure_poststop_manifest(cfg)
    complete_path = target / "phase2_complete.json"
    if complete_path.exists():
        return _read_json(complete_path)
    zero_results = {}
    for step in cfg["phase2_zero_shot"]["checkpoints"]:
        path = target / "phase2_zero_shot" / f"ckpt-{step}.json"
        zero_results[step] = _read_json(path)
    for step in cfg["phase2b"]["checkpoints"]:
        metrics = _read_json(target / "phase2b" / f"metrics_ckpt{step}.json")
        if metrics.get("n_probe") != cfg["phase2b"]["probe_questions"]:
            raise RuntimeError(f"Q artifact probe count mismatch at ckpt-{step}")
    transfer = transfer_table(cfg, zero_results)
    _write_json_once(target / "phase2_zero_shot" / "transfer_T.json", transfer)
    complete = {
        "status": "complete_truncated",
        "source_safety_stop_step": cfg["source"]["safety_stop_step"],
        "zero_shot_checkpoints": cfg["phase2_zero_shot"]["checkpoints"],
        "q_checkpoints": cfg["phase2b"]["checkpoints"],
        "claim_boundary": cfg["claim_boundary"],
        "completed_unix": time.time(),
    }
    _write_json_once(complete_path, complete)
    return complete


class CodeIOEvalCallback(TrainerCallback):
    """TRL callback for fixed 0/10/.../50 greedy CodeIO evaluation."""

    def __init__(self, rows, out_path: Path, *, every: int, batch_size: int,
                 max_prompt_length: int, max_new_tokens: int):
        self.rows = rows
        self.out_path = out_path
        self.every = every
        self.batch_size = batch_size
        self.max_prompt_length = max_prompt_length
        self.max_new_tokens = max_new_tokens
        self.tokenizer = None

    def _run(self, model, step: int) -> None:
        result = evaluate_codeio(
            model, self.tokenizer, self.rows, batch_size=self.batch_size,
            max_prompt_length=self.max_prompt_length,
            max_new_tokens=self.max_new_tokens)
        row = {"step": step, **result}
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[codeio-eval] step {step}: score={result['score']:.6f}")
        model.train()

    def on_train_begin(self, args, state, control, model=None,
                       processing_class=None, tokenizer=None, **kwargs):
        del args, control, kwargs
        self.tokenizer = processing_class or tokenizer
        if state.global_step == 0:
            self._run(model, 0)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        del args, control, kwargs
        if state.global_step % self.every == 0:
            self._run(model, int(state.global_step))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number} is not a JSON object")
            rows.append(row)
    return rows


def validate_stage_b_cell(out_dir: Path, cfg: dict, checkpoint: int,
                          seed: int) -> dict:
    spec = cfg["stage_b"]
    if (out_dir / "safety_stop.json").exists():
        raise RuntimeError(f"Stage-B safety stop exists: {out_dir}")
    summary = _read_json(out_dir / "summary.json")
    expected = {
        "checkpoint": checkpoint,
        "seed": seed,
        "actual_updates": spec["budget_updates"],
        "completion_status": "complete",
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"Stage-B summary {key} mismatch")
    eval_steps = [int(row["step"]) for row in
                  _read_jsonl(out_dir / "codeio_eval_curve.jsonl")]
    if eval_steps != spec["eval_at_updates"]:
        raise RuntimeError(f"Stage-B eval steps mismatch: {eval_steps}")
    sentinel = _read_jsonl(out_dir / "update_sentinel.jsonl")
    if [int(row["step"]) for row in sentinel] != spec["eval_at_updates"][1:]:
        raise RuntimeError("Stage-B sentinel steps mismatch")
    if not all(row.get("updates_effective") is True for row in sentinel):
        raise RuntimeError("Stage-B contains an ineffective update window")
    dashboard = [row for row in _read_jsonl(out_dir / "dashboard.jsonl")
                 if "loss" in row or "grad_norm" in row]
    if sorted({int(row["step"]) for row in dashboard}) != list(
            range(1, spec["budget_updates"] + 1)):
        raise RuntimeError("Stage-B dashboard does not cover all 50 updates")
    for row in dashboard:
        for key in ("loss", "grad_norm"):
            if key in row and not math.isfinite(float(row[key])):
                raise RuntimeError(f"non-finite Stage-B {key} at step {row['step']}")
    return summary


def _acquire_lock(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"post-stop lock exists: {path}") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(str(pid))

    def release() -> None:
        try:
            owner = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return
        if owner == pid:
            path.unlink(missing_ok=True)

    atexit.register(release)


def _curve_auc(rows: list[dict], budget: int) -> float:
    points = sorted((int(row["step"]), float(row["score"])) for row in rows)
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area / budget


def run_stage_b_cell(cfg: dict, checkpoint: int, seed: int) -> dict:
    import torch
    from transformers import set_seed
    from trl import GRPOConfig, GRPOTrainer

    _require_cuda()
    target = ensure_poststop_manifest(cfg)
    if not (target / "phase2_complete.json").exists():
        raise RuntimeError("Phase 2/2b must complete before Stage B starts")
    spec = cfg["stage_b"]
    if checkpoint not in spec["checkpoints"] or seed not in spec["seeds"]:
        raise ValueError("checkpoint/seed is outside the truncated Stage-B grid")
    out_dir = target / "stage_b" / f"seed-{seed}" / f"ckpt-{checkpoint}"
    if (out_dir / "summary.json").exists():
        return validate_stage_b_cell(out_dir, cfg, checkpoint, seed)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(
            f"refusing to resume or overwrite partial Stage-B cell: {out_dir}")
    _acquire_lock(target / "stage_b.lock")
    out_dir.mkdir(parents=True, exist_ok=False)
    set_seed(seed)

    checkpoint_path, model, tokenizer = _load_checkpoint(cfg, checkpoint)
    train_rows = _codeio_rows(cfg, "train")
    eval_rows = _codeio_rows(cfg, "eval")
    from datasets import Dataset

    train_data = Dataset.from_list(train_rows)
    trainer_dir = out_dir / "trainer"
    args = GRPOConfig(
        output_dir=str(trainer_dir), seed=seed,
        max_steps=spec["budget_updates"],
        learning_rate=spec["learning_rate"],
        per_device_train_batch_size=spec["per_device_train_batch_size"],
        gradient_accumulation_steps=spec["gradient_accumulation_steps"],
        num_generations=spec["num_generations"], beta=spec["beta"],
        temperature=spec["temperature"], top_p=spec["top_p"],
        max_completion_length=spec["max_completion_length"],
        bf16=True, fp16=False, optim=cfg["execution"]["optimizer"],
        gradient_checkpointing=cfg["execution"]["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=True, logging_steps=1,
        save_strategy="steps", save_steps=spec["save_steps"],
        save_total_limit=2, save_only_model=True, report_to="none")
    eval_every = spec["eval_at_updates"][1]
    trainer = GRPOTrainer(
        model=model, args=args, train_dataset=train_data,
        reward_funcs=guru_reward, processing_class=tokenizer,
        callbacks=[
            JsonlDashboardLogger(out_dir / "dashboard.jsonl"),
            UpdateEffectivenessSentinel(
                out_dir / "update_sentinel.jsonl", every=eval_every),
            CodeIOEvalCallback(
                eval_rows, out_dir / "codeio_eval_curve.jsonl",
                every=eval_every, batch_size=spec["eval_batch_size"],
                max_prompt_length=spec["max_prompt_length"],
                max_new_tokens=spec["max_completion_length"]),
            LocalSafetyCallback(
                out_dir / "safety_stop.json", signal_patience=5,
                max_clip_ratio=0.10),
        ])
    started = time.time()
    trainer.train()
    if (out_dir / "safety_stop.json").exists():
        raise RuntimeError("Stage-B safety-stopped; preserve the cell and report")
    if int(trainer.state.global_step) != spec["budget_updates"]:
        raise RuntimeError("Stage-B returned before the fixed 50-update budget")
    curve = _read_jsonl(out_dir / "codeio_eval_curve.jsonl")
    summary = {
        "checkpoint": checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": cfg["source"]["sha256"][
            f"ckpt-{checkpoint}/model.safetensors"],
        "seed": seed,
        "task": "Simulation-short-context CodeIO",
        "algorithm": spec["algorithm"],
        "requested_updates": spec["budget_updates"],
        "actual_updates": int(trainer.state.global_step),
        "completion_status": "complete",
        "score_before": float(curve[0]["score"]),
        "score_after": float(curve[-1]["score"]),
        "delta_score": float(curve[-1]["score"]) - float(curve[0]["score"]),
        "normalized_auc": _curve_auc(curve, spec["budget_updates"]),
        "recipe": spec,
        "wall_seconds": time.time() - started,
        "claim_boundary": cfg["claim_boundary"]["stage_b"],
    }
    _write_json_once(out_dir / "summary.json", summary)
    validate_stage_b_cell(out_dir, cfg, checkpoint, seed)
    _release(trainer, model)
    return summary


def finalize_stage_b(cfg: dict) -> dict:
    target = ensure_poststop_manifest(cfg)
    complete_path = target / "stage_b_complete.json"
    if complete_path.exists():
        return _read_json(complete_path)
    summaries = []
    for seed in cfg["stage_b"]["seeds"]:
        for checkpoint in cfg["stage_b"]["checkpoints"]:
            out = target / "stage_b" / f"seed-{seed}" / f"ckpt-{checkpoint}"
            summaries.append(validate_stage_b_cell(out, cfg, checkpoint, seed))
    complete = {
        "status": "complete_truncated",
        "cells": len(summaries),
        "checkpoints": cfg["stage_b"]["checkpoints"],
        "seeds": cfg["stage_b"]["seeds"],
        "claim_boundary": cfg["claim_boundary"],
        "completed_unix": time.time(),
    }
    _write_json_once(complete_path, complete)
    return complete


def status(cfg: dict) -> dict:
    source = validate_source(cfg)
    target = poststop_run_dir(cfg)
    result = {
        "source": source["source_run"],
        "source_safety_stop_step": source["safety_stop"]["step"],
        "poststop_run": str(target),
        "manifest": (target / "manifest.json").exists(),
        "phase2_complete": (target / "phase2_complete.json").exists(),
        "stage_b_complete": (target / "stage_b_complete.json").exists(),
        "zero_shot": {}, "phase2b": {}, "stage_b": {},
    }
    for step in cfg["phase2_zero_shot"]["checkpoints"]:
        result["zero_shot"][str(step)] = (
            target / "phase2_zero_shot" / f"ckpt-{step}.json").exists()
    for step in cfg["phase2b"]["checkpoints"]:
        result["phase2b"][str(step)] = (
            target / "phase2b" / f"metrics_ckpt{step}.json").exists()
    for seed in cfg["stage_b"]["seeds"]:
        for step in cfg["stage_b"]["checkpoints"]:
            key = f"seed-{seed}/ckpt-{step}"
            result["stage_b"][key] = (
                target / "stage_b" / key / "summary.json").exists()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=(
        "audit", "zero-shot", "q", "phase2-finalize",
        "stageb-cell", "stageb-finalize", "status"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    cfg = _read_json(args.config.resolve())
    cfg["_config_path"] = str(args.config.resolve())

    if args.action == "audit":
        target = ensure_poststop_manifest(cfg)
        result = {"status": "validated", "poststop_run": str(target),
                  **validate_source(cfg)}
    elif args.action == "zero-shot":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for zero-shot")
        result = run_zero_shot(cfg, args.checkpoint)
    elif args.action == "q":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for q")
        result = run_q_measurement(cfg, args.checkpoint)
    elif args.action == "phase2-finalize":
        result = finalize_phase2(cfg)
    elif args.action == "stageb-cell":
        if args.checkpoint is None or args.seed is None:
            parser.error("--checkpoint and --seed are required for stageb-cell")
        result = run_stage_b_cell(cfg, args.checkpoint, args.seed)
    elif args.action == "stageb-finalize":
        result = finalize_stage_b(cfg)
    else:
        result = status(cfg)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
