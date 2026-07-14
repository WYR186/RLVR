"""Run one isolated Stage-B adaptation seed x source checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
sys.path.insert(0, str(PROJECT))

from src.adaptation import run_fixed_budget_adaptation  # noqa: E402
from src.repeats import (EXPECTED_BASELINES, acquire_repeat_lock,  # noqa: E402
                         ensure_repeat_manifest, frozen_repeat_recipe,
                         repeat_output_dir, validate_repeat_directory,
                         validate_runtime_against_source,
                         validate_source_run)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(43, 44), required=True)
    parser.add_argument("--checkpoint", type=int,
                        choices=(0, 25, 50, 100, 200), required=True)
    parser.add_argument("--attempt-id",
                        help="Wrapper ownership token for safe failure cleanup")
    parser.add_argument("--status-path", type=Path,
                        help="Per-invocation wrapper status JSON")
    args = parser.parse_args()
    if args.attempt_id and not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", args.attempt_id):
        raise ValueError("attempt-id must be 8-128 URL-safe characters")
    if bool(args.attempt_id) != bool(args.status_path):
        raise ValueError("attempt-id and status-path must be supplied together")

    import torch

    source_run = args.source_run.resolve()
    config, source_manifest = validate_source_run(source_run, args.checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError("RTX CUDA is required for official Stage-B repeats")
    gpu_name = torch.cuda.get_device_name(0)
    validate_runtime_against_source(source_manifest, gpu_name)

    pilot = json.loads((PROJECT / "pilot_config.json").read_text(encoding="utf-8"))
    adaptation = pilot["adaptation"]
    execution = config["execution"]
    recipe = frozen_repeat_recipe()
    # Guard against silent drift in canonical values even though seed remains 42.
    for key in ("budget_updates", "eval_every", "learning_rate", "beta",
                "temperature", "top_p", "num_generations",
                "max_prompt_length", "max_completion_length"):
        if adaptation[key] != recipe[key]:
            raise RuntimeError(
                f"pilot adaptation {key}={adaptation[key]!r}; expected {recipe[key]!r}")

    acquire_repeat_lock(source_run)
    checkpoint_path = source_run / f"ckpt-{args.checkpoint}"
    out_dir = repeat_output_dir(source_run, args.seed, args.checkpoint)
    validation_args = {
        "seed": args.seed,
        "checkpoint_path": checkpoint_path,
        "recipe": recipe,
        "expected_acc_before": EXPECTED_BASELINES[args.checkpoint],
    }
    existing = validate_repeat_directory(
        out_dir, recipe["budget_updates"], recipe["eval_every"],
        **validation_args)
    if existing is not None:
        if args.status_path:
            args.status_path.resolve().write_text(json.dumps({
                "attempt_id": args.attempt_id,
                "mode": "existing_valid",
                "out_dir": str(out_dir),
            }, indent=1), encoding="utf-8")
        print(json.dumps(existing, indent=1))
        print(f"repeat already complete and valid: {out_dir}")
        return

    if args.attempt_id:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / ".repeat_attempt.json").write_text(json.dumps({
            "attempt_id": args.attempt_id,
            "runner_pid": os.getpid(),
            "created_unix": time.time(),
        }, indent=1), encoding="utf-8")
        args.status_path.resolve().write_text(json.dumps({
            "attempt_id": args.attempt_id,
            "mode": "new_attempt",
            "out_dir": str(out_dir),
        }, indent=1), encoding="utf-8")

    ensure_repeat_manifest(source_run, args.seed, recipe, REPO, gpu_name)

    summary = run_fixed_budget_adaptation(
        checkpoint_path=checkpoint_path,
        out_dir=out_dir,
        budget_updates=recipe["budget_updates"],
        eval_every=recipe["eval_every"],
        seed=args.seed,
        learning_rate=recipe["learning_rate"],
        num_generations=recipe["num_generations"],
        per_device_batch=recipe["per_device_train_batch_size"],
        grad_accum=recipe["gradient_accumulation_steps"],
        beta=recipe["beta"],
        temperature=recipe["temperature"],
        top_p=recipe["top_p"],
        max_prompt_length=recipe["max_prompt_length"],
        max_completion_length=recipe["max_completion_length"],
        bf16=False,
        device=execution["device"],
        dtype_name=execution["dtype"],
        autocast_dtype_name=execution["autocast_dtype"],
        optim=execution["optim"],
        gradient_checkpointing=execution["gradient_checkpointing"],
        save_steps=10,
        expected_acc_before=EXPECTED_BASELINES[args.checkpoint],
    )
    post_validation_args = dict(validation_args)
    if args.attempt_id:
        post_validation_args["allow_attempt_id"] = args.attempt_id
    else:
        post_validation_args["require_attached_telemetry"] = False
    validate_repeat_directory(
        out_dir, recipe["budget_updates"], recipe["eval_every"],
        **post_validation_args)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
