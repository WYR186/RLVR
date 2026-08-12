#!/usr/bin/env python3
"""Prepare, smoke-test, and run Stage A of the RTX 4070 exp2 variant."""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PILOT = REPO / "eaaj-pilot"
for path in (PILOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed  # noqa: E402
from transformers.trainer_utils import get_last_checkpoint  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from exp2_4070_data import dataset_for, ensure_splits, load_config, splits_path  # noqa: E402
from exp2_4070_reward import (  # noqa: E402
    boxed_format_score,
    guru_reward,
    guru_reward_boxed_01,
    score_completion,
)
from scripts.run_local_pipeline import acquire_runner_lock, setup_backend  # noqa: E402
from src.callbacks import (  # noqa: E402
    JsonlDashboardLogger,
    LocalSafetyCallback,
    SaveAtSteps,
    UpdateEffectivenessSentinel,
)
from src.repro import config_hash, runtime_manifest, sha256_file  # noqa: E402


CONFIG_PATH = HERE / "exp2_config_4070.json"


def variant_tag(cfg: dict) -> str:
    if cfg.get("variant_tag"):
        return cfg["variant_tag"]
    if "instruct_v2" in cfg["experiment"]:
        return "4070_instruct_v2"
    return "4070_instruct" if "Instruct" in cfg["model_id"] else "4070"


def smoke_root(cfg: dict) -> Path:
    return HERE / f"smoke_outputs_{variant_tag(cfg)}"


def data_marker(cfg: dict, name: str) -> Path:
    return HERE / "data" / f"exp2_{variant_tag(cfg)}_{name}.json"


def execution_profile(cfg: dict) -> dict:
    recipe = cfg["stage_a"]
    return {
        "device": "cuda",
        "dtype": "float32",
        "autocast_dtype": "bfloat16",
        "optim": "paged_adamw_8bit",
        "gradient_checkpointing": True,
        "per_device_train_batch_size": recipe["per_device_train_batch_size"],
        "gradient_accumulation_steps": recipe["gradient_accumulation_steps"],
        "num_generations": recipe["num_generations"],
    }


def run_config(cfg: dict) -> dict:
    return {
        "experiment": cfg["experiment"],
        "model": cfg["model_id"],
        "model_revision": cfg["model_revision"],
        "seed": cfg["seed"],
        "dataset_revision": cfg["dataset"]["revision"],
        "stage_a": cfg["stage_a"],
        "stage_b_contract": cfg["stage_b"],
        "execution": execution_profile(cfg),
    }


def run_dir(cfg: dict) -> Path:
    rcfg = run_config(cfg)
    return PILOT / "outputs" / f"exp2_4070_cuda_guru_math_{config_hash(rcfg)}"


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the registered RTX 4070 stratum")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected GPU does not support bf16 autocast")


def load_base(cfg: dict):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_id"], revision=cfg["model_revision"], local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        revision=cfg["model_revision"],
        dtype=torch.float32,
        local_files_only=True,
    ).to("cuda")
    return model, tokenizer


def reward_func_for(recipe: dict):
    mode = recipe.get("reward_mode", "exact")
    if mode == "exact":
        return guru_reward
    if mode == "exact_plus_boxed_format_0.1":
        return guru_reward_boxed_01
    raise ValueError(f"unsupported reward_mode: {mode}")


def release(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    torch.cuda.empty_cache()


def prepare(cfg: dict) -> dict:
    splits = ensure_splits(cfg)
    if set(splits["stage_b"]["train_ids"]) & set(splits["stage_b"]["eval_ids"]):
        raise RuntimeError("Stage-B train/eval leakage")
    audit = {
        "experiment": cfg["experiment"],
        "status": "geometry_and_split_gate_pass",
        "training_started": False,
        "stage_a": {
            "eligible": splits["stage_a"]["eligible_count"],
            "max_prompt_tokens": cfg["stage_a"]["token_filter_max"],
        },
        "stage_b": {
            "eligible": splits["stage_b"]["eligible_count"],
            "train": len(splits["stage_b"]["train_ids"]),
            "eval": len(splits["stage_b"]["eval_ids"]),
            "max_prompt_tokens": cfg["stage_b"]["token_filter_max"],
            "max_completion_tokens": cfg["stage_b"]["max_completion_length"],
            "total_sequence_budget": cfg["stage_b"]["token_filter_max"]
            + cfg["stage_b"]["max_completion_length"],
        },
        "next_gate": "real sparse-reward preflight on both stages",
    }
    path = data_marker(cfg, "phase0_audit")
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return audit


def sparse_preflight(model, tokenizer, dataset, recipe: dict, out_path: Path, seed: int) -> dict:
    if out_path.exists():
        result = json.loads(out_path.read_text(encoding="utf-8"))
        if not result.get("has_grpo_signal"):
            raise RuntimeError(f"existing sparse-reward gate is STOP: {out_path}")
        return result
    rng = random.Random(seed)
    set_seed(seed)
    indices = rng.sample(range(len(dataset)), 8)
    groups = []
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    try:
        with torch.no_grad():
            for index in indices:
                row = dataset[index]
                enc = tokenizer(row["prompt"], return_tensors="pt", truncation=False).to("cuda")
                prompt_tokens = enc["input_ids"].shape[1]
                if prompt_tokens > recipe["token_filter_max"]:
                    raise RuntimeError("frozen prompt exceeded the registered token filter")
                output = model.generate(
                    **enc,
                    do_sample=True,
                    num_return_sequences=recipe["num_generations"],
                    temperature=recipe["temperature"],
                    top_p=recipe["top_p"],
                    max_new_tokens=recipe["max_completion_length"],
                    pad_token_id=tokenizer.pad_token_id,
                )
                generated = output[:, prompt_tokens:]
                completions = tokenizer.batch_decode(generated, skip_special_tokens=True)
                exact_rewards = [
                    score_completion(
                        completion,
                        row["ground_truth"],
                        row["data_source"],
                        row["extra_info"],
                    )
                    for completion in completions
                ]
                format_rewards = [
                    boxed_format_score(completion, row["data_source"])
                    for completion in completions
                ]
                if recipe.get("reward_mode", "exact") == "exact_plus_boxed_format_0.1":
                    rewards = [
                        exact + 0.1 * format_score
                        for exact, format_score in zip(
                            exact_rewards, format_rewards, strict=True
                        )
                    ]
                else:
                    rewards = exact_rewards
                groups.append(
                    {
                        "id": row["id"],
                        "prompt_tokens": prompt_tokens,
                        "ground_truth": row["ground_truth"],
                        "rewards": rewards,
                        "exact_rewards": exact_rewards,
                        "boxed_format_rewards": format_rewards,
                        "completion_tails": [value[-800:] for value in completions],
                    }
                )
    finally:
        tokenizer.padding_side = old_side
    variable = sum(len(set(group["rewards"])) > 1 for group in groups)
    result = {
        "seed": seed,
        "generation_rng_seeded": True,
        "n_prompts": len(groups),
        "num_generations": recipe["num_generations"],
        "reward_mode": recipe.get("reward_mode", "exact"),
        "n_correct": int(sum(sum(group["exact_rewards"]) for group in groups)),
        "n_boxed": int(sum(sum(group["boxed_format_rewards"]) for group in groups)),
        "groups_with_reward_variance": variable,
        "has_grpo_signal": variable > 0,
        "groups": groups,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["has_grpo_signal"]:
        raise RuntimeError(
            "sparse-reward gate STOP: every sampled prompt group had constant reward; preserve diagnostic"
        )
    return result


def trainer_args(recipe: dict, output_dir: Path, max_steps: int, save_steps: int) -> GRPOConfig:
    return GRPOConfig(
        output_dir=str(output_dir),
        seed=42,
        max_steps=max_steps,
        learning_rate=recipe["learning_rate"],
        per_device_train_batch_size=recipe["per_device_train_batch_size"],
        gradient_accumulation_steps=recipe["gradient_accumulation_steps"],
        num_generations=recipe["num_generations"],
        beta=recipe["beta"],
        temperature=recipe["temperature"],
        top_p=recipe["top_p"],
        max_completion_length=recipe["max_completion_length"],
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        save_only_model=True,
        report_to="none",
    )


def run_smoke_stage(cfg: dict, stage: str) -> dict:
    require_cuda()
    recipe = cfg["stage_a" if stage == "a" else "stage_b"]
    out_dir = smoke_root(cfg) / f"stage_{stage}"
    complete = out_dir / "smoke_complete.json"
    if complete.exists():
        return json.loads(complete.read_text(encoding="utf-8"))
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite partial smoke directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = dataset_for(stage, "train", cfg)
    model, tokenizer = load_base(cfg)
    sparse_preflight(
        model,
        tokenizer,
        dataset,
        recipe,
        out_dir / "sparse_reward_preflight.json",
        cfg["seed"],
    )
    set_seed(cfg["seed"])
    trainer_dir = out_dir / "trainer"
    trainer = GRPOTrainer(
        model=model,
        args=trainer_args(recipe, trainer_dir, max_steps=2, save_steps=1),
        train_dataset=dataset,
        reward_funcs=reward_func_for(recipe),
        processing_class=tokenizer,
        callbacks=[
            JsonlDashboardLogger(out_dir / "dashboard.jsonl"),
            UpdateEffectivenessSentinel(out_dir / "update_sentinel.jsonl", every=1),
            LocalSafetyCallback(
                out_dir / "safety_stop.json",
                signal_patience=5,
                max_clip_ratio=0.10,
            ),
        ],
    )
    trainer.train()
    if trainer.state.global_step != 2:
        raise RuntimeError(f"smoke stopped at step {trainer.state.global_step}, expected 2")
    if (out_dir / "safety_stop.json").exists():
        raise RuntimeError("smoke fired safety stop")
    checkpoint = trainer_dir / "checkpoint-2" / "config.json"
    if not checkpoint.exists():
        raise RuntimeError("smoke checkpoint-2 is missing")
    dashboard = out_dir / "dashboard.jsonl"
    if not dashboard.exists() or len(dashboard.read_text().splitlines()) < 2:
        raise RuntimeError("smoke dashboard did not record both updates")
    dashboard_rows = [
        json.loads(line) for line in dashboard.read_text().splitlines() if line.strip()
    ]
    metrics = [
        row for row in dashboard_rows if "completions/clipped_ratio" in row
    ]
    if len(metrics) != 2:
        raise RuntimeError(
            f"smoke dashboard has {len(metrics)} update rows, expected exactly 2"
        )
    clip_ratios = [float(row["completions/clipped_ratio"]) for row in metrics]
    step_times = [float(row["step_time"]) for row in metrics]
    nonzero_reward_updates = sum(
        float(row.get("reward_std", 0.0)) > 0.0
        and abs(float(row.get("grad_norm", 0.0))) > 0.0
        for row in metrics
    )
    smoke_clip_limit = cfg["gates"].get(
        "phase0_stage_a_smoke_max_clip_ratio_each_update"
    )
    if stage == "a" and smoke_clip_limit is not None and any(
        ratio > smoke_clip_limit for ratio in clip_ratios
    ):
        failure = {
            "stage": stage,
            "status": "STOP",
            "reason": "stage_a_smoke_completion_clipping_exceeded_limit",
            "limit": smoke_clip_limit,
            "clip_ratios": clip_ratios,
            "step_times": step_times,
        }
        (out_dir / "smoke_gate_failure.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        release(trainer, model)
        raise RuntimeError(
            f"Stage-A smoke clipping gate STOP: {clip_ratios} exceeds {smoke_clip_limit}"
        )
    min_nonzero = cfg["gates"].get(
        "phase0_stage_a_smoke_min_nonzero_reward_updates"
    )
    if stage == "a" and min_nonzero is not None and nonzero_reward_updates < min_nonzero:
        failure = {
            "stage": stage,
            "status": "STOP",
            "reason": "stage_a_smoke_had_too_few_nonzero_reward_updates",
            "minimum": min_nonzero,
            "observed": nonzero_reward_updates,
            "clip_ratios": clip_ratios,
            "step_times": step_times,
        }
        (out_dir / "smoke_gate_failure.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        release(trainer, model)
        raise RuntimeError(
            f"Stage-A smoke reward gate STOP: {nonzero_reward_updates} nonzero updates, need {min_nonzero}"
        )
    result = {
        "stage": stage,
        "status": "complete",
        "global_step": trainer.state.global_step,
        "checkpoint": str(checkpoint.parent),
        "dashboard": str(dashboard),
        "clip_ratios": clip_ratios,
        "step_times": step_times,
        "nonzero_reward_updates": nonzero_reward_updates,
        "completed_unix": time.time(),
    }
    complete.write_text(json.dumps(result, indent=2), encoding="utf-8")
    release(trainer, model)
    return result


def smoke(cfg: dict) -> dict:
    prepare(cfg)
    result = {"stage_a": run_smoke_stage(cfg, "a")}
    result["stage_b"] = run_smoke_stage(cfg, "b")
    path = data_marker(cfg, "phase0_smoke_complete")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def stage1(cfg: dict) -> Path:
    require_cuda()
    smoke_marker = data_marker(cfg, "phase0_smoke_complete")
    if not smoke_marker.exists():
        raise RuntimeError("CUDA smoke has not completed; run --action smoke first")
    target = run_dir(cfg)
    target.mkdir(parents=True, exist_ok=True)
    acquire_runner_lock(target)
    complete = target / "phase1_complete.json"
    if complete.exists():
        print(f"Stage 1 already complete: {target}")
        return target
    free_gib = __import__("shutil").disk_usage(target).free / 2**30
    if free_gib < 30:
        raise RuntimeError(f"only {free_gib:.1f} GiB free; need >=30 GiB")

    rcfg = run_config(cfg)
    (target / "config.json").write_text(json.dumps(rcfg, indent=2), encoding="utf-8")
    if not (target / "manifest.json").exists():
        manifest = runtime_manifest(PILOT, rcfg)
        config_name = Path(cfg.get("_config_path", CONFIG_PATH)).name
        amendment_name = Path(cfg["amendment"]).name
        split_name = splits_path(cfg).relative_to(HERE).as_posix()
        manifest["exp2_4070_inputs_sha256"] = {
            name: sha256_file(HERE / name)
            for name in (
                config_name,
                amendment_name,
                "exp2_4070_data.py",
                "exp2_4070_reward.py",
                "run_exp2_4070.py",
                split_name,
            )
        }
        effective_examples = (
            cfg["stage_a"]["max_steps"]
            * cfg["stage_a"]["per_device_train_batch_size"]
            * cfg["stage_a"]["gradient_accumulation_steps"]
        )
        manifest["stage_a_reported_epochs"] = effective_examples / cfg["stage_a"][
            "expected_eligible_questions"
        ]
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    setup_backend(execution_profile(cfg))
    recipe = cfg["stage_a"]
    dataset = dataset_for("a", "train", cfg)
    model, tokenizer = load_base(cfg)
    ckpt0 = target / "ckpt-0"
    if not ckpt0.exists():
        model.save_pretrained(ckpt0, safe_serialization=True)
        tokenizer.save_pretrained(ckpt0)
    sparse_preflight(
        model,
        tokenizer,
        dataset,
        recipe,
        target / "sparse_reward_preflight.json",
        cfg["seed"],
    )
    set_seed(cfg["seed"])
    trainer_dir = target / "trainer"
    resume = get_last_checkpoint(str(trainer_dir)) if trainer_dir.exists() else None
    trainer = GRPOTrainer(
        model=model,
        args=trainer_args(recipe, trainer_dir, recipe["max_steps"], 25),
        train_dataset=dataset,
        reward_funcs=reward_func_for(recipe),
        processing_class=tokenizer,
        callbacks=[
            JsonlDashboardLogger(target / "dashboard.jsonl"),
            UpdateEffectivenessSentinel(target / "update_sentinel.jsonl", every=25),
            SaveAtSteps(recipe["checkpoint_steps"][1:], target, tokenizer=tokenizer),
            LocalSafetyCallback(
                target / "safety_stop.json",
                signal_patience=5,
                max_clip_ratio=0.10,
            ),
        ],
    )
    trainer.train(resume_from_checkpoint=resume)
    if (target / "safety_stop.json").exists():
        raise RuntimeError("Stage 1 safety-stopped; preserve and report")
    missing = [
        step
        for step in recipe["checkpoint_steps"]
        if not (target / f"ckpt-{step}" / "config.json").exists()
    ]
    if missing:
        raise RuntimeError(f"Stage 1 missing checkpoints: {missing}")
    complete.write_text(
        json.dumps(
            {"status": "complete", "global_step": trainer.state.global_step, "completed_unix": time.time()},
            indent=2,
        ),
        encoding="utf-8",
    )
    release(trainer, model)
    print(f"Stage 1 complete: {target}")
    return target


def status(cfg: dict) -> dict:
    target = run_dir(cfg)
    result = {
        "run_dir": str(target),
        "model": cfg["model_id"],
        "phase0_geometry": data_marker(cfg, "phase0_audit").exists(),
        "phase0_smoke": data_marker(cfg, "phase0_smoke_complete").exists(),
        "stage1_started": target.exists(),
        "stage1_complete": (target / "phase1_complete.json").exists(),
        "safety_stop": (target / "safety_stop.json").exists(),
    }
    if (target / "dashboard.jsonl").exists():
        rows = (target / "dashboard.jsonl").read_text().splitlines()
        result["dashboard_rows"] = len(rows)
        result["dashboard_last"] = json.loads(rows[-1]) if rows else None
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "smoke", "stage1", "status"), required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["_config_path"] = str(args.config.resolve())
    if args.action == "prepare":
        prepare(cfg)
    elif args.action == "smoke":
        smoke(cfg)
    elif args.action == "stage1":
        stage1(cfg)
    else:
        status(cfg)


if __name__ == "__main__":
    main()
