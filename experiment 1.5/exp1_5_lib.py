"""Shared helpers for experiment 1.5 (Stage-A dose escalation + noise fix).

Everything scientific is REUSED from eaaj-pilot/src so Q metrics, reward
parsing, prompt formatting, callbacks and validators are byte-identical to
the pilot (cross-experiment comparability). This module only adds what the
pilot did not have:

  * a v1.5 SVAMP split: the SAME frozen 256 train questions, but the eval
    set widened from the frozen 100 to the FULL 300-question test split
    (the pilot's 100 remain a strict subset -> legacy sub-score bridges the
    two experiments);
  * a fork of run_fixed_budget_adaptation that (a) evaluates the primary
    before/after delta on the 300-question set, (b) keeps the every-10-updates
    curve on the pilot's frozen 100 questions so adaptation-speed curves stay
    directly comparable to experiment 1, and (c) records both scores.

Pilot code is NOT modified: experiment 1's pre-registered analysis (seed-44
completion) is still pending and its inputs must stay frozen.
"""
from __future__ import annotations

import gc
import json
import shutil
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO = EXP_DIR.parent
PILOT = REPO / "eaaj-pilot"
for p in (str(PILOT), str(EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.adaptation import (fixed_budget_completion,          # noqa: E402
                            validate_adaptation_completion)
from src.data import format_prompt, load_svamp, svamp_grpo_dataset  # noqa: E402
from src.repro import config_hash, runtime_manifest, sha256_file    # noqa: E402

DATA_DIR = EXP_DIR / "data"
CONFIG_PATH = EXP_DIR / "exp1_5_config.json"
V15_SPLITS_PATH = DATA_DIR / "svamp_splits_v15.json"

EXP15_CODE_INPUTS = ("exp1_5_lib.py", "run_exp1_5.py",
                     "analysis_exp1_5.py")


def load_config(config_path=None) -> dict:
    path = Path(config_path) if config_path is not None else CONFIG_PATH
    return json.loads(path.resolve().read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# v1.5 SVAMP splits: same frozen 256 train, eval widened to the full test split
# ---------------------------------------------------------------------------

def ensure_v15_svamp_splits(expected_eval_questions: int = 300) -> dict:
    """Freeze the v1.5 SVAMP lists; idempotent, committed to git.

    train_idx is copied VERBATIM from the pilot's frozen svamp_splits.json —
    resampling it would break Stage-B comparability with experiment 1. The
    eval list is every index of the pinned test split (the pilot's frozen
    100-question eval_idx is a strict subset and is recorded as legacy_eval_idx).
    """
    if V15_SPLITS_PATH.exists():
        return json.loads(V15_SPLITS_PATH.read_text(encoding="utf-8"))
    pilot_splits = json.loads(
        (PILOT / "data" / "svamp_splits.json").read_text(encoding="utf-8"))
    ds = load_svamp()
    n_test = len(ds["test"])
    if n_test != expected_eval_questions:
        raise RuntimeError(
            f"pinned SVAMP test split has {n_test} questions, expected "
            f"{expected_eval_questions}; do not proceed — the pre-registered "
            "eval size would be wrong")
    splits = {
        "seed": pilot_splits["seed"],
        "svamp_train_idx": pilot_splits["svamp_train_idx"],
        "svamp_eval_idx": list(range(n_test)),
        "legacy_eval_idx": pilot_splits["svamp_eval_idx"],
        "note": ("train_idx identical to eaaj-pilot/data/svamp_splits.json; "
                 "eval widened to the full pinned test split; legacy_eval_idx "
                 "is experiment 1's frozen 100-question eval set"),
    }
    DATA_DIR.mkdir(exist_ok=True)
    V15_SPLITS_PATH.write_text(json.dumps(splits, indent=1), encoding="utf-8")
    return splits


def _svamp_question(ex) -> str:
    # Kept identical to src.data._svamp_question (Body + Question join);
    # duplicated one-liner rather than importing a private symbol.
    return f"{ex['Body'].strip()} {ex['Question'].strip()}"


def svamp_eval_sets_v15() -> dict:
    """Return the v1.5 eval material.

    prompts/golds       — full 300-question primary eval (delta on this set)
    legacy_mask         — aligned bools, True where the item is in the pilot's
                          frozen 100-question eval set (legacy sub-score)
    curve_prompts/golds — exactly the pilot's 100-question set, same order as
                          src.data.svamp_eval_set (adaptation-speed curve)
    """
    splits = ensure_v15_svamp_splits()
    ds = load_svamp()
    sub = ds["test"].select(splits["svamp_eval_idx"])
    prompts = [format_prompt(_svamp_question(ex)) for ex in sub]
    golds = [float(ex["Answer"]) for ex in sub]
    legacy = set(splits["legacy_eval_idx"])
    legacy_mask = [i in legacy for i in splits["svamp_eval_idx"]]

    legacy_sub = ds["test"].select(sorted(splits["legacy_eval_idx"]))
    curve_prompts = [format_prompt(_svamp_question(ex)) for ex in legacy_sub]
    curve_golds = [float(ex["Answer"]) for ex in legacy_sub]
    return {"prompts": prompts, "golds": golds, "legacy_mask": legacy_mask,
            "curve_prompts": curve_prompts, "curve_golds": curve_golds}


def svamp_train_ds_v15():
    """The pilot's frozen 256-question GRPO dataset, unchanged."""
    return svamp_grpo_dataset()


# ---------------------------------------------------------------------------
# Run-dir plumbing
# ---------------------------------------------------------------------------

def stage_a_run_cfg(cfg: dict, execution: dict) -> dict:
    stage_a = dict(cfg["stage_a"])
    for key in ("per_device_train_batch_size", "gradient_accumulation_steps"):
        if key in execution:
            stage_a[key] = execution[key]
    return {"experiment": cfg["experiment"], "model": cfg["model_id"],
            "model_revision": cfg["model_revision"], "seed": cfg["seed"],
            **stage_a, "execution": execution}


def stage_a_run_dir(cfg: dict, backend: str, execution: dict,
                    smoke: bool = False) -> tuple[Path, dict]:
    run_cfg = stage_a_run_cfg(cfg, execution)
    root = (EXP_DIR / "smoke_outputs") if smoke else (PILOT / "outputs")
    prefix = f"exp15_{backend}_grpo_gsm8k"
    return root / f"{prefix}_{config_hash(run_cfg)}", run_cfg


def exp15_manifest(cfg: dict, run_cfg: dict, config_path=None) -> dict:
    manifest = runtime_manifest(PILOT, run_cfg)
    manifest["experiment"] = cfg["experiment"]
    manifest["exp1_5_inputs_sha256"] = {
        name: sha256_file(EXP_DIR / name) for name in EXP15_CODE_INPUTS
        if (EXP_DIR / name).exists()}
    selected_config = (Path(config_path) if config_path is not None
                       else CONFIG_PATH).resolve()
    try:
        config_name = selected_config.relative_to(EXP_DIR).as_posix()
    except ValueError:
        config_name = str(selected_config)
    manifest["exp1_5_inputs_sha256"][config_name] = sha256_file(selected_config)
    manifest["config_source"] = config_name
    if V15_SPLITS_PATH.exists():
        manifest["exp1_5_inputs_sha256"]["data/svamp_splits_v15.json"] = (
            sha256_file(V15_SPLITS_PATH))
    return manifest


def require_free_disk(path: Path, min_free_gib: float = 30.0) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(path).free / 2 ** 30
    if free_gib < min_free_gib:
        raise RuntimeError(
            f"only {free_gib:.1f} GiB free under {path}; need >= "
            f"{min_free_gib:.0f} GiB (8 fp32 checkpoints + rolling trainer "
            "state). Free disk space before starting.")


def append_compute(run_dir: Path, phase: str, started: float, status: str,
                   execution: dict) -> None:
    row = {"phase": phase, "status": status, "started_unix": started,
           "ended_unix": time.time(), "wall_seconds": time.time() - started,
           "device": execution["device"], "dtype": execution["dtype"]}
    with (run_dir / "local_compute.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Fixed-budget adaptation, v1.5
# ---------------------------------------------------------------------------

def _masked_accuracy(details: list[dict], mask: list[bool]) -> float:
    hits = [d["correct"] for d, m in zip(details, mask) if m]
    return sum(hits) / len(hits)


def run_fixed_budget_adaptation_v15(checkpoint_path,
                                    out_dir,
                                    *,
                                    recipe: dict,
                                    execution: dict,
                                    seed: int,
                                    eval_material: dict,
                                    train_ds,
                                    reward_func=None,
                                    expected_acc_before: float | None = None,
                                    keep_trainer_dir: bool = False) -> dict:
    """Adapt one checkpoint to SVAMP under the fixed budget (v1.5 recipe).

    Fork of src.adaptation.run_fixed_budget_adaptation (2026-07 pilot). The
    training recipe, guards, sentinel, validator and artifact contract are
    unchanged; the only scientific difference is the eval material:
    before/after delta on the 300-question set, curve on the pilot's frozen
    100 questions. Both scores land in baseline.json / summary.json.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOConfig, GRPOTrainer

    from src.callbacks import (ExactAnswerEvalCallback, JsonlDashboardLogger,
                               LocalSafetyCallback, UpdateEffectivenessSentinel)
    from src.evaluate import exact_answer_accuracy
    from src.reward import exact_answer_reward

    if reward_func is None:
        reward_func = exact_answer_reward
    budget_updates = recipe["budget_updates"]
    eval_every = recipe["eval_every"]
    device = execution["device"]
    dtype_name = execution["dtype"]
    autocast_dtype_name = execution.get("autocast_dtype")
    optim = execution.get("optim")
    gradient_checkpointing = execution.get("gradient_checkpointing", False)
    per_device_batch = execution.get(
        "per_device_train_batch_size", recipe["per_device_train_batch_size"])
    grad_accum = execution.get(
        "gradient_accumulation_steps", recipe["gradient_accumulation_steps"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        return validate_adaptation_completion(
            out_dir, budget_updates=budget_updates, eval_every=eval_every)
    safety_path = out_dir / "safety_stop.json"
    if safety_path.exists():
        raise RuntimeError(
            f"refusing to resume safety-stopped adaptation in {out_dir}; "
            "preserve it as a failed attempt and start in a fresh directory")
    if (out_dir / "incomplete.json").exists():
        raise RuntimeError(
            f"refusing to overwrite incomplete adaptation in {out_dir}; "
            "preserve it as a failed attempt and start in a fresh directory")
    trainer_dir = out_dir / "trainer"
    resume_from = get_last_checkpoint(str(trainer_dir)) if trainer_dir.exists() else None
    partial = [out_dir / "dashboard.jsonl", out_dir / "svamp_eval_curve.jsonl"]
    has_partial = any(p.exists() and p.stat().st_size for p in partial)
    if has_partial and (resume_from is None or optim == "paged_adamw_8bit"):
        raise RuntimeError(
            f"partial adaptation artifacts exist in {out_dir}; move the directory "
            "aside and rerun so the fixed budget is not mixed across attempts")
    set_seed(seed)

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    compute_dtype_name = autocast_dtype_name or dtype_name
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[dtype_name]
    if (compute_dtype_name == "bfloat16" and device == "cuda"
            and not torch.cuda.is_bf16_supported()):
        raise RuntimeError("bf16 requested on a GPU without bf16 support")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, dtype=dtype)
    model.to(device)

    eval_prompts = eval_material["prompts"]
    eval_golds = eval_material["golds"]
    legacy_mask = eval_material["legacy_mask"]
    curve_prompts = eval_material["curve_prompts"]
    curve_golds = eval_material["curve_golds"]

    t0 = time.time()
    baseline_path = out_dir / "baseline.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        acc_before = baseline["acc_before"]
        acc_before_legacy = baseline.get("acc_before_legacy100")
    else:
        acc_before, details = exact_answer_accuracy(
            model, tokenizer, eval_prompts, eval_golds, return_details=True)
        acc_before_legacy = _masked_accuracy(details, legacy_mask)
        baseline_path.write_text(json.dumps(
            {"acc_before": acc_before,
             "acc_before_legacy100": acc_before_legacy,
             "n_eval": len(eval_prompts)}, indent=1), encoding="utf-8")
    print(f"[adapt-v15] {checkpoint_path}: SVAMP-300 BEFORE = {acc_before:.4f} "
          f"(legacy-100 = {acc_before_legacy:.4f})")
    if (expected_acc_before is not None
            and abs(acc_before - expected_acc_before) > 1e-12):
        (out_dir / "baseline_mismatch.json").write_text(json.dumps(
            {"expected_acc_before": expected_acc_before,
             "actual_acc_before": acc_before,
             "checkpoint": str(checkpoint_path),
             "timestamp_unix": time.time()}, indent=1), encoding="utf-8")
        raise RuntimeError(
            f"SVAMP baseline mismatch across seeds: expected "
            f"{expected_acc_before:.4f}, got {acc_before:.4f} — greedy eval "
            "must be seed-independent; investigate before continuing")

    skip_optimizer_checkpoints = optim == "paged_adamw_8bit"
    grpo_cfg = GRPOConfig(
        output_dir=str(trainer_dir),
        seed=seed,
        max_steps=budget_updates,
        learning_rate=recipe["learning_rate"],
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        num_generations=recipe["num_generations"],
        beta=recipe["beta"],
        temperature=recipe["temperature"],
        top_p=recipe["top_p"],
        max_completion_length=recipe["max_completion_length"],
        bf16=(compute_dtype_name == "bfloat16" and device == "cuda"),
        fp16=(compute_dtype_name == "float16" and device == "cuda"),
        **({"optim": optim} if optim else {}),
        use_cpu=(device == "cpu"),
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if gradient_checkpointing else None),
        dataloader_pin_memory=(device == "cpu"),
        logging_steps=1,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        save_only_model=skip_optimizer_checkpoints,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        args=grpo_cfg,
        train_dataset=train_ds,
        reward_funcs=reward_func,
        processing_class=tokenizer,
        callbacks=[
            JsonlDashboardLogger(out_dir / "dashboard.jsonl"),
            UpdateEffectivenessSentinel(out_dir / "update_sentinel.jsonl",
                                        every=eval_every),
            ExactAnswerEvalCallback(curve_prompts, curve_golds,
                                    out_dir / "svamp_eval_curve.jsonl",
                                    every=eval_every),
            LocalSafetyCallback(out_dir / "safety_stop.json"),
        ],
    )
    trainer.train(resume_from_checkpoint=resume_from)
    try:
        completion = fixed_budget_completion(
            trainer, out_dir, budget_updates, safety_path)
    except RuntimeError:
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    acc_after, details_after = exact_answer_accuracy(
        model, tokenizer, eval_prompts, eval_golds, return_details=True)
    acc_after_legacy = _masked_accuracy(details_after, legacy_mask)
    print(f"[adapt-v15] {checkpoint_path}: SVAMP-300 AFTER = {acc_after:.4f} "
          f"(legacy-100 = {acc_after_legacy:.4f})")

    summary = {
        "experiment": "exp1_5",
        "checkpoint": str(checkpoint_path),
        "task": "SVAMP",
        "train_questions": len(train_ds),
        "eval_questions": len(eval_prompts),
        "curve_eval_questions": len(curve_prompts),
        "budget_updates": budget_updates,
        **completion,
        "seed": seed,
        "algo": "grpo",
        "learning_rate": recipe["learning_rate"],
        "beta": recipe["beta"],
        "temperature": recipe["temperature"],
        "top_p": recipe["top_p"],
        "num_generations": recipe["num_generations"],
        "per_device_train_batch_size": per_device_batch,
        "gradient_accumulation_steps": grad_accum,
        "max_prompt_length": recipe["max_prompt_length"],
        "max_completion_length": recipe["max_completion_length"],
        "bf16": compute_dtype_name == "bfloat16",
        "device": device,
        "dtype": dtype_name,
        "autocast_dtype": autocast_dtype_name,
        "optim": optim,
        "gradient_checkpointing": gradient_checkpointing,
        "resume_from_checkpoint": resume_from,
        "acc_before": acc_before,
        "acc_after": acc_after,
        "delta_acc": acc_after - acc_before,
        "acc_before_legacy100": acc_before_legacy,
        "acc_after_legacy100": acc_after_legacy,
        "delta_acc_legacy100": acc_after_legacy - acc_before_legacy,
        "wall_seconds": time.time() - t0,
    }
    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    validate_adaptation_completion(
        out_dir, budget_updates=budget_updates, eval_every=eval_every)

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not keep_trainer_dir and trainer_dir.exists():
        # Scientific artifacts (summary/curve/sentinel/dashboard/baseline) are
        # all outside trainer/; the rolling trainer state is only needed for
        # mid-run resume and costs ~4 GiB per adaptation on disk.
        shutil.rmtree(trainer_dir)
    return summary
