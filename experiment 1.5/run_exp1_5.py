#!/usr/bin/env python3
"""Resumable runner for experiment 1.5 (Stage-A dose escalation + noise fix).

Same four-phase shape as the pilot's run_local_pipeline.py, with the
pre-registered v1.5 changes (see EXPERIMENT_1_5_PLAN_ZH.md):

  Phase 1  GRPO on GSM8K, lr 1e-5, 500 updates, ckpts 0/25/50/100/200/300/400/500
  Phase 2  Q metrics on the pilot's frozen 512-prompt probe set, all 8 ckpts
  Phase 3  fixed-budget SVAMP adaptation, 3 seeds x 6 ckpts, eval on 300 questions
  Phase 4  analysis: manipulation checks + rho(erank_L12, mean-of-3-seed delta)

Target platform is the Windows RTX 4070 machine (--backend cuda, the pilot's
v2 execution profile, imported verbatim). --smoke runs the whole pipeline
with tiny sizes on cpu/cuda into experiment 1.5/smoke_outputs/ — plumbing
validation only, never a result (rewards are replaced by a deterministic
length-variance dummy so GRPO always has signal, and lr is raised so the
update sentinel has measurable movement).

Deliberately NOT shared with the pilot runner: this script never touches
outputs/ACTIVE_RUN.txt, so the pilot's phase-2/3/4 tooling can never be
pointed at an exp1.5 run dir by accident (and vice versa).
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

import exp1_5_lib as lib  # noqa: E402  (also puts eaaj-pilot on sys.path)

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed  # noqa: E402
from transformers.trainer_utils import get_last_checkpoint  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from scripts.run_local_pipeline import (DTYPES, EXECUTION_PROFILES,  # noqa: E402
                                        acquire_runner_lock, setup_backend)
from src.callbacks import (ExactAnswerEvalCallback, JsonlDashboardLogger,  # noqa: E402
                           LocalSafetyCallback, SaveAtSteps,
                           UpdateEffectivenessSentinel)
from src.data import gsm8k_eval_set, gsm8k_grpo_dataset, load_probe_prompts  # noqa: E402
from src.metrics import checkpoint_q_metrics  # noqa: E402
from src.preflight import sparse_reward_preflight  # noqa: E402
from src.reward import exact_answer_reward  # noqa: E402


def smoke_variance_reward(completions, answer, **kwargs):
    """Deterministic dummy reward with guaranteed within-group variance.

    Smoke runs are plumbing-only: tiny completion budgets make the real
    exact-answer reward all-zero, which would leave GRPO without gradient and
    trip the update sentinel for reasons unrelated to the code under test.
    Never used outside --smoke.
    """
    return [(len(c) % 7) / 7.0 for c in completions]


def apply_smoke_overrides(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["experiment"] += "_smoke"
    cfg["stage_a"].update({
        "max_steps": 2, "checkpoint_steps": [0, 1, 2], "eval_every": 1,
        "learning_rate": 1e-4, "num_generations": 4,
        "per_device_train_batch_size": 2, "gradient_accumulation_steps": 2,
        "max_prompt_length": 256, "max_completion_length": 48,
    })
    cfg["measurement"].update({
        "probe_questions": 8, "batch_size": 4, "sensitivity_checkpoints": [],
    })
    cfg["adaptation"].update({
        "eval_questions": 8, "curve_eval_questions": 4, "budget_updates": 2,
        "eval_every": 1, "learning_rate": 1e-4, "num_generations": 4,
        "per_device_train_batch_size": 2, "gradient_accumulation_steps": 2,
        "max_prompt_length": 256, "max_completion_length": 48,
        "checkpoints": [0, 2], "checkpoint_order": [0, 2], "seeds": [42, 43],
    })
    return cfg


def smoke_eval_material(material: dict, cfg: dict) -> dict:
    n = cfg["adaptation"]["eval_questions"]
    k = cfg["adaptation"]["curve_eval_questions"]
    return {"prompts": material["prompts"][:n], "golds": material["golds"][:n],
            "legacy_mask": material["legacy_mask"][:n],
            "curve_prompts": material["curve_prompts"][:k],
            "curve_golds": material["curve_golds"][:k]}


def run_execution(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())["execution"]


# ---------------------------------------------------------------------------
# Phase 1 — Stage A
# ---------------------------------------------------------------------------

def phase1(cfg: dict, backend: str, smoke: bool) -> Path:
    started = time.time()
    execution = EXECUTION_PROFILES[backend]
    run_dir, run_cfg = lib.stage_a_run_dir(cfg, backend, execution, smoke)
    device = execution["device"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(run_cfg, indent=1))
    if not (run_dir / "manifest.json").exists():
        (run_dir / "manifest.json").write_text(
            json.dumps(lib.exp15_manifest(cfg, run_cfg), indent=1))
    stage_a = {k: run_cfg[k] for k in run_cfg if k != "execution"}
    ckpt_steps = stage_a["checkpoint_steps"]
    if (run_dir / "phase1_complete.json").exists() and all(
            (run_dir / f"ckpt-{n}" / "config.json").exists() for n in ckpt_steps):
        print(f"Phase 1 already complete: {run_dir}")
        return run_dir
    if not smoke:
        lib.require_free_disk(run_dir)
    setup_backend(execution)
    set_seed(stage_a["seed"])

    tok = AutoTokenizer.from_pretrained(
        stage_a["model"], revision=stage_a["model_revision"], local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        stage_a["model"], revision=stage_a["model_revision"],
        dtype=DTYPES[execution["dtype"]], local_files_only=True)
    if device != "cpu":
        model.to(device)
    ckpt0 = run_dir / "ckpt-0"
    if not ckpt0.exists():
        model.save_pretrained(ckpt0, safe_serialization=True)
        tok.save_pretrained(ckpt0)

    train_ds = gsm8k_grpo_dataset()
    eval_prompts, eval_golds, eval_metadata = gsm8k_eval_set(return_metadata=True)
    reward_func = smoke_variance_reward if smoke else exact_answer_reward
    preflight_path = run_dir / "sparse_reward_preflight.json"
    if smoke:
        preflight_path.write_text(json.dumps(
            {"smoke": True, "skipped": "plumbing-only run"}, indent=1))
    elif not preflight_path.exists():
        preflight = sparse_reward_preflight(
            model, tok, train_ds["prompt"][:8], train_ds["answer"][:8],
            num_generations=stage_a["num_generations"],
            temperature=stage_a["temperature"], top_p=stage_a["top_p"],
            max_new_tokens=stage_a["max_completion_length"])
        preflight_path.write_text(json.dumps(preflight, indent=1))
        if not preflight["has_grpo_signal"]:
            raise RuntimeError(
                "base-model exact reward preflight has no GRPO signal; "
                "preserve the diagnostic and ask the team (pre-registered gate)")
    elif not json.loads(preflight_path.read_text()).get("has_grpo_signal"):
        raise RuntimeError("existing preflight shows no GRPO signal")

    set_seed(stage_a["seed"])  # preflight must not perturb the training stream
    trainer_dir = run_dir / "trainer"
    resume_from = get_last_checkpoint(str(trainer_dir)) if trainer_dir.exists() else None
    compute_dtype = execution.get("autocast_dtype", execution["dtype"])
    skip_optimizer_checkpoints = execution.get("optim") == "paged_adamw_8bit"
    args = GRPOConfig(
        output_dir=str(trainer_dir), seed=stage_a["seed"],
        max_steps=stage_a["max_steps"], learning_rate=stage_a["learning_rate"],
        per_device_train_batch_size=stage_a["per_device_train_batch_size"],
        gradient_accumulation_steps=stage_a["gradient_accumulation_steps"],
        num_generations=stage_a["num_generations"], beta=stage_a["beta"],
        temperature=stage_a["temperature"], top_p=stage_a["top_p"],
        max_completion_length=stage_a["max_completion_length"],
        use_cpu=(device == "cpu"),
        bf16=(compute_dtype == "bfloat16"),
        fp16=(compute_dtype == "float16"),
        **({"optim": execution["optim"]} if "optim" in execution else {}),
        gradient_checkpointing=execution.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False}
            if execution.get("gradient_checkpointing") else None),
        dataloader_pin_memory=(device != "mps"),
        logging_steps=1, save_strategy="steps", save_steps=25,
        save_total_limit=2, save_only_model=skip_optimizer_checkpoints,
        report_to="none")
    trainer = GRPOTrainer(
        model=model, args=args, train_dataset=train_ds,
        reward_funcs=reward_func, processing_class=tok,
        callbacks=[
            JsonlDashboardLogger(run_dir / "dashboard.jsonl"),
            UpdateEffectivenessSentinel(run_dir / "update_sentinel.jsonl",
                                        every=min(25, stage_a["max_steps"])),
            SaveAtSteps(ckpt_steps[1:], run_dir, tokenizer=tok),
            ExactAnswerEvalCallback(
                eval_prompts, eval_golds, run_dir / "gsm8k_eval.jsonl",
                every=stage_a["eval_every"], also_at_step0=True,
                item_metadata=eval_metadata),
            LocalSafetyCallback(run_dir / "safety_stop.json"),
        ])
    trainer.train(resume_from_checkpoint=resume_from)
    missing = [n for n in ckpt_steps
               if not (run_dir / f"ckpt-{n}" / "config.json").exists()]
    if missing:
        lib.append_compute(run_dir, "phase1", started, "incomplete", execution)
        raise RuntimeError(
            f"Phase 1 stopped before required checkpoints: {missing} "
            "(if safety_stop.json exists, preserve everything and report to "
            "the team — a reward/entropy collapse at lr 1e-5 is itself a "
            "pre-registered observable, not something to silently retry)")
    (run_dir / "phase1_complete.json").write_text(json.dumps(
        {"completed_unix": time.time(), "global_step": trainer.state.global_step,
         "resume_from_checkpoint": resume_from}, indent=1))
    lib.append_compute(run_dir, "phase1", started, "complete", execution)
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Phase 1 complete: {run_dir}")
    return run_dir


# ---------------------------------------------------------------------------
# Phase 2 — Q measurement (identical metric code as the pilot)
# ---------------------------------------------------------------------------

def phase2(cfg: dict, run_dir: Path, smoke: bool) -> None:
    started = time.time()
    execution = run_execution(run_dir)
    device = execution["device"]
    # Measurement dtype follows the run's EXECUTION dtype (fp32), exactly like
    # the pilot's local pipeline did — the pilot's committed ckpt-0 reference
    # values (gate 3) were produced this way, and comparability requires the
    # same path. (pilot_config's measurement.model_dtype=float16 field was
    # never honored by run_local_pipeline.py; verified 2026-07-16 against
    # metrics_ckpt0.json: model_dtype_requested=float32 on both strata.)
    dtype = DTYPES[execution["dtype"]]
    ckpts = cfg["stage_a"]["checkpoint_steps"]
    layers = tuple(cfg["measurement"]["layers"])
    out_dir = run_dir / "measurements"
    out_dir.mkdir(exist_ok=True)
    probe = load_probe_prompts()[:cfg["measurement"]["probe_questions"]]
    sensitivity_at = set(cfg["measurement"].get("sensitivity_checkpoints", []))
    probe_big = load_probe_prompts(big=True) if sensitivity_at else None
    for n in ckpts:
        out_path = out_dir / f"metrics_ckpt{n}.json"
        if out_path.exists():
            continue
        ckpt = run_dir / f"ckpt-{n}"
        if not (ckpt / "config.json").exists():
            raise FileNotFoundError(f"missing Phase-1 checkpoint: {ckpt}")
        tok = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            ckpt, dtype=dtype, local_files_only=True)
        if device != "cpu":
            model.to(device)
        t0 = time.time()
        m = checkpoint_q_metrics(
            model, tok, probe, layers=layers,
            batch_size=cfg["measurement"]["batch_size"],
            max_length=cfg["measurement"]["max_prompt_length"])
        m.update({"checkpoint": n, "run_dir": str(run_dir),
                  "experiment": cfg["experiment"],
                  "model_dtype_requested": execution["dtype"],
                  "device": device, "wall_seconds": time.time() - t0})
        if n in sensitivity_at and not smoke:
            t_big = time.time()
            big = checkpoint_q_metrics(
                model, tok, probe_big, layers=layers,
                batch_size=cfg["measurement"]["batch_size"],
                max_length=cfg["measurement"]["max_prompt_length"])
            m["sensitivity_2048"] = {"per_layer": big["per_layer"],
                                     "wall_seconds": time.time() - t_big}
        out_path.write_text(json.dumps(m, indent=1))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Phase 2 checkpoint {n} complete")
    (run_dir / "phase2_complete.json").write_text(json.dumps(
        {"completed_unix": time.time(), "checkpoints": ckpts}, indent=1))
    lib.append_compute(run_dir, "phase2", started, "complete", execution)


# ---------------------------------------------------------------------------
# Phase 3 — fixed-budget SVAMP adaptation, 3 seeds x 6 checkpoints
# ---------------------------------------------------------------------------

def existing_baseline(run_dir: Path, seeds: list[int], ckpt: int) -> float | None:
    """Baseline from any already-run seed of this checkpoint (greedy eval is
    seed-independent, so all seeds must reproduce it exactly)."""
    for seed in seeds:
        p = run_dir / f"adaptation_seed{seed}" / f"ckpt-{ckpt}" / "baseline.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))["acc_before"]
    return None


def phase3(cfg: dict, run_dir: Path, smoke: bool,
           only_checkpoint: int | None = None, only_seed: int | None = None,
           keep_trainer_dirs: bool = False) -> None:
    from src.adaptation import validate_adaptation_completion

    started = time.time()
    execution = run_execution(run_dir)
    recipe = cfg["adaptation"]
    seeds = recipe["seeds"]
    order = recipe["checkpoint_order"]
    if sorted(order) != sorted(recipe["checkpoints"]):
        raise RuntimeError("checkpoint_order and checkpoints disagree in config")
    if only_checkpoint is not None:
        if only_checkpoint not in order:
            raise ValueError(f"checkpoint must be one of {sorted(order)}")
        order = [only_checkpoint]
    run_seeds = [only_seed] if only_seed is not None else seeds
    if only_seed is not None and only_seed not in seeds:
        raise ValueError(f"seed must be one of {seeds}")
    if not smoke:
        lib.require_free_disk(run_dir, min_free_gib=15.0)

    material = lib.svamp_eval_sets_v15()
    if smoke:
        material = smoke_eval_material(material, cfg)
    train_ds = lib.svamp_train_ds_v15()
    reward_func = smoke_variance_reward if smoke else None

    for ckpt in order:
        ckpt_path = run_dir / f"ckpt-{ckpt}"
        if not (ckpt_path / "config.json").exists():
            raise FileNotFoundError(f"missing Phase-1 checkpoint: {ckpt_path}")
        for seed in run_seeds:
            out_dir = run_dir / f"adaptation_seed{seed}" / f"ckpt-{ckpt}"
            summary = lib.run_fixed_budget_adaptation_v15(
                checkpoint_path=ckpt_path, out_dir=out_dir,
                recipe=recipe, execution=execution, seed=seed,
                eval_material=material, train_ds=train_ds,
                reward_func=reward_func,
                expected_acc_before=existing_baseline(run_dir, seeds, ckpt),
                keep_trainer_dir=keep_trainer_dirs)
            print(f"Phase 3 ckpt {ckpt} seed {seed}: "
                  f"Δacc(300)={summary['delta_acc']:+.4f}")

    complete = True
    for ckpt in recipe["checkpoints"]:
        for seed in seeds:
            try:
                validate_adaptation_completion(
                    run_dir / f"adaptation_seed{seed}" / f"ckpt-{ckpt}",
                    recipe["budget_updates"], recipe["eval_every"])
            except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
                complete = False
    if complete:
        (run_dir / "phase3_complete.json").write_text(json.dumps(
            {"completed_unix": time.time(),
             "checkpoints": recipe["checkpoints"], "seeds": seeds}, indent=1))
    lib.append_compute(
        run_dir,
        f"phase3:{only_checkpoint if only_checkpoint is not None else 'all'}"
        f":seed{only_seed if only_seed is not None else 'all'}",
        started, "complete" if complete else "partial", execution)


# ---------------------------------------------------------------------------
# Phase 4 — analysis
# ---------------------------------------------------------------------------

def phase4(cfg: dict, run_dir: Path) -> None:
    from analysis_exp1_5 import run_exp15_analysis

    started = time.time()
    summary = run_exp15_analysis(run_dir, cfg)
    lib.append_compute(run_dir, "phase4", started, "complete",
                       run_execution(run_dir))
    print(json.dumps(summary, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("1", "2", "3", "4", "all"),
                        required=True)
    parser.add_argument("--backend", choices=("cuda", "cpu"), default="cuda",
                        help="cuda = the pre-registered RTX 4070 stratum; "
                             "cpu is allowed only together with --smoke")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny plumbing-only end-to-end run, never a result")
    parser.add_argument("--adapt-checkpoint", type=int)
    parser.add_argument("--adapt-seed", type=int)
    parser.add_argument("--keep-trainer-dirs", action="store_true",
                        help="keep per-adaptation trainer state (~4 GiB each)")
    args = parser.parse_args()

    if args.backend == "cpu" and not args.smoke:
        raise SystemExit(
            "experiment 1.5 is pre-registered on the cuda (RTX 4070) stratum; "
            "cpu is only for --smoke plumbing validation "
            "(see EXPERIMENT_1_5_PLAN_ZH.md §5)")
    cfg = lib.load_config()
    if args.smoke:
        cfg = apply_smoke_overrides(cfg)

    execution = EXECUTION_PROFILES[args.backend]
    run_dir, _ = lib.stage_a_run_dir(cfg, args.backend, execution, args.smoke)
    run_dir.mkdir(parents=True, exist_ok=True)
    acquire_runner_lock(run_dir)
    if args.phase in ("1", "all"):
        run_dir = phase1(cfg, args.backend, args.smoke)
    else:
        if not (run_dir / "config.json").exists():
            raise SystemExit(f"run dir has no config.json yet: {run_dir}; "
                             "run --phase 1 first")
        setup_backend(run_execution(run_dir))
    if args.phase in ("2", "all"):
        phase2(cfg, run_dir, args.smoke)
    if args.phase in ("3", "all"):
        phase3(cfg, run_dir, args.smoke, args.adapt_checkpoint,
               args.adapt_seed, args.keep_trainer_dirs)
    if args.phase in ("4", "all"):
        phase4(cfg, run_dir)


if __name__ == "__main__":
    main()
