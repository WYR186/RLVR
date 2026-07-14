#!/usr/bin/env python3
"""Resumable runner for the complete local Phase 1–4 pilot.

Backends (--backend):
  cpu  — original float32 CPU stratum; config dict (and therefore the run-dir
         hash) is byte-identical to the first run, so it resumes
         outputs/local_grpo_gsm8k_eac028bfcc87 in place.
  mps  — float32 Apple-GPU stratum (new run dir = new stratum; never merged
         with CPU results). Uses the two mathematically identical execution
         patches from src/mps_compat.py; see LOCAL_EXPERIMENT_PLAN.md
         2026-07-08 update for the measurements behind them.
  cuda — NVIDIA-GPU stratum, v2 (new run dir = new stratum; never merged with
         other strata): fp32 MASTER weights + bf16 autocast + 8-bit paged
         AdamW. v1 loaded the params themselves in bf16 and every lr=1e-6
         update rounded to zero (run local_cuda_grpo_gsm8k_6a075c15808e is a
         no-op control — see eaaj-pilot-win4070/WIN4070_RUN_ANALYSIS.md).
         Designed for the Windows RTX 4070 Laptop machine — read
         eaaj-pilot-win4070/WIN4070_RERUN_GUIDE.md before running it.

Backend is a Phase-1 choice only: phases 2–4 read device/dtype from the
active run's config.json so measurement and adaptation always match the
stratum they operate on.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer

from src.adaptation import (run_fixed_budget_adaptation,
                            validate_adaptation_completion)
from src.analysis import run_analysis
from src.callbacks import (ExactAnswerEvalCallback, JsonlDashboardLogger,
                           LocalSafetyCallback, SaveAtSteps,
                           UpdateEffectivenessSentinel)
from src.data import gsm8k_eval_set, gsm8k_grpo_dataset, load_probe_prompts
from src.metrics import checkpoint_q_metrics
from src.preflight import sparse_reward_preflight
from src.repro import config_hash, get_active_run, runtime_manifest, set_active_run
from src.reward import exact_answer_reward

EXECUTION_PROFILES = {
    # Byte-identical to the original runner config: keeps the existing CPU
    # run dir hash (local_grpo_gsm8k_eac028bfcc87) resumable. Do not edit.
    "cpu": {
        "backend": "pytorch_cpu", "device": "cpu", "dtype": "float32",
        "torch_threads": 12, "gradient_checkpointing": False,
        "reason": "PyTorch MPS unavailable on macOS 26.5.1",
    },
    "mps": {
        "backend": "pytorch_mps", "device": "mps", "dtype": "float32",
        "torch_threads": 12, "gradient_checkpointing": False,
        "grpo_kernel_patches": ["chunked_selective_log_softmax",
                                "compiled_generation"],
        "reason": ("MPS re-validated available on macOS 26.5.1 (2026-07-08); "
                   "float32 kept to match the CPU stratum dtype. Stock TRL "
                   "kernels are pathologically slow on MPS — see "
                   "LOCAL_EXPERIMENT_PLAN.md 2026-07-08 update."),
    },
    "cuda": {
        "backend": "pytorch_cuda", "device": "cuda", "dtype": "float32",
        "autocast_dtype": "bfloat16", "optim": "paged_adamw_8bit",
        "torch_threads": 8, "gradient_checkpointing": True,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 16,
        "reason": ("Windows RTX 4070 Laptop (8 GiB VRAM) stratum, v2 "
                   "precision fix: fp32 MASTER weights + bf16 autocast, "
                   "because pure-bf16 params at lr=1e-6 round every update "
                   "to zero (v1 run local_cuda_grpo_gsm8k_6a075c15808e "
                   "trained a no-op; see eaaj-pilot-win4070/"
                   "WIN4070_RUN_ANALYSIS.md). bitsandbytes paged_adamw_8bit "
                   "keeps fp32 master + optimizer states inside 8 GiB; "
                   "micro-batch 4 x grad-accum 16 keeps the same "
                   "64-completion effective update (measured 2026-07-09)."),
    },
}

DTYPES = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}


def load_pilot() -> dict:
    return json.loads((PROJECT / "pilot_config.json").read_text())


def local_run_dir(pilot: dict, backend: str = "cpu") -> tuple[Path, dict]:
    execution = EXECUTION_PROFILES[backend]
    stage_a = dict(pilot["stage_a"])
    for key in ("per_device_train_batch_size", "gradient_accumulation_steps"):
        if key in execution:
            stage_a[key] = execution[key]
    cfg = {
        "model": pilot["model_id"], "model_revision": pilot["model_revision"],
        "seed": pilot["seed"], **stage_a,
        "execution": execution,
    }
    prefix = "local_grpo_gsm8k" if backend == "cpu" else f"local_{backend}_grpo_gsm8k"
    run_dir = PROJECT / "outputs" / f"{prefix}_{config_hash(cfg)}"
    return run_dir, cfg


def run_execution(run_dir: Path) -> dict:
    """Execution profile recorded in the run's own config.json."""
    return json.loads((run_dir / "config.json").read_text())["execution"]


def _pid_alive(pid: int) -> bool:
    """Liveness probe for the runner lock. POSIX uses signal 0; on Windows
    os.kill(pid, 0) TERMINATES the target process instead of probing it, so
    the pid is queried through OpenProcess there."""
    import os

    if sys.platform == "win32":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False  # gone, or not ours — same reclaim semantics as before
    return True


def acquire_runner_lock(run_dir: Path) -> None:
    """Refuse to start if another runner process is live on this run dir.

    Two concurrent trainers on one run dir race on trainer checkpoints and
    duplicate optimizer steps (this nearly happened on 2026-07-08). The lock
    holds the owner pid; a lock whose pid is dead is stale and is reclaimed.
    """
    import atexit
    import os

    lock = run_dir / "runner.lock"
    if lock.exists():
        try:
            other = int(lock.read_text().strip())
        except ValueError:
            other = None  # unreadable lock -> stale -> reclaim
        if other is not None and _pid_alive(other):
            raise RuntimeError(
                f"another runner (pid {other}) is already active on {run_dir}; "
                "stop it first or wait for it to finish")
    lock.write_text(str(os.getpid()))
    atexit.register(lambda: lock.unlink(missing_ok=True))


def setup_backend(execution: dict) -> None:
    torch.set_num_threads(execution.get("torch_threads", 12))
    if execution["device"] == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("mps backend requested but MPS is unavailable")
        from src.mps_compat import apply_mps_grpo_patches
        apply_mps_grpo_patches()
    elif execution["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda backend requested but CUDA is unavailable")
        wants_bf16 = "bfloat16" in (execution["dtype"],
                                    execution.get("autocast_dtype"))
        if wants_bf16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "bf16 requested but this GPU lacks bf16 support; "
                "log an fp16 deviation before switching dtypes")


def append_compute(run_dir: Path, phase: str, started: float, status: str,
                   execution: dict) -> None:
    path = run_dir / "local_compute.jsonl"
    row = {"phase": phase, "status": status, "started_unix": started,
           "ended_unix": time.time(), "wall_seconds": time.time() - started,
           "device": execution["device"], "dtype": execution["dtype"]}
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def phase1(pilot: dict, backend: str = "cpu") -> Path:
    started = time.time()
    run_dir, cfg = local_run_dir(pilot, backend)
    execution = cfg["execution"]
    device = execution["device"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    if not (run_dir / "manifest.json").exists():
        (run_dir / "manifest.json").write_text(
            json.dumps(runtime_manifest(PROJECT, cfg), indent=1))
    set_active_run(PROJECT, run_dir)
    if (run_dir / "phase1_complete.json").exists() and all(
            (run_dir / f"ckpt-{n}" / "config.json").exists()
            for n in cfg["checkpoint_steps"]):
        print(f"Phase 1 already complete: {run_dir}")
        return run_dir
    setup_backend(execution)
    set_seed(cfg["seed"])

    tok = AutoTokenizer.from_pretrained(
        cfg["model"], revision=cfg["model_revision"], local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], revision=cfg["model_revision"],
        dtype=DTYPES[execution["dtype"]], local_files_only=True)
    if device != "cpu":
        model.to(device)
    ckpt0 = run_dir / "ckpt-0"
    if not ckpt0.exists():
        model.save_pretrained(ckpt0, safe_serialization=True)
        tok.save_pretrained(ckpt0)

    train_ds = gsm8k_grpo_dataset()
    eval_prompts, eval_golds, eval_metadata = gsm8k_eval_set(return_metadata=True)
    preflight_path = run_dir / "sparse_reward_preflight.json"
    if preflight_path.exists():
        preflight = json.loads(preflight_path.read_text())
    else:
        preflight = sparse_reward_preflight(
            model, tok, train_ds["prompt"][:8], train_ds["answer"][:8],
            num_generations=cfg["num_generations"], temperature=cfg["temperature"],
            top_p=cfg["top_p"], max_new_tokens=cfg["max_completion_length"])
        preflight_path.write_text(json.dumps(preflight, indent=1))
    if not preflight["has_grpo_signal"]:
        raise RuntimeError("base-model exact reward preflight has no GRPO signal")

    # Preflight generation must not perturb the formal training RNG stream.
    set_seed(cfg["seed"])
    trainer_dir = run_dir / "trainer"
    resume_from = get_last_checkpoint(str(trainer_dir)) if trainer_dir.exists() else None
    # Autocast/compute dtype may differ from the master-weight dtype (cuda v2
    # profile); strata without autocast_dtype compute in their master dtype.
    compute_dtype = execution.get("autocast_dtype", execution["dtype"])
    skip_optimizer_checkpoints = execution.get("optim") == "paged_adamw_8bit"
    args = GRPOConfig(
        output_dir=str(trainer_dir), seed=cfg["seed"], max_steps=cfg["max_steps"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_generations=cfg["num_generations"], beta=cfg["beta"],
        temperature=cfg["temperature"], top_p=cfg["top_p"],
        max_completion_length=cfg["max_completion_length"],
        use_cpu=(device == "cpu"),
        bf16=(compute_dtype == "bfloat16"),
        fp16=(compute_dtype == "float16"),
        # only override the trainer's optimizer when a profile asks for it
        **({"optim": execution["optim"]} if "optim" in execution else {}),
        gradient_checkpointing=execution.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False}
            if execution.get("gradient_checkpointing") else None),
        dataloader_pin_memory=(device != "mps"),  # unsupported no-op on MPS
        logging_steps=1, save_strategy="steps", save_steps=25,
        save_total_limit=2, save_only_model=skip_optimizer_checkpoints,
        report_to="none")
    trainer = GRPOTrainer(
        model=model, args=args, train_dataset=train_ds,
        reward_funcs=exact_answer_reward, processing_class=tok,
        callbacks=[
            JsonlDashboardLogger(run_dir / "dashboard.jsonl"),
            UpdateEffectivenessSentinel(run_dir / "update_sentinel.jsonl",
                                        every=25),
            SaveAtSteps(cfg["checkpoint_steps"][1:], run_dir, tokenizer=tok),
            ExactAnswerEvalCallback(
                eval_prompts, eval_golds, run_dir / "gsm8k_eval.jsonl",
                every=cfg["eval_every"], also_at_step0=True,
                item_metadata=eval_metadata),
            LocalSafetyCallback(run_dir / "safety_stop.json"),
        ])
    if device == "mps":
        from src.mps_compat import enable_compiled_generation
        enable_compiled_generation(trainer)
    trainer.train(resume_from_checkpoint=resume_from)
    missing = [n for n in cfg["checkpoint_steps"] if not (run_dir / f"ckpt-{n}" / "config.json").exists()]
    if missing:
        append_compute(run_dir, "phase1", started, "incomplete", execution)
        raise RuntimeError(f"Phase 1 stopped before required checkpoints: {missing}")
    (run_dir / "phase1_complete.json").write_text(json.dumps(
        {"completed_unix": time.time(), "global_step": trainer.state.global_step,
         "resume_from_checkpoint": resume_from}, indent=1))
    append_compute(run_dir, "phase1", started, "complete", execution)
    print(f"Phase 1 complete: {run_dir}")
    return run_dir


def phase2(pilot: dict, run_dir: Path) -> None:
    started = time.time()
    execution = run_execution(run_dir)
    device, dtype_name = execution["device"], execution["dtype"]
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[dtype_name]
    ckpts = pilot["stage_a"]["checkpoint_steps"]
    layers = tuple(pilot["measurement"]["layers"])
    out_dir = run_dir / "measurements"
    out_dir.mkdir(exist_ok=True)
    probe, probe_big = load_probe_prompts(), load_probe_prompts(big=True)
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
            batch_size=pilot["measurement"]["batch_size"],
            max_length=pilot["measurement"]["max_prompt_length"])
        m.update({"checkpoint": n, "run_dir": str(run_dir),
                  "model_dtype_requested": dtype_name, "device": device,
                  "wall_seconds": time.time() - t0})
        if n in (0, 200):
            t_big = time.time()
            big = checkpoint_q_metrics(
                model, tok, probe_big, layers=layers,
                batch_size=pilot["measurement"]["batch_size"],
                max_length=pilot["measurement"]["max_prompt_length"])
            m["sensitivity_2048"] = {
                "per_layer": big["per_layer"], "wall_seconds": time.time() - t_big}
        out_path.write_text(json.dumps(m, indent=1))
        del model
        gc.collect()
        print(f"Phase 2 checkpoint {n} complete")
    (run_dir / "phase2_complete.json").write_text(json.dumps(
        {"completed_unix": time.time(), "checkpoints": ckpts}, indent=1))
    append_compute(run_dir, "phase2", started, "complete", execution)


def phase3(pilot: dict, run_dir: Path, only_checkpoint: int | None = None) -> None:
    started = time.time()
    execution = run_execution(run_dir)
    recipe = pilot["adaptation"]
    per_device_batch = execution.get(
        "per_device_train_batch_size", recipe["per_device_train_batch_size"])
    grad_accum = execution.get(
        "gradient_accumulation_steps", recipe["gradient_accumulation_steps"])
    ckpts = pilot["stage_a"]["checkpoint_steps"]
    if only_checkpoint is not None:
        if only_checkpoint not in ckpts:
            raise ValueError(f"checkpoint must be one of {ckpts}")
        ckpts = [only_checkpoint]
    root = run_dir / "adaptation"
    root.mkdir(exist_ok=True)
    for n in ckpts:
        summary = run_fixed_budget_adaptation(
            checkpoint_path=run_dir / f"ckpt-{n}", out_dir=root / f"ckpt-{n}",
            budget_updates=recipe["budget_updates"], eval_every=recipe["eval_every"],
            seed=pilot["seed"], learning_rate=recipe["learning_rate"],
            num_generations=recipe["num_generations"],
            per_device_batch=per_device_batch,
            grad_accum=grad_accum, beta=recipe["beta"],
            temperature=recipe["temperature"], top_p=recipe["top_p"],
            max_prompt_length=recipe["max_prompt_length"],
            max_completion_length=recipe["max_completion_length"],
            bf16=False, device=execution["device"],
            dtype_name=execution["dtype"],
            autocast_dtype_name=execution.get("autocast_dtype"),
            optim=execution.get("optim"),
            gradient_checkpointing=execution.get("gradient_checkpointing", False),
            save_steps=10)
        print(f"Phase 3 checkpoint {n}: Δacc={summary['delta_acc']:+.4f}")
    expected = pilot["stage_a"]["checkpoint_steps"]
    complete = True
    for n in expected:
        try:
            validate_adaptation_completion(
                root / f"ckpt-{n}", recipe["budget_updates"], recipe["eval_every"])
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
            complete = False
            break
    if complete:
        (run_dir / "phase3_complete.json").write_text(json.dumps(
            {"completed_unix": time.time(), "checkpoints": expected}, indent=1))
    append_compute(run_dir, f"phase3:{only_checkpoint or 'all'}", started,
                   "complete", execution)


def phase4(pilot: dict, run_dir: Path) -> None:
    started = time.time()
    summary = run_analysis(run_dir, pilot)
    append_compute(run_dir, "phase4", started, "complete", run_execution(run_dir))
    print(json.dumps(summary, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("1", "2", "3", "4", "all"), required=True)
    parser.add_argument("--adapt-checkpoint", type=int)
    parser.add_argument("--backend", choices=tuple(EXECUTION_PROFILES), default="cpu",
                        help="Phase-1 execution stratum; phases 2-4 always follow "
                             "the active run's recorded execution profile")
    args = parser.parse_args()
    pilot = load_pilot()
    if args.phase in ("1", "all"):
        run_dir, _ = local_run_dir(pilot, args.backend)
        run_dir.mkdir(parents=True, exist_ok=True)
        acquire_runner_lock(run_dir)
        run_dir = phase1(pilot, args.backend)
    else:
        run_dir = get_active_run(PROJECT)
        acquire_runner_lock(run_dir)
        setup_backend(run_execution(run_dir))
    if args.phase in ("2", "all"):
        phase2(pilot, run_dir)
    if args.phase in ("3", "all"):
        phase3(pilot, run_dir, args.adapt_checkpoint)
    if args.phase in ("4", "all"):
        phase4(pilot, run_dir)


if __name__ == "__main__":
    main()
