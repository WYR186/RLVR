"""Phase 3: fixed-budget SVAMP adaptation — one function, run once per
checkpoint (briefing §6 Phase 3).

The recipe is IDENTICAL across checkpoints (same data, budget, optimizer,
LR, seed): 256 fixed SVAMP train questions, 50 GRPO updates, greedy eval on
the same fixed 100 SVAMP questions before and after, accuracy logged every
10 updates for the adaptation-speed curve.

Default adaptation algorithm is GRPO (keeps 'stage B = RL' apples-to-apples
with the proposal; open question #2 for the team — SFT alternative would go
here behind a flag).
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path


def run_fixed_budget_adaptation(checkpoint_path,
                                out_dir,
                                budget_updates: int = 50,
                                eval_every: int = 10,
                                seed: int = 42,
                                learning_rate: float = 1e-6,
                                num_generations: int = 8,
                                per_device_batch: int = 8,
                                grad_accum: int = 8,
                                beta: float = 0.0,
                                temperature: float = 0.7,
                                top_p: float = 1.0,
                                max_prompt_length: int = 512,
                                max_completion_length: int = 512,
                                bf16: bool = True,
                                device: str | None = None,
                                dtype_name: str | None = None,
                                autocast_dtype_name: str | None = None,
                                optim: str | None = None,
                                gradient_checkpointing: bool = True,
                                save_steps: int = 10) -> dict:
    """Adapt one checkpoint to SVAMP under a fixed budget; returns summary.

    Writes to out_dir: svamp_eval_curve.jsonl (accuracy at 0,10,...,50),
    dashboard.jsonl (per-update signals), summary.json.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOConfig, GRPOTrainer

    from .callbacks import (ExactAnswerEvalCallback, JsonlDashboardLogger,
                            LocalSafetyCallback, UpdateEffectivenessSentinel)
    from .data import svamp_eval_set, svamp_grpo_dataset
    from .evaluate import exact_answer_accuracy
    from .reward import exact_answer_reward

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    trainer_dir = out_dir / "trainer"
    resume_from = get_last_checkpoint(str(trainer_dir)) if trainer_dir.exists() else None
    partial = [out_dir / "dashboard.jsonl", out_dir / "svamp_eval_curve.jsonl"]
    if any(p.exists() and p.stat().st_size for p in partial) and resume_from is None:
        raise RuntimeError(
            f"partial adaptation artifacts exist in {out_dir}; move the directory "
            "aside and rerun so the fixed 50-update budget is not mixed across attempts")
    set_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    if dtype_name is None:
        # 2026-07-09: bf16=True now means fp32 MASTER weights + bf16 autocast.
        # Loading the params themselves in bf16 makes every lr=1e-6 AdamW
        # update round to zero (bf16 ulp ~|w|*2^-8 >> 1e-6) — verified on run
        # local_cuda_grpo_gsm8k_6a075c15808e, see
        # eaaj-pilot-win4070/WIN4070_RUN_ANALYSIS.md.
        dtype_name = "float32"
        if bf16 and autocast_dtype_name is None:
            autocast_dtype_name = "bfloat16"
    compute_dtype_name = autocast_dtype_name or dtype_name
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[dtype_name]
    if compute_dtype_name == "bfloat16" and device == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 requested on a GPU without bf16 support; use L4/A100 or log an fp16 deviation")
    if device == "mps":
        from .mps_compat import apply_mps_grpo_patches
        apply_mps_grpo_patches()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        dtype=dtype)
    model.to(device)

    eval_prompts, eval_golds = svamp_eval_set()
    train_ds = svamp_grpo_dataset()

    t0 = time.time()
    baseline_path = out_dir / "baseline.json"
    if baseline_path.exists():
        acc_before = json.loads(baseline_path.read_text())["acc_before"]
    else:
        acc_before = exact_answer_accuracy(model, tokenizer, eval_prompts, eval_golds)
        baseline_path.write_text(json.dumps({"acc_before": acc_before}, indent=1))
    print(f"[adapt] {checkpoint_path}: SVAMP accuracy BEFORE = {acc_before:.4f}")

    skip_optimizer_checkpoints = optim == "paged_adamw_8bit"
    cfg = GRPOConfig(
        output_dir=str(trainer_dir),
        seed=seed,
        max_steps=budget_updates,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        num_generations=num_generations,
        beta=beta,
        temperature=temperature,
        top_p=top_p,
        max_completion_length=max_completion_length,
        bf16=(compute_dtype_name == "bfloat16" and device == "cuda"),
        fp16=(compute_dtype_name == "float16" and device == "cuda"),
        # only override the trainer's optimizer when a profile asks for it
        **({"optim": optim} if optim else {}),
        use_cpu=(device == "cpu"),
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if gradient_checkpointing else None),
        dataloader_pin_memory=(device == "cpu"),  # unsupported no-op on MPS
        logging_steps=1,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        save_only_model=skip_optimizer_checkpoints,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        reward_funcs=exact_answer_reward,
        processing_class=tokenizer,
        callbacks=[
            JsonlDashboardLogger(out_dir / "dashboard.jsonl"),
            UpdateEffectivenessSentinel(out_dir / "update_sentinel.jsonl",
                                        every=eval_every),
            ExactAnswerEvalCallback(eval_prompts, eval_golds,
                                    out_dir / "svamp_eval_curve.jsonl",
                                    every=eval_every),
            LocalSafetyCallback(out_dir / "safety_stop.json"),
        ],
    )
    if device == "mps":
        from .mps_compat import enable_compiled_generation
        enable_compiled_generation(trainer)
    trainer.train(resume_from_checkpoint=resume_from)

    acc_after = exact_answer_accuracy(model, tokenizer, eval_prompts, eval_golds)
    print(f"[adapt] {checkpoint_path}: SVAMP accuracy AFTER = {acc_after:.4f}")

    summary = {
        "checkpoint": str(checkpoint_path),
        "budget_updates": budget_updates,
        "seed": seed,
        "algo": "grpo",
        "learning_rate": learning_rate,
        "beta": beta,
        "temperature": temperature,
        "top_p": top_p,
        "num_generations": num_generations,
        "per_device_train_batch_size": per_device_batch,
        "gradient_accumulation_steps": grad_accum,
        "max_prompt_length": max_prompt_length,
        "max_completion_length": max_completion_length,
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
        "wall_seconds": time.time() - t0,
    }
    summary_path.write_text(json.dumps(summary, indent=1))

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return summary
