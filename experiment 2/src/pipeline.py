"""Stage-1 (Math) / stage-2 (Simulation) LoRA GRPO pipeline for exp2's Colab
variant (EXPERIMENT_2_COLAB_PLAN.md).

**Status: reviewed, not yet run against a live GPU.** Phase 0's smoke test
(2 stage-A updates + 2 stage-B updates, plan §3 Phase 0 step 7) is the first
real exercise of this code path and exists specifically to catch what a
read-through cannot — treat every function here accordingly until that smoke
test passes and is committed.

Operates on GURU row dicts (`prompt`, `ground_truth`, `data_source`,
`extra_info` — see `guru_data.py`), matching the confirmed real contract and
the WIN4070 track's `exp2_4070_data.py`/`exp2_4070_reward.py`, NOT the
`answer`/`domain` columns an earlier heuristic-discovery version of this
module assumed.

Reuses two modules from `eaaj-pilot/src` that are dataset-agnostic and have
no import-time relative-import dependencies: `metrics.py` (activation Q
metrics — works on any decoder-only model exposing `.model.layers`, PEFT
included) and four callback classes from `callbacks.py`
(`JsonlDashboardLogger`, `SaveAtSteps`, `LocalSafetyCallback`,
`UpdateEffectivenessSentinel`). They are loaded by explicit file path under
synthetic module names (`_load_pilot_module` below), NOT via
`sys.path`-based `import src...`, because both `eaaj-pilot/src` and this
directory (`experiment 2/src`) are top-level packages named `src` — putting
both roots on `sys.path` and importing `src.anything` would be ambiguous
about which `src` wins. `ExactAnswerEvalCallback` and everything in
`preflight.py`/`evaluate.py`/`adaptation.py`'s task-specific glue are NOT
reused, because they hardcode the GSM8K/SVAMP numeric reward and file names
(e.g. `svamp_eval_curve.jsonl`) — this module reimplements the small amount
of generic logic they contain against `guru_reward.py`'s vendored-verifier
wrapper instead.
"""
from __future__ import annotations

import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

EXP2_ROOT = Path(__file__).resolve().parent.parent
ALGOVERSE_ROOT = EXP2_ROOT.parent
EAAJ_PILOT_ROOT = ALGOVERSE_ROOT / "eaaj-pilot"
DATA_DIR = EXP2_ROOT / "data"

from .guru_reward import exact_correct, select_reward_fn  # noqa: E402


def _load_pilot_module(synthetic_name: str, filename: str):
    path = EAAJ_PILOT_ROOT / "src" / filename
    if not path.exists():
        raise RuntimeError(
            f"expected eaaj-pilot module at {path}, but it does not exist — "
            "this pipeline reuses eaaj-pilot/src for dataset-agnostic "
            "activation-metric and callback code; check the sibling-directory "
            "layout hasn't changed")
    spec = importlib.util.spec_from_file_location(synthetic_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[synthetic_name] = module
    spec.loader.exec_module(module)
    return module


def _pilot():
    """Lazy, memoized load of the reused eaaj-pilot pieces."""
    if not hasattr(_pilot, "_cache"):
        metrics = _load_pilot_module("_eaaj_pilot_metrics", "metrics.py")
        callbacks = _load_pilot_module("_eaaj_pilot_callbacks", "callbacks.py")
        _pilot._cache = dict(
            checkpoint_q_metrics=metrics.checkpoint_q_metrics,
            JsonlDashboardLogger=callbacks.JsonlDashboardLogger,
            SaveAtSteps=callbacks.SaveAtSteps,
            LocalSafetyCallback=callbacks.LocalSafetyCallback,
            UpdateEffectivenessSentinel=callbacks.UpdateEffectivenessSentinel,
        )
    return _pilot._cache


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_peft_model(model_id: str, peft_cfg: dict, device: str,
                     revision: str | None = None, adapter_path=None):
    """Load the frozen bf16 base + a LoRA adapter (fresh, or loaded from
    `adapter_path` if given). LoRA adapter parameters are upcast to float32
    after wrapping — same rounding-hazard discipline as the 4070 plan's
    fp32-master-weights choice, applied to the parameters that actually
    receive gradient updates here (plan §1, "LoRA adapter dtype" row).
    """
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch.bfloat16)

    if adapter_path is not None:
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=peft_cfg["r"], lora_alpha=peft_cfg["lora_alpha"],
            lora_dropout=peft_cfg["lora_dropout"],
            target_modules=peft_cfg["target_modules"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_config)

    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            param.data = param.data.float()

    model.to(device)
    return model, tokenizer


def _make_lora_sentinel_class():
    """Subclass of eaaj-pilot's UpdateEffectivenessSentinel that samples ONLY
    trainable (requires_grad, i.e. LoRA) parameters.

    The parent samples every parameter tensor, which was correct for full
    fine-tuning: the whole parameter vector is supposed to move. Under LoRA
    the 7B frozen base never moves by construction, so including it dilutes
    the relative-change measurement toward zero and the step-25 kill-gate
    could flag a perfectly healthy LoRA run as ineffective (or, symmetric
    failure, the dilution could mask a genuinely dead adapter behind
    numerical noise from dtype casts). Sampling the trainable subset restores
    the parent's intended semantics — "did the parameters that are supposed
    to move actually move" — for the LoRA case.

    NOTE the parent's reference scales (healthy ~5e-7/window, broken ~3e-9,
    measured on full-parameter 0.5B runs) do NOT transfer to this
    measurement; the warn threshold 1e-8 is kept as a conservative
    something-is-deeply-wrong floor, and Phase 0's smoke test records the
    first real healthy-LoRA reference value.
    """
    parent = _pilot()["UpdateEffectivenessSentinel"]

    class LoraUpdateEffectivenessSentinel(parent):
        def _sample(self, model):
            import torch

            chunks = []
            with torch.no_grad():
                for name, p in sorted(model.named_parameters()):
                    if not p.requires_grad:
                        continue
                    flat = p.detach().reshape(-1)
                    stride = max(1, flat.numel() // self.max_per_tensor)
                    chunks.append(flat[::stride].float().cpu())
            if not chunks:
                raise RuntimeError(
                    "no trainable parameters found — the LoRA adapter is "
                    "missing or fully frozen, so training this model would "
                    "be a guaranteed no-op")
            return torch.cat(chunks)

    return LoraUpdateEffectivenessSentinel


def unwrap_for_hooks(model):
    """`checkpoint_q_metrics`/`collect_probe_activations` (eaaj-pilot
    metrics.py) hook `model.model.layers[l]` by module reference. A
    `PeftModelForCausalLM` exposes the same underlying module tree (with
    individual Linear layers swapped for LoRA-wrapped ones, forward pass
    unchanged in shape) via `get_base_model()` — use that so hooks land on
    the *adapted* model's forward pass, not a separately merged copy.
    """
    return model.get_base_model() if hasattr(model, "get_base_model") else model


# ---------------------------------------------------------------------------
# Generic preflight / eval, operating on GURU row dicts
# ---------------------------------------------------------------------------

def guru_sparse_reward_preflight(model, tokenizer, rows: list[dict], reward_mode: str, *,
                                 num_generations: int, temperature: float = 0.7,
                                 top_p: float = 1.0, max_new_tokens: int = 512,
                                 min_variable_groups: int = 2) -> dict:
    """Sparse-reward preflight, tightened per the v9 finding
    (`FINDING_GROUP_SIZE_REWARD_VARIANCE.md`): the naive gate ("STOP iff
    EVERY group has zero variance") let a run through that was 47%
    zero-gradient updates. This version (a) defaults to requiring
    `min_variable_groups >= 2` combined-reward-variable groups, not just 1,
    matching v9's tightened threshold, and (b) tracks the EXACT-correctness
    channel separately from the combined (possibly format-shaped) reward —
    the finding's central point is that combined variance can come almost
    entirely from a shaping term, not from reasoning signal, so reporting
    only the combined count hides that.
    """
    import torch

    reward_fn = select_reward_fn(reward_mode)
    device = next(model.parameters()).device
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    groups = []
    try:
        with torch.no_grad():
            for row in rows:
                enc = tokenizer(row["prompt"], return_tensors="pt", truncation=True,
                                max_length=2048).to(device)
                out = model.generate(
                    **enc, do_sample=True, num_return_sequences=num_generations,
                    temperature=temperature, top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
                gen = out[:, enc["input_ids"].shape[1]:]
                completions = tokenizer.batch_decode(gen, skip_special_tokens=True)
                gt = [row["ground_truth"]] * len(completions)
                ds = [row["data_source"]] * len(completions)
                info = [row.get("extra_info", {})] * len(completions)
                combined = reward_fn(completions=completions, ground_truth=gt,
                                     data_source=ds, extra_info=info)
                exact = [float(exact_correct(c, row["ground_truth"], row["data_source"],
                                             row.get("extra_info")))
                        for c in completions]
                groups.append({
                    "id": row.get("id"), "ground_truth": row["ground_truth"],
                    "combined_rewards": combined, "exact_rewards": exact,
                    "combined_variable": len(set(combined)) > 1,
                    "exact_variable": len(set(exact)) > 1,
                    "completion_tails": [c[-500:] for c in completions],
                })
    finally:
        tokenizer.padding_side = old_padding_side

    n_combined_variable = sum(g["combined_variable"] for g in groups)
    n_exact_variable = sum(g["exact_variable"] for g in groups)
    return {
        "reward_mode": reward_mode, "n_prompts": len(groups), "num_generations": num_generations,
        "n_exact_correct": int(sum(sum(g["exact_rewards"]) for g in groups)),
        "groups_with_combined_variance": n_combined_variable,
        "groups_with_exact_variance": n_exact_variable,
        "min_variable_groups_required": min_variable_groups,
        "has_grpo_signal": n_combined_variable >= min_variable_groups,
        "groups": groups,
    }


def guru_greedy_accuracy(model, tokenizer, rows: list[dict], *,
                         batch_size: int = 8, max_new_tokens: int = 512,
                         max_prompt_length: int = 2048, return_details: bool = False):
    """Greedy exact-correctness accuracy over `rows` (deterministic eval,
    independent of the sampling temperature used for GRPO rollouts) using
    the vendored verifier via `guru_reward.exact_correct`."""
    import torch

    model.eval()
    device = next(model.parameters()).device
    hits, details = [], []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        with torch.no_grad():
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                prompts = [r["prompt"] for r in batch]
                enc = tokenizer(prompts, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_prompt_length).to(device)
                out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
                gen = out[:, enc["input_ids"].shape[1]:]
                texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
                for row, text in zip(batch, texts):
                    ok = exact_correct(text, row["ground_truth"], row["data_source"],
                                       row.get("extra_info"))
                    hits.append(ok)
                    if return_details:
                        details.append({"id": row.get("id"), "completion": text, "correct": ok})
    finally:
        tokenizer.padding_side = old_padding_side
    acc = sum(hits) / len(hits) if hits else 0.0
    return (acc, details) if return_details else acc


def _make_eval_callback_class():
    """Builds a `TrainerCallback` subclass for periodic greedy-eval accuracy
    during GRPO training, built lazily inside a function because
    `TrainerCallback` is only importable once torch/transformers are
    available and this module must stay importable without torch (per the
    module docstring)."""
    from transformers import TrainerCallback

    class _GuruEvalCallback(TrainerCallback):
        def __init__(self, rows, out_path, every: int = 10,
                    batch_size: int = 8, max_new_tokens: int = 512):
            self.rows = rows
            self.out_path = Path(out_path)
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            self.every = every
            self.batch_size = batch_size
            self.max_new_tokens = max_new_tokens
            self._tokenizer = None

        def _run_eval(self, model, step):
            t0 = time.time()
            acc, details = guru_greedy_accuracy(
                model, self._tokenizer, self.rows,
                batch_size=self.batch_size, max_new_tokens=self.max_new_tokens,
                return_details=True)
            row = {"step": step, "accuracy": acc, "n_eval": len(details),
                  "n_correct": sum(d["correct"] for d in details),
                  "eval_seconds": time.time() - t0}
            with self.out_path.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            print(f"[eval] step {step}: accuracy={acc:.4f}")
            model.train()

        def on_train_begin(self, args, state, control, model=None,
                           processing_class=None, tokenizer=None, **kwargs):
            self._tokenizer = processing_class or tokenizer

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if state.global_step % self.every == 0:
                self._run_eval(model, state.global_step)

    return _GuruEvalCallback


# ---------------------------------------------------------------------------
# Fixed-budget completion bookkeeping (generic; adapted from
# eaaj-pilot/src/adaptation.py with domain-neutral file names — that module's
# validator hardcodes `svamp_eval_curve.jsonl`, which would be a misleading
# name for a Simulation-domain run)
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def validate_stage_b_completion(out_dir, budget_updates: int, eval_every: int) -> dict:
    out_dir = Path(out_dir)
    safety_path = out_dir / "safety_stop.json"
    if safety_path.exists():
        raise RuntimeError(f"safety stop exists: {safety_path}")
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"summary missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("completion_status") != "complete":
        raise RuntimeError("completion_status is not complete")
    if int(summary.get("actual_updates", -1)) != budget_updates:
        raise RuntimeError("actual_updates does not match the fixed budget")

    expected_steps = list(range(eval_every, budget_updates + 1, eval_every))
    curve = _read_jsonl(out_dir / "stageb_eval_curve.jsonl")
    curve_steps = [int(row["step"]) for row in curve]
    if curve_steps != expected_steps:
        raise RuntimeError(f"eval curve steps {curve_steps} != expected {expected_steps}")
    sentinel = _read_jsonl(out_dir / "update_sentinel.jsonl")
    sentinel_steps = [int(row["step"]) for row in sentinel]
    if sentinel_steps != expected_steps:
        raise RuntimeError(
            f"sentinel steps {sentinel_steps} != expected {expected_steps}")
    if not all(row.get("updates_effective") is True for row in sentinel):
        raise RuntimeError("one or more sentinel windows are ineffective")
    return summary


def fixed_budget_completion(trainer, out_dir, requested_updates: int,
                            safety_stop_path=None) -> dict:
    out_dir = Path(out_dir)
    actual_updates = int(trainer.state.global_step)
    if actual_updates == requested_updates:
        return {"requested_updates": requested_updates, "actual_updates": actual_updates,
                "completion_status": "complete"}
    safety_stop_path = Path(safety_stop_path) if safety_stop_path else None
    incomplete = {
        "requested_updates": requested_updates, "actual_updates": actual_updates,
        "completion_status": "incomplete",
        "reason": "trainer returned before the requested fixed budget",
        "safety_stop_path": (str(safety_stop_path)
                             if safety_stop_path is not None and safety_stop_path.exists()
                             else None),
        "run_path": str(out_dir), "timestamp_unix": time.time(),
    }
    (out_dir / "incomplete.json").write_text(json.dumps(incomplete, indent=1), encoding="utf-8")
    raise RuntimeError(f"incomplete: requested {requested_updates}, got {actual_updates}")


# ---------------------------------------------------------------------------
# Phase 1 — stage-A GRPO (Math), LoRA
# ---------------------------------------------------------------------------

def run_stage_a_grpo(model_id: str, peft_cfg: dict, train_dataset, out_dir,
                     checkpoint_steps: list[int], max_steps: int, reward_mode: str,
                     learning_rate: float, per_device_batch: int, grad_accum: int,
                     num_generations: int, beta: float, temperature: float, top_p: float,
                     max_completion_length: int,
                     revision: str | None = None, seed: int = 42, device: str | None = None,
                     eval_every: int = 25):
    """Stage-1 (Math) GRPO with LoRA. `train_dataset` is a HF Dataset with
    columns `prompt`/`ground_truth`/`data_source`/`extra_info` (from
    `guru_data.to_hf_dataset`) already filtered to `token_filter_max` by
    `guru_data.build_exp2_splits` — TRL's `GRPOConfig` has no prompt-length
    field of its own (checked against the installed trl package: only
    `log_unique_prompts` matches "prompt"), so prompt length is controlled
    entirely upstream at the data layer, not here. Saves adapter-only
    checkpoints at `checkpoint_steps` (`SaveAtSteps.on_step_end` never fires
    for step 0 — the caller must save ckpt-0 itself, before `trainer.train()`,
    exactly as the 4070 plan's notebooks do for the base model)."""
    import torch
    from transformers import set_seed
    from trl import GRPOConfig, GRPOTrainer

    pilot = _pilot()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = build_peft_model(model_id, peft_cfg, device, revision=revision)
    model.save_pretrained(out_dir / "ckpt-0")  # adapter-only; identity at init

    t0 = time.time()
    reward_fn = select_reward_fn(reward_mode)
    sentinel_cls = _make_lora_sentinel_class()
    cfg = GRPOConfig(
        output_dir=str(out_dir / "trainer"), seed=seed, max_steps=max_steps,
        learning_rate=learning_rate, per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum, num_generations=num_generations,
        beta=beta, temperature=temperature, top_p=top_p,
        max_completion_length=max_completion_length,
        bf16=(device == "cuda"), optim="paged_adamw_8bit",
        use_cpu=(device == "cpu"), gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, save_strategy="no", report_to="none",
    )
    trainer = GRPOTrainer(
        model=model, args=cfg, train_dataset=train_dataset,
        reward_funcs=reward_fn, processing_class=tokenizer,
        callbacks=[
            pilot["JsonlDashboardLogger"](out_dir / "dashboard.jsonl"),
            sentinel_cls(out_dir / "update_sentinel.jsonl", every=eval_every),
            pilot["SaveAtSteps"]([s for s in checkpoint_steps if s > 0], out_dir, tokenizer=tokenizer),
            pilot["LocalSafetyCallback"](out_dir / "safety_stop.json"),
        ],
    )
    trainer.train()

    completion = fixed_budget_completion(trainer, out_dir, max_steps, out_dir / "safety_stop.json")
    summary = {"task": "Math", "algo": "grpo", "peft": "lora", "reward_mode": reward_mode,
              **completion, "learning_rate": learning_rate, "seed": seed,
              "checkpoint_steps": checkpoint_steps,
              "wall_seconds": time.time() - t0}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


# ---------------------------------------------------------------------------
# Phase 2 — T_t, zero-shot transfer control
# ---------------------------------------------------------------------------

def run_transfer_T(model_id: str, peft_cfg: dict, stage_a_out_dir, checkpoint_steps: list[int],
                   eval_rows: list[dict], out_path, revision: str | None = None,
                   device: str | None = None):
    """Score_B(M_{A,t}) for every checkpoint including ckpt-0 (= M_0, the
    pure base model — no adapter attached). `eval_rows` are the frozen
    stage-B eval rows."""
    import torch

    stage_a_out_dir = Path(stage_a_out_dir)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    scores = {}
    for step in checkpoint_steps:
        ckpt_dir = stage_a_out_dir / f"ckpt-{step}"
        model, tokenizer = build_peft_model(
            model_id, peft_cfg, device, revision=revision, adapter_path=ckpt_dir)
        score = guru_greedy_accuracy(model, tokenizer, eval_rows)
        scores[str(step)] = score
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    m0_score = scores.get("0")
    if m0_score is None:
        raise RuntimeError(
            "checkpoint_steps must include 0: T_t is defined relative to "
            "M_0 (ckpt-0 adapter is the identity, so ckpt-0 == the base model)")
    result = {
        "scores_by_checkpoint": scores,
        "M0_score": m0_score,
        "T_t": {step: score - m0_score for step, score in scores.items()},
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1))
    return result


# ---------------------------------------------------------------------------
# Phase 2b — activation Q metrics
# ---------------------------------------------------------------------------

def measure_checkpoint_q(model_id: str, peft_cfg: dict, adapter_path, probe_prompts: list[str],
                         layers: tuple[int, ...], revision: str | None = None,
                         device: str | None = None, batch_size: int = 8,
                         max_length: int = 512) -> dict:
    """`probe_prompts` are plain prompt strings (no reward needed for
    activation collection) — callers pass `[r["prompt"] for r in probe_rows]`."""
    import torch

    pilot = _pilot()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = build_peft_model(
        model_id, peft_cfg, device, revision=revision, adapter_path=adapter_path)
    unwrapped = unwrap_for_hooks(model)
    result = pilot["checkpoint_q_metrics"](
        unwrapped, tokenizer, probe_prompts, layers=layers,
        batch_size=batch_size, max_length=max_length)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Phase 3 — stage-B GRPO (Simulation) from a given stage-A checkpoint
# ---------------------------------------------------------------------------

def run_stage_b_adaptation(model_id: str, peft_cfg: dict, stage_a_checkpoint,
                           train_dataset, eval_rows: list[dict], out_dir,
                           budget_updates: int, eval_every: int, reward_mode: str,
                           learning_rate: float,
                           per_device_batch: int, grad_accum: int, num_generations: int,
                           beta: float, temperature: float, top_p: float,
                           max_completion_length: int,
                           revision: str | None = None, seed: int = 42,
                           device: str | None = None):
    """Fixed-budget Simulation-domain GRPO from `stage_a_checkpoint` (a
    stage-A adapter dir, or None for the ckpt-0 / stage-2-alone baseline —
    trains a fresh LoRA adapter on the base model in that case). Both
    `train_dataset` (HF Dataset) and `eval_rows` (list[dict]) carry
    `prompt`/`ground_truth`/`data_source`/`extra_info`, already filtered to
    `token_filter_max` upstream — see `run_stage_a_grpo`'s docstring for why
    there is no `max_prompt_length` parameter here."""
    import torch
    from transformers import set_seed
    from trl import GRPOConfig, GRPOTrainer

    pilot = _pilot()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        return validate_stage_b_completion(out_dir, budget_updates, eval_every)
    if (out_dir / "safety_stop.json").exists():
        raise RuntimeError(f"refusing to resume safety-stopped run in {out_dir}")
    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = build_peft_model(
        model_id, peft_cfg, device, revision=revision, adapter_path=stage_a_checkpoint)

    t0 = time.time()
    acc_before = guru_greedy_accuracy(model, tokenizer, eval_rows)
    (out_dir / "baseline.json").write_text(json.dumps({"acc_before": acc_before}, indent=1))
    print(f"[adapt] {stage_a_checkpoint}: Simulation accuracy BEFORE = {acc_before:.4f}")

    reward_fn = select_reward_fn(reward_mode)
    eval_callback_cls = _make_eval_callback_class()
    cfg = GRPOConfig(
        output_dir=str(out_dir / "trainer"), seed=seed, max_steps=budget_updates,
        learning_rate=learning_rate, per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum, num_generations=num_generations,
        beta=beta, temperature=temperature, top_p=top_p,
        max_completion_length=max_completion_length,
        bf16=(device == "cuda"), optim="paged_adamw_8bit",
        use_cpu=(device == "cpu"), gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, save_strategy="no", report_to="none",
    )
    trainer = GRPOTrainer(
        model=model, args=cfg, train_dataset=train_dataset,
        reward_funcs=reward_fn, processing_class=tokenizer,
        callbacks=[
            pilot["JsonlDashboardLogger"](out_dir / "dashboard.jsonl"),
            _make_lora_sentinel_class()(out_dir / "update_sentinel.jsonl", every=eval_every),
            eval_callback_cls(eval_rows, out_dir / "stageb_eval_curve.jsonl", every=eval_every),
            pilot["LocalSafetyCallback"](out_dir / "safety_stop.json"),
        ],
    )
    trainer.train()
    completion = fixed_budget_completion(trainer, out_dir, budget_updates, out_dir / "safety_stop.json")

    acc_after = guru_greedy_accuracy(model, tokenizer, eval_rows)
    print(f"[adapt] {stage_a_checkpoint}: Simulation accuracy AFTER = {acc_after:.4f}")

    summary = {
        "stage_a_checkpoint": str(stage_a_checkpoint), "task": "Simulation", "algo": "grpo",
        "peft": "lora", "reward_mode": reward_mode, "budget_updates": budget_updates,
        **completion, "seed": seed,
        "learning_rate": learning_rate, "acc_before": acc_before, "acc_after": acc_after,
        "delta_acc": acc_after - acc_before, "wall_seconds": time.time() - t0,
    }
    summary_path.write_text(json.dumps(summary, indent=1))
    validate_stage_b_completion(out_dir, budget_updates, eval_every)

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


# ---------------------------------------------------------------------------
# Gate C0 — GPU memory calibration
# ---------------------------------------------------------------------------

def gate_c0_memory_probe(model_id: str, peft_cfg: dict, smoke_dataset, reward_mode: str,
                         num_generations: int, per_device_batch: int, grad_accum: int,
                         max_completion_length: int, device: str = "cuda",
                         revision: str | None = None, min_headroom_pct: float = 15.0,
                         learning_rate: float = 2e-5) -> dict:
    """2-update smoke run on `smoke_dataset` AT THE SAME GEOMETRY the real
    run will use (num_generations/batch/grad_accum/completion length all
    passed in, not hardcoded) — the point of this gate is to measure whether
    the *actual* recipe fits, not a cheaper stand-in. Reports peak allocated
    memory vs. total device memory. Plan §3 Gate C0: PASS on L4 if headroom
    >=15%, else escalate to A100 rather than shrinking the geometry below
    the config defaults."""
    import torch
    from trl import GRPOConfig, GRPOTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("Gate C0 requires a CUDA device")
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = build_peft_model(model_id, peft_cfg, device, revision=revision)
    reward_fn = select_reward_fn(reward_mode)
    cfg = GRPOConfig(
        output_dir="/tmp/exp2_gate_c0_smoke", seed=42, max_steps=2,
        learning_rate=learning_rate, per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum, num_generations=num_generations,
        beta=0.0, max_completion_length=max_completion_length,
        bf16=True, optim="paged_adamw_8bit",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, save_strategy="no", report_to="none",
    )
    trainer = GRPOTrainer(model=model, args=cfg, train_dataset=smoke_dataset,
                          reward_funcs=reward_fn, processing_class=tokenizer)
    trainer.train()

    peak_bytes = torch.cuda.max_memory_allocated()
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    headroom_pct = 100.0 * (total_bytes - peak_bytes) / total_bytes

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "peak_allocated_gib": peak_bytes / (1024 ** 3),
        "total_device_gib": total_bytes / (1024 ** 3),
        "headroom_pct": headroom_pct,
        "gate_pass": headroom_pct >= min_headroom_pct,
        "geometry": {"num_generations": num_generations, "per_device_batch": per_device_batch,
                    "grad_accum": grad_accum, "max_completion_length": max_completion_length},
    }
