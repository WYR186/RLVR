"""Trainer callbacks: dashboard-signal logging, exact-step checkpointing,
periodic exact-answer eval.

This is Person 4's instrumentation (briefing §4): even though the pilot only
*needs* Q, we log every dashboard signal per update (reward mean/std, KL,
grad norm, entropy, completion length) so the later detector bake-off can be
run on this same data. Leakage rule: rows are written as they happen; nothing
is normalized across the run.
"""
from __future__ import annotations

import json
import math
import resource
import time
from pathlib import Path

from transformers import TrainerCallback


class JsonlDashboardLogger(TrainerCallback):
    """Append every trainer log dict (one row per logging step) to a JSONL.

    GRPOTrainer already computes reward mean/std, KL (if beta>0), entropy,
    completion length and grad norm into its logs; persisting the raw dicts
    keeps every dashboard signal without depending on TRL's exact key names.
    """

    def __init__(self, out_path):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        row = {"step": state.global_step, "wall_time": time.time(), **logs}
        with self.out_path.open("a") as f:
            f.write(json.dumps(row) + "\n")


class SaveAtSteps(TrainerCallback):
    """Save bf16 model weights (no optimizer state) at exact update counts.

    Tommy's spec: checkpoints at 0/25/50/100/200 updates. Step 0 (= base
    model) is saved by the notebook before training starts; this callback
    handles the rest. Weights-only keeps each 0.5B checkpoint ~1 GB on Drive.
    """

    def __init__(self, steps, out_dir, tokenizer=None):
        self.steps = set(int(s) for s in steps)
        self.out_dir = Path(out_dir)
        self.tokenizer = tokenizer

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step in self.steps:
            path = self.out_dir / f"ckpt-{state.global_step}"
            path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(path, safe_serialization=True)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(path)


class LocalSafetyCallback(TrainerCallback):
    """Stop a long local job when pre-declared feasibility limits are crossed.

    This callback does not alter successful updates. It only requests a clean
    trainer stop after repeated timing/reward/clipping failures, NaN/Inf, or a
    hard resident-memory ceiling. The reason is persisted for audit.
    """

    def __init__(self, out_path, max_step_seconds: float = 480.0,
                 max_rss_gib: float = 96.0, patience: int = 3,
                 signal_patience: int = 5, max_clip_ratio: float = 0.10):
        self.out_path = Path(out_path)
        self.max_step_seconds = max_step_seconds
        self.max_rss_gib = max_rss_gib
        self.patience = patience
        self.signal_patience = signal_patience
        self.max_clip_ratio = max_clip_ratio
        self.slow_steps = self.zero_signal_steps = self.clipped_steps = 0

    @staticmethod
    def _rss_gib() -> float:
        # macOS ru_maxrss is bytes; Linux is KiB.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        import sys
        return rss / (1024 ** 3) if sys.platform == "darwin" else rss / (1024 ** 2)

    def _stop(self, state, control, reason, logs):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps({
            "step": state.global_step, "reason": reason,
            "rss_gib": self._rss_gib(), "logs": logs,
            "wall_time": time.time(),
        }, indent=1))
        print(f"[local-safety] stopping at step {state.global_step}: {reason}")
        control.should_training_stop = True
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        for key in ("loss", "grad_norm"):
            value = logs.get(key)
            if value is not None and not math.isfinite(float(value)):
                return self._stop(state, control, f"non-finite {key}", logs)
        if self._rss_gib() > self.max_rss_gib:
            return self._stop(state, control, "RSS exceeded local limit", logs)

        step_time = float(logs.get("step_time", 0.0) or 0.0)
        self.slow_steps = self.slow_steps + 1 if step_time > self.max_step_seconds else 0
        zero_frac = float(logs.get("frac_reward_zero_std", 0.0) or 0.0)
        self.zero_signal_steps = self.zero_signal_steps + 1 if zero_frac >= 1.0 else 0
        clipped = float(logs.get("completions/clipped_ratio", 0.0) or 0.0)
        self.clipped_steps = self.clipped_steps + 1 if clipped > self.max_clip_ratio else 0

        if self.slow_steps >= self.patience:
            return self._stop(state, control, "three consecutive updates exceeded 8 minutes", logs)
        if self.zero_signal_steps >= self.signal_patience:
            return self._stop(state, control, "five consecutive updates had zero group reward variance", logs)
        if self.clipped_steps >= self.signal_patience:
            return self._stop(state, control, "five consecutive updates exceeded 10% completion clipping", logs)
        return control


class ExactAnswerEvalCallback(TrainerCallback):
    """Greedy exact-answer accuracy on a fixed slice every `every` updates.

    Results are appended to a JSONL (step, accuracy, wall time) — this is the
    'eval accuracy on a small fixed GSM8K slice every 25 updates' dashboard
    signal, and in the adaptation phase it records the accuracy-vs-update
    curve (adaptation speed).
    """

    def __init__(self, prompts, golds, out_path, every: int = 25,
                 batch_size: int = 16, max_new_tokens: int = 512,
                 also_at_step0: bool = False, item_metadata=None):
        self.prompts, self.golds = prompts, golds
        self.item_metadata = item_metadata
        if item_metadata is not None and len(item_metadata) != len(prompts):
            raise ValueError("item_metadata must align one-to-one with prompts")
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.every = every
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.also_at_step0 = also_at_step0
        self._tokenizer = None

    def _run_eval(self, model, step):
        from .evaluate import exact_answer_accuracy

        t0 = time.time()
        acc, details = exact_answer_accuracy(
            model, self._tokenizer, self.prompts, self.golds,
            batch_size=self.batch_size, max_new_tokens=self.max_new_tokens,
            return_details=True)
        if self.item_metadata is not None:
            for detail, metadata in zip(details, self.item_metadata):
                detail.update(metadata)
        by_calc_count = {}
        for detail in details:
            count = detail.get("gold_calculation_count")
            if count is not None:
                bucket = by_calc_count.setdefault(str(count), {"n": 0, "correct": 0})
                bucket["n"] += 1
                bucket["correct"] += int(detail["correct"])
        for bucket in by_calc_count.values():
            bucket["accuracy"] = bucket["correct"] / bucket["n"]
        row = {
            "step": step,
            "accuracy": acc,
            "n_eval": len(details),
            "n_correct": sum(d["correct"] for d in details),
            "mean_completion_chars": (
                sum(len(d["completion"]) for d in details) / len(details)
                if details else 0.0),
            # Preserve item-level predictions so later difficulty strata do
            # not require another expensive generation pass.
            "items": details,
            "accuracy_by_gold_calculation_count": by_calc_count,
            "eval_seconds": time.time() - t0,
        }
        with self.out_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[eval] step {step}: accuracy={acc:.4f}")
        model.train()

    def on_train_begin(self, args, state, control, model=None,
                       processing_class=None, tokenizer=None, **kwargs):
        self._tokenizer = processing_class or tokenizer
        if self.also_at_step0 and state.global_step == 0:
            self._run_eval(model, 0)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.every == 0:
            self._run_eval(model, state.global_step)
