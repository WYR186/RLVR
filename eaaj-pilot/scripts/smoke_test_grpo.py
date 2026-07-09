#!/usr/bin/env python3
"""One CPU GRPO update against a tiny random Qwen model.

This is a contract test for the fast-moving TRL/Transformers boundary. It
checks reward invocation, logging, evaluation callbacks, and weights-only
checkpoint saving without claiming any scientific result.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
from trl import GRPOConfig, GRPOTrainer

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.callbacks import ExactAnswerEvalCallback, JsonlDashboardLogger, SaveAtSteps
from src.reward import exact_answer_reward


def main() -> None:
    cfg = json.loads((PROJECT / "pilot_config.json").read_text())
    out = Path(tempfile.mkdtemp(prefix="eaaj-grpo-smoke-", dir="/tmp"))
    tok = AutoTokenizer.from_pretrained(
        cfg["model_id"], revision=cfg["model_revision"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_cfg = Qwen2Config(
        vocab_size=len(tok), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=128, bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    model = Qwen2ForCausalLM(model_cfg)
    ds = Dataset.from_dict({
        "prompt": ["Question: 1+1?\nAnswer: ####"], "answer": [2.0]})
    args = GRPOConfig(
        output_dir=str(out / "trainer"), max_steps=1,
        per_device_train_batch_size=2, gradient_accumulation_steps=1,
        num_generations=2, max_completion_length=4, temperature=0.7,
        beta=0.0, bf16=False, use_cpu=True, logging_steps=1,
        gradient_checkpointing=False, save_strategy="no", report_to="none")
    trainer = GRPOTrainer(
        model=model, args=args, train_dataset=ds,
        reward_funcs=exact_answer_reward, processing_class=tok,
        callbacks=[
            JsonlDashboardLogger(out / "dashboard.jsonl"),
            SaveAtSteps([1], out, tokenizer=tok),
            ExactAnswerEvalCallback(
                ["Question: 1+1?\nAnswer: ####"], [2.0], out / "eval.jsonl",
                every=1, batch_size=1, max_new_tokens=2, also_at_step0=True),
        ])
    trainer.train()

    assert (out / "dashboard.jsonl").stat().st_size > 0
    eval_rows = [json.loads(x) for x in (out / "eval.jsonl").read_text().splitlines()]
    assert [x["step"] for x in eval_rows] == [0, 1]
    assert (out / "ckpt-1" / "config.json").exists()
    print(f"GRPO contract smoke test passed: {out}")


if __name__ == "__main__":
    main()
