"""Greedy exact-answer accuracy evaluation (GSM8K / SVAMP)."""
from __future__ import annotations

from .reward import extract_pred_answer, answers_match


def exact_answer_accuracy(model, tokenizer, prompts, golds,
                          batch_size: int = 16,
                          max_new_tokens: int = 512,
                          max_prompt_length: int = 512,
                          return_details: bool = False):
    """Greedy-decode each prompt, extract final number, exact-match vs gold.

    Greedy (do_sample=False) so eval accuracy is deterministic and does not
    depend on the sampling temperature used for GRPO rollouts.
    """
    import torch

    model.eval()
    device = next(model.parameters()).device
    hits, details = [], []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # decoder-only batched generation contract
    try:
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                enc = tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_prompt_length).to(device)
                out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                     do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id
                                     or tokenizer.eos_token_id)
                gen = out[:, enc["input_ids"].shape[1]:]
                texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
                for text, gold in zip(texts, golds[i:i + batch_size]):
                    pred = extract_pred_answer(text)
                    ok = answers_match(pred, gold)
                    hits.append(ok)
                    if return_details:
                        details.append({"completion": text, "pred": pred,
                                        "gold": gold, "correct": ok})
    finally:
        tokenizer.padding_side = old_padding_side
    acc = sum(hits) / len(hits)
    return (acc, details) if return_details else acc
