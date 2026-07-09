"""Cheap, no-update checks that protect the Colab budget."""
from __future__ import annotations

from .reward import exact_answer_reward, extract_pred_answer


def sparse_reward_preflight(model, tokenizer, prompts, golds, *,
                            num_generations: int = 8,
                            temperature: float = 0.7,
                            top_p: float = 1.0,
                            max_new_tokens: int = 512) -> dict:
    """Sample fixed groups and verify exact reward has within-group variance.

    GRPO has no learning signal when every completion in every prompt group
    receives the same reward. This check performs no parameter updates and
    should run before committing to the 200-update Stage-A job.
    """
    import torch

    device = next(model.parameters()).device
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    groups = []
    try:
        with torch.no_grad():
            for prompt, gold in zip(prompts, golds):
                enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=512).to(device)
                out = model.generate(
                    **enc, do_sample=True, num_return_sequences=num_generations,
                    temperature=temperature, top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
                gen = out[:, enc["input_ids"].shape[1]:]
                completions = tokenizer.batch_decode(gen, skip_special_tokens=True)
                rewards = exact_answer_reward(
                    completions=completions, answer=[gold] * len(completions))
                groups.append({
                    "gold": gold,
                    "rewards": rewards,
                    "predictions": [extract_pred_answer(x) for x in completions],
                    "completion_tails": [x[-500:] for x in completions],
                })
    finally:
        tokenizer.padding_side = old_padding_side

    variable = sum(len(set(g["rewards"])) > 1 for g in groups)
    return {
        "n_prompts": len(groups),
        "num_generations": num_generations,
        "n_correct": int(sum(sum(g["rewards"]) for g in groups)),
        "groups_with_reward_variance": variable,
        "has_grpo_signal": variable > 0,
        "groups": groups,
    }
