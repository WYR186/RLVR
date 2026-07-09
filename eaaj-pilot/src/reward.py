"""Exact-answer reward for GRPO on GSM8K / SVAMP.

Reward = 1.0 iff the final number extracted from the completion matches the
gold answer numerically, else 0.0 (Tommy's spec: "exact-answer reward").

`exact_answer_reward` follows the TRL GRPOTrainer reward-function contract:
called with keyword args including `completions` and every extra dataset
column (we keep the gold answer in an `answer` column); returns list[float].
"""
from __future__ import annotations

import re
from numbers import Real

# number like -1,234.56 or $12 or 3/ optionally embedded in text
_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")
_GSM8K_GOLD_RE = re.compile(r"####\s*(-?\$?[\d,]*\.?\d+)")


def _canon(num_str: str) -> float | None:
    """'$1,234.50' -> 1234.5; returns None if unparseable."""
    s = num_str.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def extract_gold_answer(gsm8k_answer_field: str) -> float | None:
    """Gold answer from a GSM8K `answer` field ('... #### 72')."""
    m = _GSM8K_GOLD_RE.search(gsm8k_answer_field)
    if m:
        return _canon(m.group(1))
    # fall back to last number (lets the same code accept SVAMP-style golds,
    # where the answer field is already a bare number)
    return extract_pred_answer(gsm8k_answer_field)


def extract_pred_answer(text: str) -> float | None:
    """Model's final answer = the LAST number in the completion.

    If the completion itself uses the GSM8K '#### x' convention, that wins.
    """
    # A model may restart or revise its answer. The final marker is the
    # committed answer, matching the fallback rule of taking the last number.
    marked = list(_GSM8K_GOLD_RE.finditer(text))
    if marked:
        return _canon(marked[-1].group(1))
    nums = _NUMBER_RE.findall(text)
    for cand in reversed(nums):
        v = _canon(cand)
        if v is not None:
            return v
    return None


def answers_match(pred: float | None, gold: float | None,
                  abs_tol: float = 1e-9) -> bool:
    """Numerically canonicalized exact match.

    A relative tolerance is intentionally not used: at a gold answer of
    10,000 the old 1e-4 rule accepted errors as large as 1.0, contradicting
    the pre-registered exact-answer reward.
    """
    if pred is None or gold is None:
        return False
    return abs(pred - gold) <= abs_tol


def _completion_text(completion) -> str:
    """TRL passes either a plain string (standard format) or a list of
    chat messages (conversational format)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):  # [{'role': ..., 'content': ...}, ...]
        return "".join(m.get("content", "") for m in completion)
    return str(completion)


def exact_answer_reward(completions=None, answer=None, **kwargs) -> list[float]:
    """TRL GRPOTrainer-compatible reward function.

    `answer` is the dataset column with the gold answer, already canonical
    (float) or a raw GSM8K answer string — both handled.
    """
    rewards = []
    for comp, gold in zip(completions, answer):
        gold_val = float(gold) if isinstance(gold, Real) else extract_gold_answer(str(gold))
        pred_val = extract_pred_answer(_completion_text(comp))
        rewards.append(1.0 if answers_match(pred_val, gold_val) else 0.0)
    return rewards
