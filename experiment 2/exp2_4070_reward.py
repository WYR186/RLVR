"""TRL-compatible GURU rewards for the RTX 4070 Experiment-2 variant."""
from __future__ import annotations

from vendor.reasoning360_reward_score import codeio, naive_dapo


def boxed_format_score(completion: str, data_source: str) -> float:
    """Return 1 only for a non-empty, brace-balanced Math boxed answer."""
    if not data_source.startswith("math"):
        return 0.0
    marker = r"\boxed{"
    start = completion.rfind(marker)
    if start < 0:
        return 0.0
    depth = 1
    content_start = start + len(marker)
    for index in range(content_start, len(completion)):
        if completion[index] == "{":
            depth += 1
        elif completion[index] == "}":
            depth -= 1
            if depth == 0:
                return float(bool(completion[content_start:index].strip()))
    return 0.0


def score_completion(completion: str, ground_truth: str, data_source: str, extra_info=None) -> float:
    try:
        if data_source.startswith("math"):
            result = naive_dapo.compute_score(completion, ground_truth, extra_info or {})
        elif data_source.startswith("simulation__codeio"):
            result = codeio.compute_score(completion, ground_truth)
        else:
            raise ValueError(f"unsupported exp2 data source: {data_source}")
        if isinstance(result, dict):
            return float(result["score"])
        return float(result)
    except Exception:
        # The upstream reward router treats verifier failures as failed
        # answers in practice; a malformed completion must never crash GRPO.
        return 0.0


def guru_reward(completions, ground_truth, data_source, extra_info=None, **kwargs):
    del kwargs
    if extra_info is None:
        extra_info = [{} for _ in completions]
    return [
        score_completion(completion, truth, source, info)
        for completion, truth, source, info in zip(
            completions, ground_truth, data_source, extra_info, strict=True
        )
    ]


def guru_reward_boxed_01(completions, ground_truth, data_source, extra_info=None, **kwargs):
    """Exact GURU reward plus a registered 0.1 Math boxed-format reward."""
    del kwargs
    if extra_info is None:
        extra_info = [{} for _ in completions]
    return [
        score_completion(completion, truth, source, info)
        + 0.1 * boxed_format_score(completion, source)
        for completion, truth, source, info in zip(
            completions, ground_truth, data_source, extra_info, strict=True
        )
    ]
