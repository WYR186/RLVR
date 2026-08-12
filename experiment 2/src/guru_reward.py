"""TRL-compatible GURU rewards for exp2's Colab variant.

An earlier version of this module hand-rolled boxed/numeric extraction and a
normalized-JSON-output comparison for Math and Simulation respectively,
because no verifier was known to exist yet. One does: `vendor/
reasoning360_reward_score` (vendored from the official LLM360/Reasoning360
repo, upstream revision `13158341d2a0dfe5f3bb80e7126ff21de0d16676`, committed
by the WIN4070 track's `experiment 2/exp2_4070_reward.py`) is the same
verifier Tommy's spec and every other team member's stage-2 score is
presumably graded with. This module is a thin wrapper around it — matching
`exp2_4070_reward.py`'s contract exactly — instead of a second, divergent
implementation. Confirmed answer-format contract (data/guru_schema_audit.json):
Math = last `\\boxed{...}`, normalized string/mathematical equivalence, 1.0
correct / 0.0 incorrect (verified empirically against the vendored code —
`naive_dapo.py`'s own docstring claims -1.0 for incorrect, but its actual
`return` statement computes `1.0 if correct else 0.`; the docstring is stale,
the code is what runs); Simulation (CodeIO) = JSON in a fenced code block,
recursively normalized, exact structured equality, 1.0/0.0.

Dataset rows carry `data_source`, `ground_truth` (from the nested
`reward_model.ground_truth` field — flattened by `guru_data.py`), and
`extra_info` — NOT the `answer` column the earlier version assumed. Reward
functions here follow that contract (TRL's GRPOTrainer contract: called with
`completions=` and every extra dataset column as a kwarg).
"""
from __future__ import annotations

from vendor.reasoning360_reward_score import codeio, naive_dapo


def _completion_text(completion) -> str:
    """TRL passes either a plain string or a list of chat messages."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "".join(m.get("content", "") for m in completion)
    return str(completion)


def boxed_format_score(completion: str, data_source: str) -> float:
    """1.0 iff `completion` ends with a non-empty, brace-balanced Math
    `\\boxed{...}`; 0.0 otherwise (including for non-Math `data_source`).
    A shaping bonus, not the correctness signal — see `guru_reward_boxed_01`.
    """
    if not data_source.startswith("math"):
        return 0.0
    marker = "\\boxed{"
    start = completion.rfind(marker)
    if start < 0:
        return 0.0
    depth = 1
    content_start = start + len(marker)
    for i in range(content_start, len(completion)):
        if completion[i] == "{":
            depth += 1
        elif completion[i] == "}":
            depth -= 1
            if depth == 0:
                return float(bool(completion[content_start:i].strip()))
    return 0.0


def score_completion(completion: str, ground_truth, data_source: str, extra_info=None) -> float:
    """Route to the vendored verifier by `data_source` prefix. Any verifier
    exception scores 0.0 rather than crashing GRPO — matches the upstream
    reward router's behavior for malformed completions."""
    try:
        if data_source.startswith("math"):
            result = naive_dapo.compute_score(completion, str(ground_truth), extra_info or {})
        elif data_source.startswith("simulation__codeio"):
            result = codeio.compute_score(completion, ground_truth, extra_info)
        else:
            raise ValueError(f"unsupported exp2 data source: {data_source!r}")
        return float(result["score"]) if isinstance(result, dict) else float(result)
    except Exception:
        return 0.0


def guru_reward(completions=None, ground_truth=None, data_source=None,
                extra_info=None, **kwargs) -> list[float]:
    """Exact GURU verifier score, no shaping. Used for Simulation (Stage B) —
    matches `exp2_4070_reward.py`'s `reward_mode: "exact"`."""
    del kwargs
    texts = [_completion_text(c) for c in completions]
    if extra_info is None:
        extra_info = [{} for _ in texts]
    return [score_completion(t, gt, ds, info)
           for t, gt, ds, info in zip(texts, ground_truth, data_source, extra_info, strict=True)]


def guru_reward_boxed_01(completions=None, ground_truth=None, data_source=None,
                         extra_info=None, **kwargs) -> list[float]:
    """Exact GURU verifier score + 0.1 Math boxed-format bonus. Used for
    Math (Stage A) — matches `exp2_4070_reward.py`'s
    `reward_mode: "exact_plus_boxed_format_0.1"`. The finding behind this
    choice (`FINDING_GROUP_SIZE_REWARD_VARIANCE.md`): at this population's low
    base accuracy, pure exact-match leaves the large majority of GRPO groups
    with zero within-group variance (dead gradient); the format term recovers
    some of that but is itself mostly formatting noise, not reasoning signal
    — reported honestly in analysis, not treated as if it were free lift.
    """
    del kwargs
    texts = [_completion_text(c) for c in completions]
    if extra_info is None:
        extra_info = [{} for _ in texts]
    return [score_completion(t, gt, ds, info) + 0.1 * boxed_format_score(t, ds)
           for t, gt, ds, info in zip(texts, ground_truth, data_source, extra_info, strict=True)]


_REWARD_MODES = {
    "exact": guru_reward,
    "exact_plus_boxed_format_0.1": guru_reward_boxed_01,
}


def select_reward_fn(reward_mode: str):
    if reward_mode not in _REWARD_MODES:
        raise ValueError(f"unknown reward_mode {reward_mode!r}; expected one of {list(_REWARD_MODES)}")
    return _REWARD_MODES[reward_mode]


def exact_correct(completion: str, ground_truth, data_source: str, extra_info=None) -> bool:
    """Whether `completion` is exactly correct under the verifier (used for
    greedy-eval accuracy, and for tracking the exact-channel variance
    separately from the format-shaping channel in the sparse-reward
    preflight — plan §3 Phase 0 step 7, the v9-tightened gate)."""
    if data_source.startswith("math"):
        return score_completion(completion, ground_truth, data_source, extra_info) > 0
    return score_completion(completion, ground_truth, data_source, extra_info) >= 1.0
