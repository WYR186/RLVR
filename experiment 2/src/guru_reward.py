"""Reward / answer-extraction for exp2's two domains (Math, Simulation).

Phase 0 rule (both EXPERIMENT_2 plans, §3 step 3): "the reward function is
written from [the discovered answer format], not from a guess." What follows
is a best-effort starting point, not a verified implementation — an agent
running Phase 0 must compare `examples_by_subset` in
`data/guru_schema_audit.json` against what these extractors actually parse,
and extend them if the real format doesn't match. `select_reward_fn` and
`select_eval_fn` are the two functions Phase 1/3 code should call; which
concrete extractor they return is a decision, not a hardcoded assumption, so
it is made in one place.

Two domains, two answer shapes:
  - Math (OR1/DAPO/DeepScaler): curated RL math sets typically release a
    boxed final answer (`\\boxed{...}`) or an already-canonical numeric
    string. `numeric_exact_reward` handles both, plus the GSM8K-style
    `#### x` convention (reused so a subset that happens to use it also
    works) and a last-number fallback.
  - Simulation (CodeI/O): code input/output prediction tasks. The released
    verifier for CodeI/O is a *normalized* match against a gold output
    string (no code re-execution here — this project has no sandboxed
    executor, and reward functions run inside the training loop where a
    subprocess-per-completion would be both slow and a new failure surface).
    `code_output_exact_reward` extracts a predicted-output span, normalizes
    it (whitespace/quote/literal-structure canonicalization), and compares.

Both reward functions follow the TRL `GRPOTrainer` contract: called with
`completions=` and every extra dataset column as a kwarg (`answer=` here),
return `list[float]`.
"""
from __future__ import annotations

import ast
import json
import re
from numbers import Real

_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")
_HASH_GOLD_RE = re.compile(r"####\s*(-?\$?[\d,]*\.?\d+)")
_OUTPUT_MARKER_SPLIT_RE = re.compile(r"(?:^|\n)\s*output\s*[:=]\s*", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:\w*\n)?(.*?)```", re.DOTALL)


def _completion_text(completion) -> str:
    """TRL passes either a plain string or a list of chat messages."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "".join(m.get("content", "") for m in completion)
    return str(completion)


# ---------------------------------------------------------------------------
# Math: boxed / #### / last-number numeric extraction
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str | None:
    """Content of the LAST `\\boxed{...}` in `text`, handling nested braces.

    A model may revise its answer; the last box is the committed one, same
    "last marker wins" convention as this project's GSM8K `####` handling.
    """
    marker = "\\boxed{"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    start = idx + len(marker)
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start:i - 1]


def _canon_numeric(num_str: str) -> float | None:
    s = num_str.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def extract_pred_numeric(text: str) -> float | None:
    """Predicted numeric answer: boxed wins, then `#### x`, then last number
    in the text. Does NOT evaluate symbolic content (fractions, sqrt, pi) —
    if Phase 0 finds those in real examples, extend this before trusting it.
    """
    boxed = extract_boxed(text)
    if boxed is not None:
        v = _canon_numeric(boxed)
        if v is not None:
            return v
        nums = _NUMBER_RE.findall(boxed)
        if nums:
            v = _canon_numeric(nums[-1])
            if v is not None:
                return v
    marked = list(_HASH_GOLD_RE.finditer(text))
    if marked:
        return _canon_numeric(marked[-1].group(1))
    nums = _NUMBER_RE.findall(text)
    for cand in reversed(nums):
        v = _canon_numeric(cand)
        if v is not None:
            return v
    return None


def extract_gold_numeric(gold_field) -> float | None:
    """Gold answer from a dataset `answer` column: already-numeric, a bare
    numeric string, or text containing a boxed/#### answer — all handled by
    reusing `extract_pred_numeric` as the fallback path."""
    if isinstance(gold_field, Real):
        return float(gold_field)
    return extract_pred_numeric(str(gold_field))


def answers_match_numeric(pred: float | None, gold: float | None,
                          abs_tol: float = 1e-9) -> bool:
    """Numerically canonicalized exact match. No relative tolerance — same
    reasoning as eaaj-pilot/src/reward.py: a relative rule silently accepts
    large absolute errors on large gold answers, contradicting an
    exact-answer reward."""
    if pred is None or gold is None:
        return False
    return abs(pred - gold) <= abs_tol


def numeric_exact_reward(completions=None, answer=None, **kwargs) -> list[float]:
    """TRL GRPOTrainer-compatible reward for the Math domain."""
    rewards = []
    for comp, gold in zip(completions, answer):
        gold_val = extract_gold_numeric(gold)
        pred_val = extract_pred_numeric(_completion_text(comp))
        rewards.append(1.0 if answers_match_numeric(pred_val, gold_val) else 0.0)
    return rewards


# ---------------------------------------------------------------------------
# Simulation (CodeI/O): normalized output-string extraction
# ---------------------------------------------------------------------------

def _try_canonicalize_literal(s: str):
    """If `s` parses as a Python/JSON *structural* literal (list, dict),
    return its canonical repr so formatting differences (spacing, quote
    style, trailing commas) don't cause a false mismatch.

    Deliberately restricted to `[`/`{`-prefixed input: extending this to bare
    scalars would canonicalize JSON `true`/`false`/`null` and Python
    `True`/`False`/`None` to the same spelling, which is exactly the
    case-sensitivity this function must NOT erase for a code-output reward
    (`"True"` and `"true"` are different, real Python outputs). Bare numbers
    are handled by the plain-string path below (`"42"` vs `"42.0"` are left
    as a genuine, correctly-flagged mismatch — Phase 0 decides if the real
    verifier needs numeric-literal equivalence here).
    """
    stripped = s.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    for parser in (ast.literal_eval, json.loads):
        try:
            return repr(parser(stripped))
        except Exception:
            continue
    return None


def normalize_code_output(s: str) -> str:
    """Normalize a predicted or gold code-output string for comparison.

    Strips surrounding whitespace and matching outer quotes, tries literal
    canonicalization (see above), and otherwise collapses internal
    whitespace runs — NOT lowercased, because code output is case-sensitive
    (`"True"` != `"true"` in Python).
    """
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    canon = _try_canonicalize_literal(s)
    if canon is not None:
        return canon
    return re.sub(r"\s+", " ", s).strip()


def extract_predicted_output(text: str) -> str:
    """Best-effort predicted-output span from a completion.

    Tries, in order: an explicit `Output:`/`output =` marker (last one wins,
    same "final answer wins" convention as the Math extractor — everything
    after the LAST marker is taken, which may itself span multiple lines,
    e.g. a multi-line structural literal); the content of the last fenced
    code block; otherwise the whole completion. This is the single most
    likely thing to need revision once Phase 0's `examples_by_subset` shows
    the real CodeI/O prompt/answer convention — it is a starting point, not
    a verified parser.
    """
    parts = _OUTPUT_MARKER_SPLIT_RE.split(text)
    if len(parts) > 1:
        return parts[-1].strip()
    fenced = _CODE_FENCE_RE.findall(text)
    if fenced:
        return fenced[-1].strip()
    return text.strip()


def code_output_exact_reward(completions=None, answer=None, **kwargs) -> list[float]:
    """TRL GRPOTrainer-compatible reward for the Simulation (CodeI/O) domain."""
    rewards = []
    for comp, gold in zip(completions, answer):
        pred_norm = normalize_code_output(extract_predicted_output(_completion_text(comp)))
        gold_norm = normalize_code_output(str(gold))
        rewards.append(1.0 if pred_norm == gold_norm else 0.0)
    return rewards


# ---------------------------------------------------------------------------
# Dispatch — one place that decides which extractor a domain uses
# ---------------------------------------------------------------------------

_REWARD_FNS = {
    "Math": numeric_exact_reward,
    "Simulation": code_output_exact_reward,
}
_PRED_EXTRACTORS = {
    "Math": extract_pred_numeric,
    "Simulation": lambda text: normalize_code_output(extract_predicted_output(text)),
}
_GOLD_EXTRACTORS = {
    "Math": extract_gold_numeric,
    "Simulation": lambda gold: normalize_code_output(str(gold)),
}
_MATCH_FNS = {
    "Math": answers_match_numeric,
    "Simulation": lambda pred, gold: pred is not None and gold is not None and pred == gold,
}


def select_reward_fn(domain: str):
    if domain not in _REWARD_FNS:
        raise ValueError(f"unknown domain {domain!r}; expected one of {list(_REWARD_FNS)}")
    return _REWARD_FNS[domain]


def select_eval_fns(domain: str):
    """Returns (extract_pred, extract_gold, matches) for greedy-eval
    accuracy, mirroring eaaj-pilot/src/evaluate.py's use of
    extract_pred_answer/answers_match but domain-dispatched."""
    if domain not in _PRED_EXTRACTORS:
        raise ValueError(f"unknown domain {domain!r}; expected one of {list(_PRED_EXTRACTORS)}")
    return _PRED_EXTRACTORS[domain], _GOLD_EXTRACTORS[domain], _MATCH_FNS[domain]
