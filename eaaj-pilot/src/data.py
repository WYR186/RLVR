"""Dataset loading, prompt formatting, and frozen splits for the pilot.

All randomness is seeded and every frozen selection is written to
data/*.json so the exact prompt lists are committed (briefing §5: "freeze
the exact list in a committed file").

Splits used by the pilot (all disjoint where it matters):
  - GSM8K train slice: 512 questions for GRPO stage A       (train split)
  - GSM8K eval slice:   64 questions for accuracy-every-25   (test split)
  - probe set:         512 prompts for Q measurement         (test split,
                        disjoint from the eval slice)
  - probe sensitivity: 2048-prompt superset of the probe set (test split)
  - SVAMP train: 256 questions, SVAMP eval: 100 questions    (fixed)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 42
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
SVAMP_REVISION = "5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef"

# Prompt template — logged choice (briefing §8: deviations/choices must be
# recorded). Base model, so plain QA format with an explicit '####' answer
# convention that the exact-answer reward can parse.
PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Answer: Let's think step by step, then give the final answer after '####'.\n"
)


def format_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question.strip())


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

def load_gsm8k():
    """Returns (train, test) HF datasets with fields `question`, `answer`."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", revision=GSM8K_REVISION)
    return ds["train"], ds["test"]


def build_gsm8k_splits(n_train: int = 512, n_eval: int = 64,
                       n_probe: int = 512, n_probe_big: int = 2048,
                       seed: int = SEED) -> dict:
    """Freeze all GSM8K index lists; writes data/gsm8k_splits.json.

    Probe prompts come from the TEST split (never trained on). The eval
    slice is drawn from the test split first and the probe/probe_big sets
    from the remaining indices, so eval and probe are disjoint.
    """
    train, test = load_gsm8k()
    rng = random.Random(seed)

    train_idx = rng.sample(range(len(train)), n_train)

    test_perm = list(range(len(test)))
    rng.shuffle(test_perm)
    eval_idx = test_perm[:n_eval]
    # Primary probe set: test split only (strictest never-trained-on guarantee).
    # The GSM8K test split (1319) cannot supply 2048 prompts disjoint from the
    # eval slice, so the probe-size-sensitivity superset is topped up from a
    # held-out slice of TRAIN (disjoint from the 512 GRPO questions) — same
    # i.i.d. distribution; logged deviation from briefing §5.
    probe_big_test_idx = test_perm[n_eval:]
    probe_idx = probe_big_test_idx[:n_probe]
    need = max(0, n_probe_big - len(probe_big_test_idx))
    train_heldout = sorted(set(range(len(train))) - set(train_idx))
    probe_big_train_idx = rng.sample(train_heldout, need) if need else []

    splits = {
        "seed": seed,
        "gsm8k_train_idx": sorted(train_idx),
        "gsm8k_eval_idx": sorted(eval_idx),
        "probe_idx": probe_idx,          # order preserved: probe_big[:512] == probe
        "probe_big_test_idx": probe_big_test_idx,
        "probe_big_train_idx": probe_big_train_idx,
    }
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "gsm8k_splits.json").write_text(json.dumps(splits, indent=1))
    return splits


def freeze_probe_set(seed: int = SEED) -> list[str]:
    """Freeze the formatted probe PROMPTS themselves (belt and braces: even a
    dataset re-release cannot silently change the probe set). Writes
    data/probe_set_ids.json with indices + full prompt strings."""
    splits_path = DATA_DIR / "gsm8k_splits.json"
    if splits_path.exists():
        splits = json.loads(splits_path.read_text())
    else:
        splits = build_gsm8k_splits(seed=seed)
    train, test = load_gsm8k()

    probe_prompts = [format_prompt(test[i]["question"]) for i in splits["probe_idx"]]
    probe_big_prompts = (
        [format_prompt(test[i]["question"]) for i in splits["probe_big_test_idx"]]
        + [format_prompt(train[i]["question"]) for i in splits["probe_big_train_idx"]])

    payload = {
        "seed": splits["seed"],
        "probe_idx": splits["probe_idx"],
        "probe_prompts": probe_prompts,
        "probe_big_test_idx": splits["probe_big_test_idx"],
        "probe_big_train_idx": splits["probe_big_train_idx"],
        "probe_big_prompts": probe_big_prompts,
    }
    (DATA_DIR / "probe_set_ids.json").write_text(json.dumps(payload, indent=1))
    return probe_prompts


def load_probe_prompts(big: bool = False) -> list[str]:
    payload = json.loads((DATA_DIR / "probe_set_ids.json").read_text())
    return payload["probe_big_prompts" if big else "probe_prompts"]


def gsm8k_grpo_dataset():
    """HF dataset for GRPOTrainer: columns `prompt` (formatted) and `answer`
    (canonical float gold), restricted to the frozen 512-question train slice."""
    from .reward import extract_gold_answer

    train, _ = load_gsm8k()
    splits = json.loads((DATA_DIR / "gsm8k_splits.json").read_text())
    sub = train.select(splits["gsm8k_train_idx"])
    return sub.map(lambda ex: {
        "prompt": format_prompt(ex["question"]),
        "answer": extract_gold_answer(ex["answer"]),
    }, remove_columns=sub.column_names)


def gsm8k_eval_set(return_metadata: bool = False):
    """Fixed 64-question accuracy slice.

    With ``return_metadata=True``, also returns a pre-update difficulty proxy:
    the number of calculator annotations in the canonical gold rationale. We
    log accuracy by the exact count rather than inventing post-hoc easy/medium/
    hard thresholds.
    """
    from .reward import extract_gold_answer

    _, test = load_gsm8k()
    splits = json.loads((DATA_DIR / "gsm8k_splits.json").read_text())
    sub = test.select(splits["gsm8k_eval_idx"])
    prompts = [format_prompt(q) for q in sub["question"]]
    golds = [extract_gold_answer(a) for a in sub["answer"]]
    if not return_metadata:
        return prompts, golds
    metadata = [
        {"dataset_index": idx, "gold_calculation_count": answer.count("<<")}
        for idx, answer in zip(splits["gsm8k_eval_idx"], sub["answer"])
    ]
    return prompts, golds, metadata


# ---------------------------------------------------------------------------
# SVAMP
# ---------------------------------------------------------------------------

def load_svamp():
    """SVAMP via HF `ChilleD/SVAMP` (fields verified in notebook 00:
    Body, Question, Answer, Equation, Type, ID). Question text = Body + Question."""
    from datasets import load_dataset

    return load_dataset("ChilleD/SVAMP", revision=SVAMP_REVISION)


def _svamp_question(ex) -> str:
    return f"{ex['Body'].strip()} {ex['Question'].strip()}"


def build_svamp_splits(n_train: int = 256, n_eval: int = 100,
                       seed: int = SEED) -> dict:
    """Freeze SVAMP adaptation train/eval lists; writes data/svamp_splits.json.

    Uses the dataset's own train/test splits (train slice from train,
    eval slice from test) so the fixed-budget eval is held out.
    """
    ds = load_svamp()
    rng = random.Random(seed)
    train_idx = rng.sample(range(len(ds["train"])), min(n_train, len(ds["train"])))
    eval_idx = rng.sample(range(len(ds["test"])), min(n_eval, len(ds["test"])))
    splits = {"seed": seed,
              "svamp_train_idx": sorted(train_idx),
              "svamp_eval_idx": sorted(eval_idx)}
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "svamp_splits.json").write_text(json.dumps(splits, indent=1))
    return splits


def svamp_grpo_dataset():
    """HF dataset (prompt, answer) for the fixed-budget adaptation phase."""
    ds = load_svamp()
    splits = json.loads((DATA_DIR / "svamp_splits.json").read_text())
    sub = ds["train"].select(splits["svamp_train_idx"])
    return sub.map(lambda ex: {
        "prompt": format_prompt(_svamp_question(ex)),
        "answer": float(ex["Answer"]),
    }, remove_columns=sub.column_names)


def svamp_eval_set():
    """(prompts, golds) for the fixed 100-question SVAMP eval."""
    ds = load_svamp()
    splits = json.loads((DATA_DIR / "svamp_splits.json").read_text())
    sub = ds["test"].select(splits["svamp_eval_idx"])
    prompts = [format_prompt(_svamp_question(ex)) for ex in sub]
    golds = [float(ex["Answer"]) for ex in sub]
    return prompts, golds
