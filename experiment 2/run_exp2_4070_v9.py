#!/usr/bin/env python3
"""Contract-check and run the independent RTX 4070 Instruct v9 candidate."""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_RUNNER_PATH = HERE / "run_exp2_4070.py"
V8_CONFIG_PATH = HERE / "exp2_config_4070_instruct_v8.json"
DEFAULT_CONFIG_PATH = HERE / "exp2_config_4070_instruct_v9.json"
V8_SPLITS_PATH = HERE / "data" / "exp2_4070_instruct_v8_splits.json"


def _load_base_runner():
    spec = importlib.util.spec_from_file_location(
        "exp2_4070_v9_base_runner", BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base runner: {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base_runner()
_BASE_PREPARE = base.prepare
_BASE_SMOKE = base.smoke
_BASE_RUNTIME_MANIFEST = base.runtime_manifest
_BASE_LOCAL_SAFETY = base.LocalSafetyCallback


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def _without(value: dict, *keys: str) -> dict:
    result = deepcopy(value)
    for key in keys:
        result.pop(key, None)
    return result


def validate_contract(cfg: dict, config_path: Path) -> dict:
    v8 = _read_json(V8_CONFIG_PATH)
    if cfg.get("variant_tag") != "4070_instruct_v9":
        raise RuntimeError("v9 variant_tag mismatch")
    for key in (
            "model_id", "model_revision", "seed", "dataset", "stage_b",
            "execution", "measurement"):
        if cfg.get(key) != v8.get(key):
            raise RuntimeError(f"v9 unexpectedly changes frozen field: {key}")

    if _without(cfg["stage_a"], "per_device_train_batch_size", "num_generations") != \
            _without(v8["stage_a"], "per_device_train_batch_size", "num_generations"):
        raise RuntimeError("v9 changes Stage-A fields beyond linked group geometry")
    if cfg["stage_a"]["num_generations"] != 8 or \
            cfg["stage_a"]["per_device_train_batch_size"] != 8:
        raise RuntimeError("v9 Stage-A group and device batch must both be 8")
    unique_prompts = (
        cfg["stage_a"]["per_device_train_batch_size"]
        * cfg["stage_a"]["gradient_accumulation_steps"]
        // cfg["stage_a"]["num_generations"]
    )
    if unique_prompts != 8:
        raise RuntimeError("v9 must preserve eight unique prompts per update")

    frozen_gate_keys = (
        "phase0_stage_a_max_prompt_tokens",
        "phase0_stage_b_max_prompt_tokens",
        "phase0_exact_eligible_counts",
        "phase0_cuda_smoke_updates_each_stage",
        "phase0_stage_a_smoke_max_clip_ratio_each_update",
        "phase0_stage_a_smoke_min_nonzero_reward_updates",
        "phase1_update_sentinel_step",
        "zero_variance_streak_stop",
        "clipping_over_10pct_streak_stop",
    )
    for key in frozen_gate_keys:
        if cfg["gates"].get(key) != v8["gates"].get(key):
            raise RuntimeError(f"v9 unexpectedly changes frozen gate: {key}")
    if cfg["gates"].get("phase0_stage_a_preflight_prompts") != 16 or \
            cfg["gates"].get("phase0_stage_b_preflight_prompts") != 8 or \
            cfg["gates"].get("phase0_stage_a_preflight_min_variable_groups") != 2:
        raise RuntimeError("v9 preflight diagnostic contract mismatch")

    amendment = HERE.parent / cfg["amendment"]
    if not amendment.is_file():
        raise RuntimeError(f"v9 amendment missing: {amendment}")
    result = {
        "status": "contract_valid_candidate_not_promoted",
        "config": str(config_path.resolve()),
        "config_sha256": base.sha256_file(config_path),
        "amendment": str(amendment.resolve()),
        "amendment_sha256": base.sha256_file(amendment),
        "v9_runner_sha256": base.sha256_file(Path(__file__)),
        "base_runner_sha256": base.sha256_file(BASE_RUNNER_PATH),
        "v8_config_sha256": base.sha256_file(V8_CONFIG_PATH),
        "stage_a_group_size": 8,
        "stage_a_device_batch": 8,
        "stage_a_unique_prompts_per_update": unique_prompts,
        "stage_a_preflight_prompts": 16,
        "stage_a_preflight_generations": 8,
        "stage_a_preflight_min_variable_groups": 2,
        "formal_stage_a_eligible": False,
        "run_dir": str(base.run_dir(cfg)),
        "smoke_root": str(base.smoke_root(cfg)),
    }
    return result


def _id_projection(value):
    if isinstance(value, dict):
        return {
            key: _id_projection(item)
            for key, item in value.items()
            if key.endswith("_ids") or isinstance(item, dict)
        }
    return value


def prepare_v9(cfg: dict) -> dict:
    audit_path = base.data_marker(cfg, "phase0_audit")
    if audit_path.exists():
        result = _read_json(audit_path)
    else:
        result = _BASE_PREPARE(cfg)
    if not V8_SPLITS_PATH.is_file():
        raise RuntimeError(f"frozen v8 split file missing: {V8_SPLITS_PATH}")
    v9_splits_path = base.splits_path(cfg)
    if not v9_splits_path.is_file():
        raise RuntimeError(f"v9 split file missing after prepare: {v9_splits_path}")
    v8_ids = _id_projection(_read_json(V8_SPLITS_PATH))
    v9_ids = _id_projection(_read_json(v9_splits_path))
    if v9_ids != v8_ids:
        raise RuntimeError("v9 split IDs differ from frozen v8 split IDs")
    identity_path = base.data_marker(cfg, "phase0_split_identity")
    identity = {
        "status": "exact_id_match",
        "v8_splits": str(V8_SPLITS_PATH),
        "v8_splits_sha256": base.sha256_file(V8_SPLITS_PATH),
        "v9_splits": str(v9_splits_path),
        "v9_splits_sha256": base.sha256_file(v9_splits_path),
    }
    if identity_path.exists():
        if _read_json(identity_path) != identity:
            raise RuntimeError("existing v9 split-identity record mismatch")
    else:
        _write_json_once(identity_path, identity)
    print(json.dumps(identity, indent=2))
    return result


def _validate_existing_preflight(result: dict, prompt_count: int,
                                 min_variable_groups: int) -> dict:
    expected = {
        "runner_variant": "4070_instruct_v9",
        "n_prompts": prompt_count,
        "num_generations": 8,
        "minimum_combined_variable_groups": min_variable_groups,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(f"existing v9 preflight {key} mismatch")
    if not result.get("gate_pass"):
        raise RuntimeError("existing v9 sparse-reward preflight is STOP")
    return result


def sparse_preflight_v9(model, tokenizer, dataset, recipe: dict,
                        out_path: Path, seed: int, cfg: dict) -> dict:
    is_stage_a = recipe.get("domain") == "Math"
    prompt_count = int(cfg["gates"][
        "phase0_stage_a_preflight_prompts" if is_stage_a
        else "phase0_stage_b_preflight_prompts"])
    min_variable_groups = int(cfg["gates"].get(
        "phase0_stage_a_preflight_min_variable_groups", 1)) if is_stage_a else 1
    if out_path.exists():
        return _validate_existing_preflight(
            _read_json(out_path), prompt_count, min_variable_groups)

    rng = random.Random(seed)
    base.set_seed(seed)
    indices = rng.sample(range(len(dataset)), prompt_count)
    groups = []
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    try:
        with base.torch.no_grad():
            for index in indices:
                row = dataset[index]
                enc = tokenizer(
                    row["prompt"], return_tensors="pt", truncation=False).to("cuda")
                prompt_tokens = enc["input_ids"].shape[1]
                if prompt_tokens > recipe["token_filter_max"]:
                    raise RuntimeError("frozen prompt exceeded registered token filter")
                output = model.generate(
                    **enc,
                    do_sample=True,
                    num_return_sequences=recipe["num_generations"],
                    temperature=recipe["temperature"],
                    top_p=recipe["top_p"],
                    max_new_tokens=recipe["max_completion_length"],
                    pad_token_id=tokenizer.pad_token_id,
                )
                generated = output[:, prompt_tokens:]
                completions = tokenizer.batch_decode(
                    generated, skip_special_tokens=True)
                exact = [
                    base.score_completion(
                        completion, row["ground_truth"], row["data_source"],
                        row["extra_info"])
                    for completion in completions
                ]
                boxed = [
                    base.boxed_format_score(completion, row["data_source"])
                    for completion in completions
                ]
                combined = [
                    exact_score + 0.1 * boxed_score
                    for exact_score, boxed_score in zip(exact, boxed, strict=True)
                ] if recipe.get("reward_mode") == "exact_plus_boxed_format_0.1" else exact
                groups.append({
                    "dataset_index": index,
                    "id": row["id"],
                    "prompt_tokens": prompt_tokens,
                    "ground_truth": row["ground_truth"],
                    "registered_rewards": combined,
                    "exact_rewards": exact,
                    "boxed_format_rewards": boxed,
                    "combined_has_variance": len(set(combined)) > 1,
                    "exact_has_variance": len(set(exact)) > 1,
                    "boxed_has_variance": len(set(boxed)) > 1,
                    "completion_tails": [value[-800:] for value in completions],
                })
    finally:
        tokenizer.padding_side = old_side

    combined_variable = sum(group["combined_has_variance"] for group in groups)
    exact_variable = sum(group["exact_has_variance"] for group in groups)
    boxed_variable = sum(group["boxed_has_variance"] for group in groups)
    result = {
        "runner_variant": "4070_instruct_v9",
        "seed": seed,
        "generation_rng_seeded": True,
        "frozen_dataset_indices": indices,
        "n_prompts": len(groups),
        "num_generations": recipe["num_generations"],
        "reward_mode": recipe.get("reward_mode", "exact"),
        "n_exact_correct": int(sum(sum(group["exact_rewards"]) for group in groups)),
        "n_boxed": int(sum(sum(group["boxed_format_rewards"]) for group in groups)),
        "combined_groups_with_reward_variance": combined_variable,
        "exact_groups_with_reward_variance": exact_variable,
        "boxed_groups_with_reward_variance": boxed_variable,
        "minimum_combined_variable_groups": min_variable_groups,
        "gate_pass": combined_variable >= min_variable_groups,
        "has_grpo_signal": combined_variable >= min_variable_groups,
        "groups": groups,
    }
    _write_json_once(out_path, result)
    if not result["gate_pass"]:
        raise RuntimeError(
            "v9 sparse-reward gate STOP: too few frozen groups had variable "
            "registered reward; preserve diagnostic")
    return result


class SnapshottingLocalSafetyCallback(_BASE_LOCAL_SAFETY):
    """Keep terminal weights for measurement without making them resumable."""

    def on_log(self, args, state, control, logs=None, model=None,
               processing_class=None, tokenizer=None, **kwargs):
        existed = self.out_path.exists()
        result = super().on_log(
            args, state, control, logs=logs, model=model,
            processing_class=processing_class, tokenizer=tokenizer, **kwargs)
        if not existed and self.out_path.exists():
            if model is None:
                raise RuntimeError("v9 safety stop fired without a model to snapshot")
            snapshot = self.out_path.parent / f"safety-stop-weights-step-{state.global_step}"
            if snapshot.exists():
                raise RuntimeError(f"refusing to overwrite safety snapshot: {snapshot}")
            snapshot.mkdir(parents=False, exist_ok=False)
            model.save_pretrained(snapshot, safe_serialization=True)
            tok = processing_class or tokenizer
            if tok is not None:
                tok.save_pretrained(snapshot)
            _write_json_once(snapshot / "DIAGNOSTIC_ONLY.json", {
                "step": int(state.global_step),
                "source_safety_stop": str(self.out_path),
                "resume_allowed": False,
                "registered_checkpoint": False,
                "purpose": "post-stop representation and failure analysis only",
            })
        return result


def install_hooks(cfg: dict, config_path: Path) -> None:
    base.prepare = prepare_v9
    base.LocalSafetyCallback = SnapshottingLocalSafetyCallback
    base.sparse_preflight = lambda model, tokenizer, dataset, recipe, out_path, seed: (
        sparse_preflight_v9(
            model, tokenizer, dataset, recipe, out_path, seed, cfg))

    def runtime_manifest_v9(pilot, run_config):
        manifest = _BASE_RUNTIME_MANIFEST(pilot, run_config)
        manifest["exp2_4070_v9_contract"] = validate_contract(cfg, config_path)
        return manifest

    base.runtime_manifest = runtime_manifest_v9


def smoke_v9(cfg: dict) -> dict:
    marker = base.data_marker(cfg, "phase0_smoke_complete")
    if marker.exists():
        result = _read_json(marker)
        for stage in ("stage_a", "stage_b"):
            if stage not in result:
                raise RuntimeError("existing v9 smoke marker is incomplete")
        return result
    return _BASE_SMOKE(cfg)


def stage1_v9(cfg: dict):
    target = base.run_dir(cfg)
    if target.exists():
        if (target / "phase1_complete.json").exists():
            return base.stage1(cfg)
        raise RuntimeError(
            f"v9 output already exists but is incomplete; refusing automatic "
            f"resume or overwrite: {target}")
    return base.stage1(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action", choices=("contract", "prepare", "smoke", "stage1", "status"),
        required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = base.load_config(config_path)
    cfg["_config_path"] = str(config_path)
    contract = validate_contract(cfg, config_path)
    install_hooks(cfg, config_path)
    if args.action == "contract":
        print(json.dumps(contract, indent=2))
    elif args.action == "prepare":
        prepare_v9(cfg)
    elif args.action == "smoke":
        smoke_v9(cfg)
    elif args.action == "stage1":
        stage1_v9(cfg)
    else:
        base.status(cfg)


if __name__ == "__main__":
    main()
