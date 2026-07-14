"""Regression tests for fixed-budget Stage-B repeat completion contracts."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from src.adaptation import (fixed_budget_completion,
                            validate_adaptation_completion)
from src.repeats import (EXPECTED_CONFIG_HASH, acquire_repeat_lock,
                         ensure_repeat_manifest, frozen_repeat_recipe,
                         repeat_output_dir, sha256_file,
                         validate_repeat_directory)


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8")


def _complete_artifacts(out_dir, actual_updates=50, curve_steps=None,
                        sentinel_steps=None, dashboard_steps=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = [10, 20, 30, 40, 50]
    (out_dir / "summary.json").write_text(json.dumps({
        "requested_updates": 50,
        "actual_updates": actual_updates,
        "completion_status": "complete",
    }), encoding="utf-8")
    _write_jsonl(out_dir / "svamp_eval_curve.jsonl", [
        {"step": step, "accuracy": 0.5} for step in (curve_steps or steps)
    ])
    _write_jsonl(out_dir / "update_sentinel.jsonl", [
        {"step": step, "updates_effective": True}
        for step in (sentinel_steps or steps)
    ])
    _write_jsonl(out_dir / "dashboard.jsonl", [
        {"step": step, "loss": 0.1, "grad_norm": 1.0}
        for step in (dashboard_steps or list(range(1, 51)))
    ])


def _valid_repeat_summary(checkpoint, seed=43):
    return {
        "seed": seed,
        "task": "SVAMP",
        "train_questions": 256,
        "eval_questions": 100,
        "algo": "grpo",
        "checkpoint": str(checkpoint),
        "budget_updates": 50,
        "requested_updates": 50,
        "actual_updates": 50,
        "completion_status": "complete",
        "acc_before": 0.53,
        "acc_after": 0.55,
        "delta_acc": 0.02,
        "wall_seconds": 1.0,
        "learning_rate": 1e-6,
        "beta": 0.0,
        "temperature": 0.7,
        "top_p": 1.0,
        "num_generations": 8,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 16,
        "max_prompt_length": 512,
        "max_completion_length": 512,
        "device": "cuda",
        "dtype": "float32",
        "autocast_dtype": "bfloat16",
        "optim": "paged_adamw_8bit",
        "gradient_checkpointing": True,
        "bf16": True,
    }


def test_early_trainer_writes_incomplete_and_fails(tmp_path):
    trainer = SimpleNamespace(state=SimpleNamespace(global_step=30))
    with pytest.raises(RuntimeError, match="requested 50.*completed 30"):
        fixed_budget_completion(trainer, tmp_path, 50)
    record = json.loads((tmp_path / "incomplete.json").read_text())
    assert record["requested_updates"] == 50
    assert record["actual_updates"] == 30
    assert record["completion_status"] == "incomplete"
    assert not (tmp_path / "summary.json").exists()


def test_exact_budget_returns_complete_metadata(tmp_path):
    trainer = SimpleNamespace(state=SimpleNamespace(global_step=50))
    assert fixed_budget_completion(trainer, tmp_path, 50) == {
        "requested_updates": 50,
        "actual_updates": 50,
        "completion_status": "complete",
    }


def test_validator_rejects_actual_updates_30(tmp_path):
    _complete_artifacts(tmp_path, actual_updates=30)
    with pytest.raises(RuntimeError, match="actual_updates"):
        validate_adaptation_completion(tmp_path)


def test_seed_output_directories_are_isolated(tmp_path):
    seed43 = repeat_output_dir(tmp_path, 43, 0)
    seed44 = repeat_output_dir(tmp_path, 44, 0)
    legacy = tmp_path / "adaptation" / "ckpt-0"
    assert len({seed43, seed44, legacy}) == 3
    assert seed43.parts[-3:] == ("adaptation_repeats", "seed-43", "ckpt-0")


def test_safety_stopped_directory_cannot_validate_or_resume(tmp_path):
    _complete_artifacts(tmp_path)
    (tmp_path / "safety_stop.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="safety stop"):
        validate_repeat_directory(tmp_path)


@pytest.mark.parametrize("artifact", ["curve", "sentinel"])
def test_validator_rejects_missing_step_40_or_50(tmp_path, artifact):
    kwargs = {f"{artifact}_steps": [10, 20, 30]}
    _complete_artifacts(tmp_path, **kwargs)
    with pytest.raises(RuntimeError, match="steps"):
        validate_adaptation_completion(tmp_path)


def test_partial_repeat_directory_refuses_overwrite(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "dashboard.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="partial or failed"):
        validate_repeat_directory(tmp_path)


def test_validator_rejects_missing_dashboard_step(tmp_path):
    _complete_artifacts(tmp_path, dashboard_steps=list(range(1, 50)))
    with pytest.raises(RuntimeError, match="dashboard training steps"):
        validate_adaptation_completion(tmp_path)


@pytest.mark.parametrize("key", ["loss", "grad_norm"])
def test_validator_rejects_nonfinite_dashboard_value(tmp_path, key):
    _complete_artifacts(tmp_path)
    rows = [json.loads(line) for line in
            (tmp_path / "dashboard.jsonl").read_text().splitlines()]
    rows[24][key] = float("nan")
    _write_jsonl(tmp_path / "dashboard.jsonl", rows)
    with pytest.raises(RuntimeError, match=f"non-finite {key}"):
        validate_adaptation_completion(tmp_path)


def test_repeat_identity_rejects_wrong_seed(tmp_path):
    _complete_artifacts(tmp_path)
    recipe = frozen_repeat_recipe()
    checkpoint = tmp_path / "source" / "ckpt-0"
    (tmp_path / "summary.json").write_text(json.dumps(
        _valid_repeat_summary(checkpoint, seed=44)))
    with pytest.raises(RuntimeError, match="summary seed"):
        validate_repeat_directory(
            tmp_path, seed=43, checkpoint_path=checkpoint, recipe=recipe,
            expected_acc_before=0.53)


def test_existing_repeat_requires_finalized_telemetry(tmp_path):
    _complete_artifacts(tmp_path)
    recipe = frozen_repeat_recipe()
    checkpoint = tmp_path / "source" / "ckpt-0"
    (tmp_path / "summary.json").write_text(json.dumps(
        _valid_repeat_summary(checkpoint)))
    kwargs = {
        "seed": 43,
        "checkpoint_path": checkpoint,
        "recipe": recipe,
        "expected_acc_before": 0.53,
    }
    with pytest.raises(RuntimeError, match="telemetry is missing"):
        validate_repeat_directory(tmp_path, **kwargs)
    (tmp_path / "gpu_20260713_stageb_seed43_ckpt0.csv").write_text(
        "timestamp,memory\n2026-07-13,100\n")
    assert validate_repeat_directory(tmp_path, **kwargs)["seed"] == 43


def test_existing_repeat_rejects_unfinalized_attempt_marker(tmp_path):
    _complete_artifacts(tmp_path)
    (tmp_path / ".repeat_attempt.json").write_text(json.dumps({
        "attempt_id": "attempt_1234",
    }))
    with pytest.raises(RuntimeError, match="not finalized"):
        validate_repeat_directory(tmp_path)


def test_repeat_lock_rejects_live_owner(tmp_path):
    (tmp_path / "stageb_repeat.lock").write_text(str(os.getpid()))
    with pytest.raises(RuntimeError, match="another Stage-B repeat runner"):
        acquire_repeat_lock(tmp_path)


def test_repeat_lock_is_created_atomically(tmp_path):
    lock = acquire_repeat_lock(tmp_path)
    assert lock.read_text() == str(os.getpid())
    lock.unlink()


def test_manifest_refreshes_sha_only_before_first_complete_checkpoint(
        tmp_path, monkeypatch):
    import src.repeats as repeats

    source = tmp_path / "source"
    source.mkdir()
    for name in ("config.json", "manifest.json"):
        (source / name).write_text("{}")
    recipe = frozen_repeat_recipe()
    root = source / "adaptation_repeats" / "seed-43"
    root.mkdir(parents=True)
    manifest_path = root / "repeat_manifest.json"
    manifest_path.write_text(json.dumps({
        "source_sha256": {
            name: sha256_file(source / name)
            for name in ("config.json", "manifest.json")
        },
        "source_config_hash": EXPECTED_CONFIG_HASH,
        "seed": 43,
        "recipe": recipe,
        "git_sha": "old-sha",
    }))
    monkeypatch.setattr(repeats, "_git_sha", lambda repo: "fixed-sha")
    monkeypatch.setattr(repeats, "runtime_versions", lambda: {"torch": "test"})
    refreshed = ensure_repeat_manifest(source, 43, recipe, tmp_path, "RTX 4070")
    assert refreshed["git_sha"] == "fixed-sha"
    assert refreshed["git_sha_history"] == ["old-sha"]

    completed = root / "ckpt-0"
    completed.mkdir()
    (completed / "summary.json").write_text("{}")
    monkeypatch.setattr(repeats, "_git_sha", lambda repo: "later-artifact-sha")
    pinned = ensure_repeat_manifest(source, 43, recipe, tmp_path, "RTX 4070")
    assert pinned["git_sha"] == "fixed-sha"
